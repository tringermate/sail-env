"""Solo start environment (Phase 1-3), gymnax-style functional API.

    env = SoloStartEnv()
    obs, state = env.reset(key, params)
    obs, state, reward, done, info = env.step(key, state, action, params)

`step` auto-resets: when an episode ends, the returned obs/state already belong to a fresh episode,
while reward/done/info describe the step that just finished (as in gymnax). `step_env` is the raw
step without the reset.

Episode: clock runs T-60 .. T+10 s, one decision per second (70 steps). Terminal reward:
+1 success, -ocs_penalty if still OCS at the end, 0 otherwise.

Optional potential-based shaping (Ng, Harada & Russell 1999), coefficient `reward_shaping`:
    Phi(s) = -| time_to_line(s) - time_to_gun(s) | / 60        (0 once legally started)
    time_to_line = distance below the line / best close-hauled VMG at the current wind speed
                   [+ tau_accel * (1 - current VMG towards the line / best VMG)  if shaping_speed_aware]
The optional term is the time a slow boat loses getting up to speed: with the first-order speed lag,
distance covered after t seconds is ~ V t - (V - v0) tau, so arrival is delayed by tau (1 - v0/V).
    F(s, s') = gamma * Phi(s') * (1 - done) - Phi(s)
Summed over an episode the F terms telescope to (roughly) -Phi(s_0), a constant of the start
state, so they cannot change which policy is optimal; they only tell the agent earlier whether a
step brought it closer to "arriving at the line with the gun".

Success: legal start between T=0 and T+success_window, at >= success_speed_frac of polar target
speed at the moment of crossing (speed measured at the end of the crossing step).
"""

import jax
import jax.numpy as jnp
from flax import struct

from sail_env.config import EnvParams
from sail_env.geometry import heading_vector, signed_twa
from sail_env.physics import N_ACTIONS, BoatState, bow_position, physics_step
from sail_env.polar import beat_vmg, load_polar, polar_speed
from sail_env.rules import Line, StartStatus, init_status, line_frame, make_line, signed_distance, update_status
from sail_env.weather import (WEATHER_FEATURES, Weather, any_weather, current_velocity, initial_log, push_log,
                              sample_weather, water_wind, wave_speed_factor, wave_state)
from sail_env.weather import observe as observe_weather
from sail_env.wind import Wind, sample_wind, wind_at

# Observation layout: OWN_FEATURES about the boat itself, then N_NEIGHBOURS slots of
# NEIGHBOUR_FEATURES for the nearest other boats. In solo training the slots are all zeros; they
# exist so the same network can later be fine-tuned in the fleet environment (Phase 4).
OWN_FEATURES = 15
N_NEIGHBOURS = 5
NEIGHBOUR_FEATURES = 9  # present, rel. fwd, rel. stbd, sin/cos rel. heading, speed, tack, I have ROW, in their shadow
OBS_DIM = OWN_FEATURES + N_NEIGHBOURS * NEIGHBOUR_FEATURES


@struct.dataclass
class EnvState:
    boat: BoatState
    status: StartStatus
    wind: Wind
    line: Line
    bias: jax.Array  # line bias, radians (> 0 committee end upwind)
    t: jax.Array  # clock, seconds relative to the gun
    start_speed_ratio: jax.Array  # speed / polar at the legal crossing (0 until started)
    start_on_starboard: jax.Array  # bool: tack at the legal crossing (False until started)
    gun_below: jax.Array  # bow distance below the line when the gun went (m; < 0 = over), nan before
    gun_inside: jax.Array  # bool: bow between the ends (along the line) when the gun went
    rule_w: jax.Array  # rule strictness of this episode in [0, 1] (1 = full rules; used with per_episode_rules)
    vmg_sum: jax.Array  # sum over steps after the legal start of min(VMG / best VMG, 1)
    vmg_steps: jax.Array  # number of those steps
    weather: Weather  # shifts, pressure, current, waves (weather.py); unused unless switched on in EnvParams
    wind_log: jax.Array  # (60, 2) wind the boat sailed in, 1..60 s ago (direction, speed); (0, 2) without observe_weather
    log_origin: jax.Array  # (3,) x, y, t where the episode (and the log) began; (0,) without observe_weather


RULE_FEATURES = 5  # appended with observe_rules: window, speed fraction, gun distance, OCS penalty, port penalty


class SoloStartEnv:
    def __init__(self, params: EnvParams | None = None):
        params = params or EnvParams()
        self.polar = load_polar(params.no_go)  # constant arrays, closed over by jit
        self._obs_dim = (OBS_DIM + (RULE_FEATURES if params.observe_rules else 0)
                         + (WEATHER_FEATURES if params.observe_weather else 0))

    @property
    def num_actions(self) -> int:
        return N_ACTIONS

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @staticmethod
    def episode_params(state: EnvState, params: EnvParams) -> EnvParams:
        """The rules of this episode: interpolated between easy and full by state.rule_w."""
        if not params.per_episode_rules:  # static
            return params
        w = state.rule_w

        def lerp(easy, full):
            return easy + w * (full - easy)

        return params.replace(
            success_window=lerp(params.easy_success_window, params.success_window),
            success_speed_frac=lerp(params.easy_success_speed_frac, params.success_speed_frac),
            gun_distance_max=lerp(params.easy_gun_distance_max, params.gun_distance_max),
            ocs_penalty=lerp(params.easy_ocs_penalty, params.ocs_penalty),
            port_penalty=lerp(params.easy_port_penalty, params.port_penalty),
            speed_bonus=lerp(params.easy_speed_bonus, params.speed_bonus),
        )

    # --- reset -------------------------------------------------------------------------------

    def reset(self, key: jax.Array, params: EnvParams):
        k_wind, k_bias, k_dist, k_lat, k_head, k_speed, k_time = jax.random.split(key, 7)
        wind = sample_wind(k_wind, params)
        # Its own key stream, so the base wind, line and boat are the same as without weather.
        weather = sample_weather(jax.random.fold_in(key, 1), wind.base_dir, params, wind.tws)
        bias = jax.random.uniform(k_bias, minval=-params.line_bias_max, maxval=params.line_bias_max)
        line = make_line(0.0, 0.0, params.wind_dir, bias, params.line_length)

        # Start clock: always T-60, or (easier starts in the mix) a random time to gun.
        if params.random_start_time:  # static flag
            # Whole seconds, so the gun falls on a decision-step boundary (as with T-60).
            time_to_gun = jnp.round(
                jax.random.uniform(k_time, minval=params.init_time_to_gun_min, maxval=-params.t_start) / params.dt
            ) * params.dt
        else:
            time_to_gun = jnp.float32(-params.t_start)

        # Position in the line frame: `dist` downwind of the line, `lat` along it from the centre.
        # Both are capped so the line can be reached by the gun (every start is winnable).
        dx, dy, nx, ny, _ = line_frame(line)
        vmg = beat_vmg(self.polar, wind.tws)
        dist_max = jnp.minimum(params.init_dist_max, params.init_reach_frac * vmg * time_to_gun)
        dist_min = jnp.minimum(params.init_dist_min, 0.5 * dist_max)
        dist = jax.random.uniform(k_dist, minval=dist_min, maxval=dist_max)
        lat_max = jnp.minimum(params.init_lateral_max, 0.5 * params.line_length + params.init_lateral_speed * time_to_gun)
        lat = jax.random.uniform(k_lat, minval=-lat_max, maxval=lat_max)
        x, y = lat * dx - dist * nx, lat * dy - dist * ny

        heading = jax.random.uniform(k_head, minval=-jnp.pi, maxval=jnp.pi)
        if any_weather(params):  # static
            wind_dir, tws = water_wind(wind, weather, -time_to_gun, x, y, params)
        else:
            wind_dir, tws = wind_at(wind, -time_to_gun)
        v_polar = polar_speed(self.polar, signed_twa(heading, wind_dir), tws)
        speed = jax.random.uniform(k_speed, maxval=params.init_speed_frac_max) * v_polar

        boat = BoatState(x=x, y=y, heading=heading, speed=speed, target_heading=heading, manoeuvring=jnp.bool_(False))
        state = EnvState(
            boat=boat, status=init_status(), wind=wind, line=line, bias=bias,
            t=-time_to_gun, start_speed_ratio=jnp.float32(0.0), start_on_starboard=jnp.bool_(False),
            gun_below=jnp.float32(jnp.nan), gun_inside=jnp.bool_(False), rule_w=jnp.float32(1.0),
            vmg_sum=jnp.float32(0.0), vmg_steps=jnp.float32(0.0), weather=weather,
            **dict(zip(("wind_log", "log_origin"), initial_log(-time_to_gun, x, y, params))),
        )
        return self.get_obs(state, params), state

    # --- step --------------------------------------------------------------------------------

    def step_env(self, key: jax.Array, state: EnvState, action: jax.Array, params: EnvParams,
                 tws_scale=1.0, extra_ok=True):
        """One decision. The fleet environment reuses this per boat with tws_scale (wind shadow:
        the boat sails in tws_scale x the true wind) and extra_ok (additional success conditions:
        clear air, no unserved penalty). Defaults reproduce the solo environment."""
        del key  # dynamics are deterministic given the reset
        params = self.episode_params(state, params)  # per-episode rules (joint curriculum), if enabled
        phi0 = self.potential(state, params)
        t0 = state.t
        t1 = t0 + params.dt
        wind_dir, tws = self.local_wind(state, params)  # at the boat, over the water
        waves = wave_state(state.weather, params)
        boat, bow_path = physics_step(state.boat, action, self.polar, wind_dir, tws * tws_scale, params,
                                      current=current_velocity(state.weather, params), waves=waves)
        status = update_status(state.status, state.line, bow_path, t0, t1)

        # Speed quality at the moment of the legal crossing.
        started_now = status.started & ~state.status.started
        v_polar = polar_speed(self.polar, signed_twa(boat.heading, wind_dir), tws)
        if waves is not None:  # static: full speed in waves is the polar x the wave factor
            v_polar = v_polar * wave_speed_factor(boat.heading, waves, params)
        ratio = boat.speed / jnp.maximum(v_polar, 1e-3)
        start_speed_ratio = jnp.where(started_now, ratio, state.start_speed_ratio)
        on_starboard = signed_twa(boat.heading, wind_dir) > 0.0
        start_on_starboard = jnp.where(started_now, on_starboard, state.start_on_starboard)

        # Where the bow is when the gun goes (the step whose clock reaches T=0).
        gun_now = (t0 < 0.0) & (t1 >= 0.0)
        bx, by = bow_position(boat, params)
        ldx, ldy, _, _, llen = line_frame(state.line)
        along = (bx - state.line.pin_x) * ldx + (by - state.line.pin_y) * ldy
        gun_below = jnp.where(gun_now, -signed_distance(state.line, bx, by), state.gun_below)
        gun_inside = jnp.where(gun_now, (along >= 0.0) & (along <= llen), state.gun_inside)
        # The instruments log the wind this boat sailed in during this second, where it was.
        state = state.replace(wind_log=push_log(state.wind_log, wind_dir, tws))
        state = state.replace(boat=boat, status=status, t=t1, start_speed_ratio=start_speed_ratio,
                              start_on_starboard=start_on_starboard, gun_below=gun_below, gun_inside=gun_inside)
        vmg_frac = self.vmg_fraction(state)  # velocity towards the line / best VMG (speed bonus)
        up_frac = self.upwind_fraction(state, params)  # VMG along the current wind axis / best VMG, capped at 1
        state = state.replace(vmg_sum=state.vmg_sum + jnp.where(status.started, up_frac, 0.0),
                              vmg_steps=state.vmg_steps + status.started.astype(jnp.float32))
        success = self.is_success(state, params) & extra_ok
        done = t1 >= params.t_end - 1e-6
        efficiency = self.upwind_efficiency(state, params)
        success_reward = 1.0 - params.port_penalty * (~start_on_starboard)  # port start: reduced reward
        success_reward = success_reward + params.upwind_bonus * efficiency  # height gained after the start
        terminal = jnp.where(done, jnp.where(success, success_reward, jnp.where(status.ocs, -params.ocs_penalty, 0.0)), 0.0)
        shaping = params.shaping_gamma * self.potential(state, params) * (1.0 - done) - phi0
        in_window = (t1 >= params.speed_bonus_from) & (t1 <= params.speed_bonus_until)
        bonus = params.speed_bonus * vmg_frac * in_window
        ramp = jnp.clip((t1 - params.starboard_bonus_from) / jnp.maximum(-params.starboard_bonus_from, 1e-6), 0.0, 1.0)
        starboard_approach = params.starboard_bonus * ramp * (on_starboard & ~status.started)
        terminal = params.start_reward_weight * terminal
        reward = terminal + params.reward_shaping * shaping + bonus + starboard_approach
        s = status
        strict = (s.started & (s.start_time >= 0.0) & (s.start_time <= params.strict_success_window)
                  & (start_speed_ratio >= params.strict_success_speed_frac) & extra_ok)
        on_line_strict = (gun_below >= 0.0) & (gun_below <= params.strict_gun_distance_max) & gun_inside
        info = {
            "success": success & done,
            "success_strict": strict & done,
            "good_start": strict & start_on_starboard & done,  # full rules and on starboard
            "good_start_on_line": strict & start_on_starboard & on_line_strict & done,  # ... and 0-3 m at the gun
            "on_line_at_gun": on_line_strict & done,
            "start_on_starboard": status.started & start_on_starboard & done,
            "upwind_efficiency": jnp.where(done & status.started, efficiency, 0.0),
            "ocs": status.ocs & done,
            "started": status.started & done,
            "start_time": status.start_time,
            "start_speed_ratio": start_speed_ratio,
        }
        return self.get_obs(state, params), state, reward.astype(jnp.float32), done, info

    def step(self, key: jax.Array, state: EnvState, action: jax.Array, params: EnvParams):
        """step_env + auto-reset on done (gymnax convention)."""
        k_step, k_reset = jax.random.split(key)
        obs, state, reward, done, info = self.step_env(k_step, state, action, params)
        obs_re, state_re = self.reset(k_reset, params)
        state = jax.tree.map(lambda a, b: jnp.where(done, a, b), state_re, state)
        obs = jnp.where(done, obs_re, obs)
        return obs, state, reward, done, info

    # --- wind, success and observation -------------------------------------------------------

    @staticmethod
    def local_wind(state: EnvState, params: EnvParams):
        """(direction, speed) of the wind the boat sails in: over the water at its position (weather.py).
        Without weather this is the uniform base wind, exactly as in sail_env 0.1."""
        if not any_weather(params):  # static
            return wind_at(state.wind, state.t)
        return water_wind(state.wind, state.weather, state.t, state.boat.x, state.boat.y, params)

    def vmg_fraction(self, state: EnvState) -> jax.Array:
        """Velocity towards the line as a fraction of the best close-hauled VMG, clipped to [0, 1]."""
        _, _, nx, ny, _ = line_frame(state.line)
        fx, fy = heading_vector(state.boat.heading)
        v_towards = state.boat.speed * (fx * nx + fy * ny)
        return jnp.clip(v_towards / jnp.maximum(beat_vmg(self.polar, state.wind.tws), 0.1), 0.0, 1.0)

    def upwind_fraction(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Upwind VMG (along the current true wind axis) as a fraction of the best close-hauled VMG, in [0, 1]."""
        wind_dir, _ = self.local_wind(state, params)
        fx, fy = heading_vector(state.boat.heading)
        v_up = state.boat.speed * (fx * jnp.sin(wind_dir) + fy * jnp.cos(wind_dir))
        return jnp.clip(v_up / jnp.maximum(beat_vmg(self.polar, state.wind.tws), 0.1), 0.0, 1.0)

    def upwind_efficiency(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Average over the seconds since the legal start of min(VMG / best close-hauled VMG, 1), in [0, 1].

        Capping each second at the polar optimum means momentum tricks cannot pay: shooting head to
        wind scores at most 1 while coasting, then less and less as the boat stalls.
        """
        tail = params.efficiency_tail_seconds  # extra seconds at the current VMG (the boat keeps sailing)
        total = state.vmg_sum + tail * self.upwind_fraction(state, params)
        return jnp.where(state.status.started, total / jnp.maximum(state.vmg_steps + tail, 1.0), 0.0)

    def potential(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Phi(s) = -|time_to_line - time_to_gun| / 60 before a legal start, 0 after it."""
        bx, by = bow_position(state.boat, params)
        below = -signed_distance(state.line, bx, by)  # metres below the line (< 0 if over)
        vmg = jnp.maximum(beat_vmg(self.polar, state.wind.tws), 0.1)
        time_to_line = below / vmg
        if params.shaping_speed_aware:  # static flag: chosen when the env is compiled
            _, _, nx, ny, _ = line_frame(state.line)
            fx, fy = heading_vector(state.boat.heading)
            v_towards = state.boat.speed * (fx * nx + fy * ny)  # velocity component towards the line
            missing = 1.0 - jnp.clip(v_towards / vmg, 0.0, 1.0)
            time_to_line = time_to_line + params.tau_accel * missing
        time_to_gun = jnp.maximum(-state.t, 0.0)
        phi = -jnp.abs(time_to_line - time_to_gun) / 60.0
        return jnp.where(state.status.started, 0.0, phi)

    @staticmethod
    def is_success(state: EnvState, params: EnvParams) -> jax.Array:
        s = state.status
        in_window = (s.start_time >= 0.0) & (s.start_time <= params.success_window)
        on_line = (state.gun_below >= 0.0) & (state.gun_below <= params.gun_distance_max) & state.gun_inside
        on_line = on_line | (params.gun_distance_max >= 1e6)  # check disabled
        return s.started & in_window & (state.start_speed_ratio >= params.success_speed_frac) & on_line

    def get_obs(self, state: EnvState, params: EnvParams) -> jax.Array:
        """OWN_FEATURES (15), all in the boat's own frame and scaled to roughly [-1, 1]:

        speed, sin/cos TWA, tack (+1 stbd / -1 port), time to gun, pin and committee vectors
        (forward, starboard), signed distance to the line, line bias, manoeuvring, started, OCS,
        true wind speed; then N_NEIGHBOURS x NEIGHBOUR_FEATURES zeros (no other boats in solo).
        Optional blocks after that: the episode's rules (observe_rules), then the weather (observe_weather,
        weather.observe; always last, also in the race). Wind and TWA are the wind at the boat (local_wind).
        """
        b, line = state.boat, state.line
        wind_dir, tws = self.local_wind(state, params)
        twa = signed_twa(b.heading, wind_dir)
        fx, fy = heading_vector(b.heading)  # forward unit vector
        rx, ry = fy, -fx  # starboard (right) unit vector

        def to_boat_frame(px, py):
            ddx, ddy = px - b.x, py - b.y
            return (ddx * fx + ddy * fy) / 100.0, (ddx * rx + ddy * ry) / 100.0

        pin_f, pin_r = to_boat_frame(line.pin_x, line.pin_y)
        cb_f, cb_r = to_boat_frame(line.cb_x, line.cb_y)
        return jnp.stack([
            b.speed / 5.0,
            jnp.sin(twa),
            jnp.cos(twa),
            jnp.where(twa >= 0, 1.0, -1.0),
            -state.t / 60.0,
            pin_f, pin_r, cb_f, cb_r,
            signed_distance(line, b.x, b.y) / 100.0,
            state.bias / jnp.maximum(params.line_bias_max, 1e-6),
            b.manoeuvring.astype(jnp.float32),
            state.status.started.astype(jnp.float32),
            state.status.ocs.astype(jnp.float32),
            tws / 8.0,  # a sailor knows the wind strength; full speed depends on it
            *([jnp.float32(0.0)] * (N_NEIGHBOURS * NEIGHBOUR_FEATURES)),
            *self._rule_features(state, params),
            *self._weather_features(state, params),
        ]).astype(jnp.float32)

    @staticmethod
    def _weather_features(state, params):
        """With observe_weather: WEATHER_FEATURES (weather.observe), after the rule features."""
        if not params.observe_weather:  # static
            return []
        b = state.boat
        return observe_weather(state.wind, state.weather, state.wind_log, state.log_origin, state.t, b.x, b.y, b.heading,
                               params)

    def _rule_features(self, state, params):
        """With observe_rules: this episode's rule values, scaled to ~[0, 1] (after the neighbour slots)."""
        if not params.observe_rules:  # static
            return []
        ep = self.episode_params(state, params)
        return [ep.success_window / 10.0, ep.success_speed_frac, jnp.minimum(ep.gun_distance_max, 30.0) / 30.0,
                ep.ocs_penalty, ep.port_penalty]
