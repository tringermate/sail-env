"""Fleet start environment (Phase 4): several boats, one shared policy, gymnax-style API.

    env = FleetStartEnv(params)                       # params.fleet_size boats
    obs, state = env.reset(key, params)               # obs: (N, OBS_DIM)
    obs, state, reward, done, info = env.step(key, state, actions, params)   # actions (N,), reward/done (N,)

Each boat is simulated by the solo environment's step (same physics, polar, start rules, success
and reward), with three fleet additions, every decision:
    1. wind shadow: each boat sails in (min over other boats' shadows) x the true wind;
    2. a boat serving a penalty is forced to luff (sails eased, no steering) for penalty_seconds, or
       (penalty_turn) turns a full circle, at once or after getting clear (penalty_clear_distance);
    3. after the move, new contacts are checked; the boat that had to keep clear is penalised
       (both if no rule separates them).
A good fleet start also needs clear air (no shadow) at the end and no penalty left to serve; a
penalty still unserved at the end costs -foul_penalty.

Observation: the solo features, plus for each of the 5 nearest other boats (zero-padded):
present, relative position (forward, starboard; /100 m), sin/cos relative heading, speed /5,
tack (+1 stbd / -1 port), "I have right of way over it", "I am in its wind shadow".
"""

import jax
import jax.numpy as jnp
from flax import struct

from sail_env.config import EnvParams
from sail_env.fleet_rules import contacts, must_keep_clear, shadow_factors
from sail_env.geometry import heading_vector, signed_twa, wrap_angle
from sail_env.physics import HOLD, LUFF, bow_position
from sail_env.polar import beat_vmg
from sail_env.rules import line_frame, signed_distance
from sail_env.start_env import N_NEIGHBOURS, NEIGHBOUR_FEATURES, OBS_DIM, OWN_FEATURES, EnvState, SoloStartEnv
from sail_env.weather import any_weather, reference_wind
from sail_env.wind import wind_at


@struct.dataclass
class FleetState:
    boats: EnvState  # every field stacked over boats (shared fields such as wind, line, t repeated)
    penalty_left: jax.Array  # (N,) penalty still to serve: seconds (luff) or radians of turning (penalty_turn)
    penalty_dir: jax.Array  # (N,) +1 / -1: direction of the penalty turn
    penalty_turning: jax.Array  # (N,) bool: the penalty turn is under way (actions ignored)
    penalty_wait: jax.Array  # (N,) s since the penalty while not yet turning (getting clear)
    in_contact: jax.Array  # (N, N) bool, contact at the previous step (a contact is penalised once)
    fouls: jax.Array  # (N,) number of penalties incurred this episode
    mark_contact: jax.Array  # (N, M) bool, touching each mark at the previous step (M = env.n_marks)
    contacts: jax.Array  # (N,) number of contacts (boats or marks) this boat was involved in
    mark_hits: jax.Array  # (N,) number of times this boat touched a mark
    # Used by the full race (RaceEnv); unused by the start environments.
    leg: jax.Array  # (N,) int: 0 beat 1, 1 run 1, 2 beat 2, 3 run 2 to the finish, 4 finished
    finish_time: jax.Array  # (N,) clock at the finish (inf until finished)
    in_zone: jax.Array  # (N,) bool: inside the 3-length zone of the mark being rounded
    room: jax.Array  # (N, N) bool: [i, j] = i must give j mark-room
    turned: jax.Array  # (N,) rad, heading change during the previous step (RRS 16: changing course)
    gate_through: jax.Array  # (N,) bool: on the run, through the leeward gate, still to round a gate mark
    mark_through: jax.Array  # (N,) bool: on a beat, across the line upwind of the windward mark, still to get round it
    # Fleet-racing measures (reported in info at the end of an episode; no effect on the dynamics).
    shadow_time: jax.Array  # (N,) s sailed after the gun in another boat's wind shadow
    mark1_order: jax.Array  # (N,) int: 0 = first round the first windward mark, ...; -1 = not yet (race only)


def _first(x):
    return jnp.broadcast_to(x[0], x.shape)


def rules_boats(boats, taking, wind_dir, params: EnvParams):
    """Boats as the right-of-way rules see them: `manoeuvring` = keeps clear like a tacking boat (RRS 13),
    which includes a boat taking a penalty (RRS 22.2). With rrs13_tack_only a gybe does not count."""
    turning = boats.manoeuvring
    if params.rrs13_tack_only:  # static
        turning = turning & (jnp.abs(signed_twa(boats.heading, wind_dir)) < 0.5 * jnp.pi)
    return boats.replace(manoeuvring=turning | taking)


class FleetStartEnv:
    def __init__(self, params: EnvParams):
        self.solo = SoloStartEnv(params)
        self.polar = self.solo.polar
        self.num_agents = params.fleet_size
        self.num_actions = self.solo.num_actions
        self.obs_dim = self.solo.obs_dim
        self.n_marks = 2  # pin and committee boat; the race adds its course marks

    # --- reset ---------------------------------------------------------------------------------

    def reset(self, key, params: EnvParams):
        n = self.num_agents
        k_boats, k_perm, k_jit, k_dist = jax.random.split(key, 4)
        _, st = jax.vmap(self.solo.reset, in_axes=(0, None))(jax.random.split(k_boats, n), params)
        # One wind, line and clock for everybody (taken from boat 0).
        st = st.replace(wind=jax.tree.map(_first, st.wind), line=jax.tree.map(_first, st.line),
                        bias=_first(st.bias), t=_first(st.t), weather=jax.tree.map(_first, st.weather))
        # Spread the boats along the line so they never start overlapping: one slot each across the
        # allowed width, shuffled, with jitter of a third of the slot spacing.
        time_to_gun = -st.t[0]
        vmg = beat_vmg(self.polar, st.wind.tws[0])
        dist_max = jnp.minimum(params.init_dist_max, params.init_reach_frac * vmg * time_to_gun)
        dist_min = jnp.minimum(params.init_dist_min, 0.5 * dist_max)
        lat_max = jnp.minimum(params.init_lateral_max, 0.5 * params.line_length + params.init_lateral_speed * time_to_gun)
        slots = (jnp.linspace(-1.0, 1.0, n) if n > 1 else jnp.zeros(1)) * lat_max  # a single boat starts mid-line
        spacing = 2.0 * lat_max / max(n - 1, 1)
        lat = jax.random.permutation(k_perm, slots) + jax.random.uniform(k_jit, (n,), minval=-1, maxval=1) * spacing / 3
        dist = jax.random.uniform(k_dist, (n,), minval=dist_min, maxval=dist_max)
        line0 = jax.tree.map(lambda x: x[0], st.line)
        dx, dy, nx, ny, _ = line_frame(line0)
        st = st.replace(boat=st.boat.replace(x=lat * dx - dist * nx, y=lat * dy - dist * ny))
        fs = FleetState(boats=st, penalty_left=jnp.zeros(n), penalty_dir=jnp.ones(n),
                        penalty_turning=jnp.zeros(n, bool), penalty_wait=jnp.zeros(n), in_contact=jnp.zeros((n, n), bool),
                        fouls=jnp.zeros(n, jnp.int32), mark_contact=jnp.zeros((n, self.n_marks), bool),
                        contacts=jnp.zeros(n, jnp.int32), mark_hits=jnp.zeros(n, jnp.int32),
                        leg=jnp.zeros(n, jnp.int32), finish_time=jnp.full(n, jnp.inf), in_zone=jnp.zeros(n, bool),
                        room=jnp.zeros((n, n), bool), turned=jnp.zeros(n), gate_through=jnp.zeros(n, bool),
                        mark_through=jnp.zeros(n, bool),
                        shadow_time=jnp.zeros(n),
                        mark1_order=jnp.full(n, -1, jnp.int32))
        return self.get_obs(fs, params), fs

    # --- step ----------------------------------------------------------------------------------

    # --- hooks (overridden by the full race) ------------------------------------------------------

    def marks(self, fs: FleetState, params: EnvParams):
        """(centres (M, 2), radii (M,)) of the marks that are obstacles: here pin and committee boat."""
        line = jax.tree.map(lambda x: x[0], fs.boats.line)
        centres = jnp.stack([jnp.stack([line.pin_x, line.pin_y]), jnp.stack([line.cb_x, line.cb_y])])
        return centres, jnp.array([params.pin_radius, params.committee_radius])

    def keep_clear_matrix(self, fs: FleetState, boats, wind_dir, params: EnvParams):
        """[i, j] = boat i must keep clear of boat j (right of way; the race adds mark-room)."""
        return must_keep_clear(boats, wind_dir, params.boat_length)

    def post_step(self, fs_prev: FleetState, fs: FleetState, reward, done, info, params: EnvParams):
        """Extra logic after each step (the race: roundings, finishing). Returns (fs, reward, done, info)."""
        return fs, reward, done, info

    def mark_contacts(self, fs: FleetState, boats, params: EnvParams):
        """(N, M) bool: boat capsule touches a mark circle."""
        from sail_env.fleet_rules import bow_stern, point_segment_distance

        centres, radii = self.marks(fs, params)
        (bx, by), (sx, sy) = bow_stern(boats, params.boat_length)
        d = point_segment_distance(centres[None, :, 0], centres[None, :, 1],
                                   sx[:, None], sy[:, None], bx[:, None], by[:, None])
        return d < 0.5 * params.boat_beam + radii[None, :]

    # --- step ----------------------------------------------------------------------------------

    def local_winds(self, st, params: EnvParams):
        """(N,) direction and speed of the wind each boat sails in. Without weather: one shared wind, computed
        exactly as in sail_env 0.1 (a scalar; vectorised sin/cos can differ in the last bit)."""
        if not any_weather(params):  # static
            return wind_at(jax.tree.map(lambda x: x[0], st.wind), st.t[0])
        return jax.vmap(self.solo.local_wind, in_axes=(0, None))(st, params)

    def step_env(self, key, fs: FleetState, actions, params: EnvParams):
        st = fs.boats
        wind_dir, tws = self.local_winds(st, params)
        scale = shadow_factors(st.boat, wind_dir, params, tws).min(axis=1)  # wind shadow per boat
        pending = fs.penalty_left > 0.0
        if params.penalty_turn:  # static: turn the circle, one decision at a time
            # Start the turn once clear of other boats and marks (or at once / after the timeout).
            d = params.penalty_clear_distance
            b0 = st.boat
            gap_b = jnp.hypot(b0.x[None, :] - b0.x[:, None], b0.y[None, :] - b0.y[:, None])
            gap_b = jnp.where(jnp.eye(self.num_agents, dtype=bool), jnp.inf, gap_b).min(axis=1)
            centres, radii = self.marks(fs, params)
            gap_m = (jnp.hypot(centres[None, :, 0] - b0.x[:, None], centres[None, :, 1] - b0.y[:, None])
                     - radii[None, :]).min(axis=1)
            clear = (gap_b > d) & (gap_m > d)
            serving = pending & (fs.penalty_turning | clear | (fs.penalty_wait >= params.penalty_clear_timeout) | (d <= 0.0))
            # Turn direction: start by bearing away (away from the wind).
            away0 = -jnp.sign(signed_twa(b0.heading, wind_dir))
            penalty_dir0 = jnp.where(serving & ~fs.penalty_turning & (d > 0.0), jnp.where(away0 == 0.0, 1.0, away0),
                                     fs.penalty_dir)
            wait = jnp.where(pending & ~serving, fs.penalty_wait + params.dt, 0.0)
            turn = jnp.minimum(params.penalty_turn_rate * params.dt, fs.penalty_left)
            h_new = wrap_angle(st.boat.heading + penalty_dir0 * turn)
            b = st.boat
            st = st.replace(boat=b.replace(
                heading=jnp.where(serving, h_new, b.heading),
                target_heading=jnp.where(serving, h_new, b.target_heading),
                manoeuvring=jnp.where(serving, False, b.manoeuvring)))
            actions = jnp.where(serving, HOLD, actions)
            left = fs.penalty_left - jnp.where(serving, turn, 0.0)
            will_be_served = left <= 0.0
        else:
            serving = pending
            penalty_dir0 = fs.penalty_dir
            wait = jnp.zeros_like(fs.penalty_wait)
            actions = jnp.where(serving, LUFF, actions)  # penalty: sails eased, no steering
            left = jnp.maximum(fs.penalty_left - params.dt, 0.0)
            will_be_served = fs.penalty_left <= params.dt
        clear_air = scale >= 1.0
        _, st2, reward, done, info = jax.vmap(self.solo.step_env, in_axes=(None, 0, 0, None, 0, 0))(
            key, st, actions, params, scale, clear_air & will_be_served
        )

        # Contacts after the move; a new contact penalises the boat that had to keep clear.
        touching = contacts(st2.boat, params.boat_length, params.boat_beam)
        new = touching & ~fs.in_contact
        # A boat taking a penalty keeps clear like a tacking boat (RRS 22.2); one still getting clear
        # has normal rights.
        taking = (serving & (left > 0.0)) if params.penalty_turn else (left > 0.0)
        keep_clear = self.keep_clear_matrix(fs, rules_boats(st2.boat, taking, wind_dir, params), wind_dir, params)
        at_fault = keep_clear & ~keep_clear.T
        undecided = keep_clear == keep_clear.T  # no rule (or both): penalise both
        turned = jnp.abs(wrap_angle(st2.boat.heading - fs.boats.boat.heading))  # course change this step
        if params.rrs16:  # static. RRS 16.1: a right-of-way boat that changed course must give room
            changed = (turned + fs.turned) > params.rrs16_turn  # over the last 2 s
            excused = at_fault & changed[None, :] & ~fs.room  # [i, j]: i kept clear, j changed course into it
            at_fault = (at_fault & ~excused) | excused.T
        # Marks are obstacles too: touching a mark is a penalty (RRS 31).
        on_mark = self.mark_contacts(fs, st2.boat, params)
        new_mark = jnp.any(on_mark & ~fs.mark_contact, axis=1)
        # A contact while turning the penalty circle (the boat has no control then) is part of the same
        # incident: it counts as a contact (and costs contact_cost) but adds no further penalty.
        penalised = (jnp.any(new & (at_fault | undecided), axis=1) | new_mark) & ~serving
        if params.penalty_turn:
            full = params.penalty_turns * 2.0 * jnp.pi
            penalty_left = jnp.where(penalised, left + full, left)  # another penalty adds circles
            # Start by bearing away (turn away from the wind), unless already turning.
            away = -jnp.sign(signed_twa(st2.boat.heading, wind_dir))
            away = jnp.where(away == 0.0, 1.0, away)
            penalty_dir = jnp.where(penalised & (left <= 0.0), away, penalty_dir0)
            penalty_turning = serving & (left > 0.0)
        else:
            penalty_left = jnp.where(penalised, params.penalty_seconds, left)
            penalty_dir = fs.penalty_dir
            penalty_turning = penalty_left > 0.0
        fouls = fs.fouls + penalised.astype(jnp.int32)
        unserved = penalty_left > 0.0
        reward = reward - jnp.where(done & unserved, params.foul_penalty, 0.0)
        # Racing each other: rank by height at the end (only legal starters with no penalty to serve).
        bow_x, bow_y = bow_position(st2.boat, params)
        wind_end, _ = reference_wind(jax.tree.map(lambda x: x[0], st2.wind), jax.tree.map(lambda x: x[0], st2.weather),
                                     st2.t[0], params)
        line_c = jax.tree.map(lambda x: x[0], st2.line)
        cx, cy = 0.5 * (line_c.pin_x + line_c.cb_x), 0.5 * (line_c.pin_y + line_c.cb_y)
        # Height = distance upwind of the line centre along the current wind axis.
        height = (bow_x - cx) * jnp.sin(wind_end) + (bow_y - cy) * jnp.cos(wind_end)
        valid = st2.status.started & ~st2.status.ocs & ~unserved
        score = jnp.where(valid, height, -1e6)
        higher = (score[None, :] > score[:, None]).sum(axis=1)  # boats strictly ahead of me
        ties = (score[None, :] == score[:, None]).sum(axis=1) - 1  # others level with me
        rank = higher + 0.5 * ties  # 0 = leader; tied boats share the average place
        n = self.num_agents
        rank_r = params.rank_reward * ((n - 1 - rank) / max(n - 1, 1) - 0.5)
        reward = reward + jnp.where(done, rank_r, 0.0)

        # Avoid contact at all times: every boat involved in a new contact pays per contact; the boat at
        # fault (or both, if no rule decides) and a boat hitting a mark pay contact_cost, the boat
        # that had right of way pays contact_cost_row.
        fault_contacts = (new & (at_fault | undecided)).sum(axis=1) + new_mark.astype(jnp.int32)
        row_contacts = (new & ~(at_fault | undecided)).sum(axis=1)
        cost_row = jnp.where(params.contact_cost_row < 0.0, params.contact_cost, params.contact_cost_row)
        reward = reward - params.contact_cost * fault_contacts - cost_row * row_contacts
        n_new = fault_contacts + row_contacts
        contacts_total = fs.contacts + n_new
        mark_hits = fs.mark_hits + new_mark.astype(jnp.int32)

        info = dict(info)
        info["fouls"] = jnp.where(done, fouls, 0)
        info["unserved_foul"] = unserved & done
        info["clear_air"] = clear_air & done
        info["contact_now"] = penalised
        info["rank"] = jnp.where(done, rank, 0.0)
        info["height_end"] = jnp.where(done, height, 0.0)
        info["contacts"] = jnp.where(done, contacts_total, 0)
        info["mark_hits"] = jnp.where(done, mark_hits, 0)
        shadow_time = fs.shadow_time + params.dt * ((scale < 0.999) & (st.t >= 0.0))
        info["shadow_time"] = jnp.where(done, shadow_time, 0.0)
        fs2 = fs.replace(boats=st2, penalty_left=penalty_left, penalty_dir=penalty_dir, penalty_turning=penalty_turning,
                         penalty_wait=wait, in_contact=touching, turned=turned,
                         fouls=fouls, mark_contact=on_mark, contacts=contacts_total, mark_hits=mark_hits,
                         shadow_time=shadow_time)
        fs2, reward, done, info = self.post_step(fs, fs2, reward, done, info, params)
        return self.get_obs(fs2, params), fs2, reward.astype(jnp.float32), done, info

    def step(self, key, fs: FleetState, actions, params: EnvParams):
        """step_env + auto-reset of the whole fleet when the episode ends."""
        k_step, k_reset = jax.random.split(key)
        obs, fs, reward, done, info = self.step_env(k_step, fs, actions, params)
        obs_re, fs_re = self.reset(k_reset, params)
        end = done[0]
        fs = jax.tree.map(lambda a, b: jnp.where(end, a, b), fs_re, fs)
        obs = jnp.where(end, obs_re, obs)
        return obs, fs, reward, done, info

    # --- observation ---------------------------------------------------------------------------

    def get_obs(self, fs: FleetState, params: EnvParams):
        st = fs.boats
        n = self.num_agents
        own = jax.vmap(self.solo.get_obs, in_axes=(0, None))(st, params)  # (N, OBS_DIM), neighbour slots 0
        b = st.boat
        wind_dir, tws = self.local_winds(st, params)
        taking = fs.penalty_turning | ((fs.penalty_left > 0.0) & (params.penalty_clear_distance <= 0.0))
        keep_clear = self.keep_clear_matrix(fs, rules_boats(b, taking, wind_dir, params), wind_dir, params)  # [i, j]: i keeps clear of j
        shadow = shadow_factors(b, wind_dir, params, tws)  # [i, j]: factor i gets from j

        fx, fy = heading_vector(b.heading)
        rx, ry = fy, -fx
        dx = b.x[None, :] - b.x[:, None]  # [i, j]: j relative to i
        dy = b.y[None, :] - b.y[:, None]
        fwd = dx * fx[:, None] + dy * fy[:, None]
        stbd = dx * rx[:, None] + dy * ry[:, None]
        rel_h = b.heading[None, :] - b.heading[:, None]
        tack_j = jnp.where(signed_twa(b.heading, wind_dir) >= 0.0, 1.0, -1.0)
        feats = jnp.stack([
            jnp.ones((n, n)),
            fwd / 100.0, stbd / 100.0,
            jnp.sin(rel_h), jnp.cos(rel_h),
            jnp.broadcast_to(b.speed[None, :] / 5.0, (n, n)),
            jnp.broadcast_to(tack_j[None, :], (n, n)),
            keep_clear.T.astype(jnp.float32),  # [i, j] = j keeps clear of i: I have right of way
            (shadow < 1.0).astype(jnp.float32),  # [i, j]: I am in j's shadow
        ], axis=-1)  # (N, N, NEIGHBOUR_FEATURES)

        dist = jnp.hypot(dx, dy) + jnp.where(jnp.eye(n, dtype=bool), jnp.inf, 0.0)
        k = min(N_NEIGHBOURS, n - 1)
        order = jnp.argsort(dist, axis=1)[:, :k]  # nearest other boats first
        near = jnp.take_along_axis(feats, order[:, :, None], axis=1)  # (N, k, F)
        near = jnp.concatenate([near, jnp.zeros((n, N_NEIGHBOURS - k, NEIGHBOUR_FEATURES))], axis=1)
        return own.at[:, OWN_FEATURES:OWN_FEATURES + N_NEIGHBOURS * NEIGHBOUR_FEATURES].set(near.reshape(n, -1))


def rollout_fleet(env: FleetStartEnv, params, key, policy_fn):
    """One full fleet episode without auto-reset. policy_fn(key, obs (N, D)) -> actions (N,).
    Returns (trajectory with a leading time axis, actions, rewards, final info)."""
    k_reset, k_run = jax.random.split(key)
    obs, fs = env.reset(k_reset, params)

    def body(carry, k):
        obs, fs, finished = carry
        k_act, k_step = jax.random.split(k)
        acts = policy_fn(k_act, obs)
        new_obs, new_fs, reward, done, info = env.step_env(k_step, fs, acts, params)
        new_fs = jax.tree.map(lambda a, b: jnp.where(finished, b, a), new_fs, fs)
        new_obs = jnp.where(finished, obs, new_obs)
        reward = jnp.where(finished, 0.0, reward)
        return (new_obs, new_fs, finished | done[0]), (new_fs, acts, reward, info, done[0] & ~finished)

    (_, _, _), (traj, acts, rewards, infos, ends) = jax.lax.scan(
        body, (obs, fs, jnp.bool_(False)), jax.random.split(k_run, params.n_steps))
    traj = jax.tree.map(lambda a, b: jnp.concatenate([a[None], b]), fs, traj)
    last = jnp.argmax(ends)
    return traj, acts, rewards, jax.tree.map(lambda x: x[last], infos)
