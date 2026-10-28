"""Digital sailor: a hand-written start controller (no learning), as a baseline for the RL boats.

Classic "time on distance": pick a spot on the line, use the starboard close-hauled approach to it,
and every second steer towards where the boat should be on that approach given the time left:

    target  T  = pin + spot x (committee - pin)                 (spot in [0, 1] along the line)
    approach u = starboard close-hauled heading (wind - optimal beat angle, from the polar)
    schedule Pd = T - v_cl x (time to gun + timing) x u         (where to be now; v_cl = close-hauled speed)

Steering towards Pd makes the boat sail away from the line when it is early, come back when the
schedule catches up, luff when it is on the approach but ahead of schedule, and sail flat out on
starboard for the final seconds. It reads the true state (position, wind, clock), like a sailor with
good judgement and instruments, and acts with the same 7 discrete actions as the RL boats.

digital_expert: the same controller, but for each start it tries a grid of (spot, timing) settings in
the simulator and keeps the best one, a strong reference that knows the future (solo only).
"""

import jax
import jax.numpy as jnp

from sail_env.geometry import heading_vector, signed_twa, wrap_angle
from sail_env.physics import DOWN5, DOWN15, HOLD, LUFF, TACK, UP5, UP15
from sail_env.polar import polar_speed
from sail_env.start_env import SoloStartEnv


def beat_angle(polar, tws):
    """Optimal close-hauled TWA (rad) at wind speed tws: argmax over the polar grid of speed x cos(TWA)."""
    speeds = jax.vmap(lambda row: jnp.interp(tws, polar.tws_grid, row))(polar.speed)  # (n_twa,)
    return polar.twa_grid[jnp.argmax(speeds * jnp.cos(polar.twa_grid))]


def steer(heading, desired, wind_dir):
    """Discrete action turning `heading` towards `desired` (tack or gybe if it is on the other tack)."""
    twa_now = signed_twa(heading, wind_dir)
    twa_des = signed_twa(desired, wind_dir)
    other_tack = jnp.sign(twa_now) != jnp.sign(twa_des)
    up = jnp.where(twa_now >= 0.0, 1.0, -1.0)  # heading change that turns towards the wind
    k = jnp.degrees(wrap_angle(desired - heading)) * up  # > 0: need to head up
    act = jnp.where(k > 10.0, UP15, jnp.where(k > 2.5, UP5, jnp.where(k < -10.0, DOWN15, jnp.where(k < -2.5, DOWN5, HOLD))))
    return jnp.where(other_tack & (jnp.abs(jnp.degrees(wrap_angle(desired - heading))) > 20.0), TACK, act)


def unstick(act, speed, heading, wind_dir, beat, params):
    """Never get stuck head to wind. A tack started too slowly runs out of speed and is abandoned, so:
    - in irons (slower than steerage way, inside the beat angle): bear away ("back the jib");
    - a tack below full steerage speed waits: sail close-hauled on this tack until there is speed for it.
    Without this the agents could try to tack forever and sit head to wind until the time limit."""
    twa = jnp.abs(signed_twa(heading, wind_dir))
    in_irons = (speed < params.min_steer_speed) & (twa < beat)
    slow_tack = (act == TACK) & (speed < params.full_steer_speed) & (twa < 0.5 * jnp.pi)
    act = jnp.where(slow_tack, jnp.where(twa < beat, DOWN5, HOLD), act)
    return jnp.where(in_irons, DOWN15, act)


def digital_sailor_action(state, polar, params, spot=0.3, timing=-3.0, final_window=6.0):
    """One decision for one boat. state: EnvState of that boat (solo state, or one boat of the fleet)."""
    b = state.boat
    wind_dir, tws = SoloStartEnv.local_wind(state, params)  # at this boat (weather.py), else the base wind
    beat = beat_angle(polar, tws)
    v_cl = jnp.maximum(polar_speed(polar, beat, tws), 0.5)
    h_cl = wrap_angle(wind_dir - beat)  # starboard close-hauled heading (TWA = +beat)
    ux, uy = heading_vector(h_cl)
    tx = state.line.pin_x + spot * (state.line.cb_x - state.line.pin_x)
    ty = state.line.pin_y + spot * (state.line.cb_y - state.line.pin_y)
    time_to_gun = -state.t
    lead = jnp.maximum(time_to_gun + timing, 0.0)  # seconds of close-hauled sailing to the target
    # Account for the bow: the schedule is for the bow, the state for the boat's centre.
    half = 0.5 * params.boat_length
    px, py = tx - (v_cl * lead + half) * ux, ty - (v_cl * lead + half) * uy  # where the centre should be now
    dx, dy = px - b.x, py - b.y
    dist = jnp.hypot(dx, dy)

    # Along the approach: > 0 when the boat is ahead of schedule (closer to the target than planned).
    ahead = -(dx * ux + dy * uy)
    lateral = jnp.abs(dx * uy - dy * ux)
    on_approach = (lateral < 6.0) & (jnp.abs(wrap_angle(b.heading - h_cl)) < jnp.radians(20.0))

    bearing = jnp.arctan2(dx, dy)
    # Inside the no-go zone the boat cannot sail straight there: beat on the CURRENT tack until the
    # target is on the layline (its bearing leaves the no-go zone on the other side), then tack once.
    # (Choosing the side nearer the bearing every second made the boat tack back and forth.)
    twa_b = signed_twa(bearing, wind_dir)
    on_stbd = signed_twa(b.heading, wind_dir) >= 0.0
    current_tack_cl = jnp.where(on_stbd, wind_dir - beat, wind_dir + beat)
    desired = jnp.where(jnp.abs(twa_b) < beat, current_tack_cl, bearing)
    # Close to the schedule point: line up on the approach heading.
    desired = jnp.where(dist < 8.0, h_cl, desired)
    # Final seconds: flat out on starboard close-hauled at the line.
    desired = jnp.where(time_to_gun < final_window, h_cl, desired)
    act = steer(b.heading, desired, wind_dir)
    # On the approach: how long will the run to the line take from the current speed? Distance along
    # the approach / close-hauled speed, plus the time lost getting up to speed (first-order lag:
    # ~tau_accel x missing fraction of speed). Luff only while there is spare time ("time to burn").
    bow_x, bow_y = b.x + half * jnp.sin(b.heading), b.y + half * jnp.cos(b.heading)
    d_along = jnp.maximum((tx - bow_x) * ux + (ty - bow_y) * uy, 0.0)
    t_need = d_along / v_cl + params.tau_accel * jnp.clip(1.0 - b.speed / v_cl, 0.0, 1.0)
    burn = time_to_gun + timing - t_need
    early = on_approach & (burn > 1.0) & (time_to_gun > 2.0)
    del ahead
    act = jnp.where(early, LUFF, act)
    return unstick(act, b.speed, b.heading, wind_dir, beat, params).astype(jnp.int32)


def solo_policy(env, params, spot=0.3, timing=-3.0):
    """policy(state) -> action for the solo environment."""
    return lambda state: digital_sailor_action(state, env.polar, params, spot, timing)


def rollout_solo(env, params, state, spot, timing):
    """Sail one solo episode from `state` with the digital sailor; returns the final info and total reward."""

    def body(carry, _):
        s, finished, total, final = carry
        a = digital_sailor_action(s, env.polar, params, spot, timing)
        _, s2, r, done, info = env.step_env(jax.random.PRNGKey(0), s, a, params)
        s2 = jax.tree.map(lambda n, o: jnp.where(finished, o, n), s2, s)
        total = total + jnp.where(finished, 0.0, r)
        final = jax.tree.map(lambda n, o: jnp.where(done & ~finished, n, o), info, final)
        return (s2, finished | done, total, final), None

    _, _, _, _, info0 = env.step_env(jax.random.PRNGKey(0), state, jnp.int32(0), params)
    init = (state, jnp.bool_(False), jnp.float32(0.0), jax.tree.map(jnp.zeros_like, info0))
    (_, _, total, final), _ = jax.lax.scan(body, init, None, length=params.n_steps)
    return final, total


def digital_expert(env, params, state, spots=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), timings=(-6.0, -3.0, -1.0, 1.0, 3.0)):
    """Try every (spot, timing) in the simulator from this start and keep the best (by total reward)."""
    grid = jnp.array([(s, t) for s in spots for t in timings])
    finals, totals = jax.vmap(lambda g: rollout_solo(env, params, state, g[0], g[1]))(grid)
    best = jnp.argmax(totals)
    return jax.tree.map(lambda x: x[best], finals), totals[best], grid[best]
