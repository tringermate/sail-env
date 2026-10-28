"""Digital racer: a hand-written controller for the whole race (no learning), a baseline and opponent.

    start     the digital sailor's "time on distance" start (agents/digital_sailor.py)
    upwind    beat on the current tack until the mark is on the layline, then tack once; within 50 m
              of the windward mark, come in on a starboard layline that passes it to windward (on
              port until then), and bear away round it, down its left side, which rounds it to port
    downwind  sail the polar's best downwind VMG angle and gybe on the layline
    gate      through the gate at the lane's spot, then round the mark on that side; finish through the line
    late      a boat that has not started after the gun (late, or OCS) goes back below the line if
              needed and crosses it

Obstacles: before steering, the racer checks the next few boat lengths of its course and, if that
would pass too close to a mark or to a boat it must keep clear of (from the observation's neighbour
slots, where they will be in 2 s), picks the nearest heading that clears them. Right-of-way boats hold
their course.

Each boat of a fleet keeps its own lane (approach point at the windward mark, spot in the gate and on
the finish line, from its `spot`), so digital racers do not all aim at the same point. It reads the
current leg from the observation and the geometry from the (privileged) state.
"""

import jax
import jax.numpy as jnp

from sail_env.agents.digital_sailor import beat_angle, digital_sailor_action, steer, unstick
from sail_env.geometry import signed_twa, wrap_angle
from sail_env.geometry import heading_vector
from sail_env.race_env import N_LEGS, RACE_FEATURES, course_geometry
from sail_env.start_env import N_NEIGHBOURS, NEIGHBOUR_FEATURES, OWN_FEATURES, SoloStartEnv
from sail_env.weather import WEATHER_FEATURES


def run_angle(polar, tws):
    """Best downwind VMG TWA (rad): argmax over the polar of speed x (-cos TWA)."""
    speeds = jax.vmap(lambda row: jnp.interp(tws, polar.tws_grid, row))(polar.speed)
    return polar.twa_grid[jnp.argmax(speeds * -jnp.cos(polar.twa_grid))]


AVOID_OFFSETS = jnp.radians(jnp.array([0.0, 20.0, -20.0, 40.0, -40.0, 70.0, -70.0, 110.0, -110.0]))
MAX_TURN = jnp.radians(15.0)  # largest heading change of one decision (UP15 / DOWN15)


def give_way_obstacles(b, obs, params, horizon=2.0, reach=40.0):
    """(centres (K, 2), radii (K,)) of the neighbours this boat must keep clear of, where they will be
    in `horizon` s, from the observation's neighbour slots (others are moved far away)."""
    nb = obs[OWN_FEATURES:OWN_FEATURES + N_NEIGHBOURS * NEIGHBOUR_FEATURES].reshape(N_NEIGHBOURS, NEIGHBOUR_FEATURES)
    fx, fy = heading_vector(b.heading)
    rx, ry = fy, -fx
    fwd, stbd = 100.0 * nb[:, 1], 100.0 * nb[:, 2]
    h_j = b.heading + jnp.arctan2(nb[:, 3], nb[:, 4])
    v_j = 5.0 * nb[:, 5]
    x = b.x + fwd * fx + stbd * rx + horizon * v_j * jnp.sin(h_j)
    y = b.y + fwd * fy + stbd * ry + horizon * v_j * jnp.cos(h_j)
    relevant = (nb[:, 0] > 0.5) & (nb[:, 7] < 0.5) & (jnp.hypot(fwd, stbd) < reach)
    x = jnp.where(relevant, x, b.x + 1.0e6)
    return jnp.stack([x, y], axis=1), jnp.full(N_NEIGHBOURS, 0.6 * params.boat_length)


def avoid_marks(b, desired, wind_dir, beat, marks, radii, params, margin=1.5):
    """The heading closest to `desired` whose next few boat lengths keep `margin` m of water from every
    obstacle (marks, boats; candidates in AVOID_OFFSETS; headings in the no-go zone are skipped)."""
    look = 0.5 * params.boat_length + 4.0 + 3.0 * b.speed  # m of track ahead of the boat's centre
    cand = wrap_angle(desired + AVOID_OFFSETS)
    mx, my = marks[:, 0] - b.x, marks[:, 1] - b.y  # (M,)

    def clear(h, length):  # (C,) bool: the straight track along headings h keeps clear of every obstacle
        dx, dy = jnp.sin(h), jnp.cos(h)
        along = jnp.clip(mx[None, :] * dx[:, None] + my[None, :] * dy[:, None], 0.0, length)  # (C, M)
        gap = jnp.hypot(mx[None, :] - along * dx[:, None], my[None, :] - along * dy[:, None])
        return jnp.all(gap > radii[None, :] + 0.5 * params.boat_beam + margin, axis=1)

    # The boat turns at most ~15 deg per decision: also check the heading it will have on the way.
    on_the_way = wrap_angle(b.heading + jnp.clip(wrap_angle(cand - b.heading), -MAX_TURN, MAX_TURN))
    safe = clear(cand, look) & clear(on_the_way, 0.6 * look)
    sailable = jnp.abs(signed_twa(cand, wind_dir)) >= beat - jnp.radians(5.0)
    ok = safe & (sailable | (jnp.arange(cand.shape[0]) == 0))
    return jnp.where(jnp.any(ok), cand[jnp.argmax(ok)], desired)


def race_sailor_action(state, obs, polar, params, spot=0.3, timing=-3.0):
    """One decision for one boat in the race. state: that boat's EnvState; obs: its race observation."""
    from sail_env.race_env import TEAM_FEATURES

    # The race features come before the teammate flags (team racing) and the weather block (both last).
    end = obs.shape[0] - (WEATHER_FEATURES if params.observe_weather else 0) - (TEAM_FEATURES if params.team_size else 0)
    leg = jnp.argmax(obs[end - RACE_FEATURES:end - RACE_FEATURES + N_LEGS + 1])
    b = state.boat
    wind_dir, tws = SoloStartEnv.local_wind(state, params)  # at this boat (weather.py), else the base wind
    k = course_geometry(state.line, state.wind.base_dir, params)
    (ux, uy), (rx, ry), (wx, wy) = k["u"], k["r"], k["w"]
    beat = beat_angle(polar, tws)
    run = run_angle(polar, tws)

    lane = jnp.clip((spot - 0.15) / 0.7, 0.0, 1.0)  # 0 .. 1: this boat's lane
    line = state.line
    lx, ly = line.pin_x + spot * (line.cb_x - line.pin_x), line.pin_y + spot * (line.cb_y - line.pin_y)

    # Where to go on this leg.
    up_from_w = (b.x - wx) * ux + (b.y - wy) * uy  # > 0: above the windward mark
    right_of_w = (b.x - wx) * rx + (b.y - wy) * ry
    # Upwind: aim to the right of the mark (8-20 m, by lane); once level with it on that side, aim
    # above-left of it (crossing the line straight upwind of the mark from right to left = port rounding).
    off = 8.0 + 12.0 * lane
    a_x, a_y = wx + off * rx, wy + off * ry
    c_x, c_y = wx + 10.0 * ux - 10.0 * rx, wy + 10.0 * uy - 10.0 * ry
    at_mark = (up_from_w > -2.0) & (right_of_w > 0.0)  # level with the mark, on its right
    tgt_x = jnp.where(at_mark, c_x, a_x)
    tgt_y = jnp.where(at_mark, c_y, a_y)
    gate_off = (lane - 0.5) * 1.2 * k["hw"]  # spread across the gate
    g_x, g_y = k["g"][0] + gate_off * rx, k["g"][1] + gate_off * ry
    # Through the gate at the lane's spot, then round the gate mark on that side: out past it below the
    # gate line, then up outside it (coming back up across the line outside a mark completes the run).
    side = jnp.where(lane < 0.5, -1.0, 1.0)  # -1: the left gate mark (looking upwind), +1: the right one
    gm_x, gm_y = k["g"][0] + side * k["hw"] * rx, k["g"][1] + side * k["hw"] * ry
    below_gate = (b.x - k["g"][0]) * ux + (b.y - k["g"][1]) * uy < 0.0
    outside = side * ((b.x - k["g"][0]) * rx + (b.y - k["g"][1]) * ry) > k["hw"] + 3.0
    round_x = jnp.where(outside, gm_x + side * 8.0 * rx + 15.0 * ux, gm_x + side * 10.0 * rx - 8.0 * ux)
    round_y = jnp.where(outside, gm_y + side * 8.0 * ry + 15.0 * uy, gm_y + side * 10.0 * ry - 8.0 * uy)
    g_x, g_y = jnp.where(below_gate, round_x, g_x), jnp.where(below_gate, round_y, g_y)
    tgt_x = jnp.select([leg == 1, leg == 3], [g_x, lx], tgt_x)
    tgt_y = jnp.select([leg == 1, leg == 3], [g_y, ly], tgt_y)
    # Just past the windward mark (the observation already shows the run): bear away down its left side,
    # since the rounding only counts once the boat is back below the mark's level on that side.
    leaving = ((leg == 1) | (leg == 3)) & (up_from_w > -3.0) & (jnp.hypot(b.x - wx, b.y - wy) < 60.0)
    tgt_x = jnp.where(leaving, wx - 15.0 * rx - 20.0 * ux, tgt_x)
    tgt_y = jnp.where(leaving, wy - 15.0 * ry - 20.0 * uy, tgt_y)
    # Late (or OCS) after the gun: below the line -> cross it at the lane's spot; above it -> go back below.
    nx, ny = -(line.cb_y - line.pin_y), line.cb_x - line.pin_x
    norm = jnp.hypot(nx, ny)
    above_line = ((b.x - line.pin_x) * nx + (b.y - line.pin_y) * ny) / norm > -1.0
    late = ~state.status.started & (state.t > 0.0)
    late_x = jnp.where(above_line, lx - 20.0 * ux, lx + 15.0 * ux)
    late_y = jnp.where(above_line, ly - 20.0 * uy, ly + 15.0 * uy)
    tgt_x = jnp.where(late, late_x, tgt_x)
    tgt_y = jnp.where(late, late_y, tgt_y)

    bearing = jnp.arctan2(tgt_x - b.x, tgt_y - b.y)
    twa_b = signed_twa(bearing, wind_dir)
    on_stbd = signed_twa(b.heading, wind_dir) >= 0.0
    # Upwind no-go: beat on the current tack until the layline.
    cl = jnp.where(on_stbd, wind_dir - beat, wind_dir + beat)
    # Downwind: deeper than the best VMG angle is slower; sail the VMG angle on the current gybe.
    vmg_down = jnp.where(on_stbd, wind_dir - run, wind_dir + run)
    desired = jnp.where(jnp.abs(twa_b) < beat, cl, jnp.where(jnp.abs(twa_b) > run, vmg_down, bearing))
    # Near the windward mark: approach on the starboard layline of a point just above-right of the mark;
    # below that layline, sail on port until reaching it (a port boat cannot round to port off the layline).
    beating = ((leg == 0) | (leg == 2)) & ~at_mark & ~late
    near = jnp.hypot(b.x - wx, b.y - wy) < 50.0
    n_x, n_y = wx + 6.0 * rx + 2.0 * ux, wy + 6.0 * ry + 2.0 * uy
    twa_n = signed_twa(jnp.arctan2(n_x - b.x, n_y - b.y), wind_dir)
    fetch = (twa_n >= beat - jnp.radians(2.0)) | (jnp.abs(twa_n) > 0.5 * jnp.pi)
    approach = jnp.where(fetch, jnp.arctan2(n_x - b.x, n_y - b.y), wind_dir + beat)  # else port close-hauled
    desired = jnp.where(beating & near, approach, desired)
    marks = jnp.stack([jnp.stack([line.pin_x, line.pin_y]), jnp.stack([line.cb_x, line.cb_y]),
                       jnp.stack(k["w"]), jnp.stack(k["g1"]), jnp.stack(k["g2"])])
    radii = jnp.array([params.pin_radius, params.committee_radius] + [params.course_mark_radius] * 3)
    boats, boat_radii = give_way_obstacles(b, obs, params)
    marks, radii = jnp.concatenate([marks, boats]), jnp.concatenate([radii, boat_radii])
    desired = avoid_marks(b, wrap_angle(desired), wind_dir, beat, marks, radii, params)
    act = unstick(steer(b.heading, desired, wind_dir), b.speed, b.heading, wind_dir, beat, params)

    racing = (state.status.started & (leg < N_LEGS)) | late
    start_act = digital_sailor_action(state, polar, params, spot, timing)
    return jnp.where(racing, act, start_act).astype(jnp.int32)
