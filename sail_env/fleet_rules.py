"""Boat-to-boat interactions for fleet starts: right of way, wind shadow and contact.

All functions take stacked boats (leading axis = boat) and return pairwise (N, N) arrays, so they
work under jit/vmap. Conventions as in geometry.py (compass headings, TWA > 0 = starboard tack).

Right of way (a subset of the Racing Rules of Sailing, as in the brief):
    - a boat that is tacking (or gybing) keeps clear                  (RRS 13)
    - port keeps clear of starboard                                    (RRS 10)
    - same tack, overlapped: windward keeps clear                      (RRS 11)
    - same tack, not overlapped: the boat clear astern keeps clear     (RRS 12)
Out of scope: room, proper course, acquiring right of way (RRS 15, 17, 19). Changing course (RRS 16.1) is
optional: EnvParams.rrs16, applied in fleet_env.step_env.

Wind shadow: a cone downwind of each boat, shadow_length long with half-angle shadow_half_angle.
Inside it the wind speed is multiplied by a factor rising linearly from shadow_min_factor at the
boat to 1 at the tip; a boat in several shadows takes the strongest one.

Contact: each boat is a capsule (its centreline from stern to bow, radius beam / 2).
"""

import jax.numpy as jnp

from sail_env.geometry import heading_vector, signed_twa


def _pairwise(a):
    """(N,) -> (N, N) arrays a_i (rows) and a_j (columns)."""
    return a[:, None], a[None, :]


def bow_stern(boats, length):
    fx, fy = heading_vector(boats.heading)
    h = 0.5 * length
    return (boats.x + h * fx, boats.y + h * fy), (boats.x - h * fx, boats.y - h * fy)


def clear_astern(boats, length):
    """A[i, j] = True if boat i is clear astern of boat j (i's bow behind the line abeam j's stern)."""
    (bx, by), (sx, sy) = bow_stern(boats, length)
    fx, fy = heading_vector(boats.heading)
    bow_i_x, stern_j_x = bx[:, None], sx[None, :]
    bow_i_y, stern_j_y = by[:, None], sy[None, :]
    return (bow_i_x - stern_j_x) * fx[None, :] + (bow_i_y - stern_j_y) * fy[None, :] < 0.0


def must_keep_clear(boats, wind_dir, length):
    """K[i, j] = True if boat i must keep clear of boat j (diagonal False).

    If K[i, j] and K[j, i] are both False (or both True), no rule separates them.
    """
    n = boats.x.shape[0]
    twa = signed_twa(boats.heading, wind_dir)
    stbd = twa > 0.0
    stbd_i, stbd_j = _pairwise(stbd)
    tacking_i, tacking_j = _pairwise(boats.manoeuvring)

    astern = clear_astern(boats, length)  # [i, j]: i clear astern of j
    overlapped = ~astern & ~astern.T
    # i is to windward of j: on j's windward side, i.e. on j's starboard side when j is on
    # starboard tack (wind over the starboard side), on its port side when j is on port.
    fx, fy = heading_vector(boats.heading)
    rx, ry = fy, -fx  # starboard unit vectors
    dx = boats.x[:, None] - boats.x[None, :]  # i relative to j
    dy = boats.y[:, None] - boats.y[None, :]
    side = dx * rx[None, :] + dy * ry[None, :]  # > 0: i on j's starboard side
    windward = jnp.where(stbd_j, side > 0.0, side < 0.0)

    same_tack = stbd_i == stbd_j
    rule = jnp.where(
        same_tack,
        jnp.where(overlapped, windward, astern),  # RRS 11 / RRS 12
        ~stbd_i,  # RRS 10: port keeps clear
    )
    # RRS 13: a tacking boat keeps clear of one that is not; if both are tacking, no rule here.
    rule = jnp.where(tacking_i & ~tacking_j, True, jnp.where(tacking_j & ~tacking_i, False, rule))
    rule = jnp.where(tacking_i & tacking_j, False, rule)
    return rule & ~jnp.eye(n, dtype=bool)


def shadow_direction(boats, wind_dir, tws, params):
    """Unit vector (per boat) along which its wind shadow is cast.

    True wind: straight downwind. Apparent wind (params.shadow_apparent): the direction the air
    moves relative to the boat, true wind velocity minus boat velocity, so a close-hauled boat's
    shadow swings towards its leeward quarter.
    """
    down_x = -jnp.sin(wind_dir) * jnp.ones_like(boats.x)
    down_y = -jnp.cos(wind_dir) * jnp.ones_like(boats.x)
    if params.shadow_apparent:  # static flag
        fx, fy = heading_vector(boats.heading)
        ax, ay = tws * down_x - boats.speed * fx, tws * down_y - boats.speed * fy
        norm = jnp.maximum(jnp.hypot(ax, ay), 1e-6)
        down_x, down_y = ax / norm, ay / norm
    return down_x, down_y


def shadow_factors(boats, wind_dir, params, tws=0.0):
    """S[i, j] = wind-speed factor boat i gets from boat j's shadow (1 = clear); factor_i = min_j S[i, j]."""
    n = boats.x.shape[0]
    down_x, down_y = shadow_direction(boats, wind_dir, tws, params)
    down_x, down_y = down_x[None, :], down_y[None, :]  # the caster j's direction
    dx = boats.x[:, None] - boats.x[None, :]  # i relative to j (the shadow caster)
    dy = boats.y[:, None] - boats.y[None, :]
    along = dx * down_x + dy * down_y  # distance downstream of j
    across = jnp.abs(dx * down_y - dy * down_x)
    length = params.shadow_length_bl * params.boat_length
    inside = (along > 0.0) & (along < length) & (across < along * jnp.tan(params.shadow_half_angle))
    factor = params.shadow_min_factor + (1.0 - params.shadow_min_factor) * along / length
    s = jnp.where(inside, factor, 1.0)
    return jnp.where(jnp.eye(n, dtype=bool), 1.0, s)


def _segment_distance(p1, q1, p2, q2):
    """Shortest distance between 2D segments p1-q1 and p2-q2 (arrays broadcast elementwise)."""

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    d1 = (q1[0] - p1[0], q1[1] - p1[1])
    d2 = (q2[0] - p2[0], q2[1] - p2[1])
    r = (p1[0] - p2[0], p1[1] - p2[1])
    a, e, f = dot(d1, d1), dot(d2, d2), dot(d2, r)
    c, b = dot(d1, r), dot(d1, d2)
    denom = a * e - b * b
    s = jnp.where(denom > 1e-9, jnp.clip((b * f - c * e) / jnp.where(denom > 1e-9, denom, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / e
    # Clamp t to [0, 1] and recompute s for the clamped t.
    s = jnp.where(t < 0.0, jnp.clip(-c / a, 0.0, 1.0), jnp.where(t > 1.0, jnp.clip((b - c) / a, 0.0, 1.0), s))
    t = jnp.clip(t, 0.0, 1.0)
    cx1, cy1 = p1[0] + d1[0] * s, p1[1] + d1[1] * s
    cx2, cy2 = p2[0] + d2[0] * t, p2[1] + d2[1] * t
    return jnp.hypot(cx1 - cx2, cy1 - cy2)


def point_segment_distance(px, py, ax, ay, bx, by):
    """Distance from point (px, py) to segment a-b (arrays broadcast)."""
    vx, vy = bx - ax, by - ay
    t = jnp.clip(((px - ax) * vx + (py - ay) * vy) / jnp.maximum(vx * vx + vy * vy, 1e-9), 0.0, 1.0)
    return jnp.hypot(ax + t * vx - px, ay + t * vy - py)


def mark_contacts(boats, line, params):
    """(N, 2) bool: boat touches the pin (column 0) / the committee boat (column 1)."""
    (bx, by), (sx, sy) = bow_stern(boats, params.boat_length)
    half_beam = 0.5 * params.boat_beam
    d_pin = point_segment_distance(line.pin_x, line.pin_y, sx, sy, bx, by)
    d_cb = point_segment_distance(line.cb_x, line.cb_y, sx, sy, bx, by)
    return jnp.stack([d_pin < half_beam + params.pin_radius, d_cb < half_beam + params.committee_radius], axis=1)


def contacts(boats, length, beam):
    """C[i, j] = True if the capsules of boats i and j touch (symmetric, diagonal False)."""
    n = boats.x.shape[0]
    (bx, by), (sx, sy) = bow_stern(boats, length)
    dist = _segment_distance(
        (sx[:, None], sy[:, None]), (bx[:, None], by[:, None]), (sx[None, :], sy[None, :]), (bx[None, :], by[None, :])
    )
    return (dist < beam) & ~jnp.eye(n, dtype=bool)
