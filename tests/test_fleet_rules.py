"""Right of way, wind shadow and contact on the brief's canonical scenarios. Wind from north."""

import jax.numpy as jnp
import numpy as np
import pytest

from sail_env.config import EnvParams, deg
from sail_env.fleet_rules import contacts, must_keep_clear, shadow_factors
from sail_env.physics import BoatState

P = EnvParams()
L, B = P.boat_length, P.boat_beam


def boats(*specs):
    """specs: (x, y, heading_deg[, tacking])."""
    xs, ys, hs, ts = [], [], [], []
    for s in specs:
        xs.append(s[0]), ys.append(s[1]), hs.append(deg(s[2])), ts.append(s[3] if len(s) > 3 else False)
    h = jnp.array(hs, jnp.float32)
    return BoatState(x=jnp.array(xs, jnp.float32), y=jnp.array(ys, jnp.float32), heading=h,
                     speed=jnp.full(len(specs), 3.0, jnp.float32), target_heading=h,
                     manoeuvring=jnp.array(ts, bool))


def keep_clear(bs):
    return np.asarray(must_keep_clear(bs, 0.0, L))


def test_port_keeps_clear_of_starboard_when_crossing():
    # A on starboard (heading NW), B on port (heading NE), on converging courses.
    k = keep_clear(boats((0.0, 0.0, -45.0), (-15.0, 0.0, 45.0)))
    assert k[1, 0] and not k[0, 1]


def test_windward_keeps_clear_when_leeward_luffs():
    # Both on starboard (NW), overlapped; B 3 m on A's starboard (= windward) side, A luffing up.
    sx, sy = np.cos(deg(-45)), -np.sin(deg(-45))  # A's starboard unit vector (towards NE)
    k = keep_clear(boats((0.0, 0.0, -35.0), (3.0 * sx, 3.0 * sy, -45.0)))
    assert k[1, 0] and not k[0, 1]


def test_tacking_boat_keeps_clear():
    # A sailing on starboard; B tacks right in front of her. Even though B ends up on starboard,
    # while tacking she must keep clear.
    k = keep_clear(boats((0.0, 0.0, -45.0), (-5.0, 8.0, -10.0, True)))
    assert k[1, 0] and not k[0, 1]


def test_clear_astern_keeps_clear():
    # Both on starboard, same heading; B 10 m straight behind A (clear astern) and catching up.
    fx, fy = np.sin(deg(-45)), np.cos(deg(-45))
    k = keep_clear(boats((0.0, 0.0, -45.0), (-10.0 * fx, -10.0 * fy, -45.0)))
    assert k[1, 0] and not k[0, 1]


def test_wind_shadow_cone():
    p = P
    # A at the origin; boats 2 BL and 7 BL straight downwind (south), and 2 BL downwind but 45 deg off.
    bs = boats((0.0, 0.0, -45.0), (0.0, -2 * L, -45.0), (0.0, -7 * L, -45.0), (2 * L, -2 * L, -45.0))
    s = np.asarray(shadow_factors(bs, 0.0, p))
    assert s[1, 0] == pytest.approx(0.75 + 0.25 * 2 / 6, abs=1e-4)  # in A's cone, 1/3 of the way
    assert s[2, 0] == 1.0 and s[3, 0] == 1.0  # beyond the tip, outside the 12 deg cone
    assert s[0, 1] == 1.0  # A is upwind of B: not shadowed by B
    assert np.all(np.diag(s) == 1.0)


def test_contact_between_capsules():
    # Side by side 2 m apart (beam 2.25 m): touching; 3 m apart: not.
    c = np.asarray(contacts(boats((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (5.0, 0.0, 0.0)), L, B))
    assert c[0, 1] and c[1, 0] and not c[1, 2] and not c[0, 2]
    # T-bone: B's bow 0.5 m from A's side.
    c = np.asarray(contacts(boats((0.0, 0.0, 0.0), (0.5 + 1.125 + L / 2, 0.0, -90.0)), L, B))
    assert c[0, 1]
    c = np.asarray(contacts(boats((0.0, 0.0, 0.0), (3.0 + L / 2, 0.0, -90.0)), L, B))
    assert not c[0, 1]


def test_apparent_wind_shadow_swings_to_the_leeward_quarter():
    import math

    from sail_env.config import kn
    from sail_env.fleet_rules import shadow_direction

    p = P.replace(shadow_apparent=True)
    tws = kn(10)
    # Close-hauled on starboard (heading NW) at 6 kn: leeward is the port side (towards SW).
    a = boats((0.0, 0.0, -42.0)).replace(speed=jnp.array([kn(6)], jnp.float32))
    dx, dy = (float(v[0]) for v in shadow_direction(a, 0.0, tws, p))
    bearing = math.degrees(math.atan2(dx, dy)) % 360  # compass bearing of the shadow
    assert 150 < bearing < 175  # between straight downwind (180) and astern (138): leeward quarter
    # A stationary boat casts its shadow straight downwind, as with true wind.
    s = boats((0.0, 0.0, -42.0)).replace(speed=jnp.zeros(1, jnp.float32))
    dx, dy = (float(v[0]) for v in shadow_direction(s, 0.0, tws, p))
    assert abs(dx) < 1e-6 and dy == pytest.approx(-1.0)
    # The boat sitting in the apparent-wind shadow is shadowed; it is not with the true-wind cone.
    sx, sy = 3 * L * math.sin(math.radians(bearing)), 3 * L * math.cos(math.radians(bearing))
    pair = boats((0.0, 0.0, -42.0), (sx, sy, -42.0)).replace(speed=jnp.array([kn(6), kn(6)], jnp.float32))
    assert float(shadow_factors(pair, 0.0, p, tws)[1, 0]) < 1.0
    assert float(shadow_factors(pair, 0.0, P, tws)[1, 0]) == 1.0


def test_rrs13_tack_only_gybing_boat_keeps_its_rights():
    from sail_env.fleet_env import rules_boats

    # Running: A on port (heading SSE, TWA -160), B on starboard (heading SSW, TWA +160) in the middle of a
    # gybe (manoeuvring). RRS 13 is about tacking only: B on starboard keeps right of way over port.
    bs = boats((0.0, 0.0, 160.0), (6.0, 0.0, 200.0, True))
    none = jnp.zeros(2, bool)
    old = np.asarray(must_keep_clear(rules_boats(bs, none, 0.0, P), 0.0, L))
    new = np.asarray(must_keep_clear(rules_boats(bs, none, 0.0, P.replace(rrs13_tack_only=True)), 0.0, L))
    assert old[1, 0] and not old[0, 1]  # sail_env 0.1: any manoeuvre keeps clear
    assert new[0, 1] and not new[1, 0]  # RRS 10: port keeps clear of the gybing starboard boat
    # A tacking boat (|TWA| < 90) still keeps clear, and so does a boat taking a penalty.
    tack = boats((0.0, 0.0, -45.0), (-5.0, 8.0, -10.0, True))
    k = np.asarray(must_keep_clear(rules_boats(tack, none, 0.0, P.replace(rrs13_tack_only=True)), 0.0, L))
    assert k[1, 0] and not k[0, 1]
    k = np.asarray(must_keep_clear(rules_boats(bs, jnp.array([False, True]), 0.0, P.replace(rrs13_tack_only=True)), 0.0, L))
    assert k[1, 0] and not k[0, 1]
