import math

import jax
import jax.numpy as jnp
import pytest

from sail_env.config import EnvParams, deg, kn
from sail_env.geometry import signed_twa
from sail_env.physics import (
    DOWN15,
    HOLD,
    LUFF,
    TACK,
    UP5,
    UP15,
    BoatState,
    physics_step,
    relax_speed,
    turn_rate,
)
from sail_env.polar import load_polar, polar_speed

P = EnvParams()
POLAR = load_polar(P.no_go)
WIND_DIR, TWS = jnp.float32(0.0), jnp.float32(kn(10))


def boat(heading_deg, speed_kn=0.0):
    h = jnp.float32(deg(heading_deg))
    return BoatState(
        x=jnp.float32(0.0), y=jnp.float32(0.0), heading=h, speed=jnp.float32(kn(speed_kn)),
        target_heading=h, manoeuvring=jnp.bool_(False),
    )


def target_kn(heading_deg):
    return float(polar_speed(POLAR, signed_twa(jnp.float32(deg(heading_deg)), WIND_DIR), TWS))


step = jax.jit(lambda b, a: physics_step(b, a, POLAR, WIND_DIR, TWS, P)[0])


def run(b, actions):
    trace = [b]
    for a in actions:
        b = step(b, jnp.int32(a))
        trace.append(b)
    return b, trace


# --- turn rate and speed lag -------------------------------------------------------------------


def test_turn_rate_profile():
    assert float(turn_rate(jnp.float32(kn(0.4)), P)) == 0.0
    assert float(turn_rate(jnp.float32(kn(3.0)), P)) == pytest.approx(P.max_turn_rate, rel=1e-5)
    assert float(turn_rate(jnp.float32(kn(8.0)), P)) == pytest.approx(P.max_turn_rate, rel=1e-5)
    assert float(turn_rate(jnp.float32(kn(1.75)), P)) == pytest.approx(P.max_turn_rate / 2, rel=1e-4)


def test_relax_speed_formula_and_asymmetry():
    up = float(relax_speed(jnp.float32(0.0), jnp.float32(4.0), 1.0, P))
    down = float(relax_speed(jnp.float32(4.0), jnp.float32(0.0), 1.0, P))
    assert up == pytest.approx(4.0 * (1 - math.exp(-1 / P.tau_accel)), rel=1e-5)
    assert down == pytest.approx(4.0 * math.exp(-1 / P.tau_decel), rel=1e-5)
    assert up > 4.0 - down  # accelerating is faster than decelerating


def test_speed_lag_from_standstill_on_a_beam_reach():
    # Beam reach (heading 90 = east, wind from north). Start at a crawl so the boat can steer.
    b, _ = run(boat(90, speed_kn=0.0), [HOLD] * 4)
    expected = target_kn(90) * (1 - math.exp(-4.0 / P.tau_accel))  # one time constant
    assert float(b.speed) == pytest.approx(expected, rel=1e-3)
    b, _ = run(b, [HOLD] * 40)
    assert float(b.speed) == pytest.approx(target_kn(90), rel=1e-3)


# --- no-go zone ---------------------------------------------------------------------------------


def test_head_to_wind_coasts_down_old_model():
    p_old = EnvParams(drag_decel=False)
    step_old = jax.jit(lambda b, a: physics_step(b, a, POLAR, WIND_DIR, TWS, p_old)[0])
    b = boat(0, speed_kn=6.0)
    for _ in range(8):
        b = step_old(b, jnp.int32(HOLD))
    assert float(b.speed) == pytest.approx(kn(6.0) * math.exp(-8.0 / P.tau_decel), rel=1e-3)


def _coast_head_to_wind(tws_kn, seconds):
    tws = jnp.float32(kn(tws_kn))
    step_w = jax.jit(lambda b, a: physics_step(b, a, POLAR, WIND_DIR, tws, P)[0])
    b, speeds = boat(0, speed_kn=6.0), []
    for _ in range(seconds):
        b = step_w(b, jnp.int32(HOLD))
        speeds.append(float(b.speed))
    return speeds


def test_head_to_wind_drag_stops_faster_in_a_breeze():
    light, medium, strong = (_coast_head_to_wind(w, 30) for w in (6, 10, 16))
    assert strong[3] < medium[3] < light[3]  # windage of flogging sails grows with the wind
    assert medium[-1] < kn(0.5) and strong[11] < kn(0.5)  # stops within 30 s (10 kn), 12 s (16 kn)
    assert all(b <= a + 1e-6 for a, b in zip(medium, medium[1:]))  # monotonic, never below 0
    assert min(medium) >= 0.0


def test_drag_force_signs():
    from sail_env.physics import drag_force

    assert float(drag_force(jnp.float32(3.0), jnp.float32(0.0), jnp.float32(5.0), P)) > 0  # head to wind
    # Dead downwind, stopped: the wind pushes the boat forward (negative resistance).
    assert float(drag_force(jnp.float32(0.0), jnp.float32(math.pi), jnp.float32(5.0), P)) < 0


def test_cannot_head_up_when_stopped():
    b, _ = run(boat(-20, speed_kn=0.0), [UP15] * 3)  # in irons on starboard
    assert math.degrees(float(b.heading)) == pytest.approx(-20, abs=1e-3)


def test_can_bear_away_out_of_irons():
    b0 = boat(-10, speed_kn=0.0)  # stopped, nearly head to wind, starboard side
    b1, _ = run(b0, [DOWN15])
    # Bears away (heading decreases on starboard) at the irons rate, 5 deg/s, while still stopped.
    assert math.degrees(float(b1.heading)) == pytest.approx(-10 - math.degrees(P.irons_bear_away_rate), abs=0.1)
    b, _ = run(b1, [DOWN15] * 10)
    assert float(b.speed) > kn(1.0)  # out of the no-go zone and moving again


def test_stalled_tack_unlocks_controls():
    b = step(boat(-35, speed_kn=0.4), jnp.int32(TACK))  # too slow to complete the tack
    assert not bool(b.manoeuvring)
    b2 = step(b, jnp.int32(DOWN15))
    assert math.degrees(float(b2.target_heading)) != pytest.approx(math.degrees(float(b.target_heading)), abs=1e-3)


# --- steering ------------------------------------------------------------------------------------


def test_head_up_depends_on_tack():
    # Starboard close reach (heading -60): heading up turns clockwise, towards north.
    b, _ = run(boat(-60, 6.0), [UP5])
    assert math.degrees(float(b.heading)) == pytest.approx(-55, abs=1e-3)
    # Port (heading +60): heading up turns anticlockwise.
    b, _ = run(boat(60, 6.0), [UP15])
    assert math.degrees(float(b.heading)) == pytest.approx(45, abs=1e-3)
    b, _ = run(boat(60, 6.0), [DOWN15])
    assert math.degrees(float(b.heading)) == pytest.approx(75, abs=1e-3)


def test_turn_limited_by_rate_when_slow():
    # Starts at half the max turn rate (10 deg/s); it accelerates during the step so the rate grows,
    # but it cannot complete the 15 deg turn in 1 s.
    b, _ = run(boat(-90, 1.75), [UP15])
    turned = math.degrees(float(b.heading)) + 90
    assert 10.0 < turned < 14.0


# --- tack and gybe -------------------------------------------------------------------------------


def test_tack_mirrors_heading_and_costs_speed():
    b0 = boat(-45, 0.0).replace(speed=jnp.float32(target_kn(-45)))  # steady starboard close-hauled
    b1 = step(b0, jnp.int32(TACK))
    assert bool(b1.manoeuvring)
    assert math.degrees(float(b1.target_heading)) == pytest.approx(45, abs=1e-3)
    b, trace = run(b1, [HOLD] * 6)
    assert not bool(b.manoeuvring)
    assert math.degrees(float(b.heading)) == pytest.approx(45, abs=0.5)
    # It turned through the wind (heading passed 0), not the long way round.
    assert all(abs(math.degrees(float(s.heading))) <= 45.5 for s in trace)
    assert float(b1.speed) < (1 - P.tack_speed_loss) * float(b0.speed)


def test_actions_ignored_during_tack():
    b1 = step(boat(-45, 6.0), jnp.int32(TACK))
    b2 = step(b1, jnp.int32(DOWN15))
    assert math.degrees(float(b2.target_heading)) == pytest.approx(45, abs=1e-3)


def test_gybe_turns_through_downwind():
    b1 = step(boat(-150, 7.0), jnp.int32(TACK))  # starboard broad reach -> port broad reach
    assert math.degrees(float(b1.target_heading)) == pytest.approx(150, abs=1e-3)
    b, trace = run(b1, [HOLD] * 5)
    assert math.degrees(float(b.heading)) == pytest.approx(150, abs=0.5)
    assert all(abs(math.degrees(float(s.heading))) >= 149.5 for s in trace)


# --- luff ----------------------------------------------------------------------------------------


def test_luffing_slows_towards_fraction_of_polar():
    v0 = target_kn(-60)
    b, _ = run(boat(-60, 0.0).replace(speed=jnp.float32(v0)), [LUFF] * 15)
    assert float(b.speed) < 0.5 * v0  # half speed or less within 15 s of easing the sails
    b, _ = run(b, [LUFF] * 45)
    # Quadratic drag is small at low speed, so the last part of the slowdown is gradual.
    assert P.luff_factor * v0 <= float(b.speed) < 1.5 * P.luff_factor * v0
    assert math.degrees(float(b.heading)) == pytest.approx(-60, abs=1e-3)


# --- batching ------------------------------------------------------------------------------------


def test_vmap_over_boats():
    n = 4096
    boats = jax.vmap(lambda h: boat(0).replace(heading=h, target_heading=h, speed=jnp.float32(2.0)))(
        jnp.linspace(-3.0, 3.0, n)
    )
    actions = jnp.arange(n) % 7
    out, path = jax.jit(jax.vmap(lambda b, a: physics_step(b, a, POLAR, WIND_DIR, TWS, P)))(boats, actions)
    assert out.x.shape == (n,) and path[0].shape == (n,)
    assert bool(jnp.all(jnp.isfinite(out.speed)))
