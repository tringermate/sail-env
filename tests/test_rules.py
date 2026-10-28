import math

import jax
import jax.numpy as jnp
import pytest

from sail_env.config import deg
from sail_env.rules import crossing, init_status, make_line, signed_distance, update_status

L = 20.0
SQUARE = make_line(0.0, 0.0, 0.0, 0.0, L)  # wind from north: pin west, committee east


def f(x):
    return jnp.float32(x)


# --- geometry ------------------------------------------------------------------------------------


def test_square_line_geometry():
    assert float(SQUARE.pin_x) == pytest.approx(-10) and float(SQUARE.cb_x) == pytest.approx(10)
    assert float(SQUARE.pin_y) == pytest.approx(0, abs=1e-6) and float(SQUARE.cb_y) == pytest.approx(0, abs=1e-6)


def test_positive_bias_puts_committee_upwind():
    line = make_line(0.0, 0.0, 0.0, deg(10), L)
    assert float(line.cb_y) == pytest.approx(10 * math.sin(deg(10)), abs=1e-5)
    assert float(line.pin_y) == pytest.approx(-10 * math.sin(deg(10)), abs=1e-5)
    assert float(math.hypot(line.cb_x - line.pin_x, line.cb_y - line.pin_y)) == pytest.approx(L, rel=1e-6)


def test_line_follows_wind_direction():
    # Wind from east (90 deg): upwind is east, the starboard end is south.
    line = make_line(0.0, 0.0, deg(90), 0.0, L)
    assert float(line.cb_y) == pytest.approx(-10, abs=1e-5)
    assert float(signed_distance(line, f(5.0), f(0.0))) == pytest.approx(5.0, abs=1e-5)


def test_signed_distance_positive_upwind():
    assert float(signed_distance(SQUARE, f(3.0), f(4.0))) == pytest.approx(4.0)
    assert float(signed_distance(SQUARE, f(-3.0), f(-2.0))) == pytest.approx(-2.0)


# --- crossings -----------------------------------------------------------------------------------


def test_crossing_up_through_middle():
    up, down, frac = crossing(SQUARE, f(0.0), f(-1.0), f(0.0), f(3.0))
    assert bool(up) and not bool(down)
    assert float(frac) == pytest.approx(0.25)


def test_crossing_down():
    up, down, _ = crossing(SQUARE, f(2.0), f(1.0), f(2.0), f(-1.0))
    assert bool(down) and not bool(up)


def test_crossing_outside_the_ends_does_not_count():
    up, down, _ = crossing(SQUARE, f(12.0), f(-1.0), f(12.0), f(1.0))  # beyond the committee boat
    assert not bool(up) and not bool(down)
    up, _, _ = crossing(SQUARE, f(-10.5), f(-1.0), f(-10.5), f(1.0))  # just outside the pin
    assert not bool(up)


def test_diagonal_path_meeting_line_inside():
    # Starts outside the ends in x but meets the line at x = 9 (inside).
    up, _, frac = crossing(SQUARE, f(8.0), f(-1.0), f(11.0), f(2.0))
    assert bool(up) and float(frac) == pytest.approx(1 / 3, abs=1e-5)


def test_parallel_path_no_crossing():
    up, down, _ = crossing(SQUARE, f(-5.0), f(-1.0), f(5.0), f(-1.0))
    assert not bool(up) and not bool(down)


# --- start state machine -------------------------------------------------------------------------


def step(status, path, t0):
    return update_status(status, SQUARE, tuple(f(v) for v in path), f(t0), f(t0 + 1.0))


def test_crossing_before_gun_is_not_a_start():
    s = step(init_status(), (0, -1, 0, 1), -5.0)
    assert not bool(s.started) and not bool(s.ocs)


def test_clean_start_records_interpolated_time():
    s = step(init_status(), (0, -2, 0, -1), -1.0)  # gun: behind the line
    assert not bool(s.ocs)
    s = step(s, (0, -1, 0, 1), 1.0)  # crosses halfway through the step 1 -> 2
    assert bool(s.started) and float(s.start_time) == pytest.approx(1.5)


def test_over_at_gun_is_ocs_and_must_dip_back():
    s = step(init_status(), (0, 0.5, 0, 1.0), -1.0)  # bow over the line at T=0
    assert bool(s.ocs)
    s = step(s, (0, 1.0, 0, 2.0), 0.0)  # keeps sailing on: still OCS, not started
    assert bool(s.ocs) and not bool(s.started)
    s = step(s, (0, 2.0, 0, -0.5), 1.0)  # dips back below the line: cleared
    assert not bool(s.ocs) and not bool(s.started)
    s = step(s, (0, -0.5, 0, 0.5), 2.0)  # re-crosses: legal start
    assert bool(s.started) and float(s.start_time) == pytest.approx(2.5)


def test_ocs_boat_crossing_back_up_without_clearing_first_does_not_start():
    # Over at the gun; within one step it is still over at the end, so no clearing happens.
    s = step(init_status(), (0, -1.0, 0, 0.2), -1.0)
    assert bool(s.ocs)
    s = step(s, (0, 0.2, 0, 1.0), 0.0)
    assert not bool(s.started)


def test_vmapped_status_update():
    n = 1000
    ys = jnp.linspace(-2.0, 2.0, n)
    s = jax.vmap(lambda y: update_status(init_status(), SQUARE, (f(0), y, f(0), y + 1.0), f(1.0), f(2.0)))(ys)
    assert s.started.shape == (n,)
    # Starts exactly where the path goes from <= 0 to > 0: y0 in (-1, 0].
    assert bool(jnp.all(s.started == ((ys > -1.0) & (ys <= 0.0))))
