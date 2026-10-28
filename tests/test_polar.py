import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sail_env.config import KNOT, EnvParams, deg, kn
from sail_env.polar import load_polar, polar_speed, read_orc_csv

P = load_polar(EnvParams().no_go)
ORC = read_orc_csv()


def speed_kn(twa_deg, tws_kn):
    return float(polar_speed(P, jnp.float32(deg(twa_deg)), jnp.float32(kn(tws_kn)))) / KNOT


@pytest.mark.parametrize("twa", [52, 60, 90, 135, 150])
@pytest.mark.parametrize("k", [0, 2, 5])
def test_grid_points_match_orc_table(twa, k):
    tws = ORC["tws"][k]
    expected = ORC["table"][ORC["angles"].index(twa)][k]
    assert speed_kn(twa, tws) == pytest.approx(expected, abs=1e-3)


def test_bilinear_midpoint_is_mean_of_corners():
    # Halfway between TWA 60/75 and TWS 8/10: corners 5.78, 6.21, 5.89, 6.36.
    assert speed_kn(67.5, 9.0) == pytest.approx((5.78 + 6.21 + 5.89 + 6.36) / 4, abs=1e-3)


def test_beat_point_matches_orc_beat_vmg():
    k = ORC["tws"].index(10.0)
    a, vmg = ORC["beat_angle"][k], ORC["beat_vmg"][k]
    # The beat angle is generally off-grid and sits on a kink, so allow a small interpolation error.
    assert speed_kn(a, 10.0) == pytest.approx(vmg / math.cos(math.radians(a)), abs=0.05)


@pytest.mark.parametrize("twa", [0, 10, 20, 29, 30])
def test_no_go_zone_is_zero(twa):
    assert speed_kn(twa, 12.0) == pytest.approx(0.0, abs=1e-6)  # float32 noise at the knot


def test_pinching_ramp_is_monotonic():
    s = [speed_kn(a, 12.0) for a in range(30, 39)]
    assert s[1] > 0.0
    assert all(b > a for a, b in zip(s, s[1:]))


def test_port_starboard_symmetry_and_wrap():
    for a in [35.0, 45.0, 100.0, 170.0]:
        assert speed_kn(-a, 10.0) == pytest.approx(speed_kn(a, 10.0), abs=1e-4)
        assert speed_kn(360.0 - a, 10.0) == pytest.approx(speed_kn(a, 10.0), abs=1e-3)


def test_light_air_scales_linearly_to_zero():
    assert speed_kn(60, 3.0) == pytest.approx(4.93 / 2, abs=1e-3)
    assert speed_kn(60, 0.0) == 0.0


def test_strong_wind_clamps_to_top_row():
    assert speed_kn(90, 30.0) == pytest.approx(speed_kn(90, 24.0), abs=1e-4)


def test_dead_run_slower_than_broad_reach():
    for tws in ORC["tws"]:
        assert speed_kn(180, tws) < speed_kn(150, tws)


def test_jit_and_vmap():
    twa = jnp.linspace(-np.pi, np.pi, 1000)
    tws = jnp.full_like(twa, kn(10.0))
    out = jax.jit(jax.vmap(polar_speed, in_axes=(None, 0, 0)))(P, twa, tws)
    assert out.shape == (1000,)
    assert bool(jnp.all(jnp.isfinite(out))) and bool(jnp.all(out >= 0))
