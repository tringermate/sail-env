import jax
import jax.numpy as jnp
import pytest

from sail_env.config import EnvParams, deg, kn
from sail_env.geometry import signed_twa, wrap_angle
from sail_env.wind import Wind, sample_wind, wind_at


def make_wind(amp=0.0, period=120.0, phase=0.0):
    return Wind(jnp.float32(0.0), jnp.float32(kn(10)), jnp.float32(amp), jnp.float32(period), jnp.float32(phase))


def test_steady_wind_is_constant():
    w = make_wind()
    for t in [-60.0, 0.0, 33.3]:
        d, s = wind_at(w, t)
        assert float(d) == 0.0 and float(s) == pytest.approx(kn(10), rel=1e-6)


def test_oscillating_shift():
    w = make_wind(amp=deg(10), period=120.0)
    assert float(wind_at(w, 30.0)[0]) == pytest.approx(deg(10), abs=1e-5)  # quarter period: +amp
    assert float(wind_at(w, 90.0)[0]) == pytest.approx(-deg(10), abs=1e-5)
    assert float(wind_at(w, 120.0)[0]) == pytest.approx(0.0, abs=1e-5)


def test_sampled_wind_in_range():
    p = EnvParams(shift_amp_max=deg(8))
    ws = jax.vmap(lambda k: sample_wind(k, p))(jax.random.split(jax.random.PRNGKey(0), 1000))
    assert float(ws.tws.min()) >= p.tws_min and float(ws.tws.max()) <= p.tws_max
    assert float(ws.shift_amp.max()) <= p.shift_amp_max


def test_twa_sign_convention():
    # Wind from north. Heading NW (-45 deg): wind over the starboard bow -> starboard, TWA > 0.
    assert float(signed_twa(jnp.float32(-deg(45)), 0.0)) == pytest.approx(deg(45), abs=1e-6)
    assert float(signed_twa(jnp.float32(deg(45)), 0.0)) == pytest.approx(-deg(45), abs=1e-6)
    assert float(abs(signed_twa(jnp.float32(deg(180)), 0.0))) == pytest.approx(deg(180), abs=1e-6)
    assert float(wrap_angle(jnp.float32(deg(270)))) == pytest.approx(-deg(90), abs=1e-6)
