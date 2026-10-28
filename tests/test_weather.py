import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sail_env as se
from sail_env.config import EnvParams, deg, kn
from sail_env.physics import HOLD, TACK, BoatState, physics_step
from sail_env.polar import load_polar
from sail_env.geometry import wrap_angle
from sail_env.weather import (DIR_SCALE, WEATHER_FEATURES, HISTORY_LAGS, reference_wind, sample_weather, water_wind,
                              wave_speed_factor, wind_at_point)
from sail_env.wind import Wind, wind_at

ON = dict(weather_shifts=True, weather_pressure=True, water_current=True, waves=True)
POLAR = load_polar(EnvParams().no_go)


def base_wind(tws_kn=10.0, amp=0.0):
    return Wind(jnp.float32(0.0), jnp.float32(kn(tws_kn)), jnp.float32(amp), jnp.float32(120.0), jnp.float32(0.3))


def calm_weather(params, **fields):
    """A sampled Weather with every feature zeroed, then the given fields set."""
    w = sample_weather(jax.random.PRNGKey(0), 0.0, params)
    w = w.replace(drift=jnp.float32(0.0), shift_amp=jnp.zeros_like(w.shift_amp), bend=jnp.float32(0.0),
                  grad=jnp.float32(0.0), puff_amp=jnp.zeros_like(w.puff_amp), puff_dir=jnp.zeros_like(w.puff_dir),
                  current_x=jnp.float32(0.0), current_y=jnp.float32(0.0), wave_h=jnp.float32(0.0),
                  wave_dir=jnp.float32(0.0))
    return w.replace(**{k: jnp.asarray(v, jnp.float32) for k, v in fields.items()})


def test_off_by_default_is_the_base_wind():
    p = EnvParams(shift_amp_max=deg(5))
    wind = base_wind(amp=deg(5))
    weather = sample_weather(jax.random.PRNGKey(1), 0.0, EnvParams(**ON))
    for t, x, y in [(-30.0, 0.0, 0.0), (100.0, -120.0, 300.0), (500.0, 80.0, -40.0)]:
        ref = wind_at(wind, t)
        for f in (wind_at_point, water_wind):
            d, s = f(wind, weather, t, x, y, p)
            assert float(d) == float(ref[0]) and float(s) == float(ref[1])
        assert float(reference_wind(wind, weather, t, p)[0]) == float(ref[0])


def test_presets_keep_their_observation_and_add_weather_variants():
    for name, dim in [("solo", 60), ("fleet", 60), ("race-solo", 72), ("race", 72)]:
        env, p = se.make(name)
        assert env.obs_dim == dim and not (p.weather_shifts or p.weather_pressure or p.water_current or p.waves)
        env_w, p_w = se.make(f"{name}-weather")
        assert env_w.obs_dim == dim + WEATHER_FEATURES
        assert p_w.weather_shifts and p_w.weather_pressure and p_w.water_current and p_w.waves and p_w.observe_weather
    _, p = se.make("race-weather", waves=False)
    assert not p.waves and p.water_current


def test_weather_does_not_change_the_rest_of_the_reset():
    """Weather has its own random stream: same key, same boats, line and base wind."""
    key = jax.random.PRNGKey(7)
    _, fs = se.make("fleet")[0].reset(key, se.make("fleet")[1])
    env_w, p_w = se.make("fleet-weather")
    _, fs_w = env_w.reset(key, p_w)
    for a, b in [(fs.boats.boat.x, fs_w.boats.boat.x), (fs.boats.boat.heading, fs_w.boats.boat.heading),
                 (fs.boats.wind.tws, fs_w.boats.wind.tws), (fs.boats.line.pin_x, fs_w.boats.line.pin_x)]:
        assert bool(jnp.all(a == b))


def test_field_moves_downwind():
    """What is at distance a upwind now arrives at the line centre a / U seconds later (shifts and puffs)."""
    p = EnvParams(weather_shifts=True, weather_pressure=True, shift_amp_max=deg(5))
    wind = base_wind(amp=deg(4))
    weather = sample_weather(jax.random.PRNGKey(3), 0.0, p)
    u = p.weather_advection * float(wind.tws)
    for t, a in [(0.0, 200.0), (150.0, 400.0), (-60.0, 50.0)]:
        d_up, s_up = wind_at_point(wind, weather, t, 0.0, a, p)  # wind from north: upwind = +y
        d_0, s_0 = wind_at_point(wind, weather, t + a / u, 0.0, 0.0, p)
        assert float(d_up) == pytest.approx(float(d_0), abs=2e-4)
        assert float(s_up) == pytest.approx(float(s_0), rel=2e-4)


def test_persistent_drift_is_linear():
    p = EnvParams(weather_shifts=True)
    w = calm_weather(p, drift=deg(6) / 600)
    d1, _ = wind_at_point(base_wind(), w, 100.0, 0.0, 0.0, p)
    d2, _ = wind_at_point(base_wind(), w, 700.0, 0.0, 0.0, p)
    assert float(d2 - d1) == pytest.approx(deg(6), abs=1e-5)


def test_random_shifts_have_the_requested_spread():
    p = EnvParams(weather_shifts=True, drift_max=0.0, bend_max=0.0, random_shift_std=deg(4))
    ws = jax.vmap(lambda k: sample_weather(k, 0.0, p))(jax.random.split(jax.random.PRNGKey(0), 4000))
    d, _ = jax.vmap(lambda w: wind_at_point(base_wind(), w, 37.0, 0.0, 0.0, p))(ws)
    assert float(jnp.std(d)) == pytest.approx(deg(4), rel=0.1)
    assert abs(float(jnp.mean(d))) < deg(0.3)


def test_bend_and_pressure_gradient_across_the_course():
    p = EnvParams(weather_shifts=True, weather_pressure=True)
    w = calm_weather(p, bend=deg(2), grad=0.05)
    # Wind from north: right, looking upwind, is east (+x).
    d0, s0 = wind_at_point(base_wind(), w, 0.0, 0.0, 0.0, p)
    d1, s1 = wind_at_point(base_wind(), w, 0.0, 100.0, 0.0, p)
    assert float(d1 - d0) == pytest.approx(deg(2), abs=1e-5)
    assert float(s1 / s0) == pytest.approx(1.05, rel=1e-5)


def test_a_puff_and_the_floor():
    p = EnvParams(weather_pressure=True)
    n = p.n_puffs
    one = jnp.zeros(n).at[0].set(1.0)
    w = calm_weather(p, puff_xi=jnp.full(n, 1000.0), puff_c=jnp.zeros(n), puff_r=jnp.full(n, 80.0),
                     puff_amp=0.2 * one, puff_dir=deg(3) * one)
    wind = base_wind()
    u = p.weather_advection * float(wind.tws)
    t_centre = 1000.0 / u  # the puff centre reaches the line centre (air coordinate xi = U t)
    d, s = wind_at_point(wind, w, t_centre, 0.0, 0.0, p)
    assert float(s) == pytest.approx(1.2 * float(wind.tws), rel=1e-4)
    assert float(d) == pytest.approx(deg(3), abs=1e-5)
    _, far = wind_at_point(wind, w, t_centre, 400.0, 0.0, p)  # 400 m across: outside the puff
    assert float(far) == pytest.approx(float(wind.tws), rel=1e-3)
    # Periodic cell: the same puff comes back one cell length later.
    _, again = wind_at_point(wind, w, t_centre + p.puff_cell_length / u, 0.0, 0.0, p)
    assert float(again) == pytest.approx(float(s), rel=1e-3)
    # A deep lull is held at the floor.
    deep = w.replace(puff_amp=-0.95 * one)
    _, s_lull = wind_at_point(wind, deep, t_centre, 0.0, 0.0, p)
    assert float(s_lull) == pytest.approx(p.tws_floor_frac * float(wind.tws), rel=1e-4)


def test_current_changes_the_wind_over_the_water():
    p = EnvParams(water_current=True)
    wind = base_wind(tws_kn=10.0)  # from the north
    against = calm_weather(p, current_y=kn(1.0))  # water flows north, into the wind: more wind
    d, s = water_wind(wind, against, 0.0, 0.0, 0.0, p)
    assert float(s) == pytest.approx(kn(11.0), rel=1e-5) and float(d) == pytest.approx(0.0, abs=1e-6)
    with_wind = calm_weather(p, current_y=-kn(1.0))
    assert float(water_wind(wind, with_wind, 0.0, 0.0, 0.0, p)[1]) == pytest.approx(kn(9.0), rel=1e-5)
    # Current to the east under a 5 m/s northerly: the wind over the water comes from east of north.
    across = calm_weather(p, current_x=1.0)
    d, s = water_wind(base_wind(tws_kn=5.0 / 0.5144), across, 0.0, 0.0, 0.0, p)
    assert float(d) == pytest.approx(math.atan2(1.0, 5.0), abs=1e-5)
    assert float(s) == pytest.approx(math.hypot(1.0, 5.0), rel=1e-5)


def boat(speed=3.0, heading=-deg(45)):
    return BoatState(x=jnp.float32(0.0), y=jnp.float32(0.0), heading=jnp.float32(heading), speed=jnp.float32(speed),
                     target_heading=jnp.float32(heading), manoeuvring=jnp.bool_(False))


def test_current_carries_the_boat_over_the_ground():
    p = EnvParams()
    still, _ = physics_step(boat(), HOLD, POLAR, 0.0, kn(10), p)
    moved, _ = physics_step(boat(), HOLD, POLAR, 0.0, kn(10), p, current=(jnp.float32(0.4), jnp.float32(-0.3)))
    assert float(moved.speed) == float(still.speed) and float(moved.heading) == float(still.heading)
    assert float(moved.x - still.x) == pytest.approx(0.4 * p.dt, abs=1e-5)
    assert float(moved.y - still.y) == pytest.approx(-0.3 * p.dt, abs=1e-5)


def test_waves_slow_upwind_help_downwind_and_cost_on_tacks():
    p = EnvParams()
    waves = (jnp.float32(0.8), jnp.float32(0.0))  # 0.8 m from the north
    assert float(wave_speed_factor(0.0, waves, p)) == pytest.approx(1 - 0.8 * p.wave_upwind_loss)
    assert float(wave_speed_factor(math.pi, waves, p)) == pytest.approx(1 + 0.8 * p.wave_downwind_gain)
    assert float(wave_speed_factor(math.pi / 2, waves, p)) == pytest.approx(1.0, abs=1e-6)
    flat, _ = physics_step(boat(speed=5.0), HOLD, POLAR, 0.0, kn(10), p)
    rough, _ = physics_step(boat(speed=5.0), HOLD, POLAR, 0.0, kn(10), p, waves=waves)
    assert float(rough.speed) < float(flat.speed)  # close-hauled into the waves
    t_flat, _ = physics_step(boat(speed=5.0), TACK, POLAR, 0.0, kn(10), p)
    t_rough, _ = physics_step(boat(speed=5.0), TACK, POLAR, 0.0, kn(10), p, waves=(jnp.float32(0.8), jnp.float32(math.pi)))
    assert float(t_rough.speed) < float(t_flat.speed)  # extra loss on the tack (waves from behind: no upwind loss)


def test_weather_observation():
    env, p = se.make("solo-weather")
    obs, s = env.reset(jax.random.PRNGKey(2), p)
    assert obs.shape == (env.obs_dim,) and bool(jnp.all(jnp.isfinite(obs)))
    feats = obs[-WEATHER_FEATURES:]
    for i, lag in enumerate(HISTORY_LAGS):  # at reset: as if the boat had been at its starting position
        d, v = water_wind(s.wind, s.weather, s.t - lag, s.boat.x, s.boat.y, p)
        assert float(feats[2 * i + 1]) == pytest.approx(float(v) / 8.0, rel=1e-5)
    # The wind the boat feels now is what the base features report.
    d, v = env.local_wind(s, p)
    assert float(obs[14]) == pytest.approx(float(v) / 8.0, rel=1e-6)


def test_weather_block_comes_last_and_nothing_else_moves():
    key = jax.random.PRNGKey(4)
    env, p = se.make("race")
    env_w, p_w = se.make("race-weather")
    obs, _ = env.reset(key, p)
    obs_w, _ = env_w.reset(key, p_w)
    assert obs_w.shape[1] == obs.shape[1] + WEATHER_FEATURES
    assert bool(jnp.allclose(obs[:, 60:72], obs_w[:, 60:72]))  # race features at the same indices, unchanged
    assert bool(jnp.allclose(obs[:, 5:9], obs_w[:, 5:9]))  # line ends: same boats, same line
    from sail_env.weather import observe
    _, fs = env_w.reset(key, p_w)
    b0 = jax.tree.map(lambda x: x[0], fs.boats)
    expected = jnp.stack(observe(b0.wind, b0.weather, b0.wind_log, b0.log_origin, b0.t, b0.boat.x, b0.boat.y,
                                 b0.boat.heading, p_w))
    assert bool(jnp.allclose(obs_w[0, -WEATHER_FEATURES:], expected, atol=1e-6))


def test_history_is_what_the_boat_itself_sailed_in():
    """The history features are the boat's own log: the wind where it was, when it was there."""
    env, p = se.make("solo-weather")
    _, s = env.reset(jax.random.PRNGKey(5), p)
    felt = []  # (direction, speed) the boat sails in at the start of each step, at its position then
    step = jax.jit(lambda s, a: env.step_env(jax.random.PRNGKey(0), s, a, p))
    for i in range(25):
        felt.append([float(v) for v in env.local_wind(s, p)])
        obs, s, *_ = step(s, jnp.int32(4 if i < 3 else 0))  # bear away, then sail on: the boat moves
    feats = obs[-WEATHER_FEATURES:]
    for i, lag in enumerate(HISTORY_LAGS[:2]):  # 10 and 20 s ago: inside the episode
        d, v = felt[-lag]
        assert float(feats[2 * i]) == pytest.approx(float(wrap_angle(d - s.wind.base_dir)) / DIR_SCALE, abs=1e-5)
        assert float(feats[2 * i + 1]) == pytest.approx(v / 8.0, rel=1e-5)
    # 30 s ago, before the episode began (25 steps): as if at the starting position then.
    x0, y0, t0 = (float(v) for v in s.log_origin)
    d, v = water_wind(s.wind, s.weather, s.t - 30, x0, y0, p)
    assert float(feats[5]) == pytest.approx(float(v) / 8.0, rel=1e-5)
    assert t0 == pytest.approx(float(s.t) - 25.0)
    # Not the wind at the boat's current position back then (where it was not).
    d_here, _ = water_wind(s.wind, s.weather, s.t - 20, s.boat.x, s.boat.y, p)
    assert float(d_here) != pytest.approx(felt[-20][0], abs=1e-6)


def test_no_log_without_weather_observations():
    env, p = se.make("fleet")
    _, fs = env.reset(jax.random.PRNGKey(0), p)
    assert fs.boats.wind_log.shape == (5, 0, 2) and fs.boats.log_origin.shape == (5, 0)


def test_weather_presets_run_under_jit_and_vmap():
    for name in ["fleet-weather", "race-weather"]:
        env, p = se.make(name)
        n_env = 8
        _, s = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(0), n_env), p)

        @jax.jit
        def run(s, key):
            def body(s, k):
                k_a, k_s = jax.random.split(k)
                a = jax.random.randint(k_a, (n_env, env.num_agents), 0, 7)
                obs, s, r, d, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.random.split(k_s, n_env), s, a, p)
                return s, (jnp.isfinite(obs).all() & jnp.isfinite(r).all())
            return jax.lax.scan(body, s, jax.random.split(key, 60))

        _, finite = run(s, jax.random.PRNGKey(1))
        assert bool(np.all(np.asarray(finite)))


def test_bundled_start_policy_sails_with_weather_blind():
    from sail_env.agents import pretrained_agent

    env, p = se.make("fleet-weather")
    policy = pretrained_agent("fleet")(env, p)
    obs, fs = env.reset(jax.random.PRNGKey(0), p)
    a = jax.vmap(policy, in_axes=(None, 0, 0, 0))(jax.random.PRNGKey(1), obs, fs.boats, jnp.arange(5))
    assert a.shape == (5,)


def test_current_capped_by_wind_and_off_by_default():
    """current_max_frac caps the current at that fraction of the wind speed; 0 (default) changes nothing."""
    keys = jax.random.split(jax.random.PRNGKey(3), 512)
    tws = jnp.linspace(3.0, 8.0, 512)
    speed = lambda p, t: jax.vmap(lambda k, w: jnp.hypot(*(lambda x: (x.current_x, x.current_y))(
        sample_weather(k, 0.0, p, w))))(keys, t)
    off = EnvParams(**ON)
    np.testing.assert_array_equal(speed(off, tws), jax.vmap(lambda k: jnp.hypot(
        sample_weather(k, 0.0, off).current_x, sample_weather(k, 0.0, off).current_y))(keys))
    capped = speed(EnvParams(**ON, current_max_frac=0.06), tws)
    assert np.all(np.asarray(capped) <= 0.06 * np.asarray(tws) + 1e-5)
    assert float(capped.max()) > 0.04 * 8.0  # still uses the allowed range


def test_race_env_with_current_cap_resets():
    env, params = se.make("race-solo-weather", current_max_frac=0.06)
    _, st = env.reset(jax.random.PRNGKey(0), params)
    c = float(jnp.hypot(st.boats.weather.current_x[0], st.boats.weather.current_y[0]))
    assert c <= 0.06 * float(st.boats.wind.tws[0]) + 1e-5
