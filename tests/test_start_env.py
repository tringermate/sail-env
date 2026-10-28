import math

import jax
import jax.numpy as jnp
import pytest

from sail_env.config import EnvParams, deg, kn
from sail_env.geometry import signed_twa
from sail_env.physics import DOWN15, HOLD, LUFF
from sail_env.polar import polar_speed
from sail_env.rules import make_line, signed_distance
from sail_env.start_env import OBS_DIM, SoloStartEnv
from sail_env.wind import Wind

P = EnvParams()
ENV = SoloStartEnv(P)
step_env = jax.jit(lambda s, a: ENV.step_env(jax.random.PRNGKey(0), s, a, P))
HEADING = -deg(42)  # starboard close-hauled, wind from north


def run(state, actions):
    total, out = 0.0, None
    for a in actions:
        _, state, r, done, info = step_env(state, jnp.int32(a))
        total += float(r)
        out = (state, done, info)
    return total, out


def timed_run_state(cross_time):
    """A steady close-hauled boat at full speed placed so its bow reaches the line centre at `cross_time`."""
    _, s = ENV.reset(jax.random.PRNGKey(0), P)
    wind = Wind(jnp.float32(0.0), jnp.float32(kn(10)), jnp.float32(0.0), jnp.float32(120.0), jnp.float32(0.0))
    v = polar_speed(ENV.polar, signed_twa(jnp.float32(HEADING), 0.0), jnp.float32(kn(10)))
    t0 = -10.0
    travel = float(v) * (cross_time - t0)
    half = 0.5 * P.boat_length
    # Bow at (0, 0) after `travel` metres along the heading.
    x0 = -(travel + half) * math.sin(HEADING)
    y0 = -(travel + half) * math.cos(HEADING)
    boat = s.boat.replace(
        x=jnp.float32(x0), y=jnp.float32(y0), heading=jnp.float32(HEADING),
        target_heading=jnp.float32(HEADING), speed=v, manoeuvring=jnp.bool_(False),
    )
    return s.replace(boat=boat, wind=wind, line=make_line(0.0, 0.0, 0.0, 0.0, P.line_length),
                     bias=jnp.float32(0.0), t=jnp.float32(t0))


def test_reset_shapes_and_initial_position():
    obs, s = jax.jit(lambda k: ENV.reset(k, P))(jax.random.PRNGKey(1))
    assert obs.shape == (OBS_DIM,) and bool(jnp.all(jnp.isfinite(obs)))
    assert float(s.t) == P.t_start
    d = -float(signed_distance(s.line, s.boat.x, s.boat.y))
    assert P.init_dist_min - 1e-3 <= d <= P.init_dist_max + 1e-3


def test_episode_length_and_auto_reset():
    _, s = ENV.reset(jax.random.PRNGKey(2), P)
    step = jax.jit(lambda k, s: ENV.step(k, s, jnp.int32(HOLD), P))
    dones = []
    for i in range(P.n_steps):
        _, s, r, done, _ = step(jax.random.PRNGKey(i), s)
        dones.append(bool(done))
        if not done:
            assert float(r) == 0.0  # sparse: no reward before the end
    assert dones == [False] * (P.n_steps - 1) + [True]
    assert float(s.t) == P.t_start  # already reset to a new episode


def test_well_timed_start_is_success():
    total, (s, done, info) = run(timed_run_state(cross_time=1.5), [HOLD] * 20)
    assert bool(done)
    assert float(info["start_time"]) == pytest.approx(1.5, abs=0.1)
    assert bool(info["success"]) and total == 1.0


def test_early_start_is_ocs_and_penalised():
    total, (s, done, info) = run(timed_run_state(cross_time=-0.5), [HOLD] * 20)
    assert bool(info["ocs"]) and not bool(info["started"])
    assert total == -1.0


def test_late_start_is_not_success():
    total, (_, _, info) = run(timed_run_state(cross_time=5.0), [HOLD] * 20)
    assert bool(info["started"]) and not bool(info["success"]) and total == 0.0


def test_slow_start_is_not_success():
    # Luff for the last seconds before the gun: crosses on time but well below polar speed.
    total, (_, _, info) = run(timed_run_state(cross_time=0.5), [HOLD] * 6 + [LUFF] * 14)
    assert float(info["start_speed_ratio"]) < P.success_speed_frac
    assert not bool(info["success"]) and total <= 0.0


def test_vmap_over_envs():
    n = 1024
    keys = jax.random.split(jax.random.PRNGKey(3), n)
    obs, s = jax.vmap(ENV.reset, in_axes=(0, None))(keys, P)
    acts = jnp.arange(n) % ENV.num_actions
    obs, s, r, d, info = jax.jit(jax.vmap(ENV.step, in_axes=(0, 0, 0, None)))(keys, s, acts, P)
    assert obs.shape == (n, OBS_DIM) and r.shape == (n,)


# --- reward variants -----------------------------------------------------------------------------


def test_beat_vmg_matches_orc():
    from sail_env.polar import beat_vmg, read_orc_csv

    orc = read_orc_csv()
    k = orc["tws"].index(10.0)
    assert float(beat_vmg(ENV.polar, jnp.float32(kn(10)))) / kn(1) == pytest.approx(orc["beat_vmg"][k], abs=0.05)


def test_potential_zero_when_on_schedule():
    from sail_env.physics import bow_position
    from sail_env.polar import beat_vmg

    s = timed_run_state(cross_time=1.5)
    vmg = float(beat_vmg(ENV.polar, s.wind.tws))
    bx, by = bow_position(s.boat, P)
    # Move the boat straight down so the bow is exactly vmg * time_to_gun below the line.
    s = s.replace(boat=s.boat.replace(y=s.boat.y - by - vmg * 10.0))
    assert float(ENV.potential(s, P)) == pytest.approx(0.0, abs=1e-4)
    s_late = s.replace(boat=s.boat.replace(y=s.boat.y - vmg * 6.0))  # 6 s too far away
    assert float(ENV.potential(s_late, P)) == pytest.approx(-6.0 / 60.0, abs=1e-3)


def test_shaping_telescopes_with_gamma_one():
    # With gamma = 1 the shaping terms over a whole episode sum to -Phi(s_0).
    p = EnvParams(reward_shaping=1.0, shaping_gamma=1.0)
    _, s0 = ENV.reset(jax.random.PRNGKey(5), p)
    phi0 = float(ENV.potential(s0, p))
    step = jax.jit(lambda s, a: ENV.step_env(jax.random.PRNGKey(0), s, a, p))
    s, total, actions = s0, 0.0, jax.random.randint(jax.random.PRNGKey(1), (p.n_steps,), 0, 7)
    for a in actions:
        _, s, r, done, info = step(s, a)
        total += float(r)
    terminal = 1.0 if bool(info["success"]) else (-1.0 if bool(info["ocs"]) else 0.0)
    assert total == pytest.approx(terminal - phi0, abs=1e-3)


def test_shaping_off_by_default_and_ocs_penalty_switch():
    total, _ = run(timed_run_state(cross_time=-0.5), [HOLD] * 20)
    assert total == -1.0  # default: sparse, OCS penalised
    p = EnvParams(ocs_penalty=0.0)
    step = jax.jit(lambda s, a: ENV.step_env(jax.random.PRNGKey(0), s, a, p))
    s, total = timed_run_state(cross_time=-0.5), 0.0
    for _ in range(20):
        _, s, r, _, info = step(s, jnp.int32(HOLD))
        total += float(r)
    assert bool(info["ocs"]) and total == 0.0


def test_speed_aware_potential_charges_slow_boats():
    from sail_env.physics import bow_position
    from sail_env.polar import beat_vmg

    p = EnvParams(shaping_speed_aware=True)
    s = timed_run_state(cross_time=1.5)
    vmg = float(beat_vmg(ENV.polar, s.wind.tws))
    _, by = bow_position(s.boat, p)
    s = s.replace(boat=s.boat.replace(y=s.boat.y - by - vmg * 10.0))  # on schedule at full VMG
    # Close-hauled at full polar speed: VMG towards the line = beat VMG (approx.), no extra time.
    assert float(ENV.potential(s, p)) == pytest.approx(0.0, abs=0.01)
    # Same place, stopped: charged tau_accel = 4 s of acceleration.
    stopped = s.replace(boat=s.boat.replace(speed=jnp.float32(0.0)))
    assert float(ENV.potential(stopped, p)) == pytest.approx(-P.tau_accel / 60.0, abs=0.01)
    # The position-only potential cannot tell them apart.
    assert float(ENV.potential(stopped, P)) == pytest.approx(float(ENV.potential(s, P)), abs=1e-6)


# --- start distribution and observation layout ---------------------------------------------------


def _resets(p, n=4096):
    env = SoloStartEnv(p)
    obs, s = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(jax.random.split(jax.random.PRNGKey(9), n), p)
    return env, obs, s


@pytest.mark.parametrize("random_start", [False, True])
def test_every_start_is_reachable(random_start):
    from sail_env.polar import beat_vmg

    p = EnvParams(random_start_time=random_start)
    env, _, s = _resets(p)
    below = -jax.vmap(signed_distance)(s.line, s.boat.x, s.boat.y)
    vmg = jax.vmap(lambda w: beat_vmg(env.polar, w))(s.wind.tws)
    time_to_gun = -s.t
    assert bool(jnp.all(below <= p.init_reach_frac * vmg * time_to_gun + 1e-3))
    assert bool(jnp.all(below > 0))
    if random_start:
        assert float(time_to_gun.min()) >= p.init_time_to_gun_min - 1e-4 and float(time_to_gun.max()) <= 60.0
        assert float(time_to_gun.std()) > 10.0  # really spread out
    else:
        assert bool(jnp.all(time_to_gun == 60.0))


def test_observation_layout():
    from sail_env.start_env import N_NEIGHBOURS, NEIGHBOUR_FEATURES, OWN_FEATURES

    _, obs, s = _resets(EnvParams(), n=256)
    assert obs.shape == (256, OWN_FEATURES + N_NEIGHBOURS * NEIGHBOUR_FEATURES)
    assert bool(jnp.all(obs[:, OWN_FEATURES:] == 0.0))  # no other boats in solo
    assert bool(jnp.allclose(obs[:, OWN_FEATURES - 1], s.wind.tws / 8.0))  # wind speed is observed


def test_rollout_stops_at_episode_end():
    from sail_env.render import rollout

    p = EnvParams(random_start_time=True)
    env = SoloStartEnv(p)
    for k in range(5):
        traj, actions, rewards, info = rollout(env, p, jax.random.PRNGKey(k), lambda kk, o, i: jnp.int32(0))
        assert float(traj.t.max()) == pytest.approx(P.t_end)  # clock never runs past T+10
        assert float(jnp.abs(rewards).sum()) <= 1.0  # at most one terminal reward


def test_speed_bonus_only_around_the_gun_and_strict_success():
    p = EnvParams(speed_bonus=0.1, success_window=10.0)
    step = jax.jit(lambda s, a: ENV.step_env(jax.random.PRNGKey(0), s, a, p))
    s = timed_run_state(cross_time=5.0)  # close-hauled at full speed towards the line, from T-10
    rewards, info = [], None
    for _ in range(20):
        _, s, r, done, info = step(s, jnp.int32(HOLD))
        rewards.append(float(r))
    clocks = list(range(-9, 11))  # clock after each step
    for t, r in zip(clocks, rewards):
        if t > 3:
            continue  # bonus window ends at T+3; the terminal reward lands at T+10
        assert r == pytest.approx(0.1, abs=0.015)  # close-hauled at full speed: VMG ~ best VMG
    assert bool(info["success"]) and not bool(info["success_strict"])  # T+5: ok for 10 s window, not for 3 s


# --- starboard and height after the start ------------------------------------------------------


def mirrored_port_state(cross_time):
    """The starboard timed run mirrored east-west: same run on port tack (heading +42 deg)."""
    s = timed_run_state(cross_time)
    h = -s.boat.heading
    return s.replace(boat=s.boat.replace(x=-s.boat.x, heading=h, target_heading=h))


def _episode(state, params, actions):
    step = jax.jit(lambda s, a: ENV.step_env(jax.random.PRNGKey(0), s, a, params))
    total, info = 0.0, None
    for a in actions:
        _, state, r, done, info = step(state, jnp.int32(a))
        total += float(r)
    return total, info


def test_port_start_loses_reward_and_is_not_a_good_start():
    p = EnvParams(port_penalty=1.0)
    total_s, info_s = _episode(timed_run_state(1.5), p, [HOLD] * 20)
    total_p, info_p = _episode(mirrored_port_state(1.5), p, [HOLD] * 20)
    assert bool(info_s["success_strict"]) and bool(info_p["success_strict"])  # both on time and fast
    assert bool(info_s["good_start"]) and not bool(info_p["good_start"])  # only starboard is "good"
    assert total_s == pytest.approx(1.0) and total_p == pytest.approx(0.0)
    half, _ = _episode(mirrored_port_state(1.5), EnvParams(port_penalty=0.5), [HOLD] * 20)
    assert half == pytest.approx(0.5)


def test_upwind_efficiency_rewards_height_after_the_start():
    p = EnvParams(upwind_bonus=0.5)
    # Close-hauled at full speed all the way: nearly the polar's best VMG.
    total_high, info_high = _episode(timed_run_state(1.5), p, [HOLD] * 20)
    assert float(info_high["upwind_efficiency"]) > 0.85
    assert total_high == pytest.approx(1.0 + 0.5 * float(info_high["upwind_efficiency"]), abs=1e-4)
    # Bear away onto a reach right after crossing: fast, but little height.
    total_low, info_low = _episode(timed_run_state(1.5), p, [HOLD] * 13 + [DOWN15] * 3 + [HOLD] * 4)
    assert bool(info_low["success_strict"])
    assert float(info_low["upwind_efficiency"]) < float(info_high["upwind_efficiency"]) - 0.1
    assert total_low < total_high


def test_starboard_approach_bonus_ramps_towards_the_gun():
    p = EnvParams(starboard_bonus=0.01)
    step = jax.jit(lambda s, a: ENV.step_env(jax.random.PRNGKey(0), s, a, p))

    def rewards(state, n):
        out = []
        for _ in range(n):
            _, state, r, _, _ = step(state, jnp.int32(HOLD))
            out.append(float(r))
        return out

    stbd = rewards(timed_run_state(1.5), 12)  # starboard, from T-10; crosses at T+1.5
    port = rewards(mirrored_port_state(1.5), 12)
    for t, r in zip(range(-9, 3), stbd):  # clock after each step: T-9 .. T+2
        expected = 0.01 * min((t + 60) / 60.0, 1.0) if t <= 1 else 0.0
        assert r == pytest.approx(expected, abs=1e-4)
    assert stbd[-1] == pytest.approx(0.0, abs=1e-6)  # T+2: already started, no more bonus
    assert all(r == pytest.approx(0.0, abs=1e-6) for r in port)  # nothing on port


def test_on_the_line_at_the_gun_rule():
    from sail_env.polar import polar_speed as ps

    p = EnvParams(gun_distance_max=3.0)
    v = float(ps(ENV.polar, signed_twa(jnp.float32(HEADING), 0.0), jnp.float32(kn(10))))
    # Crossing at T+0.8 s means the bow is ~0.8 s x v (~2.4 m) below the line at the gun: on the line.
    _, info_close = _episode(timed_run_state(0.8), p, [HOLD] * 20)
    assert 0.0 <= 0.8 * v <= 3.0
    assert bool(info_close["success"]) and bool(info_close["good_start_on_line"]) and bool(info_close["on_line_at_gun"])
    # Crossing at T+2 s: still inside the 3 s window, but ~6 m short at the gun: not on the line.
    _, info_far = _episode(timed_run_state(2.0), p, [HOLD] * 20)
    assert bool(info_far["success_strict"]) and not bool(info_far["on_line_at_gun"])
    assert not bool(info_far["success"]) and not bool(info_far["good_start_on_line"])
    # Default params: rule disabled for training success, as before.
    _, info_default = _episode(timed_run_state(2.0), P, [HOLD] * 20)
    assert bool(info_default["success"])


def test_shooting_head_to_wind_after_the_start_does_not_pay():
    from sail_env.physics import UP15

    p = EnvParams(upwind_bonus=0.5)
    # Close-hauled all the way vs. turning head to wind ("shooting") right after crossing at T+1.5.
    total_ch, info_ch = _episode(timed_run_state(1.5), p, [HOLD] * 20)
    total_sh, info_sh = _episode(timed_run_state(1.5), p, [HOLD] * 12 + [UP15] * 3 + [HOLD] * 5)
    assert bool(info_sh["success_strict"])  # still a good start ...
    assert float(info_ch["upwind_efficiency"]) > 0.95  # ... but sailing the polar optimum scores ~1
    assert float(info_sh["upwind_efficiency"]) < float(info_ch["upwind_efficiency"]) - 0.05
    assert total_sh < total_ch


def test_efficiency_tail_makes_stalling_at_t10_costly():
    from sail_env.physics import UP15

    shoot = [HOLD] * 12 + [UP15] * 3 + [HOLD] * 5
    gaps = []
    for tail in (0.0, 10.0):
        p = EnvParams(upwind_bonus=1.0, efficiency_tail_seconds=tail)
        _, info_ch = _episode(timed_run_state(1.5), p, [HOLD] * 20)
        _, info_sh = _episode(timed_run_state(1.5), p, shoot)
        gaps.append(float(info_ch["upwind_efficiency"]) - float(info_sh["upwind_efficiency"]))
    assert gaps[1] > 1.5 * gaps[0] > 0  # the tail makes ending up stalled much more costly (0.18 -> 0.33)


def test_height_after_start_is_measured_along_the_wind_not_the_line():
    from sail_env.config import deg as _deg

    # A biased line (committee end 10 deg up): a close-hauled boat still sails the polar optimum, so
    # its upwind efficiency stays ~1 even though it does not head straight at the line's normal.
    p = EnvParams(upwind_bonus=1.0, efficiency_tail_seconds=10.0)
    s = timed_run_state(1.5)
    s = s.replace(line=make_line(0.0, 0.0, 0.0, _deg(10.0), P.line_length), bias=jnp.float32(_deg(10.0)))
    _, info = _episode(s, p, [HOLD] * 20)
    if bool(info["started"]):
        assert float(info["upwind_efficiency"]) > 0.95
