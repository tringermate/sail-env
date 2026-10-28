import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sail_env.train.ppo import (
    PPOConfig,
    categorical_entropy,
    categorical_log_prob,
    compute_gae,
    make_train,
)
from sail_env.train.analysis import mean_ci
from sail_env.train.cartpole import CartPole, CartPoleParams
from sail_env.wrappers import LogWrapper


def naive_gae(r, v, d, last_v, gamma, lam):
    """Straightforward double loop, the textbook definition."""
    T, E = r.shape
    adv = np.zeros((T, E))
    for e in range(E):
        next_adv, next_v = 0.0, last_v[e]
        for t in reversed(range(T)):
            nd = 1.0 - d[t, e]
            delta = r[t, e] + gamma * next_v * nd - v[t, e]
            next_adv = delta + gamma * lam * nd * next_adv
            adv[t, e] = next_adv
            next_v = v[t, e]
    return adv


def test_gae_matches_naive_loop():
    rng = np.random.default_rng(0)
    T, E = 50, 3
    r, v = rng.normal(size=(T, E)), rng.normal(size=(T, E))
    d = (rng.random((T, E)) < 0.1).astype(np.float32)
    last_v = rng.normal(size=E)
    adv, targets = compute_gae(jnp.array(r), jnp.array(v), jnp.array(d), jnp.array(last_v), 0.99, 0.95)
    np.testing.assert_allclose(np.asarray(adv), naive_gae(r, v, d, last_v, 0.99, 0.95), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(targets), np.asarray(adv) + v, rtol=1e-5, atol=1e-5)


def test_gae_lambda_one_gives_discounted_return():
    # lambda = 1, single episode ending at the last step: A_t + V_t = sum_k gamma^k r_{t+k}.
    r = jnp.ones((5, 1))
    v = jnp.zeros((5, 1))
    d = jnp.array([[0], [0], [0], [0], [1]], dtype=jnp.float32)
    _, targets = compute_gae(r, v, d, jnp.array([123.0]), 0.9, 1.0)
    expected = [sum(0.9**k for k in range(5 - t)) for t in range(5)]
    np.testing.assert_allclose(np.asarray(targets[:, 0]), expected, rtol=1e-5)


def test_categorical_helpers():
    logits = jnp.zeros((7,))
    assert float(categorical_entropy(logits)) == pytest.approx(math.log(7), rel=1e-5)
    assert float(categorical_log_prob(logits, jnp.int32(3))) == pytest.approx(-math.log(7), rel=1e-5)
    peaked = jnp.array([10.0, 0.0, 0.0])
    assert float(categorical_entropy(peaked)) < 0.01


def test_cartpole_terminates_without_control():
    env, p = CartPole(), CartPoleParams()
    _, s = env.reset(jax.random.PRNGKey(0), p)
    for t in range(500):
        _, s, r, done, _ = env.step_env(jax.random.PRNGKey(t), s, jnp.int32(1), p)  # always push right
        if bool(done):
            break
    assert 5 < t < 100  # falls over quickly


def test_mean_ci_uses_t_distribution():
    x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    m, h = mean_ci(x)
    assert m[0] == 3.0 and h[0] == pytest.approx(2.776 * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5))


def test_ppo_chunk_runs_and_is_finite():
    cfg = PPOConfig(num_envs=2, num_steps=16, total_timesteps=64, num_seeds=2, updates_per_chunk=2)
    init_fn, chunk_fn = make_train(cfg, LogWrapper(CartPole()), CartPoleParams())
    runner = jax.vmap(init_fn)(jax.random.split(jax.random.PRNGKey(0), cfg.num_seeds))
    runner, metrics = jax.jit(jax.vmap(chunk_fn))(runner)
    for k, v in metrics.items():
        assert v.shape == (2, 2), k
        assert not bool(jnp.any(jnp.isinf(v))), k  # episode metrics may be NaN (no episode finished)


def test_stagger_desynchronises_episode_clocks():
    from sail_env.config import EnvParams
    from sail_env.start_env import SoloStartEnv

    p = EnvParams()
    env = LogWrapper(SoloStartEnv(p))
    for stagger, expect_spread in [(0, False), (p.n_steps, True)]:
        cfg = PPOConfig(num_envs=256, num_steps=8, total_timesteps=2048, num_seeds=1, stagger_steps=stagger)
        init_fn, _ = make_train(cfg, env, p)
        runner = init_fn(jax.random.PRNGKey(0))
        clocks = np.asarray(runner.env_state.env_state.t)
        n_distinct = len(np.unique(np.round(clocks, 3)))
        if expect_spread:
            assert n_distinct > 50  # clocks spread over most of the 70 possible phases
        else:
            assert n_distinct == 1  # all envs at T-60
