"""Kickstarting: PPO with an extra loss pulling the policy towards a teacher's actions."""

import jax
import jax.numpy as jnp
import numpy as np

import sail_env as se
from sail_env.train.cartpole import CartPole, CartPoleParams
from sail_env.train.curriculum.config_env import ConfigResetEnv, make_race_start_bank
from sail_env.train.ppo import PPOConfig, make_train
from sail_env.train.run import make_teacher
from sail_env.wrappers import LogWrapper


def always_left(key, env_state, obs):
    return jnp.zeros(obs.shape[:-1], jnp.int32)


def run(cfg, env, params, teacher=None, chunks=1):
    init_fn, chunk_fn = make_train(cfg, env, params, teacher=teacher)
    runner = jax.vmap(init_fn)(jax.random.split(jax.random.PRNGKey(0), cfg.num_seeds))
    chunk = jax.jit(jax.vmap(chunk_fn))
    out = []
    for _ in range(chunks):
        runner, m = chunk(runner)
        out.append(m)
    return runner, {k: np.concatenate([np.asarray(m[k]) for m in out], axis=1) for k in out[0]}


def test_zero_weight_teacher_changes_nothing():
    # kick_coef = 0 (the default): the teacher is never called and training is exactly the plain PPO run.
    cfg = PPOConfig(num_envs=4, num_steps=32, total_timesteps=4 * 32 * 4, num_seeds=2, updates_per_chunk=4)
    env, p = LogWrapper(CartPole()), CartPoleParams()
    r0, m0 = run(cfg, env, p)
    r1, m1 = run(cfg, env, p, teacher=always_left)
    assert set(m0) == set(m1) and "teacher_agreement" not in m1
    for a, b in zip(jax.tree.leaves(r0.params), jax.tree.leaves(r1.params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_policy_learns_to_copy_the_teacher():
    cfg = PPOConfig(num_envs=4, num_steps=32, total_timesteps=4 * 32 * 20, num_seeds=1, updates_per_chunk=10,
                    kick_coef=5.0, kick_until=10.0)  # strong, (almost) constant weight
    _, m = run(cfg, LogWrapper(CartPole()), CartPoleParams(), teacher=always_left, chunks=2)
    agree, ce = m["teacher_agreement"][0], m["teacher_ce"][0]
    assert agree[-3:].mean() > 0.9 and ce[-3:].mean() < ce[:3].mean()


def test_kick_weight_decays_to_zero():
    cfg = PPOConfig(num_envs=2, num_steps=8, total_timesteps=2 * 8 * 10, num_seeds=1, updates_per_chunk=10,
                    kick_coef=1.0, kick_until=0.5)
    _, m = run(cfg, LogWrapper(CartPole()), CartPoleParams(), teacher=always_left)
    k = m["kick_coef"][0]
    assert k[0] == 1.0 and k[4] > 0.0 and np.all(k[5:] == 0.0) and np.all(np.diff(k) <= 0)


def test_digital_teacher_in_the_race_with_a_start_bank():
    # The run.py teacher unwraps LogWrapper + ConfigResetEnv and asks the digital racer for every boat.
    raw_env, p = se.make("race-solo", race_legs=2)
    bank = make_race_start_bank(raw_env, p, jax.random.PRNGKey(1), 4)
    env = LogWrapper(ConfigResetEnv(raw_env, bank))
    from sail_env.train.curriculum.config_env import ConfigParams  # params of the bank env

    params = ConfigParams(p, jnp.linspace(0.25, 1.0, 4))  # cumulative: uniform over the 4 starts
    cfg = PPOConfig(num_envs=2, num_steps=8, total_timesteps=2 * 8 * 2, num_seeds=1, updates_per_chunk=2,
                    kick_coef=1.0)
    _, m = run(cfg, env, params, teacher=make_teacher("digital", raw_env, p))
    assert np.all(np.isfinite(m["teacher_ce"])) and np.all((m["teacher_agreement"] >= 0) & (m["teacher_agreement"] <= 1))
