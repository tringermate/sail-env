"""Opponent pool (PPOConfig.opp_frac): learner boats only in the loss, opponents from snapshots and fixed agents."""

import jax
import jax.numpy as jnp
import numpy as np

import sail_env as se
from sail_env.train.ppo import Actor, Critic, PPOConfig, Transition, make_train, ppo_loss, train
from sail_env.train.run import make_teacher


def test_masked_loss_equals_the_loss_on_the_learner_entries():
    cfg = PPOConfig(hidden=16)
    actor, critic = Actor(7, 16), Critic(16)
    k = jax.random.split(jax.random.PRNGKey(0), 6)
    obs = jax.random.normal(k[0], (8, 10))
    params = {"actor": actor.init(k[1], obs[0]), "critic": critic.init(k[2], obs[0])}
    batch = Transition(obs, jax.random.randint(k[3], (8,), 0, 7), -jnp.ones(8) * 1.9, jnp.zeros(8), jnp.zeros(8),
                       jnp.zeros(8, bool), None)
    adv, targ = jax.random.normal(k[4], (8,)), jax.random.normal(k[5], (8,))
    mask = jnp.arange(8) < 5
    masked, _ = ppo_loss(params, actor, critic, batch._replace(learner=mask), adv, targ, cfg, 0.01)
    first = jax.tree.map(lambda x: x[:5] if x is not None else None, batch)
    plain, _ = ppo_loss(params, actor, critic, first, adv[:5], targ[:5], cfg, 0.01)
    np.testing.assert_allclose(float(masked), float(plain), rtol=1e-5)


def test_league_trains_against_snapshots_and_a_fixed_agent():
    env, p = se.make("race", fleet_size=4, t_end=-60.0)  # 30-step "races" (ranked at the time limit)
    from sail_env.wrappers import LogWrapper

    wrapped = LogWrapper(env)
    cfg = PPOConfig(num_envs=8, num_steps=32, total_timesteps=8 * 32 * 6, num_minibatches=2, hidden=16, num_seeds=1,
                    updates_per_chunk=2, opp_frac=0.5, opp_learners=2, opp_snapshots=2, opp_snapshot_every=1)
    digital = make_teacher("digital", env, p)
    runner, metrics = train(cfg, wrapped, p, jax.random.PRNGKey(0), opponents=[digital])
    assert np.isfinite(np.nanmean(metrics["pool_place"])) and np.isfinite(np.nanmean(metrics["pool_fixed_place"]))
    snaps = runner.league["snaps"]["params"]["Dense_0"]["kernel"]  # (seeds, K, in, out)
    assert not np.allclose(np.asarray(snaps[0, 0]), np.asarray(snaps[0, 1]))  # refreshed: newest != older
    choice = np.asarray(runner.league["choice"])
    assert choice.min() >= 0 and choice.max() < 3  # 2 snapshots + 1 fixed agent


def test_learner_mask_marks_the_pool_envs():
    env, p = se.make("race", fleet_size=4, t_end=-80.0)
    from sail_env.wrappers import LogWrapper

    cfg = PPOConfig(num_envs=4, num_steps=4, total_timesteps=16, num_minibatches=1, hidden=16, opp_frac=0.5,
                    opp_learners=1, opp_snapshots=1)
    init_fn, chunk_fn = make_train(cfg, LogWrapper(env), p)
    runner = init_fn(jax.random.PRNGKey(0))
    assert runner.league["snaps"]["params"]["Dense_0"]["kernel"].shape[0] == 1
    _, metrics = jax.jit(chunk_fn)(runner)
    assert "pool_place" in metrics


def test_annealed_params_phase_the_shaping_out():
    from sail_env.train.run import make_annealed_params

    env, p = se.make("race", race_shaping=0.1, race_time_cost=1 / 600)
    fn = make_annealed_params(p, {"race_shaping": [0.1, 0.0, 0.2, 0.6], "race_time_cost": [1 / 600, 0.0, 0.2, 0.6]})
    for progress, want in [(0.0, 0.1), (0.2, 0.1), (0.4, 0.05), (0.6, 0.0), (1.0, 0.0)]:
        params, logged = fn(jnp.float32(progress))
        assert float(params.race_shaping) == np.float32(want) or abs(float(params.race_shaping) - want) < 1e-6
        assert abs(float(logged["race_time_cost"]) - want / 0.1 / 600) < 1e-7


def test_mixed_pool_draws_an_opponent_per_boat():
    env, p = se.make("race", fleet_size=4, t_end=-60.0)
    from sail_env.wrappers import LogWrapper

    cfg = PPOConfig(num_envs=64, num_steps=4, total_timesteps=64 * 4, num_minibatches=1, hidden=16, opp_frac=0.5,
                    opp_learners=2, opp_snapshots=2, opp_mix=True)
    init_fn, chunk_fn = make_train(cfg, LogWrapper(env), p, opponents=[make_teacher("digital", env, p)])
    runner = init_fn(jax.random.PRNGKey(0))
    choice = np.asarray(runner.league["choice"])
    assert choice.shape == (64, 4) and (choice[:, 2] != choice[:, 3]).any()  # pool boats differ within an env
    _, metrics = jax.jit(chunk_fn)(runner)
    assert "pool_fixed_place" in metrics


def test_a_run_seed_can_be_named_as_an_agent():
    from sail_env.agents import load_agent

    env, p = se.make("race", fleet_size=4)
    policy = load_agent("racer#1")(env, p)
    obs, s = env.reset(jax.random.PRNGKey(0), p)
    a = policy(jax.random.PRNGKey(1), obs[0], jax.tree.map(lambda x: x[0], s.boats), 0)
    assert 0 <= int(a) < env.num_actions
