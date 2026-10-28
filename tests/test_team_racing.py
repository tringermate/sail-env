"""Team racing (EnvParams.team_size): two teams, sum of places, ties to the team without first place."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sail_env as se
from sail_env.race_env import TEAM_FEATURES, team_result


@pytest.mark.parametrize("rank, win0", [
    ([0, 1, 2, 3], True),    # 1+2 vs 3+4
    ([0, 3, 1, 2], False),   # 1+4 = 5 vs 2+3 = 5: tie, team 0 had first place -> team 1 wins
    ([1, 2, 0, 3], True),    # 2+3 = 5 vs 1+4 = 5: tie, team 1 had first place -> team 0 wins
    ([2, 3, 0, 1], False),   # 3+4 vs 1+2
    ([0, 2, 1, 3], True),    # 1+3 = 4 vs 2+4 = 6
])
def test_team_result(rank, win0):
    rank = jnp.asarray(rank, jnp.float32)
    win, score = team_result(rank, 2)
    places = np.asarray(rank) + 1
    s0, s1 = places[:2].sum(), places[2:].sum()
    first_team = int(np.argmin(np.asarray(rank)) >= 2)
    expect = s0 < s1 or (s0 == s1 and first_team == 1)
    assert expect == win0
    assert bool(win[0]) == bool(win[1]) == expect and bool(win[2]) == bool(win[3]) == (not expect)
    assert np.allclose(np.asarray(score), [s0, s0, s1, s1])


def test_tie_goes_to_the_team_without_first_place():
    win, _ = team_result(jnp.asarray([0.0, 3.0, 1.0, 2.0]), 2)  # 1+4 vs 2+3: team 0 has first place
    assert not bool(win[0]) and bool(win[2])


def test_teammate_flags_and_obs_dim():
    env, p = se.make("race", fleet_size=4, team_size=2)
    plain, _ = se.make("race", fleet_size=4)
    assert env.obs_dim == plain.obs_dim + TEAM_FEATURES
    obs, s = env.reset(jax.random.PRNGKey(0), p)
    assert obs.shape == (4, env.obs_dim)
    flags = np.asarray(obs[:, plain.obs_dim:])
    assert flags.shape == (4, TEAM_FEATURES)
    # each boat has exactly one teammate among its 3 neighbours (all near at the start), none in the empty slots
    assert np.all(flags[:, :3].sum(axis=1) == 1) and np.all(flags[:, 3:] == 0)
    # the observation without the flags is the plain race observation
    obs_plain, _ = plain.reset(jax.random.PRNGKey(0), p.replace(team_size=0) if hasattr(p, "replace") else p)
    np.testing.assert_allclose(np.asarray(obs[:, :plain.obs_dim]), np.asarray(obs_plain), atol=1e-6)


def test_team_reward_at_the_end():
    env, p = se.make("race", fleet_size=4, team_size=2, t_end=-80.0)  # end the episode almost at once
    obs, s = env.reset(jax.random.PRNGKey(1), p)
    done = jnp.zeros(4, bool)
    for _ in range(20):
        obs, s, r, done, info = env.step_env(jax.random.PRNGKey(0), s, jnp.zeros(4, jnp.int32), p)
        if bool(done[0]):
            break
    assert bool(done[0])
    win = np.asarray(info["team_win"])
    assert win[0] == win[1] and win[2] == win[3] and win[0] != win[2]
    # +race_rank_reward/2 to the winners, -race_rank_reward/2 to the losers, on top of the other terms
    r = np.asarray(r)
    assert (r[win] - r[~win]).min() > 0


def test_team_racing_needs_two_equal_teams():
    with pytest.raises(ValueError):
        se.make("race", fleet_size=5, team_size=2)


def test_digital_racers_finish_a_team_race():
    """The race sailor finds the race features before the teammate flags."""
    from sail_env.agents import load_agent

    env, p = se.make("race", fleet_size=4, team_size=2, rrs16=True)
    pol = load_agent("digital")(env, p)
    obs, s = env.reset(jax.random.PRNGKey(3), p)

    def body(c, i):
        o, s, fin = c
        a = jax.vmap(pol)(jax.random.split(jax.random.PRNGKey(i), 4), o, s.boats, jnp.arange(4))
        o2, s2, _, d, _ = env.step_env(jax.random.PRNGKey(0), s, a, p)
        s2 = jax.tree.map(lambda new, old: jnp.where(fin, old, new), s2, s)
        return (o2, s2, fin | d[0]), None

    (_, s_end, done), _ = jax.jit(lambda: jax.lax.scan(body, (obs, s, jnp.bool_(False)), jnp.arange(p.n_steps)))()
    assert bool(done) and np.all(np.isfinite(np.asarray(s_end.finish_time)))


def test_close_finishes_are_ranked_not_tied():
    """Boats finishing 0.3 s apart get different places (a float32 score 1e7 - t used to tie them)."""
    from sail_env.race_env import N_LEGS

    env, p = se.make("race", fleet_size=4)
    obs, s = env.reset(jax.random.PRNGKey(0), p)
    # all four boats finished, 0.3 s apart, in the order 2, 0, 3, 1; the race ends this step
    s = s.replace(leg=jnp.full(4, N_LEGS, jnp.int32), finish_time=jnp.asarray([640.3, 640.9, 640.0, 640.6]))
    _, _, _, done, info = env.step_env(jax.random.PRNGKey(0), s, jnp.zeros(4, jnp.int32), p)
    assert bool(done[0])
    np.testing.assert_array_equal(np.asarray(info["race_rank"]), [1.0, 3.0, 0.0, 2.0])


def _drop_team_block(obs, p):
    from sail_env.race_env import TEAM_FEATURES, team_block_start

    at = team_block_start(p, obs.shape[-1])
    return np.concatenate([obs[..., :at], obs[..., at + TEAM_FEATURES:]], axis=-1)


def test_team_flags_sit_before_the_weather_block_and_blind_racers_skip_them():
    from sail_env.agents import load_agent

    env_t, p_t = se.make("race-weather", fleet_size=4, team_size=2)
    env_i, p_i = se.make("race-weather", fleet_size=4)
    key = jax.random.PRNGKey(3)
    obs_t, s_t = env_t.reset(key, p_t)
    obs_i, s_i = env_i.reset(key, p_i)
    np.testing.assert_allclose(_drop_team_block(np.asarray(obs_t), p_t), np.asarray(obs_i), atol=1e-6)
    pol_t, pol_i = load_agent("rl-racer-weather")(env_t, p_t), load_agent("rl-racer-weather")(env_i, p_i)
    for b in range(4):
        boat_t, boat_i = jax.tree.map(lambda x: x[b], s_t.boats), jax.tree.map(lambda x: x[b], s_i.boats)
        assert int(pol_t(key, obs_t[b], boat_t, b)) == int(pol_i(key, obs_i[b], boat_i, b))


def test_team_racer_sails_an_individual_race_with_no_teammates(tmp_path):
    from flax import serialization

    import yaml

    from sail_env.agents import PRETRAINED, pretrained_agent
    from sail_env.agents.policy import load_policy
    from sail_env.train.run import widen_input_layer

    # a "team racer" = the weather racer widened with zero-weight teammate flags: with no teammates it must sail
    # exactly as the weather racer (both without the start policy handover, so the racer network itself acts)
    env_t, p_t = se.make("race-weather", fleet_size=4, team_size=2)
    env_i, p_i = se.make("race-weather", fleet_size=4)
    cfg, _, _, params, _ = load_policy(PRETRAINED / "racer-weather")
    cfg.pop("start_bank", None)
    wide, _ = widen_input_layer(params, env_t.obs_dim, p_t)
    for name, tree in (("racer", params), ("team", wide)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        (tmp_path / name / "params.msgpack").write_bytes(serialization.msgpack_serialize(jax.tree.map(np.asarray, tree)))
    pol_racer = pretrained_agent(str(tmp_path / "racer"))(env_i, p_i)
    pol_team = pretrained_agent(str(tmp_path / "team"))(env_i, p_i)
    key = jax.random.PRNGKey(5)
    obs, s = env_i.reset(key, p_i)
    obs = obs + jax.random.normal(key, obs.shape)  # varied inputs, so the comparison is not just one action
    for b in range(4):
        boat = jax.tree.map(lambda x: x[b], s.boats)
        assert int(pol_team(key, obs[b], boat, b)) == int(pol_racer(key, obs[b], boat, b))


def test_widened_network_acts_as_before_in_a_weather_team_race():
    from sail_env.train.ppo import Actor, Critic
    from sail_env.train.run import widen_input_layer

    env_t, p_t = se.make("race-weather", fleet_size=4, team_size=2)
    n_in = env_t.obs_dim - 5
    k = jax.random.split(jax.random.PRNGKey(0), 3)
    actor, critic = Actor(env_t.num_actions, 16), Critic(16)
    params = {"actor": actor.init(k[0], jnp.zeros(n_in)), "critic": critic.init(k[1], jnp.zeros(n_in))}
    params = jax.tree.map(lambda x: x[None], params)  # one seed
    wide, at = widen_input_layer(params, env_t.obs_dim, p_t)
    assert at == env_t.obs_dim - 5 - 22
    obs = jax.random.normal(k[2], (env_t.obs_dim,))
    narrow_obs = jnp.asarray(_drop_team_block(np.asarray(obs), p_t))
    one = lambda t: jax.tree.map(lambda x: x[0], t)  # noqa: E731
    np.testing.assert_allclose(actor.apply(one(wide["actor"]), obs), actor.apply(one(params["actor"]), narrow_obs),
                               rtol=1e-5, atol=1e-5)
