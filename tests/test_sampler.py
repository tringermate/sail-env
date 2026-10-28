import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sail_env.train.curriculum import sampler as smp
from sail_env.train.curriculum.sampler import SamplerConfig


def with_rates(rates, h=10):
    """A sampler state whose buffers are full with the given success rates (multiples of 1/h)."""
    n = len(rates)
    out = np.zeros((n, h), np.float32)
    for i, r in enumerate(rates):
        out[i, : int(round(r * h))] = 1.0
    return smp.SamplerState(outcomes=jnp.asarray(out), count=jnp.full(n, h, jnp.int32), ptr=jnp.zeros(n, jnp.int32))


@pytest.mark.parametrize("method", ["uniform", "sfl", "sgs"])
def test_empty_buffers_sample_uniformly(method):
    s = smp.init(8, 5)
    np.testing.assert_allclose(np.asarray(smp.probs(s, SamplerConfig(method=method))), 1 / 8, rtol=1e-5)


def test_update_counts_every_episode_and_wraps_around():
    s = smp.init(4, 3)
    # Config 1 finishes twice in the same batch, config 2 once; masked entries are ignored.
    s = smp.update(s, jnp.array([1, 1, 2, 3]), jnp.array([1, 0, 1, 1]), jnp.array([True, True, True, False]))
    np.testing.assert_array_equal(np.asarray(s.count), [0, 2, 1, 0])
    np.testing.assert_allclose(np.asarray(smp.success_rate(s)), [0, 0.5, 1, 0])
    # Overfill config 0 (H = 3): only the last 3 outcomes count.
    s = smp.update(s, jnp.zeros(5, jnp.int32), jnp.array([1, 1, 0, 0, 0]), jnp.ones(5, bool))
    assert int(s.count[0]) == 3 and float(smp.success_rate(s)[0]) == 0.0


def test_sgs_favours_sometimes_successful_configs():
    s = with_rates([0.0, 0.2, 0.5, 0.8, 1.0])
    p_sgs = np.asarray(smp.probs(s, SamplerConfig(method="sgs", kappa=5.0, temperature=1.0)))
    assert np.argmax(p_sgs) == 2  # peak at p = 0.5
    assert p_sgs[0] < p_sgs[1] and p_sgs[4] < p_sgs[3]  # never-solved and always-solved get least
    p_sfl = np.asarray(smp.probs(s, SamplerConfig(method="sfl")))
    assert np.argmax(p_sfl) == 2 and p_sfl[0] < 1e-3 and p_sfl[4] < 1e-3
    # Target t = 0.66 shifts SGS towards configs that succeed more often than not.
    p_66 = np.asarray(smp.probs(s, SamplerConfig(method="sgs", target=0.66, kappa=5.0, temperature=1.0)))
    assert p_66[3] / p_66[1] > p_sgs[3] / p_sgs[1]


def test_sampling_from_cdf_matches_probabilities():
    probs = jnp.array([0.1, 0.0, 0.6, 0.3])
    ids = smp.sample(jnp.cumsum(probs), jax.random.PRNGKey(0), (200_000,))
    freq = np.bincount(np.asarray(ids), minlength=4) / 200_000
    np.testing.assert_allclose(freq, np.asarray(probs), atol=0.005)


def test_curriculum_runs_inside_ppo():
    from sail_env.train.ppo import PPOConfig, make_train
    from sail_env.config import EnvParams
    from sail_env.train.curriculum.config_env import ConfigResetEnv, Curriculum, make_config_set
    from sail_env.start_env import SoloStartEnv
    from sail_env.wrappers import LogWrapper

    p = EnvParams(gun_distance_max=3.0, port_penalty=1.0)
    base = SoloStartEnv(p)
    sc = SamplerConfig(method="sgs", num_configs=64, history=10)
    configs = make_config_set(base, p, jax.random.PRNGKey(1), sc.num_configs)
    env = LogWrapper(ConfigResetEnv(base, configs))
    cfg = PPOConfig(num_envs=64, num_steps=32, total_timesteps=64 * 32 * 4, num_seeds=1, updates_per_chunk=4,
                    stagger_steps=70)
    init_fn, chunk_fn = make_train(cfg, env, p, curriculum=Curriculum(sc, tracked_ids=[0, 1]))
    runner, metrics = jax.jit(chunk_fn)(init_fn(jax.random.PRNGKey(0)))
    assert int(runner.sampler_state.count.sum()) > 0  # finished episodes were recorded
    for k in ["sampler_mean_p", "sampler_frac_learnable", "tracked0_weight", "tracked1_p"]:
        assert k in metrics and metrics[k].shape == (4,)


def test_per_episode_rules_and_joint_reset():
    from sail_env.config import EnvParams
    from sail_env.train.curriculum.config_env import ConfigParams, ConfigResetEnv, make_config_set
    from sail_env.start_env import OBS_DIM, RULE_FEATURES, SoloStartEnv

    p = EnvParams(per_episode_rules=True, observe_rules=True, gun_distance_max=3.0, port_penalty=1.0)
    env = SoloStartEnv(p)
    assert env.obs_dim == OBS_DIM + RULE_FEATURES
    _, s = env.reset(jax.random.PRNGKey(0), p)
    easy = env.episode_params(s.replace(rule_w=jnp.float32(0.0)), p)
    full = env.episode_params(s.replace(rule_w=jnp.float32(1.0)), p)
    assert float(easy.success_window) == 10.0 and float(full.success_window) == 3.0
    assert float(easy.gun_distance_max) == 30.0 and float(full.gun_distance_max) == 3.0
    obs_easy = env.get_obs(s.replace(rule_w=jnp.float32(0.0)), p)
    obs_full = env.get_obs(s.replace(rule_w=jnp.float32(1.0)), p)
    np.testing.assert_allclose(np.asarray(obs_full[-RULE_FEATURES:]), [0.3, 0.9, 0.1, 1.0, 1.0], rtol=1e-5)
    assert float(obs_easy[-RULE_FEATURES]) == pytest.approx(1.0)
    # Joint ids: id = situation x levels + level -> rule_w = level / (levels - 1).
    configs = make_config_set(env, p, jax.random.PRNGKey(1), 8)
    jenv = ConfigResetEnv(env, configs, num_levels=6)
    probs = jnp.zeros(48).at[3 * 6 + 2].set(1.0)  # only situation 3, level 2
    _, cs = jenv.reset(jax.random.PRNGKey(2), ConfigParams(env=p, cdf=jnp.cumsum(probs)))
    assert int(cs.config_id) == 20 and float(cs.env_state.rule_w) == pytest.approx(0.4)
    np.testing.assert_allclose(float(cs.env_state.boat.x), float(configs.boat.x[3]))


def test_easy_rules_accept_a_late_start_that_full_rules_reject():
    import sys
    sys.path.insert(0, "tests")
    from test_start_env import ENV as SOLO, HOLD, timed_run_state
    from sail_env.config import EnvParams

    p = EnvParams(per_episode_rules=True)
    step = jax.jit(lambda s, a: SOLO.step_env(jax.random.PRNGKey(0), s, a, p))
    results = {}
    for w in (0.0, 1.0):
        s = timed_run_state(5.0).replace(rule_w=jnp.float32(w))  # crosses at T+5
        for _ in range(20):
            _, s, r, done, info = step(s, jnp.int32(HOLD))
        results[w] = bool(info["success"])
    assert results[0.0] and not results[1.0]
