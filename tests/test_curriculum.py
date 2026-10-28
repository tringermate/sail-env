import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sail_env.train.ppo import PPOConfig, make_train
from sail_env.config import EnvParams
from sail_env.train.curriculum.staged_rules import make_staged_params
from sail_env.start_env import SoloStartEnv
from sail_env.wrappers import LogWrapper

EASY = {"success_window": 10.0, "success_speed_frac": 0.0, "ocs_penalty": 0.0, "speed_bonus": 0.02}


def test_staged_schedule_values():
    full = EnvParams()
    fn = make_staged_params(full, EASY, easy_until=0.2, full_from=0.7)
    p0, v0 = fn(0.1)
    assert float(p0.success_window) == 10.0 and float(p0.ocs_penalty) == 0.0
    assert float(p0.speed_bonus) == pytest.approx(0.02)
    pm, _ = fn(0.45)  # halfway through tightening
    assert float(pm.success_window) == pytest.approx(6.5) and float(pm.success_speed_frac) == pytest.approx(0.45)
    p1, v1 = fn(0.9)
    assert float(p1.success_window) == 3.0 and float(p1.ocs_penalty) == 1.0 and float(p1.speed_bonus) == 0.0
    assert set(v1) == set(EASY)


def test_ppo_uses_scheduled_params_and_logs_them():
    p = EnvParams(random_start_time=True)
    cfg = PPOConfig(num_envs=64, num_steps=8, total_timesteps=64 * 8 * 4, num_seeds=1, updates_per_chunk=4)
    fn = make_staged_params(p, EASY, easy_until=0.0, full_from=1.0)
    init_fn, chunk_fn = make_train(cfg, LogWrapper(SoloStartEnv(p)), p, fn)
    runner, metrics = jax.jit(chunk_fn)(init_fn(jax.random.PRNGKey(0)))
    w = np.asarray(metrics["curriculum_success_window"])
    np.testing.assert_allclose(w, [10.0, 8.25, 6.5, 4.75], rtol=1e-5)  # progress 0, 1/4, 2/4, 3/4
    assert "success_strict_rate" in metrics


def test_entropy_bonus_decay_is_logged():
    p = EnvParams()
    cfg = PPOConfig(num_envs=32, num_steps=8, total_timesteps=32 * 8 * 4, num_seeds=1, updates_per_chunk=4,
                    ent_coef=0.01, ent_coef_final=0.001)
    init_fn, chunk_fn = make_train(cfg, LogWrapper(SoloStartEnv(p)), p)
    _, metrics = jax.jit(chunk_fn)(init_fn(jax.random.PRNGKey(0)))
    np.testing.assert_allclose(np.asarray(metrics["ent_coef"]), [0.01, 0.00775, 0.0055, 0.00325], rtol=1e-4)
