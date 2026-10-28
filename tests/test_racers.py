"""The bundled racers (rl-racer, rl-racer-weather): they load by name and actually race the full course."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import sail_env as se
from sail_env.agents import PRETRAINED, load_agent
from sail_env.train.curriculum.config_env import make_race_start_bank


def bundled_env(name):
    """The preset and race settings the bundled racer was trained on (from its config.yaml)."""
    cfg = yaml.safe_load((PRETRAINED / name / "config.yaml").read_text(encoding="utf-8"))
    return se.make(cfg["env"], **cfg["env_params"])


@pytest.mark.parametrize("spec, name", [("rl-racer", "racer"), ("rl-racer-weather", "racer-weather")])
def test_bundled_racer_finishes_races(spec, name):
    env, p = bundled_env(name)
    policy = load_agent(spec)(env, p)
    bank = make_race_start_bank(env, p, jax.random.PRNGKey(4321), 8, "fleet")

    def race(s):
        def body(c, i):
            o, s = c
            a = jax.vmap(policy, in_axes=(None, 0, 0, 0))(jax.random.PRNGKey(i), o, s.boats, jnp.arange(env.num_agents))
            o, s, _, _, _ = env.step_env(jax.random.PRNGKey(0), s, a, p)
            return (o, s), None

        (_, s_end), _ = jax.lax.scan(body, (env.get_obs(s, p), s), jnp.arange(1100))
        return s_end.finish_time[0]

    t = np.asarray(jax.jit(jax.vmap(race))(bank))
    assert np.isfinite(t).sum() >= 7  # the racers finish 98-100% of races
    assert np.nanmean(np.where(np.isfinite(t), t, np.nan)) < 760.0  # digital racer: ~673 s (steady), ~685 s (weather)


def test_racer_acts_in_the_fleet_race():
    """In a fleet race (other boats as neighbours, unseen in its solo training) it still gives valid actions."""
    env, p = se.make("race")
    policy = load_agent("rl-racer")(env, p)
    obs, fs = env.reset(jax.random.PRNGKey(0), p)
    n = env.num_agents
    acts = jax.jit(jax.vmap(policy))(jax.random.split(jax.random.PRNGKey(1), n), obs, fs.boats, jnp.arange(n))
    assert acts.shape == (n,) and bool(jnp.all((acts >= 0) & (acts < 7)))
