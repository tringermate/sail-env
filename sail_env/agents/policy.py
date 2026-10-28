"""Load a trained run (config, env params, actor network, per-seed parameters) from its results folder."""

from pathlib import Path

import jax
import jax.numpy as jnp
import yaml
from flax import serialization

from sail_env.train.ppo import Actor
from sail_env.config import EnvParams
from sail_env.physics import N_ACTIONS


def load_policy(run: Path):
    """Returns (cfg, env_params, actor, params, num_seeds); params has a leading seed axis."""
    run = Path(run)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    env_params = EnvParams(**(cfg.get("env_params") or {}))
    hidden, seeds = cfg["ppo"]["hidden"], cfg["ppo"]["num_seeds"]
    actor = Actor(N_ACTIONS, hidden)
    # The saved tree carries its own shapes (the input size depends on the task: 60 for a start, 72 for
    # the race, + weather or rule features), so restore it as it is.
    raw = serialization.msgpack_restore((run / "params.msgpack").read_bytes())
    params = jax.tree.map(jnp.asarray, raw)
    return cfg, env_params, actor, params, seeds
