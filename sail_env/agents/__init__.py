"""Agents: anything that picks one of the 7 actions for a boat.

An agent is a factory  make_policy(env, params) -> policy  where

    policy(key, obs, state, boat) -> action        (pure JAX, so it can be jitted and vmapped)

    key    a jax.random key
    obs    the boat's observation, shape (env.obs_dim,)   <- what a learned agent should use
    state  the boat's full simulator state (EnvState)     <- privileged; hand-written agents may use it
    boat   the boat's index in the fleet (0 in solo)
    action int in [0, 7): 0 hold, 1 up 5, 2 down 5, 3 up 15, 4 down 15, 5 tack/gybe, 6 luff

Agents that only read `obs` are directly comparable with the bundled RL policies. Write your own as a
module with a make_policy function and evaluate it with:

    python -m sail_env.evaluate --agent mypackage.my_agent:make_policy --mode fleet-vs
"""

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp

from sail_env.agents.digital_sailor import digital_sailor_action
from sail_env.physics import N_ACTIONS
from sail_env.weather import WEATHER_FEATURES

PRETRAINED = Path(__file__).resolve().parents[1] / "pretrained"


def random_agent(env, params):
    """Uniformly random actions."""
    return lambda key, obs, state, boat: jax.random.randint(key, (), 0, N_ACTIONS)


def hold_agent(env, params):
    """Always hold course (a trivial reference)."""
    return lambda key, obs, state, boat: jnp.int32(0)


def digital_sailor_agent(env, params, timing=-3.0):
    """Hand-written sailor (privileged state). Starts: 'time on distance', each boat of a fleet aiming at
    its own spot on the line (0.15 ... 0.85; solo 0.3). In the race it then sails the course: layline
    tacking upwind, VMG-angle gybing downwind, port roundings, gate and finish (agents/race_sailor.py)."""
    from sail_env.race_env import RaceEnv

    n = getattr(env, "num_agents", 1)
    spots = jnp.linspace(0.15, 0.85, n) if n > 1 else jnp.array([0.3])
    polar = env.polar
    if isinstance(env, RaceEnv):
        from sail_env.agents.race_sailor import race_sailor_action

        return lambda key, obs, state, boat: race_sailor_action(state, obs, polar, params, spots[boat], timing)
    return lambda key, obs, state, boat: digital_sailor_action(state, polar, params, spots[boat], timing)


def pretrained_agent(name="fleet", seed=None, greedy=True):
    """A bundled RL policy ('solo', 'fleet', 'racer', 'racer-weather'), or a training run directory
    (config.yaml + params.msgpack).

    seed: which of the trained seeds to use (default: the first). greedy: most likely action
    (True) or sample from the policy's probabilities (False: weaker, more human-like mistakes)."""

    def make_policy(env, params):
        from sail_env.agents.policy import load_policy

        path = Path(name) if Path(name).exists() else PRETRAINED / name
        cfg, _, actor, params_all, seeds = load_policy(path)
        s = 0 if seed is None else seed % seeds
        actor_params = jax.tree.map(lambda x: x[s], params_all["actor"])
        n_in = actor_params["params"]["Dense_0"]["kernel"].shape[0]
        # A policy trained without weather observations can still sail with weather: the weather block
        # always comes last, so it sees the same first n_in features and ignores the rest.
        blind = params.observe_weather and n_in == env.obs_dim - WEATHER_FEATURES
        # Likewise a racer trained without team racing in a team race: it does not see the teammate flags
        # (after the race features, before the weather block).
        from sail_env.race_env import TEAM_FEATURES, team_block_start

        team_blind = bool(params.team_size) and n_in == env.obs_dim - TEAM_FEATURES
        at = team_block_start(params, env.obs_dim) if team_blind else n_in
        # And a team racer in an individual race: no boat is its teammate (all flags 0), where the block would be.
        no_mates = not params.team_size and n_in == env.obs_dim + TEAM_FEATURES
        if no_mates:
            at = team_block_start(params, env.obs_dim + TEAM_FEATURES)
        blind = blind or team_blind or no_mates
        if n_in != env.obs_dim and not blind:
            raise ValueError(f"policy {name!r} expects {n_in} observation features but this environment gives "
                             f"{env.obs_dim}: the bundled start policies (solo, fleet) need the start presets, the racers "
                             f"(racer, racer-weather) the race presets")

        def own(key, obs):
            if team_blind:
                seen = jnp.concatenate([obs[:at], obs[at + TEAM_FEATURES:]])[:n_in]
            elif no_mates:
                seen = jnp.concatenate([obs[:at], jnp.zeros(TEAM_FEATURES, obs.dtype), obs[at:]])
            else:
                seen = obs[:n_in]
            logits = actor.apply(actor_params, seen)
            return jnp.argmax(logits) if greedy else jax.random.categorical(key, logits)

        if not (cfg.get("start_bank") and hasattr(env, "course")):
            return lambda key, obs, state, boat: own(key, obs)
        # A racer trained from just after the start (start_bank): the start policy it was trained behind
        # sails the start, until this boat has started (or T+10, as in training); then the racer takes over.
        from sail_env.start_env import OBS_DIM

        _, _, s_actor, s_params_all, _ = load_policy(PRETRAINED / cfg["start_bank"].get("policy", "fleet"))
        s_params = jax.tree.map(lambda x: x[0], s_params_all["actor"])

        def policy(key, obs, state, boat):
            start = jnp.argmax(s_actor.apply(s_params, obs[:OBS_DIM]))
            use_start = ~state.status.started & (state.t < 10.0)
            return jnp.where(use_start, start, own(key, obs))

        return policy

    return make_policy


BUILTIN = {
    "random": random_agent,
    "hold": hold_agent,
    "digital": digital_sailor_agent,
    "rl": pretrained_agent("fleet"),
    "rl-solo": pretrained_agent("solo"),
    "rl-racer": pretrained_agent("racer"),  # the full race, steady wind (race-solo, race)
    "rl-racer-weather": pretrained_agent("racer-weather"),  # the full race with weather (race-solo-weather)
}


def load_agent(spec: str):
    """'random' | 'hold' | 'digital' | 'rl' | 'rl-solo' | 'rl-racer' | 'rl-racer-weather' | 'path/to/run_dir' |
    'path/to/run_dir#seed' | 'racer#seed' | 'module.sub:make_policy'."""
    if spec in BUILTIN:
        return BUILTIN[spec]
    if ":" in spec:
        module, fn = spec.split(":", 1)
        return getattr(importlib.import_module(module), fn)
    if Path(spec).exists():
        return pretrained_agent(spec)
    path, _, seed = spec.rpartition("#")  # 'path/to/run_dir#1' or 'racer#1': that seed of a run or bundled policy
    if seed.isdigit() and (Path(path).exists() or (PRETRAINED / path).exists()):
        return pretrained_agent(path, seed=int(seed))
    raise ValueError(f"unknown agent {spec!r}")
