"""Train the reference PPO agent from a YAML config, over several seeds in parallel.

Usage:
    python -m sail_env.train.run configs/cartpole.yaml
    python -m sail_env.train.run configs/cartpole.yaml --no-wandb

Writes runs/train/<name>_<timestamp>/ with config.yaml, metrics.npz, params.msgpack, curves.png.
"""

import argparse
import dataclasses
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import serialization

from sail_env.train.ppo import Actor, PPOConfig, categorical_sample, train
from sail_env.train.analysis import plot_curves
from sail_env.config import EnvParams
from sail_env.train.curriculum.config_env import (ConfigResetEnv, Curriculum, make_config_set, make_race_situation_bank,
                                                  make_race_start_bank, mix_banks, pick_tracked)
from sail_env.train.curriculum.sampler import SamplerConfig
from sail_env.train.curriculum.staged_rules import make_staged_params
from sail_env.train.cartpole import CartPole, CartPoleParams
from sail_env.fleet_env import FleetStartEnv, rollout_fleet
from sail_env.render import render_fleet_gif, render_gif, rollout
from sail_env.start_env import SoloStartEnv
from sail_env.wrappers import LogWrapper
from sail_env.train.logger import Logger

RESULTS = Path("runs") / "train"  # relative to where you run it
GIF_KEY = jax.random.PRNGKey(12345)  # same start scenario in every GIF, so progress is comparable


def make_gif_callback(cfg, env, env_params, out: Path, logger, every: int):
    """After every `every` updates: roll out seed 0's current policy once and log a GIF."""
    actor = Actor(env.num_actions, cfg.hidden)
    (out / "gifs").mkdir(exist_ok=True)

    fleet = hasattr(env, "num_agents")

    @jax.jit
    def episode(actor_params):
        if fleet:  # most likely action for every boat
            return rollout_fleet(env, env_params, GIF_KEY, lambda k, obs: jnp.argmax(actor.apply(actor_params, obs), -1))
        policy = lambda k, obs, i: categorical_sample(k, actor.apply(actor_params, obs))  # noqa: E731
        return rollout(env, env_params, GIF_KEY, policy)

    def callback(update, runner):
        if (update + 1) % every:
            return
        actor_params = jax.tree.map(lambda x: x[0], runner.params["actor"])  # seed 0
        traj, actions, rewards, info = episode(actor_params)
        path = out / "gifs" / f"update_{update + 1:05d}.gif"
        if fleet:
            good = int(np.asarray(info["good_start_on_line"]).sum())
            title = f"update {update + 1} (seed 0) - good starts {good}/{env.num_agents}"
            render_fleet_gif(traj, env_params, path, title=title)
        else:
            outcome = "success" if bool(info["success"]) else ("OCS" if bool(info["ocs"]) else "no success")
            title = f"update {update + 1} (seed 0) - {outcome}"
            render_gif(traj, env_params, path, actions=actions, title=title)
        logger.log_gif(update, path, caption=title)

    return callback


def make_race_gif_callback(cfg, env, env_params, start_state, out: Path, logger, every: int):
    """After every `every` updates: seed 0's current policy (most likely action) sails one race from a
    fixed state of the start bank; the race is rendered with the course and logged as a GIF."""
    from sail_env.runs import params_dict, render_replay, snapshot

    actor = Actor(env.num_actions, cfg.hidden)
    (out / "gifs").mkdir(exist_ok=True)
    n_steps = int(round((env_params.t_end - float(start_state.boats.t[0])) / env_params.dt)) + 1

    @jax.jit
    def race(actor_params):
        def body(carry, _):
            obs, s, finished = carry
            a = jnp.argmax(actor.apply(actor_params, obs), axis=-1)
            obs2, s2, _, done, info = env.step_env(jax.random.PRNGKey(0), s, a, env_params)
            s2 = jax.tree.map(lambda new, old: jnp.where(finished, old, new), s2, s)
            return (obs2, s2, finished | done[0]), (snapshot(env, env_params, s2), info, done[0] & ~finished)

        obs = env.get_obs(start_state, env_params)
        _, (snaps, infos, ends) = jax.lax.scan(body, (obs, start_state, jnp.bool_(False)), None, length=n_steps)
        first = snapshot(env, env_params, start_state)
        record = jax.tree.map(lambda a, b: jnp.concatenate([a[None], b]), first, snaps)
        return record, jax.tree.map(lambda x: x[jnp.argmax(ends)], infos)

    def callback(update, runner):
        if (update + 1) % every:
            return
        actor_params = jax.tree.map(lambda x: x[0], runner.params["actor"])  # seed 0
        record, info = race(actor_params)
        record = jax.tree.map(np.asarray, record)
        fin, t = bool(np.asarray(info["race_finished"])[0]), float(np.asarray(info["finish_time"])[0])
        title = (f"update {update + 1} (seed 0): " + (f"finished in {t:.0f} s" if fin else
                 f"not finished, {int(np.asarray(info['legs_completed'])[0])} of 4 legs"))
        path = out / "gifs" / f"update_{update + 1:05d}.gif"
        render_replay({"record": record, "params": params_dict(env_params), "name": title}, path, title=title)
        logger.log_gif(update, path, caption=title)

    return callback


def make_env(name: str, overrides: dict):
    from sail_env.presets import PRESETS, make

    if name in PRESETS:  # a named preset ("race-solo", "race-weather", ...) with overrides on top
        return make(name, **overrides)
    if name == "cartpole":
        return CartPole(), CartPoleParams(**overrides)
    if name == "solo_start":
        params = EnvParams(**overrides)
        return SoloStartEnv(params), params
    if name == "fleet_start":
        params = EnvParams(**overrides)
        return FleetStartEnv(params), params
    raise ValueError(f"unknown env {name!r}")


def make_annealed_params(env_params, anneal):
    """env_params_fn for ppo.make_train: each field moves linearly from `start` to `end` between the training
    fractions `from` and `until` (e.g. race_shaping: [0.1, 0.0, 0.1, 0.6] phases the shaping out)."""
    def fn(progress):
        values = {}
        for name, (start, end, a, b) in anneal.items():
            x = jnp.clip((progress - a) / max(b - a, 1e-9), 0.0, 1.0)
            values[name] = start + (end - start) * x
        return env_params.replace(**values), values

    return fn


def widen_input_layer(params, obs_dim, env_params):
    """Widen the first layer of actor and critic (params: {"actor", "critic"}, kernels (seeds, in, out)) to obs_dim
    inputs. The new features get zero input weights, so the network starts out acting exactly as before and learns
    to use them. They are inserted where they appear in the observation: the weather block comes last, but the
    teammate flags (team racing) come before the weather block. Returns (params, index of the first new input)."""
    from sail_env.race_env import TEAM_FEATURES, team_block_start

    n_in = params["actor"]["params"]["Dense_0"]["kernel"].shape[1]
    at = n_in
    if env_params.team_size and n_in == obs_dim - TEAM_FEATURES:
        at = team_block_start(env_params, obs_dim)
    params = jax.tree.map(lambda x: x, params)  # a copy: the caller's dict is left as it was
    for net in ("actor", "critic"):
        k = params[net]["params"]["Dense_0"]["kernel"]
        pad = jnp.zeros(k.shape[:1] + (obs_dim - n_in,) + k.shape[2:], k.dtype)
        params[net]["params"]["Dense_0"]["kernel"] = jnp.concatenate([k[:, :at], pad, k[:, at:]], axis=1)
    return params, at


def make_teacher(spec, raw_env, env_params):
    """Kickstarting teacher: an agent (any load_agent spec, e.g. 'digital') acting in the training batch.

    Returns teacher(key, env_state, obs) -> (num_envs, N) actions for the fleet/race envs: it unwraps the
    LogWrapper (and ConfigResetEnv) state and asks the agent for every boat of every env."""
    from sail_env.agents import load_agent
    from sail_env.train.curriculum.config_env import ConfigState

    if not hasattr(raw_env, "num_agents"):
        raise ValueError("a teacher needs a fleet or race env (the agent API takes the fleet state)")
    policy = load_agent(spec)(raw_env, env_params)
    boats = jnp.arange(raw_env.num_agents)

    def teacher(key, env_state, obs):
        fs = env_state.env_state  # LogState -> the env's state
        if isinstance(fs, ConfigState):
            fs = fs.env_state

        def one_env(o, f):
            return jax.vmap(policy, in_axes=(None, 0, 0, 0))(key, o, f.boats, boats)

        return jax.vmap(one_env)(obs, fs)

    return teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-timesteps", type=int, help="override ppo.total_timesteps (e.g. for a quick test)")
    args = ap.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.total_timesteps:
        raw.setdefault("ppo", {})["total_timesteps"] = args.total_timesteps
    cfg = PPOConfig(**raw.get("ppo", {}))
    env, env_params = make_env(raw["env"], raw.get("env_params", {}))
    raw_env = env
    curriculum = None
    start_bank = None
    if "start_bank" in raw:  # race: every episode starts just after the start, sailed by a start policy
        sb = raw["start_bank"]
        key_bank = jax.random.PRNGKey(sb.get("seed", 4321))
        start_bank = make_race_start_bank(raw_env, env_params, key_bank, sb.get("num", 4096), sb.get("policy", "fleet"))
        started = float(np.mean(np.asarray(start_bank.boats.status.started)))
        print(f"start bank: {sb.get('num', 4096)} races after the start ({100 * started:.0f}% of boats started)")
        raw["start_bank_started_fraction"] = started
        if sb.get("situations") == "anywhere":  # curriculum: situations all over the course, some starts
            gif_state = jax.tree.map(lambda x: x[0], start_bank)  # GIFs always sail the full race
            situations = make_race_situation_bank(raw_env, env_params, jax.random.fold_in(key_bank, 7), sb.get("num", 4096))
            start_bank = mix_banks(situations, start_bank, sb.get("start_fraction", 0.25), jax.random.fold_in(key_bank, 8))
            print(f"situation bank: {sb.get('num', 4096)} situations anywhere on the course, "
                  f"{100 * sb.get('start_fraction', 0.25):.0f}% of them just after the start")
        else:
            gif_state = jax.tree.map(lambda x: x[0], start_bank)
        sc = SamplerConfig(method=sb.get("sampler", "uniform"), num_configs=sb.get("num", 4096))
        curriculum = Curriculum(sc, success_key="race_finished")
        env = ConfigResetEnv(raw_env, start_bank)
    if "curriculum" in raw:  # reset to configurations from a fixed set, chosen by uniform / SFL / SGS
        cur = dict(raw["curriculum"])
        success_key = cur.pop("success_key", "good_start_on_line")  # which outcome feeds the buffers
        sc = SamplerConfig(**cur)
        n_situations = sc.num_configs // sc.num_levels
        configs = make_config_set(raw_env, env_params, jax.random.PRNGKey(raw.get("config_seed", 1234)), n_situations)
        tracked, reach = pick_tracked(raw_env, configs)
        if sc.num_levels > 1:  # track the full-rules version of each tracked situation
            tracked = [t * sc.num_levels + sc.num_levels - 1 for t in tracked]
        print("tracked configurations (reach fraction, > 1 = cannot make it):", [round(r, 2) for r in reach])
        raw["tracked_reach_fraction"] = reach
        curriculum = Curriculum(sc, success_key=success_key, tracked_ids=tracked)
        env = ConfigResetEnv(raw_env, configs, num_levels=sc.num_levels)
    env = LogWrapper(env)
    env_params_fn = None
    if "staged_rules" in raw:  # curriculum: easy success rules first, tightened to env_params
        sr = raw["staged_rules"]
        env_params_fn = make_staged_params(env_params, sr["easy"], sr.get("easy_until", 0.2), sr.get("full_from", 0.7))
    if "anneal" in raw:  # env parameters changing linearly over training: {field: [start, end, from, until]}
        if env_params_fn is not None:
            raise ValueError("anneal and staged_rules cannot be combined")
        env_params_fn = make_annealed_params(env_params, raw["anneal"])
    use_wandb = raw.get("wandb", False) and not args.no_wandb

    out = RESULTS / f"{raw['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    # Save everything needed to reproduce the run: resolved PPO and env settings and the git commit.
    resolved_env = {f.name: getattr(env_params, f.name) for f in dataclasses.fields(env_params)}
    resolved_env = {k: (float(v) if hasattr(v, "dtype") else v) for k, v in resolved_env.items()}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    full_config = {**raw, "ppo": dataclasses.asdict(cfg), "seed": args.seed,
                   "env_params_resolved": resolved_env, "git_commit": commit}
    (out / "config.yaml").write_text(yaml.safe_dump(full_config, sort_keys=False), encoding="utf-8")

    print(f"{raw['name']}: {cfg.num_seeds} seeds x {cfg.num_updates} updates x {cfg.batch_size} steps "
          f"= {cfg.num_updates * cfg.batch_size:,} steps per seed on {jax.devices()[0]}  ->  {out}")
    logger = Logger(out.name, full_config, use_wandb, project=raw.get("wandb_project", "sail_env"))
    t0 = time.perf_counter()
    gif_every = raw.get("gif_every", 0)
    if gif_every and start_bank is not None:
        callback = make_race_gif_callback(cfg, raw_env, env_params, gif_state, out, logger, gif_every)
    elif gif_every and raw["env"] in ("solo_start", "fleet_start"):
        callback = make_gif_callback(cfg, raw_env, env_params, out, logger, gif_every)
    else:
        callback = None
    # Checkpoint: save the parameters every checkpoint_every updates (params.msgpack, overwritten), so a
    # run that is stopped or crashes keeps its latest policy. 0 = only at the end.
    checkpoint_every = raw.get("checkpoint_every", 100)
    gif_callback = callback

    def callback(update, runner):
        if gif_callback is not None:
            gif_callback(update, runner)
        if checkpoint_every and (update + 1) % checkpoint_every == 0:
            (out / "params.msgpack").write_bytes(serialization.to_bytes(jax.device_get(runner.params)))
            (out / "checkpoint.txt").write_text(f"update {update + 1}, {(update + 1) * cfg.batch_size:,} steps per seed\n",
                                                encoding="utf-8")
    init_params = None
    if raw.get("init_from"):  # warm start from a trained run (e.g. the fleet from the best solo policy)
        from sail_env.agents.policy import load_policy

        _, _, _, src_params, src_seeds = load_policy(Path(raw["init_from"]))
        pick = jnp.arange(cfg.num_seeds) % src_seeds
        init_params = jax.tree.map(lambda x: x[pick], src_params)
        print(f"initialised from {raw['init_from']} ({src_seeds} seeds)")
        n_in = init_params["actor"]["params"]["Dense_0"]["kernel"].shape[1]
        if n_in < env.obs_dim:  # e.g. a racer trained without weather, now with the weather features
            init_params, at = widen_input_layer(init_params, env.obs_dim, env_params)
            print(f"  input layer widened {n_in} -> {env.obs_dim} (new inputs at {at}..{at + env.obs_dim - n_in - 1}, "
                  f"weight 0)")
    teacher = make_teacher(raw["teacher"], raw_env, env_params) if raw.get("teacher") else None
    # Opponent pool (ppo.opp_frac > 0): fixed agents next to the policy's own frozen snapshots.
    opponents = [make_teacher(spec, raw_env, env_params) for spec in raw.get("opponents", [])]
    if cfg.opp_frac > 0:
        print(f"opponent pool: {cfg.opp_frac:.0%} of the envs, boats 0-{cfg.opp_learners - 1} learn; opponents: "
              f"{cfg.opp_snapshots} snapshots (every {cfg.opp_snapshot_every} updates) + {raw.get('opponents', [])}")
    if teacher is not None:
        print(f"kickstarting from teacher {raw['teacher']!r}: weight {cfg.kick_coef} -> 0 by "
              f"{100 * cfg.kick_until:.0f}% of training")
    runner, metrics = train(cfg, env, env_params, jax.random.PRNGKey(args.seed), log_fn=logger.log,
                            chunk_fn_callback=callback, env_params_fn=env_params_fn, curriculum=curriculum,
                            init_params=init_params, teacher=teacher, opponents=opponents)
    elapsed = time.perf_counter() - t0

    # Save results BEFORE closing W&B: on Windows wandb.finish() can raise a spurious
    # KeyboardInterrupt while shutting down its helper process, which used to lose these files.
    steps = (np.arange(cfg.num_updates) + 1) * cfg.batch_size
    np.savez(out / "metrics.npz", steps=steps, **metrics)
    (out / "params.msgpack").write_bytes(serialization.to_bytes(jax.device_get(runner.params)))
    keys = ["episode_return"] + [k for k in ["success_rate", "success_strict_rate"] if k in metrics] + \
           ["entropy", "approx_kl", "explained_variance"]
    plot_curves(metrics, steps, keys, out / "curves.png", title=raw["name"])

    final = np.nanmean(metrics["episode_return"][:, -50:], axis=1)  # episodes finished in the last 50 updates
    print(f"done in {elapsed:.0f}s. final return per seed (last 50 updates): {np.round(final, 1)}", flush=True)
    try:
        logger.finish()
    except KeyboardInterrupt:
        print("warning: W&B shutdown was interrupted; results were already saved locally", flush=True)


if __name__ == "__main__":
    main()
