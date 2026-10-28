"""Export a trained policy as a small JSON file: to race against it in the browser version (web/, "Your agent"),
or to share it.

    python -m sail_env.agents.export runs/train/<run> -o my_agent.json [--seed 0] [--name "My racer"]
    python -m sail_env.agents.export racer -o racer.json          # a bundled policy: solo, fleet, racer, racer-weather

The file holds the actor network of one seed (the most likely action is played) and what it expects:

    {"format": "sail_env-agent", "version": 1, "name": ..., "kind": "start" | "race",
     "obs_dim": 60 | 72 | 94, "weather": false, "layers": [{"W": [[...]] (in x out), "b": [...]}, ...]}

kind "start": trained on a start preset (solo, fleet); it sails starts. kind "race": trained on a race preset
(race, race-solo, from a start bank); before its boat has started (or T+10) the bundled fleet start policy
sails the start, as in training and as `pretrained_agent` does in Python. The browser version has no weather,
so it plays agents with weather == false only (Python plays both: `load_agent(run_dir)`).
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from sail_env.agents import PRETRAINED
from sail_env.start_env import OBS_DIM as START_OBS_DIM
from sail_env.weather import WEATHER_FEATURES

FORMAT, VERSION = "sail_env-agent", 1
RACE_FEATURES = 12  # race_env.get_obs: leg one-hot (5), next mark(s) in the boat's frame (4), clock, penalty owed, turning


def export_agent(run, seed=0, name=None, decimals=6):
    """The JSON-able dict for a run directory or a bundled policy name ('solo', 'fleet', 'racer', 'racer-weather')."""
    from sail_env.agents.policy import load_policy

    path = Path(run) if Path(run).exists() else PRETRAINED / run
    cfg, _, _, params, seeds = load_policy(path)
    if not 0 <= seed < seeds:
        raise ValueError(f"{run}: seed {seed} not in 0..{seeds - 1}")
    p = jax.tree.map(lambda x: np.asarray(x[seed]), params["actor"])["params"]
    layers = [{"W": np.round(p[k]["kernel"], decimals).tolist(), "b": np.round(p[k]["bias"], decimals).tolist()}
              for k in sorted(p, key=lambda s: int(s.split("_")[1]))]
    obs_dim = len(layers[0]["W"])
    weather = obs_dim in (START_OBS_DIM + WEATHER_FEATURES, START_OBS_DIM + RACE_FEATURES + WEATHER_FEATURES)
    kind = "race" if obs_dim - (WEATHER_FEATURES if weather else 0) == START_OBS_DIM + RACE_FEATURES else "start"
    return {"format": FORMAT, "version": VERSION, "name": name or cfg.get("name", path.name), "kind": kind,
            "obs_dim": obs_dim, "weather": weather, "actions": len(layers[-1]["b"]), "env": cfg.get("env"),
            "seed": seed, "source": str(run), "layers": layers}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="training run directory (config.yaml + params.msgpack) or a bundled policy name")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0, help="which trained seed (default 0)")
    ap.add_argument("--name", help="the name shown in the game (default: the run's name)")
    args = ap.parse_args()
    agent = export_agent(args.run, args.seed, args.name)
    args.out.write_text(json.dumps(agent, separators=(",", ":")), encoding="utf-8")
    note = "" if not agent["weather"] else "  (weather inputs: Python only, the browser version has no weather)"
    print(f"wrote {args.out}: {agent['name']!r}, {agent['kind']} agent, {agent['obs_dim']} inputs, "
          f"{len(agent['layers'])} layers, {args.out.stat().st_size / 1024:.0f} KB{note}")


if __name__ == "__main__":
    main()
