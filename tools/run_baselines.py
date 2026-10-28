"""Reproduce the baseline tables in the README: every built-in agent on every standard mode.

    python tools/run_baselines.py --out docs/baselines.json

Starts: 512 episodes per row, races: 128 (seed 0). Each metric is a mean with a 95% confidence
interval over evaluated boats (normal approximation).
"""

import argparse
import json
import time
from pathlib import Path

from sail_env.evaluate import evaluate

ROWS = [
    ("solo", "random", None, 512), ("solo", "digital", None, 512), ("solo", "rl-solo", None, 512),
    ("fleet-self", "digital", None, 512), ("fleet-self", "rl", None, 512),
    ("fleet-vs", "digital", "rl", 512), ("fleet-vs", "rl", "digital", 512),
    ("race-solo", "random", None, 128), ("race-solo", "examples.my_agent:make_policy", None, 128),
    ("race-solo", "digital", None, 128),
    ("race-self", "digital", None, 128),
    ("race-vs", "random", "digital", 128), ("race-vs", "examples.my_agent:make_policy", "digital", 128),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/baselines.json"))
    args = ap.parse_args()
    results = []
    for mode, agent, opponents, episodes in ROWS:
        t = time.time()
        s = evaluate(agent, mode, opponents, episodes=episodes, seed=0)
        s["seconds"] = round(time.time() - t, 1)
        results.append(s)
        print(mode, agent, opponents or "", {k: round(v["mean"], 3) for k, v in s["metrics"].items()}, flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
