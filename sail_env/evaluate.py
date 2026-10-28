"""Evaluate any agent on the standard starts or on the full race.

    python -m sail_env.evaluate --agent digital --mode solo
    python -m sail_env.evaluate --agent rl --mode fleet-self
    python -m sail_env.evaluate --agent mypkg.my_agent:make_policy --mode fleet-vs --opponents rl
    python -m sail_env.evaluate --agent mypkg.my_agent:make_policy --mode race-vs --episodes 128
    python -m sail_env.evaluate --agent digital --mode race-self --weather        # with shifts, pressure, current, waves

Modes:
    solo        one boat, preset "solo"
    fleet-self  the agent sails all five boats, preset "fleet"
    fleet-vs    the agent sails boat 0 against four opponents (--opponents: rl [default], digital, ...)
    race-solo   the whole race alone, preset "race-solo"
    race-self   the agent sails all five boats of the race, preset "race"
    race-vs     the agent sails boat 0 of the race against four opponents (--opponents: digital [default])

Reports, for the evaluated boat(s). Starts: good start (0-3 m below the line at the gun, crossing
within 3 s at >= 90% polar speed, on starboard; fleet: also clear air and no unserved penalty),
started, OCS, penalties, contacts and rank. Race: finished, finish time (s after the gun, finishers),
place (1 = winner), legs completed, penalties, contacts, mark hits. Writes a JSON summary and
optionally replays to --out.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from sail_env.agents import load_agent
from sail_env.presets import make
from sail_env.runs import outcomes, run_episode, save_run


def rate_ci(x):
    """Mean and 95% normal-approximation half-width of a 0/1 or numeric array."""
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


PRESET_OF_MODE = {"solo": "solo", "fleet-self": "fleet", "fleet-vs": "fleet",
                  "race-solo": "race-solo", "race-self": "race", "race-vs": "race"}
METRICS = ["good_start_on_line", "success_strict", "started", "ocs", "race_finished", "finish_time", "race_rank",
           "legs_completed", "fouls", "contacts", "mark_hits", "rank", "upwind_efficiency"]
RATES = {"good_start_on_line", "success_strict", "started", "ocs", "race_finished"}


def evaluate(agent_spec, mode="solo", opponents=None, episodes=512, seed=0, out=None, save_replays=0, weather=False):
    race = mode.startswith("race")
    if opponents is None:
        opponents = "digital" if race else "rl"
    env, params = make(PRESET_OF_MODE[mode] + ("-weather" if weather else ""))
    agent = load_agent(agent_spec)(env, params)
    if mode.endswith("-vs"):
        policies, boat_policy = [agent, load_agent(opponents)(env, params)], None
        evaluated = [0]
    else:
        n = getattr(env, "num_agents", 1)
        policies, boat_policy = [agent], [0] * n
        evaluated = list(range(n))
    run = jax.jit(jax.vmap(lambda k: run_episode(env, params, k, policies, boat_policy)))
    keys = jax.random.split(jax.random.PRNGKey(seed), episodes)
    records, infos = run(keys)
    n = records["x"].shape[2]
    per_ep = [outcomes(jax.tree.map(lambda a: a[i], infos), n) for i in range(episodes)]
    summary = {"agent": agent_spec, "mode": mode, "weather": weather, "opponents": opponents if mode.endswith("-vs") else None,
               "episodes": episodes, "seed": seed, "metrics": {}}
    for k in METRICS:
        if k in per_ep[0]:
            if race and k in ("rank", "ocs", "good_start_on_line", "success_strict", "upwind_efficiency"):
                continue  # start-only measures
            vals = np.concatenate([np.asarray(o[k])[evaluated] for o in per_ep]).astype(float)
            if k == "upwind_efficiency":
                ok = np.concatenate([np.asarray(o["success_strict"])[evaluated] for o in per_ep]).astype(bool)
                vals = vals[ok] if ok.any() else np.array([np.nan])
            if k == "finish_time":
                vals = vals[np.isfinite(vals)] if np.isfinite(vals).any() else np.array([np.nan])
            if k == "race_rank":
                vals = vals + 1.0  # place: 1 = winner
            m, h = rate_ci(vals)
            summary["metrics"][k] = {"mean": m, "ci95": h}
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (out / f"eval_{stamp}_{mode}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        for i in range(min(save_replays, episodes)):
            rec = jax.tree.map(lambda a: a[i], records)
            save_run(out, rec, jax.tree.map(lambda a: a[i], infos), params,
                     meta={"mode": mode, "agent": agent_spec.replace(":", "_").replace("/", "_"),
                           "opponents": opponents if mode.endswith("-vs") else ""})
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="digital")
    ap.add_argument("--mode", choices=list(PRESET_OF_MODE), default="solo")
    ap.add_argument("--opponents", default=None, help="opponents in the -vs modes (default: rl for starts, digital for the race)")
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs", help="directory for the JSON summary and replays")
    ap.add_argument("--save-replays", type=int, default=0, help="also save the first N episodes as replays")
    ap.add_argument("--weather", action="store_true", help="the -weather preset: moving shifts and pressure, current, waves")
    args = ap.parse_args()
    s = evaluate(args.agent, args.mode, args.opponents, args.episodes, args.seed, args.out, args.save_replays, args.weather)
    vs = f" vs {s['opponents']}" if s["opponents"] else ""
    print(f"{s['agent']} | {s['mode']}{' (weather)' if s['weather'] else ''}{vs} | {s['episodes']} {'races' if s['mode'].startswith('race') else 'starts'}")
    for k, v in s["metrics"].items():
        pct = k in RATES
        print(f"  {k:20s} {100 * v['mean']:6.1f}% +- {100 * v['ci95']:4.1f}%" if pct else f"  {k:20s} {v['mean']:6.2f} +- {v['ci95']:.2f}")


if __name__ == "__main__":
    main()
