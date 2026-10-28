"""Render a saved run as a GIF, or list saved runs.

    python -m sail_env.replay runs/20261019-101500_fleet_mate.npz            # -> same name .gif
    python -m sail_env.replay runs/20261019-101500_fleet_mate.npz --gif out.gif
    python -m sail_env.replay --list runs
"""

import argparse
import json
from pathlib import Path

from sail_env.runs import load_run, render_replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", type=Path)
    ap.add_argument("--gif", type=Path)
    ap.add_argument("--list", type=Path, metavar="DIR")
    args = ap.parse_args()
    if args.list:
        for side in sorted(args.list.glob("*.json")):
            if side.name.startswith("eval_"):
                continue
            d = json.loads(side.read_text(encoding="utf-8"))
            o = d.get("outcome", {})
            if "race_finished" in o:
                ft = o["finish_time"][0]
                done = f"finished in {ft:.0f} s" if o["race_finished"][0] else f"{o['legs_completed'][0]} legs"
                print(f"{d.get('name', side.stem):55s} {done}  place: {o['race_rank'][0] + 1:.1f}")
            else:
                good = o.get("good_start_on_line", [None])[0]
                print(f"{d.get('name', side.stem):55s} good start: {good}  rank: {o.get('rank', ['-'])[0]}")
        return
    if args.run is None:
        ap.error("give a run file or --list DIR")
    gif = args.gif or args.run.with_suffix(".gif")
    render_replay(load_run(args.run), gif)
    print(f"wrote {gif}")


if __name__ == "__main__":
    main()
