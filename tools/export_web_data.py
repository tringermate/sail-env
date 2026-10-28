"""Data for the browser version (web/): the bundled RL policies as JSON (the fleet start policy and the racer),
and reference episodes.

    python tools/export_web_data.py                                  # web/weights.json
    python tools/export_web_data.py --traces web/ref/traces.json     # + reference episodes for web/check.js

The reference episodes (fleet start with the RL policy, solo start, fleet start, race and solo race with
the digital sailors) record the initial state, every action and the resulting state, so the JavaScript
simulator can be checked step by step against this one (web/check.js; run in CI).
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import sail_env as se
from sail_env.agents import PRETRAINED, load_agent
from sail_env.agents.policy import load_policy

TRACES = [("fleet", 5, "rl"), ("solo", 3, "digital"), ("fleet", 8, "digital"), ("race", 11, "digital"),
          ("race-solo", 2, "digital"), ("race", 7, "rl-racer")]


def actor_json(name, seed=0, decimals=5):
    """Actor MLP of a bundled policy (one seed): [{"W": (in, out), "b": (out,)}, ...]."""
    _, _, _, params, _ = load_policy(PRETRAINED / name)
    p = jax.tree.map(lambda x: np.asarray(x[seed]), params["actor"])["params"]
    return [{"W": np.round(p[k]["kernel"], decimals).tolist(), "b": np.round(p[k]["bias"], decimals).tolist()}
            for k in sorted(p, key=lambda s: int(s.split("_")[1]))]


def boats_dict(st):
    b = st.boat
    f = lambda a: np.asarray(a, dtype=float).tolist()  # noqa: E731
    return {"x": f(b.x), "y": f(b.y), "heading": f(b.heading), "speed": f(b.speed), "target": f(b.target_heading),
            "man": np.asarray(b.manoeuvring).tolist()}


def trace(preset, key, agent):
    env, p = se.make(preset)
    fleet = hasattr(env, "num_agents")
    n = env.num_agents if fleet else 1
    policy = load_agent(agent)(env, p)
    obs, s = env.reset(jax.random.PRNGKey(key), p)
    as_fleet = lambda s: s.boats if fleet else jax.tree.map(lambda x: x[None], s)  # noqa: E731
    st = as_fleet(s)
    first = lambda tree: jax.tree.map(lambda x: float(np.asarray(x).ravel()[0]), tree)  # noqa: E731
    w, line = first(st.wind), first(st.line)
    init = {"boats": boats_dict(st),
            "wind": {k: getattr(w, k) for k in ["base_dir", "tws", "shift_amp", "shift_period", "shift_phase"]},
            "line": {k: getattr(line, k) for k in ["pin_x", "pin_y", "cb_x", "cb_y"]},
            "bias": float(np.asarray(st.bias).ravel()[0]), "t": float(np.asarray(st.t).ravel()[0])}
    act = jax.jit(lambda k, o, s: jax.vmap(policy)(jax.random.split(k, n), o if fleet else o[None], as_fleet(s), jnp.arange(n)))
    step = jax.jit(lambda s, a: env.step_env(jax.random.PRNGKey(0), s, a if fleet else a[0], p))
    obs0 = np.asarray(obs if fleet else obs[None]).tolist()
    steps = []
    for i in range(p.n_steps):
        a = act(jax.random.PRNGKey(i), obs, s).astype(jnp.int32)
        obs, s, _, done, info = step(s, a)
        st = as_fleet(s)
        rec = {"a": np.asarray(a).tolist(), **boats_dict(st), "started": np.asarray(st.status.started).tolist(),
               "ocs": np.asarray(st.status.ocs).tolist()}
        if fleet:
            rec.update(pen=np.asarray(s.penalty_left).tolist(), fouls=np.asarray(s.fouls).tolist())
            if hasattr(s, "leg"):
                rec["leg"] = np.asarray(s.leg).tolist()
        steps.append(rec)
        if bool(np.asarray(done).ravel()[0]):
            fin = lambda v: [x if np.isfinite(x) else None for x in v.ravel()] if v.ndim else float(v)  # noqa: E731
            rec["info"] = {k: fin(np.asarray(v).astype(float)) for k, v in info.items()
                           if k in ("good_start_on_line", "rank", "race_rank", "finish_time", "race_finished")}
            break
    return {"preset": preset, "agent": agent, "init": init, "obs0": obs0, "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("web/weights.json"))
    ap.add_argument("--traces", type=Path)
    args = ap.parse_args()
    weights = {"fleet": actor_json("fleet"), "racer": actor_json("racer")}
    args.weights.write_text(json.dumps(weights, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {args.weights}")
    if args.traces:
        args.traces.parent.mkdir(parents=True, exist_ok=True)
        traces = [trace(*t) for t in TRACES]
        args.traces.write_text(json.dumps(traces, separators=(",", ":")), encoding="utf-8")
        print(f"wrote {args.traces} ({len(traces)} episodes)")


if __name__ == "__main__":
    main()
