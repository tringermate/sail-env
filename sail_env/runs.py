"""Run episodes with any mix of agents, record them, save them as replays, render them.

    record, info = run_episode(env, params, key, [policy_for_boat0, policy_for_others])
    path = save_run("runs", record, info, params, meta={"mode": "fleet-vs", "agent": "mine"})
    run = load_run(path); render_replay(run, "replay.gif")

A replay is an .npz file (per-step positions, headings, speeds, start status, penalties, wind, actions;
in the race also legs, finish times and the course marks) with a .json sidecar (settings, agents,
outcome per boat). runs/leaderboard.csv collects one line per
saved run, so human players and agents can be compared.
"""

import csv
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sail_env.config import KNOT
from sail_env.fleet_env import FleetStartEnv
from sail_env.physics import bow_position
from sail_env.race_env import LEG_NAMES, PARK_X, RaceEnv
from sail_env.rules import signed_distance
from sail_env.weather import reference_wind

OUTCOME_KEYS = ["good_start_on_line", "success_strict", "started", "ocs", "on_line_at_gun", "upwind_efficiency"]
FLEET_KEYS = ["fouls", "contacts", "mark_hits", "unserved_foul", "clear_air", "rank"]
RACE_KEYS = ["race_finished", "finish_time", "race_rank", "legs_completed"]
LEADERBOARD_FIELDS = ["time", "run", "mode", "player", "opponents", "good_start", "started", "ocs", "start_time",
                      "rank", "penalties", "finished", "finish_time", "race_rank", "legs"]


def _per_boat(x, n):
    return x if x.ndim >= 1 and x.shape[0] == n else jnp.broadcast_to(x, (n,))


def snapshot(env, params, s):
    """What a replay records about one moment: per-boat position, heading, speed, start status, penalty,
    plus the clock, wind and line. Works for solo (one boat) and fleet states."""
    fleet = isinstance(env, FleetStartEnv)
    st = s.boats if fleet else jax.tree.map(lambda x: x[None], s)
    b = st.boat
    first = lambda tree: jax.tree.map(lambda x: x[0], tree)  # noqa: E731
    wd, tws = reference_wind(first(st.wind), first(st.weather), st.t[0], params)  # at the line centre
    line = jax.tree.map(lambda x: x[0], st.line)
    bx, by = bow_position(b, params)
    out = {
        "t": st.t[0], "x": b.x, "y": b.y, "heading": b.heading, "speed": b.speed,
        "below": -signed_distance(line, bx, by),
        "started": st.status.started, "ocs": st.status.ocs,
        "penalty_left": s.penalty_left if fleet else jnp.zeros(1),
        "contacts": s.contacts if fleet else jnp.zeros(1, jnp.int32),
        "wind_dir": wd, "tws": tws,
        "pin": jnp.stack([line.pin_x, line.pin_y]), "committee": jnp.stack([line.cb_x, line.cb_y]),
    }
    if isinstance(env, RaceEnv):
        k = env.course(s, params)
        out["leg"] = s.leg
        out["finish_time"] = s.finish_time
        out["marks"] = jnp.stack([jnp.stack(k["w"]), jnp.stack(k["g1"]), jnp.stack(k["g2"])])
    return out


def run_episode(env, params, key, policies, boat_policy=None):
    """One full episode (no auto-reset). policies: list of policy functions; boat_policy[i] = index into
    policies for boat i (default: boat 0 uses policies[0], all others policies[-1]).

    Returns (record, info): record holds per-step arrays with a leading time axis (n_steps + 1 frames,
    frozen after the episode ends); info is the final step's info."""
    fleet = isinstance(env, FleetStartEnv)
    n = env.num_agents if fleet else 1
    if boat_policy is None:
        boat_policy = [0] + [len(policies) - 1] * (n - 1)
    boat_policy = jnp.asarray(boat_policy)
    idx = jnp.arange(n)
    k_reset, k_run = jax.random.split(key)
    obs, state = env.reset(k_reset, params)

    def boats_of(s):
        return s.boats if fleet else jax.tree.map(lambda x: x[None], s)

    def act(k, obs, s):
        obs_b = obs if fleet else obs[None]
        keys = jax.random.split(k, n)
        cand = jnp.stack([jax.vmap(p)(keys, obs_b, boats_of(s), idx) for p in policies])  # (P, N)
        a = cand[boat_policy, idx].astype(jnp.int32)
        return a if fleet else a[0]

    snap = lambda s: snapshot(env, params, s)  # noqa: E731

    def body(carry, k):
        obs, s, finished = carry
        k_act, k_step = jax.random.split(k)
        a = act(k_act, obs, s)
        obs2, s2, r, done, info = env.step_env(k_step, s, a, params)
        end = done[0] if fleet else done
        s2 = jax.tree.map(lambda new, old: jnp.where(finished, old, new), s2, s)
        obs2 = jnp.where(finished, obs, obs2)
        a = jnp.where(finished, -1, a)
        return (obs2, s2, finished | end), (snap(s2), a, info, end & ~finished)

    (_, _, _), (snaps, actions, infos, ends) = jax.lax.scan(
        body, (obs, state, jnp.bool_(False)), jax.random.split(k_run, params.n_steps))
    first = snap(state)
    record = jax.tree.map(lambda a, b: jnp.concatenate([a[None], b]), first, snaps)
    record["actions"] = actions if fleet else actions[:, None]
    last = jnp.argmax(ends)
    info = jax.tree.map(lambda x: x[last], infos)
    return record, info


def outcomes(info, n):
    """Per-boat outcome dict (numpy) from a final info."""
    out = {}
    for k in OUTCOME_KEYS + FLEET_KEYS + RACE_KEYS:
        if k in info:
            out[k] = np.asarray(_per_boat(jnp.asarray(info[k]), n))
    if "start_time" in info:
        out["start_time"] = np.asarray(_per_boat(jnp.asarray(info["start_time"]), n))
    return out


def params_dict(params):
    d = {}
    for f in dataclasses.fields(params):
        v = getattr(params, f.name)
        d[f.name] = float(v) if hasattr(v, "dtype") or isinstance(v, float) else v
    return d


def save_run(directory, record, info, params, meta=None):
    """Save a replay (npz + json) and append a line to directory/leaderboard.csv. Returns the npz path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}_{meta.get('mode', 'run')}_{meta.get('player', meta.get('agent', 'agent'))}".replace(" ", "-")
    name, k = base, 1
    while (directory / f"{name}.npz").exists():  # several runs saved within the same second
        k += 1
        name = f"{base}-{k}"
    rec = {k: np.asarray(v) for k, v in record.items()}
    n = rec["x"].shape[1]
    out = outcomes(info, n)
    np.savez_compressed(directory / f"{name}.npz", **rec)
    summary = {k: v.tolist() for k, v in out.items()}
    side = {"name": name, "time": stamp, "meta": meta, "outcome": summary, "params": params_dict(params)}
    (directory / f"{name}.json").write_text(json.dumps(side, indent=1, default=str), encoding="utf-8")
    # Leaderboard line for boat 0 (the player / the evaluated agent).
    race = "race_finished" in out
    row = {"time": stamp, "run": name, "mode": meta.get("mode", ""), "player": meta.get("player", meta.get("agent", "")),
           "opponents": meta.get("opponents", ""),
           "good_start": "" if race else int(out.get("good_start_on_line", [0])[0]),
           "started": int(out.get("started", [0])[0]), "ocs": "" if race else int(out.get("ocs", [0])[0]),
           "start_time": f"{float(out['start_time'][0]):.1f}" if "start_time" in out and np.isfinite(out["start_time"][0]) else "",
           "rank": f"{float(out['rank'][0]):.1f}" if "rank" in out and not race else "",
           "penalties": int(out["fouls"][0]) if "fouls" in out else "",
           "finished": int(out["race_finished"][0]) if race else "",
           "finish_time": f"{float(out['finish_time'][0]):.1f}" if race and np.isfinite(out["finish_time"][0]) else "",
           "race_rank": f"{float(out['race_rank'][0]):.1f}" if race else "",
           "legs": int(out["legs_completed"][0]) if race else ""}
    board = directory / "leaderboard.csv"
    new = not board.exists()
    with board.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LEADERBOARD_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return directory / f"{name}.npz"


def load_run(path):
    path = Path(path)
    rec = dict(np.load(path))
    side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) if path.with_suffix(".json").exists() else {}
    return {"record": rec, **side}


def episode_frames(rec):
    """Number of recorded frames up to the end of the episode (records are frozen afterwards)."""
    return int(np.argmax(np.asarray(rec["t"]))) + 1


def render_replay(run, gif_path, fps=6, title=None, stride=None):
    """Render a saved run (from load_run) as a GIF: hulls, tracks, line (and in the race the course),
    turning wind arrow. stride: draw every stride-th second (default: about 240 frames at most)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from sail_env.render import FLEET_COLORS, add_wind_arrow, draw_course, hull_polygon

    rec = run["record"]
    p = run.get("params", {})
    length, beam = p.get("boat_length", 6.9), p.get("boat_beam", 2.25)
    n_t = episode_frames(rec)
    x, y, h = rec["x"][:n_t].astype(float), rec["y"][:n_t].astype(float), rec["heading"][:n_t]
    parked = x > 0.5 * PARK_X  # finished boats leave the course
    x, y = np.where(parked, np.nan, x), np.where(parked, np.nan, y)
    n_b = x.shape[1]
    race = "marks" in rec
    pin, cb = rec["pin"][0], rec["committee"][0]
    marks = rec["marks"][0] if race else None
    fixed = np.stack([pin, cb] + (list(marks) if race else []))
    xs = np.concatenate([x[np.isfinite(x)], fixed[:, 0]])
    ys = np.concatenate([y[np.isfinite(y)], fixed[:, 1]])
    pad = 30.0 if race else 15.0
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    scale = max(1.0, max(x1 - x0, y1 - y0) / 200.0)  # enlarge hulls and marks on a big course
    fig, ax = plt.subplots(figsize=(7, 7 * min(max((y1 - y0) / (x1 - x0), 0.6), 1.4)))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    draw_course(ax, pin, cb, p, marks, scale)
    update_wind = add_wind_arrow(ax, x0, x1, y0, y1)
    colors = [FLEET_COLORS[i % len(FLEET_COLORS)] for i in range(n_b)]
    trails = [ax.plot([], [], "-", color=colors[i], lw=1, alpha=0.5)[0] for i in range(n_b)]
    hulls = [plt.Polygon(hull_polygon(rec["x"][0, i], rec["y"][0, i], h[0, i], length * scale, beam * scale),
                         closed=True, fc=colors[i], ec="k", zorder=3) for i in range(n_b)]
    for hull in hulls:
        ax.add_patch(hull)
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace", fontsize=8)
    ax.set_title(title or run.get("name", ""), fontsize=9)
    stride = stride or max(1, n_t // 240)
    frames = list(range(0, n_t, stride)) + ([n_t - 1] if (n_t - 1) % stride else [])

    def frame(k):
        update_wind(float(rec["wind_dir"][k]), float(rec["tws"][k]) / KNOT)
        for i in range(n_b):
            hulls[i].set_visible(not parked[k, i])
            if not parked[k, i]:
                hulls[i].set_xy(hull_polygon(x[k, i], y[k, i], h[k, i], length * scale, beam * scale))
            edge = "tab:red" if rec["ocs"][k, i] else ("tab:green" if rec["started"][k, i] else "k")
            hulls[i].set_edgecolor(edge)
            hulls[i].set_linewidth(3.0 if rec["penalty_left"][k, i] > 0 else (2.0 if edge != "k" else 0.8))
            trails[i].set_data(x[: k + 1, i], y[: k + 1, i])
        t = float(rec["t"][k])
        clock = f"T{'-' if t < 0 else '+'}{abs(t):4.0f} s"
        if race:
            leg = rec["leg"][k]
            text.set_text(f"{clock}   boat 0: {LEG_NAMES[int(leg[0])]}   finished {int((leg >= len(LEG_NAMES) - 1).sum())}/{n_b}")
        else:
            text.set_text(f"{clock}   boat 0: {rec['speed'][k, 0] / KNOT:4.1f} kn   started {int(rec['started'][k].sum())}/{n_b}")
        return hulls + trails + [text]

    anim = FuncAnimation(fig, frame, frames=frames, blit=False)
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
