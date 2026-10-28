"""Play yourself: steer boat 0 with the keyboard, a start or the whole race.

    python -m sail_env.play                         # a 5-boat fleet start vs the RL boats
    python -m sail_env.play --opponents digital     # vs hand-written digital sailors
    python -m sail_env.play --mode solo             # alone on the line
    python -m sail_env.play --mode race             # the whole race vs 4 digital racers
    python -m sail_env.play --mode race-solo        # the whole race alone
    python -m sail_env.play --mode race-weather     # ... with moving shifts and pressure, current and waves
    python -m sail_env.play --name Mate --speed 1.5

Keys (each press is one action for the next second; no key = hold course):
    Up / w           head up 5 deg            Shift+Up / W     head up 15 deg
    Down / s         bear away 5 deg          Shift+Down / S   bear away 15 deg
    Space / t        tack or gybe             l / e            luff (ease the sails)
    p                pause / resume           Esc / q          quit

Your boat is red. When the episode ends the run is saved to runs/ (replay + leaderboard line).
A good start: bow 0-3 m below the line and between the ends when the gun goes, then cross within
3 s at >= 90% of full speed, on starboard (in the fleet also: clear air, no penalty left to serve).
The race: start, beat to the windward mark (leave it to port), run through the leeward gate, beat
again, run to the finish (the start line, crossed downwind). The race runs at 4x by default (--speed).
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from sail_env.agents import load_agent
from sail_env.config import KNOT
from sail_env.fleet_env import FleetStartEnv
from sail_env.geometry import signed_twa
from sail_env.physics import DOWN5, DOWN15, HOLD, LUFF, TACK, UP5, UP15
from sail_env.presets import RACE_PRESETS, make
from sail_env.race_env import LEG_NAMES, N_LEGS, PARK_X
from sail_env.runs import outcomes, save_run, snapshot

KEYS = {
    "up": UP5, "w": UP5, "shift+up": UP15, "W": UP15,
    "down": DOWN5, "s": DOWN5, "shift+down": DOWN15, "S": DOWN15,
    " ": TACK, "t": TACK, "l": LUFF, "e": LUFF,
}
ACTION_NAMES = ["hold", "head up 5", "bear away 5", "head up 15", "bear away 15", "tack/gybe", "luff"]


class PlaySession:
    """Game logic without any window: press(key) queues an action, tick() advances one second."""

    def __init__(self, mode="fleet", opponents=None, player="player", seed=None):
        self.env, self.params = make(mode)
        self.race = mode in RACE_PRESETS
        if opponents is None:
            opponents = "digital" if self.race else "rl"  # in fleet races the solo-trained rl-racer is the weaker opponent
        self.fleet = isinstance(self.env, FleetStartEnv)
        self.mode, self.player = mode, player
        self.n = self.env.num_agents if self.fleet else 1
        self.opponents_name = opponents if self.n > 1 else ""
        env, params = self.env, self.params
        opp = load_agent(opponents)(env, params) if self.n > 1 else (lambda k, o, s, i: jnp.int32(0))
        idx = jnp.arange(self.n)

        def step(key, obs, state, player_action):
            k_opp, k_env = jax.random.split(key)
            if self.fleet:
                acts = jax.vmap(opp)(jax.random.split(k_opp, self.n), obs, state.boats, idx).astype(jnp.int32)
                acts = acts.at[0].set(player_action)
            else:
                acts = jnp.int32(player_action)
            obs2, state2, r, done, info = env.step_env(k_env, state, acts, params)
            return obs2, state2, (done[0] if self.fleet else done), info, acts

        self._step = jax.jit(step)
        self._snap = jax.jit(lambda s: snapshot(env, params, s))
        seed = int(time.time()) if seed is None else seed
        self.key = jax.random.PRNGKey(seed)
        self.key, k = jax.random.split(self.key)
        self.obs, self.state = env.reset(k, params)
        self.pending = HOLD
        self.last_action = HOLD
        self.done = False
        self.info = None
        self.frames = [self._snap(self.state)]
        self.actions = []

    def press(self, key_name):
        """Queue the action for a key (ignored if the key is not mapped). Returns the action or None."""
        if key_name in KEYS:
            self.pending = KEYS[key_name]
            return self.pending
        return None

    def tick(self):
        """Advance one decision (one second). Returns True when the episode has ended."""
        if self.done:
            return True
        self.key, k = jax.random.split(self.key)
        self.obs, self.state, done, info, acts = self._step(k, self.obs, self.state, jnp.int32(self.pending))
        self.last_action, self.pending = self.pending, HOLD
        self.frames.append(self._snap(self.state))
        self.actions.append(np.atleast_1d(np.asarray(acts)))
        if bool(done):
            self.done, self.info = True, info
        return self.done

    def record(self):
        rec = {k: np.stack([np.asarray(f[k]) for f in self.frames]) for k in self.frames[0]}
        acts = np.stack(self.actions) if self.actions else np.zeros((0, self.n), np.int32)
        rec["actions"] = acts
        return rec

    def result(self):
        """Outcome of boat 0 (the player) as a dict, once the episode has ended."""
        return {k: v[0] for k, v in outcomes(self.info, self.n).items()} if self.done else None

    def save(self, directory="runs"):
        meta = {"mode": self.mode, "player": self.player, "opponents": self.opponents_name, "human": True}
        return save_run(directory, self.record(), self.info, self.params, meta)

    def hud(self):
        """Text for the heads-up display about the player's boat."""
        s = self.state.boats if self.fleet else jax.tree.map(lambda x: x[None], self.state)
        f = self.frames[-1]
        t = float(f["t"])
        speed = float(f["speed"][0]) / KNOT
        twa = float(np.degrees(signed_twa(s.boat.heading[0], f["wind_dir"])))
        tack = "starboard" if twa >= 0 else "port"
        status = "OCS - go back below the line!" if bool(f["ocs"][0]) else ("STARTED" if bool(f["started"][0]) else "")
        pen = float(f["penalty_left"][0])
        if self.fleet and pen > 0:
            turning = bool(self.state.penalty_turning[0])
            status += "  PENALTY TURN" if turning or self.params.penalty_clear_distance <= 0 else "  PENALTY: get clear"
        text = (f"T{'-' if t < 0 else '+'}{abs(t):3.0f} s   {speed:4.1f} kn   TWA {abs(twa):3.0f} deg {tack}\n"
                f"last action: {ACTION_NAMES[self.last_action]}   {status}")
        if self.race:
            leg = int(f["leg"][0])
            if leg >= N_LEGS:
                text += f"\nFINISHED in {float(f['finish_time'][0]):.0f} s"
            elif not bool(f["started"][0]):
                text += "\nleg: start"
            else:
                tx, ty = self.next_mark(f, leg)
                d = float(np.hypot(tx - f["x"][0], ty - f["y"][0]))
                ahead = int((np.asarray(f["leg"]) > leg).sum())
                text += f"\nleg: {LEG_NAMES[leg]}   next mark {d:4.0f} m   boats ahead on the course: {ahead}"
        return text

    @staticmethod
    def next_mark(f, leg):
        """Where boat 0 is heading on leg `leg`: windward mark, gate centre or finish line centre."""
        w, g1, g2 = np.asarray(f["marks"])
        pin, cb = np.asarray(f["pin"]), np.asarray(f["committee"])
        return {0: w, 1: 0.5 * (g1 + g2), 2: w}.get(leg, 0.5 * (pin + cb))


def play(mode="fleet", opponents=None, player="player", speed=None, seed=None, out="runs"):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    from sail_env.render import FLEET_COLORS, add_wind_arrow, draw_course, hull_polygon

    for k in list(plt.rcParams):  # free the keys matplotlib uses for its own shortcuts
        if k.startswith("keymap."):
            plt.rcParams[k] = []
    session = PlaySession(mode, opponents, player, seed)
    if speed is None:
        speed = 4.0 if session.race else 1.0
    p = session.params
    f0 = session.frames[0]
    pin, cb = np.asarray(f0["pin"]), np.asarray(f0["committee"])
    marks = np.asarray(f0["marks"]) if session.race else None
    fixed = np.stack([pin, cb] + (list(marks) if session.race else []))
    xs = np.concatenate([np.asarray(f0["x"]), fixed[:, 0]])
    ys = np.concatenate([np.asarray(f0["y"]), fixed[:, 1]])
    pad = 40.0
    if session.race:  # the whole course, with room for the laylines on both sides
        half_w = max(0.6 * (ys.max() - ys.min()), 0.5 * (xs.max() - xs.min()) + pad)
        xm = 0.5 * (xs.min() + xs.max())
        xs = np.concatenate([xs, [xm - half_w, xm + half_w]])
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, max(ys.max(), 0.0) + pad
    scale = max(1.0, max(x1 - x0, y1 - y0) / 200.0)  # bigger hulls and marks on the whole-course view
    fig, ax = plt.subplots(figsize=(8, 8 * min(max((y1 - y0) / (x1 - x0), 0.6), 1.3)))
    fig.canvas.manager.set_window_title("sail_env: your race" if session.race else "sail_env: your start")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    draw_course(ax, pin, cb, p, marks, scale)
    update_wind = add_wind_arrow(ax, x0, x1, y0, y1)
    length, beam = p.boat_length * scale, p.boat_beam * scale
    n = session.n
    colors = ["tab:red"] + [FLEET_COLORS[i % len(FLEET_COLORS)] for i in range(n - 1)]
    trails = [ax.plot([], [], "-", color=colors[i], lw=1.2 if i == 0 else 0.8, alpha=0.6)[0] for i in range(n)]
    hulls = []
    for i in range(n):
        poly = plt.Polygon(hull_polygon(f0["x"][i], f0["y"][i], f0["heading"][i], length, beam),
                           closed=True, fc=colors[i], ec="k", zorder=3)
        ax.add_patch(poly)
        hulls.append(poly)
    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace", fontsize=9,
                  bbox=dict(fc="white", alpha=0.8, ec="none"))
    ax.set_title(f"{player}: you are the red boat  |  Up/Down steer (Shift = 15 deg), Space tack, L luff, P pause",
                 fontsize=9)
    state = {"paused": False, "saved": None}

    def on_key(event):
        if event.key in ("escape", "q"):
            plt.close(fig)
        elif event.key == "p":
            state["paused"] = not state["paused"]
        else:
            session.press(event.key)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def frame(_):
        if not state["paused"] and not session.done:
            session.tick()
        f = session.frames[-1]
        update_wind(float(f["wind_dir"]), float(f["tws"]) / KNOT)
        x = np.stack([np.asarray(fr["x"]) for fr in session.frames]).astype(float)
        y = np.stack([np.asarray(fr["y"]) for fr in session.frames]).astype(float)
        parked = x > 0.5 * PARK_X  # finished boats leave the course
        x, y = np.where(parked, np.nan, x), np.where(parked, np.nan, y)
        for i in range(n):
            hulls[i].set_visible(not parked[-1, i])
            if not parked[-1, i]:
                hulls[i].set_xy(hull_polygon(f["x"][i], f["y"][i], f["heading"][i], length, beam))
            edge = "tab:red" if bool(f["ocs"][i]) else ("tab:green" if bool(f["started"][i]) else "k")
            hulls[i].set_edgecolor(edge)
            hulls[i].set_linewidth(3.0 if float(f["penalty_left"][i]) > 0 else (2.0 if edge != "k" else 0.8))
            trails[i].set_data(x[:, i], y[:, i])
        text = session.hud()
        if session.done:
            if state["saved"] is None:
                state["saved"] = session.save(out)
                print(f"saved {state['saved']}")
            r = session.result()
            if session.race:
                verdict = (f"FINISHED in {r['finish_time']:.0f} s" if r["race_finished"]
                           else f"did not finish ({int(r['legs_completed'])} of {N_LEGS} legs)")
                extra = f"   place {r['race_rank'] + 1:.1f} of {n}" if n > 1 else ""
            else:
                verdict = "GOOD START!" if r.get("good_start_on_line") else "no good start"
                extra = f"   rank {r['rank'] + 1:.1f} of {n}" if "rank" in r else ""
            text += f"\n{verdict}{extra}   (saved to {out}/; close the window to exit)"
        hud.set_text(text)
        return hulls + trails + [hud]

    anim = FuncAnimation(fig, frame, interval=int(1000 / max(speed, 0.1)), blit=False, cache_frame_data=False)
    plt.show()
    del anim
    return state["saved"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["fleet", "solo", "race", "race-solo", "fleet-weather", "solo-weather", "race-weather",
                                       "race-solo-weather"], default="fleet")
    ap.add_argument("--opponents", default=None,
                    help="rl | digital | random | hold | path or module:function (default: rl for starts, digital for the race)")
    ap.add_argument("--name", default="player", help="your name for the leaderboard")
    ap.add_argument("--speed", type=float, default=None, help="simulated seconds per real second (default 1, race 4)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    play(args.mode, args.opponents, args.name, args.speed, args.seed, args.out)


if __name__ == "__main__":
    main()
