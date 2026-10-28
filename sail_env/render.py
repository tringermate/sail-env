"""Roll out an episode and render it as a GIF with matplotlib.

    traj = rollout(env, params, key, policy_fn)          # EnvState with a leading time axis
    render_gif(traj, params, "docs/figures/episode.gif")

`policy_fn(key, obs, t_index) -> action` lets the same code render a fixed action sequence,
a random policy or (later) a trained network.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from sail_env.config import KNOT, EnvParams  # noqa: E402


def rollout(env, params: EnvParams, key, policy_fn, init_state=None):
    """Run one full episode (no auto-reset). Returns (trajectory, final_info)."""
    k_reset, k_run = jax.random.split(key)
    obs, state = env.reset(k_reset, params)
    if init_state is not None:
        state = init_state
        obs = env.get_obs(state, params)

    def body(carry, inp):
        obs, state, finished = carry
        k, i = inp
        k_act, k_step = jax.random.split(k)
        action = policy_fn(k_act, obs, i)
        new_obs, new_state, reward, done, info = env.step_env(k_step, state, action, params)
        # Episodes can be shorter than n_steps (random start times): freeze once finished.
        new_state = jax.tree.map(lambda n, o: jnp.where(finished, o, n), new_state, state)
        new_obs = jnp.where(finished, obs, new_obs)
        reward = jnp.where(finished, 0.0, reward)
        return (new_obs, new_state, finished | done), (new_state, action, reward, info, done & ~finished)

    keys = jax.random.split(k_run, params.n_steps)
    init = (obs, state, jnp.bool_(False))
    _, (states, actions, rewards, infos, ends) = jax.lax.scan(body, init, (keys, jnp.arange(params.n_steps)))
    # Prepend the initial state so frame 0 shows the boat at reset.
    traj = jax.tree.map(lambda s0, s: jnp.concatenate([s0[None], s]), state, states)
    last = jnp.argmax(ends)  # the step where the episode ended
    return traj, actions, rewards, jax.tree.map(lambda x: x[last], infos)


def hull_polygon(x, y, heading, length, beam):
    """Simple sportboat outline: pointed bow, widest a third back, slightly narrower transom."""
    pts = np.array([[0.5, 0.0], [0.15, 0.5], [-0.5, 0.4], [-0.5, -0.4], [0.15, -0.5]])  # (fwd, stbd)
    fwd, stbd = pts[:, 0] * length, pts[:, 1] * beam
    fx, fy = np.sin(heading), np.cos(heading)
    rx, ry = fy, -fx
    return np.stack([x + fwd * fx + stbd * rx, y + fwd * fy + stbd * ry], axis=1)


ACTION_NAMES = ["hold", "up 5", "down 5", "up 15", "down 15", "tack/gybe", "luff"]


def wind_dirs(wind, t):
    """Current true wind direction (rad, blowing FROM) for numpy wind arrays and clock t."""
    return wind.base_dir + wind.shift_amp * np.sin(2 * np.pi * t / wind.shift_period + wind.shift_phase)


def add_wind_arrow(ax, x0, x1, y0, y1):
    """A quiver arrow in the top-right corner plus a label; returns an update(direction, tws_kn) function."""
    ax_w, ay_w = x1 - 0.1 * (x1 - x0), y1 - 0.14 * (y1 - y0)
    length = 0.08 * (y1 - y0)
    q = ax.quiver([ax_w], [ay_w], [0.0], [-length], angles="xy", scale_units="xy", scale=1, color="tab:blue",
                  width=0.008, zorder=5)
    label = ax.text(ax_w, ay_w + 0.05 * (y1 - y0), "", ha="center", fontsize=8, color="tab:blue")

    def update(direction, tws_kn):
        # The arrow points where the wind blows TO (opposite of the FROM direction).
        q.set_UVC([-length * np.sin(direction)], [-length * np.cos(direction)])
        label.set_text(f"wind {tws_kn:.1f} kn, {round(float(np.degrees(direction))) % 360:03d}°")

    return update


def add_weather_layer(ax, params: EnvParams, x0, x1, y0, y1, n=48, n_arrows=7):
    """With weather_shifts / weather_pressure: shade the wind speed (darker = more wind, like dark water on
    the course) and draw arrows of the local wind direction. Returns update(wind, weather, t), or None
    without weather. wind, weather: one episode's (unbatched) Wind and Weather."""
    if not (params.weather_shifts or params.weather_pressure):
        return None
    from sail_env.weather import wind_at_point

    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    ax_, ay_ = np.meshgrid(np.linspace(x0, x1, n_arrows + 2)[1:-1], np.linspace(y0, y1, n_arrows + 2)[1:-1])
    field = jax.jit(lambda w, we, t, x, y: wind_at_point(w, we, t, x, y, params))
    img = ax.imshow(np.ones_like(gx), extent=(x0, x1, y0, y1), origin="lower", cmap="Blues", vmin=0.7, vmax=1.3,
                    alpha=0.45, zorder=-2, interpolation="bilinear")
    size = 0.05 * (y1 - y0)
    q = ax.quiver(ax_, ay_, np.zeros_like(ax_), -size * np.ones_like(ax_), angles="xy", scale_units="xy", scale=1,
                  color="tab:blue", alpha=0.35, width=0.004, zorder=-1)

    def update(wind, weather, t):
        _, s = field(wind, weather, t, jnp.asarray(gx, jnp.float32), jnp.asarray(gy, jnp.float32))
        img.set_data(np.asarray(s) / float(wind.tws))
        d, _ = field(wind, weather, t, jnp.asarray(ax_, jnp.float32), jnp.asarray(ay_, jnp.float32))
        d = np.asarray(d)
        q.set_UVC(-size * np.sin(d), -size * np.cos(d))  # pointing where the wind blows to

    return update


def weather_caption(weather, params: EnvParams) -> str:
    """'current 0.6 kn to 045, waves 0.4 m' for the parts of the weather that are on (or '')."""
    parts = []
    if params.water_current:
        cx, cy = float(weather.current_x), float(weather.current_y)
        parts.append(f"current {np.hypot(cx, cy) / KNOT:.1f} kn to {round(np.degrees(np.arctan2(cx, cy))) % 360:03d}")
    if params.waves:
        parts.append(f"waves {float(weather.wave_h):.1f} m")
    return ", ".join(parts)


def _frame_wind(wind, weather, t, params: EnvParams):
    """(direction, speed) for the wind arrow: at the line centre, with the weather if it is on."""
    from sail_env.weather import any_weather, reference_wind

    if any_weather(params):
        return tuple(float(v) for v in reference_wind(wind, weather, t, params))
    return float(wind_dirs(wind, t)), float(wind.tws)


def _param(params, name, default):
    return params.get(name, default) if isinstance(params, dict) else getattr(params, name, default)


def draw_course(ax, pin, cb, params, marks=None, scale=1.0):
    """Start line (dashed) with pin and committee boat; with `marks` ((3, 2): windward mark, left and
    right gate mark, as recorded by runs.snapshot in the race) also the course. Marks are drawn at least
    `scale` m in radius so they stay visible on a whole-course view."""
    ax.plot([pin[0], cb[0]], [pin[1], cb[1]], "k--", lw=1)
    ax.add_patch(plt.Circle(tuple(pin), max(_param(params, "pin_radius", 1.0), scale), color="tab:orange", zorder=2))
    ax.add_patch(plt.Circle(tuple(cb), max(_param(params, "committee_radius", 4.0), 2.0 * scale), color="dimgray",
                            zorder=2))
    if marks is None:
        return
    r = max(_param(params, "course_mark_radius", 1.0), scale)
    w, g1, g2 = (np.asarray(m) for m in marks)
    ax.add_patch(plt.Circle(tuple(w), r, color="gold", ec="k", lw=0.5, zorder=2))
    ax.annotate("W", tuple(w), xytext=(6, 6), textcoords="offset points", fontsize=8)
    for g in (g1, g2):
        ax.add_patch(plt.Circle(tuple(g), r, color="tab:orange", ec="k", lw=0.5, zorder=2))
    ax.plot([g1[0], g2[0]], [g1[1], g2[1]], ":", color="tab:orange", lw=1)
    ax.annotate("gate", tuple(g2), xytext=(6, -10), textcoords="offset points", fontsize=8)


FLEET_COLORS = ["tab:blue", "tab:purple", "tab:orange", "tab:cyan", "tab:brown", "tab:pink", "tab:olive", "tab:gray"]


def render_fleet_gif(traj, params: EnvParams, path, fps=6, title=""):
    """traj: FleetState with a leading time axis (from fleet_env.rollout_fleet).
    One hull colour per boat; edge red = OCS, green = started, thick = serving a penalty."""
    tr = jax.tree.map(np.asarray, traj)
    st = tr.boats
    b, line = st.boat, st.line
    n_t, n_b = b.x.shape
    xs = np.concatenate([b.x.ravel(), line.pin_x[0], line.cb_x[0]])
    ys = np.concatenate([b.y.ravel(), line.pin_y[0], line.cb_y[0]])
    pad = 15.0
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    fig, ax = plt.subplots(figsize=(7, 7 * min(max((y1 - y0) / (x1 - x0), 0.6), 1.4)))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.plot([line.pin_x[0, 0], line.cb_x[0, 0]], [line.pin_y[0, 0], line.cb_y[0, 0]], "k--", lw=1)
    ax.add_patch(plt.Circle((line.pin_x[0, 0], line.pin_y[0, 0]), params.pin_radius, color="tab:orange"))
    ax.add_patch(plt.Circle((line.cb_x[0, 0], line.cb_y[0, 0]), params.committee_radius, color="dimgray"))
    update_wind = add_wind_arrow(ax, x0, x1, y0, y1)  # turns with the wind shifts
    update_weather = add_weather_layer(ax, params, x0, x1, y0, y1)  # None without weather
    caption = weather_caption(jax.tree.map(lambda a: a[0, 0], st.weather), params)
    colors = [FLEET_COLORS[i % len(FLEET_COLORS)] for i in range(n_b)]
    trails = [ax.plot([], [], "-", color=colors[i], lw=1, alpha=0.5)[0] for i in range(n_b)]
    hulls = []
    for i in range(n_b):
        hull = plt.Polygon(hull_polygon(b.x[0, i], b.y[0, i], b.heading[0, i], params.boat_length, params.boat_beam),
                           closed=True, fc=colors[i], ec="k")
        ax.add_patch(hull)
        hulls.append(hull)
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace", fontsize=8)
    ax.set_title(title, fontsize=9)

    # Wind shadow cones (faint, in each boat's colour), along the true or apparent wind.
    from sail_env.fleet_rules import shadow_direction

    sh_len = params.shadow_length_bl * params.boat_length
    sh_half = float(params.shadow_half_angle)
    cones = [plt.Polygon(np.zeros((3, 2)), closed=True, fc=colors[i], ec="none", alpha=0.12, zorder=0)
             for i in range(n_b)]
    for c in cones:
        ax.add_patch(c)

    def cone_points(k, i, dx, dy):
        x, y = b.x[k, i], b.y[k, i]
        pts = [(x, y)]
        for s in (-1.0, 1.0):
            c, sn = np.cos(s * sh_half), np.sin(s * sh_half)
            pts.append((x + sh_len * (c * dx - sn * dy), y + sh_len * (sn * dx + c * dy)))
        return np.array(pts)

    from sail_env.weather import any_weather, water_wind

    def frame(k):
        boats_k = jax.tree.map(lambda a: a[k], tr.boats.boat)
        w_k = jax.tree.map(lambda a: a[k, 0], st.wind)
        we_k = jax.tree.map(lambda a: a[k, 0], st.weather)
        wdir, wtws = _frame_wind(w_k, we_k, st.t[k, 0], params)
        update_wind(wdir, wtws / KNOT)
        if update_weather is not None:
            update_weather(w_k, we_k, st.t[k, 0])
        if any_weather(params):  # each boat's shadow follows the wind where it is
            wdir, wtws = (np.asarray(v) for v in water_wind(w_k, we_k, st.t[k, 0], boats_k.x, boats_k.y, params))
        ddx, ddy = (np.asarray(v) for v in shadow_direction(boats_k, wdir, wtws, params))
        for i in range(n_b):
            cones[i].set_xy(cone_points(k, i, ddx[i], ddy[i]))
        for i in range(n_b):
            hulls[i].set_xy(hull_polygon(b.x[k, i], b.y[k, i], b.heading[k, i], params.boat_length, params.boat_beam))
            edge = "tab:red" if st.status.ocs[k, i] else ("tab:green" if st.status.started[k, i] else "k")
            hulls[i].set_edgecolor(edge)
            hulls[i].set_linewidth(3.0 if tr.penalty_left[k, i] > 0 else (2.0 if edge != "k" else 0.8))
            trails[i].set_data(b.x[: k + 1, i], b.y[: k + 1, i])
        t = round(float(st.t[k, 0]), 3)
        sign = "-" if t < 0 else "+"
        started = int(st.status.started[k].sum())
        text.set_text(f"T{sign}{abs(t):4.0f} s   started {started}/{n_b}   penalties {int(tr.fouls[k].sum())}"
                      f"   contacts {int(tr.contacts[k].sum())}" + (f"\n{caption}" if caption else ""))
        return hulls + trails + [text]

    anim = FuncAnimation(fig, frame, frames=n_t, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def render_gif(traj, params: EnvParams, path, actions=None, fps=6, title=""):
    tr = jax.tree.map(np.asarray, traj)
    b, line, st = tr.boat, tr.line, tr.status
    n = b.x.shape[0]

    xs = np.concatenate([b.x, line.pin_x, line.cb_x])
    ys = np.concatenate([b.y, line.pin_y, line.cb_y])
    pad = 15.0
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad

    fig, ax = plt.subplots(figsize=(6, 6 * min(max((y1 - y0) / (x1 - x0), 0.6), 1.6)))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.plot([line.pin_x[0], line.cb_x[0]], [line.pin_y[0], line.cb_y[0]], "k--", lw=1)
    ax.plot(line.pin_x[0], line.pin_y[0], "o", color="tab:orange", ms=8, label="pin")
    ax.plot(line.cb_x[0], line.cb_y[0], "s", color="dimgray", ms=10, label="committee")
    ax.legend(loc="lower right", fontsize=8)

    update_wind = add_wind_arrow(ax, x0, x1, y0, y1)  # turns with the wind shifts
    update_weather = add_weather_layer(ax, params, x0, x1, y0, y1)  # None without weather
    caption = weather_caption(jax.tree.map(lambda a: a[0], tr.weather), params)

    (trail,) = ax.plot([], [], "-", color="tab:blue", lw=1, alpha=0.5)
    hull = plt.Polygon(hull_polygon(b.x[0], b.y[0], b.heading[0], params.boat_length, params.boat_beam),
                       closed=True, fc="tab:blue", ec="k")
    ax.add_patch(hull)
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace", fontsize=8)
    ax.set_title(title, fontsize=9)

    def frame(i):
        w_i = jax.tree.map(lambda a: a[i], tr.wind)
        we_i = jax.tree.map(lambda a: a[i], tr.weather)
        wdir, wtws = _frame_wind(w_i, we_i, tr.t[i], params)
        update_wind(wdir, wtws / KNOT)
        if update_weather is not None:
            update_weather(w_i, we_i, tr.t[i])
        hull.set_xy(hull_polygon(b.x[i], b.y[i], b.heading[i], params.boat_length, params.boat_beam))
        color = "tab:red" if st.ocs[i] else ("tab:green" if st.started[i] else "tab:blue")
        hull.set_facecolor(color)
        trail.set_data(b.x[: i + 1], b.y[: i + 1])
        t = round(float(tr.t[i]), 3)
        clock = f"T{'-' if t < 0 else '+'}{abs(t):4.0f} s"
        status = "OCS" if st.ocs[i] else (f"started T+{st.start_time[i]:.1f}s" if st.started[i] else "")
        act = f"action: {ACTION_NAMES[int(actions[i])]}" if actions is not None and i < len(actions) else ""
        text.set_text(f"{clock}   boat {b.speed[i] / KNOT:4.1f} kn\n{act}\n{status}" + (f"\n{caption}" if caption else ""))
        return hull, trail, text

    anim = FuncAnimation(fig, frame, frames=n, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
