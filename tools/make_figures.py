"""Static figures for the README and docs, generated from the simulator itself.

    python tools/make_figures.py          # docs/figures/polar.png, course.png, weather.png

polar.png    the J/70 polar used by the physics (ORC VPP median, with the no-go ramp and dead-run tail)
course.png   the race course of the "race" preset: start line, windward mark, leeward gate, zones, legs
weather.png  one episode of the "race-weather" preset: pressure and shifts moving down the course
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import sail_env as se  # noqa: E402
from sail_env.agents.digital_sailor import beat_angle  # noqa: E402
from sail_env.agents.race_sailor import run_angle  # noqa: E402
from sail_env.config import KNOT  # noqa: E402
from sail_env.polar import polar_speed  # noqa: E402
from sail_env.race_env import course_geometry  # noqa: E402
from sail_env.rules import make_line  # noqa: E402
from sail_env.weather import wind_at_point  # noqa: E402

OUT = Path("docs/figures")
INK, MUTED, ACCENT, YELLOW, ORANGE = "#0f2329", "#587077", "#df4f27", "#f0bd22", "#ec7a1e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 200, "savefig.bbox": "tight"})


def polar_figure(env):
    polar = env.polar
    twa = np.radians(np.linspace(0, 180, 361))
    fig = plt.figure(figsize=(5.2, 5.0))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, 6))
    for tws_kn, c in zip([6, 8, 10, 12, 16, 20], colors):
        tws = tws_kn * KNOT
        v = np.array([float(polar_speed(polar, a, tws)) for a in twa]) / KNOT
        ax.plot(twa, v, color=c, lw=1.6, label=f"{tws_kn} kn")
        for ang in (float(beat_angle(polar, tws)), float(run_angle(polar, tws))):
            ax.plot([ang], [float(polar_speed(polar, ang, tws)) / KNOT], "o", color=c, ms=3.5)
    ax.axvspan(0, np.radians(30), color=MUTED, alpha=0.12, lw=0)
    ax.text(np.radians(15), 7.5, "no-go\nzone", color=MUTED, ha="center", fontsize=8)
    ax.set_rlabel_position(170)
    ax.set_xticks(np.radians(np.arange(0, 181, 30)))
    ax.set_ylim(0, 11.5)
    ax.set_ylabel("boat speed (kn)", labelpad=24)
    ax.legend(title="true wind", loc="lower left", bbox_to_anchor=(0.86, 0.0), frameon=False, fontsize=8)
    ax.set_title("J/70 polar: boat speed vs true wind angle\n(dots: best upwind and downwind VMG angle)", fontsize=10, color=INK, pad=12)
    fig.savefig(OUT / "polar.png")
    plt.close(fig)


def course_figure(params, polar):
    line = make_line(0.0, 0.0, 0.0, 0.0, params.line_length)
    k = course_geometry(line, 0.0, params)
    w, g1, g2, c = (np.asarray(k[n], float) for n in ("w", "g1", "g2", "c"))
    zone = params.zone_bl * params.boat_length
    fig, ax = plt.subplots(figsize=(4.6, 6.4))
    ax.set_aspect("equal")
    ax.plot([line.pin_x, line.cb_x], [line.pin_y, line.cb_y], "--", color=INK, lw=1)
    ax.add_patch(plt.Circle((line.pin_x, line.pin_y), 3.0, color=ORANGE))
    ax.add_patch(plt.Circle((line.cb_x, line.cb_y), 5.0, color="#57666b"))
    ax.annotate("start / finish line", (line.cb_x, line.cb_y), xytext=(10, -14), textcoords="offset points", fontsize=8)
    ax.annotate("committee boat", (line.cb_x, line.cb_y), xytext=(10, 2), textcoords="offset points", fontsize=8, color=MUTED)
    ax.annotate("pin", (line.pin_x, line.pin_y), xytext=(-22, -12), textcoords="offset points", fontsize=8, color=MUTED)
    for m, col in ((w, YELLOW), (g1, ORANGE), (g2, ORANGE)):
        ax.add_patch(plt.Circle(tuple(m), 3.0, color=col, zorder=3))
        ax.add_patch(plt.Circle(tuple(m), zone, fill=False, ls=":", color=MUTED, lw=0.8))
    ax.plot([g1[0], g2[0]], [g1[1], g2[1]], ":", color=ORANGE, lw=1)
    ax.annotate("windward mark\n(leave to port)", tuple(w), xytext=(12, 0), textcoords="offset points", fontsize=8, va="center")
    ax.annotate("leeward gate", tuple(g2), xytext=(10, -4), textcoords="offset points", fontsize=8)
    ax.annotate(f"zone: {params.zone_bl:.0f} lengths", (w[0] - zone * 0.7, w[1] - zone * 0.7), xytext=(-70, -26),
                textcoords="offset points", fontsize=7, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    # laylines into the windward mark at the best beat angle (10 kn)
    b = float(beat_angle(polar, 10 * KNOT))
    for s in (-1, 1):
        L = 230.0
        ax.plot([w[0], w[0] + s * L * np.sin(b)], [w[1], w[1] - L * np.cos(b)], color=MUTED, lw=0.7, alpha=0.7)
    ax.text(w[0] + 150 * np.sin(b) + 6, w[1] - 150 * np.cos(b), "layline", fontsize=7, color=MUTED, rotation=-np.degrees(b) + 90 - 90)
    # legs: beats up the middle, run 1 through the gate, run 2 straight to the finish line
    arrow = dict(arrowstyle="-|>", color=ACCENT, lw=1.4, shrinkA=6, shrinkB=10)
    ax.annotate("", xy=(w[0] - 20, w[1] - 6), xytext=(c[0] - 20, c[1] + 6), arrowprops=arrow)
    ax.annotate("", xy=(c[0] + 2, c[1] + params.gate_offset + 4), xytext=(w[0] + 2, w[1] - 6), arrowprops=arrow)
    ax.annotate("", xy=(c[0] + 18, c[1] + 4), xytext=(w[0] + 10, w[1] - 6),
                arrowprops=dict(arrow, ls="--", connectionstyle="arc3,rad=-0.22"))
    ax.text(c[0] - 26, 230, "legs 1, 3\nbeat", fontsize=8, color=ACCENT, ha="right")
    ax.text(c[0] + 7, 150, "leg 2\nrun to\nthe gate", fontsize=7.5, color=ACCENT, ha="left")
    ax.text(c[0] + 66, 235, "leg 4\nrun to\nthe finish", fontsize=7.5, color=ACCENT, ha="left")
    ax.annotate("", xy=(-95, 360), xytext=(-95, 410), arrowprops=dict(arrowstyle="-|>", color="#2c69ad", lw=2))
    ax.text(-95, 418, "wind", ha="center", fontsize=8, color="#2c69ad")
    ax.set_xlim(-140, 140)
    ax.set_ylim(-40, 450)
    ax.set_xlabel("m")
    ax.set_ylabel("m upwind of the line")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"Race course: beat {params.course_beat_length:.0f} m, 2 laps, gate {params.gate_width_bl:.0f} lengths",
                 fontsize=10, color=INK)
    fig.savefig(OUT / "course.png")
    plt.close(fig)


def weather_figure(seed=11):
    """Four snapshots of the wind over the course 40 s apart, and the shift at the line and the mark."""
    env, params = se.make("race-weather")
    _, fs = env.reset(jax.random.PRNGKey(seed), params)
    wind = jax.tree.map(lambda x: x[0], fs.boats.wind)
    weather = jax.tree.map(lambda x: x[0], fs.boats.weather)
    line = jax.tree.map(lambda x: x[0], fs.boats.line)
    k = course_geometry(line, wind.base_dir, params)
    w, g1, g2 = (np.asarray(k[n], float) for n in ("w", "g1", "g2"))
    field = jax.jit(lambda t, x, y: wind_at_point(wind, weather, t, x, y, params))
    gx, gy = np.meshgrid(np.linspace(-250, 250, 120), np.linspace(-60, 460, 125))
    qx, qy = np.meshgrid(np.linspace(-200, 200, 6), np.linspace(-10, 420, 7))
    tws = float(wind.tws)
    u = params.weather_advection * tws
    fig = plt.figure(figsize=(10, 5.9))
    grid = fig.add_gridspec(2, 4, height_ratios=[2.3, 1.0], hspace=0.3, wspace=0.06)
    maps = []
    for i, t in enumerate([0.0, 40.0, 80.0, 120.0]):
        ax = fig.add_subplot(grid[0, i])
        maps.append(ax)
        _, s = field(t, jnp.asarray(gx, jnp.float32), jnp.asarray(gy, jnp.float32))
        im = ax.imshow(np.asarray(s) / tws, extent=(-250, 250, -60, 460), origin="lower", cmap="Blues", vmin=0.7,
                       vmax=1.3, interpolation="bilinear")
        d, _ = field(t, jnp.asarray(qx, jnp.float32), jnp.asarray(qy, jnp.float32))
        d = np.asarray(d)
        ax.quiver(qx, qy, -np.sin(d), -np.cos(d), color=INK, alpha=0.7, scale=14, width=0.012)
        ax.plot([line.pin_x, line.cb_x], [line.pin_y, line.cb_y], "-", color=INK, lw=1.2)
        for m, col in ((w, YELLOW), (g1, ORANGE), (g2, ORANGE)):
            ax.add_patch(plt.Circle(tuple(m), 7.0, color=col, ec=INK, lw=0.5, zorder=3))
        ax.set_title(f"T+{t:.0f} s", fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([] if i else [0, 200, 400])
        if i == 0:
            ax.set_ylabel("m upwind of the line")
    cb = fig.colorbar(im, ax=maps, fraction=0.02, pad=0.01)
    cb.set_label("wind speed / mean (darker = more wind)")
    ax = fig.add_subplot(grid[1, :])
    t = np.linspace(-90, 600, 700).astype(np.float32)
    for (x, y), col, lab in [((0.0, 0.0), ACCENT, "line centre"), (tuple(w), "#2c69ad", "windward mark")]:
        d, _ = field(jnp.asarray(t), jnp.full_like(jnp.asarray(t), x), jnp.full_like(jnp.asarray(t), y))
        ax.plot(t, np.degrees(np.asarray(d - wind.base_dir)), color=col, lw=1.4, label=lab)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("seconds after the gun")
    ax.set_ylabel("shift (deg, + = veer)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"The same shifts reach the windward mark {400 / u:.0f} s before the line "
                 f"(they travel downwind at {u / KNOT:.1f} kn)", fontsize=9, color=INK)
    ax.set_xlim(t[0], t[-1])
    fig.suptitle(f"race-weather, one episode: {tws / KNOT:.1f} kn mean wind, pressure and shifts moving down the course",
                 fontsize=10, color=INK)
    fig.savefig(OUT / "weather.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env, params = se.make("race")
    polar_figure(env)
    course_figure(params, env.polar)
    weather_figure()
    print(f"wrote {OUT}/polar.png, {OUT}/course.png, {OUT}/weather.png")


if __name__ == "__main__":
    main()
