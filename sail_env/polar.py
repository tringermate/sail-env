"""J/70 polar: target boat speed as a function of true wind angle (TWA) and true wind speed (TWS).

Two stages:
1. `load_polar` (NumPy, runs once, outside jit) reads the ORC table and fills the gaps the VPP does
   not cover, producing a dense grid at TWA_STEP_DEG resolution:
     - 0 .. no_go:           speed 0 (head to wind, sails flapping)
     - no_go .. beat angle:  linear ramp from 0 up to the speed at the optimal beat angle
                             (beat speed = beat_vmg / cos(beat_angle)); this is "pinching"
     - beat .. 52 .. 150:    ORC VPP table, linear between rows
     - 150 .. 180:           linear down to DEAD_RUN_VMG_FACTOR x best run VMG (an assumption:
                             an asymmetric-spinnaker boat is slow dead downwind)
     - TWS axis gets an extra 0 kn column with speed 0, so light air (e.g. inside a wind shadow)
       scales speed linearly instead of clamping at the 6 kn row.
2. `polar_speed` (pure JAX) does bilinear interpolation on that grid. Works under jit and vmap.

Everything returned is in SI units (radians, m/s).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from sail_env.config import KNOT
from sail_env.geometry import wrap_angle

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "polar_j70_orc.csv"
# 0.25 deg keeps the bilinear error at the beat-angle peak (steep pinching ramp on one side) ~0.03 kn.
TWA_STEP_DEG = 0.25
DEAD_RUN_VMG_FACTOR = 0.9  # speed at TWA 180 = this x best run VMG (assumption, not ORC data)


@struct.dataclass
class Polar:
    twa_grid: jax.Array  # (n_twa,) radians, 0 .. pi
    tws_grid: jax.Array  # (n_tws,) m/s, ascending, starts at 0
    speed: jax.Array  # (n_twa, n_tws) target boat speed, m/s


def read_orc_csv(path: Path = DEFAULT_CSV) -> dict:
    """Parse the CSV into {'tws': [...], 'angles': [...], 'table': 2D list, 'beat_angle': [...], ...}."""
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
    header, body = lines[0].split(","), lines[1:]
    out = {"tws": [float(x) for x in header[1:]], "angles": [], "table": []}
    for line in body:
        key, *vals = line.split(",")
        vals = [float(v) for v in vals]
        if key.replace(".", "").isdigit():
            out["angles"].append(float(key))
            out["table"].append(vals)
        else:
            out[key] = vals
    return out


def load_polar(no_go: float, path: Path = DEFAULT_CSV) -> Polar:
    """Build the dense polar grid. `no_go` in radians (EnvParams.no_go)."""
    d = read_orc_csv(path)
    no_go_deg = np.degrees(no_go)
    twa_deg = np.linspace(0.0, 180.0, int(round(180.0 / TWA_STEP_DEG)) + 1)
    table = np.array(d["table"])  # (n_angles, n_tws), knots

    columns = []
    for k in range(len(d["tws"])):
        beat_angle = d["beat_angle"][k]
        assert beat_angle > no_go_deg, "no-go angle must be below the optimal beat angle"
        beat_speed = d["beat_vmg"][k] / np.cos(np.radians(beat_angle))
        dead_run_speed = DEAD_RUN_VMG_FACTOR * d["run_vmg"][k]
        # Knots of a piecewise-linear curve speed(TWA) for this wind speed.
        xs = [0.0, no_go_deg, beat_angle, *d["angles"], 180.0]
        ys = [0.0, 0.0, beat_speed, *table[:, k], dead_run_speed]
        columns.append(np.interp(twa_deg, xs, ys))

    speed_kn = np.stack(columns, axis=1)  # (n_twa, n_tws)
    speed_kn = np.concatenate([np.zeros((len(twa_deg), 1)), speed_kn], axis=1)  # add TWS = 0 column
    tws_kn = np.array([0.0, *d["tws"]])
    return Polar(
        twa_grid=jnp.asarray(np.radians(twa_deg), dtype=jnp.float32),
        tws_grid=jnp.asarray(tws_kn * KNOT, dtype=jnp.float32),
        speed=jnp.asarray(speed_kn * KNOT, dtype=jnp.float32),
    )


def _locate(grid: jax.Array, x: jax.Array):
    """Index i and fraction f such that x = (1 - f) * grid[i] + f * grid[i + 1], clamped to the grid ends."""
    i = jnp.clip(jnp.searchsorted(grid, x, side="right") - 1, 0, grid.shape[0] - 2)
    f = jnp.clip((x - grid[i]) / (grid[i + 1] - grid[i]), 0.0, 1.0)
    return i, f


def beat_vmg(polar: Polar, tws: jax.Array) -> jax.Array:
    """Best upwind VMG (m/s) at wind speed tws: max over TWA of speed(TWA) * cos(TWA)."""
    vmg_per_tws = jnp.max(polar.speed * jnp.cos(polar.twa_grid)[:, None], axis=0)  # (n_tws,)
    return jnp.interp(tws, polar.tws_grid, vmg_per_tws)


def polar_speed(polar: Polar, twa: jax.Array, tws: jax.Array) -> jax.Array:
    """Target boat speed (m/s) for TWA (radians, any sign or wrap) and TWS (m/s).

    The polar is symmetric between tacks, so we look up |TWA| wrapped into [0, pi].
    TWS above the table's top row is clamped to it.
    """
    twa_abs = jnp.abs(wrap_angle(twa))
    i, fa = _locate(polar.twa_grid, twa_abs)
    j, fs = _locate(polar.tws_grid, tws)
    s = polar.speed
    # Bilinear: interpolate along TWS at the two bracketing TWA rows, then along TWA.
    lo = (1 - fs) * s[i, j] + fs * s[i, j + 1]
    hi = (1 - fs) * s[i + 1, j] + fs * s[i + 1, j + 1]
    return (1 - fa) * lo + fa * hi
