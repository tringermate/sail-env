"""Start line, line crossings and OCS (individual recall). Racing rules between boats come in Phase 4.

Line geometry: centred at (cx, cy), perpendicular to the base wind, committee boat at the starboard
end (the right-hand end looking upwind). Bias rotates it: bias > 0 puts the committee end upwind
(the starboard end is favoured).

    d   = unit vector pin -> committee boat
    n   = d rotated 90 deg anticlockwise = points to the course (upwind) side
    signed distance of a point p:  (p - pin) . n     (> 0 on the course side)

A start is a crossing of the segment pin -> committee by the bow's path, from the pre-start side
(<= 0) to the course side (> 0), after the gun.
"""

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Line:
    pin_x: jax.Array
    pin_y: jax.Array
    cb_x: jax.Array  # committee boat (starboard end)
    cb_y: jax.Array


@struct.dataclass
class StartStatus:
    started: jax.Array  # bool: crossed legally after the gun
    ocs: jax.Array  # bool: was over at the gun and has not yet dipped back
    start_time: jax.Array  # clock time of the legal crossing (inf until started)


def make_line(cx, cy, wind_dir, bias, length) -> Line:
    upwind = jnp.sin(wind_dir), jnp.cos(wind_dir)
    right = jnp.cos(wind_dir), -jnp.sin(wind_dir)  # 90 deg clockwise from upwind
    dx = right[0] * jnp.cos(bias) + upwind[0] * jnp.sin(bias)
    dy = right[1] * jnp.cos(bias) + upwind[1] * jnp.sin(bias)
    h = 0.5 * length
    return Line(pin_x=cx - h * dx, pin_y=cy - h * dy, cb_x=cx + h * dx, cb_y=cy + h * dy)


def line_frame(line: Line):
    """(dx, dy, nx, ny, length): unit direction pin -> committee, course-side normal, line length."""
    vx, vy = line.cb_x - line.pin_x, line.cb_y - line.pin_y
    length = jnp.hypot(vx, vy)
    dx, dy = vx / length, vy / length
    return dx, dy, -dy, dx, length


def signed_distance(line: Line, x, y):
    """Distance from the line (infinite extension), > 0 on the course side."""
    _, _, nx, ny, _ = line_frame(line)
    return (x - line.pin_x) * nx + (y - line.pin_y) * ny


def crossing(line: Line, x0, y0, x1, y1):
    """Did the path (x0,y0) -> (x1,y1) cross the line segment?

    Returns (crossed_up, crossed_down, frac): up = pre-start -> course side, down = the reverse,
    frac = fraction along the path where it meets the line (valid only when crossed).
    """
    dx, dy, _, _, length = line_frame(line)
    d0, d1 = signed_distance(line, x0, y0), signed_distance(line, x1, y1)
    frac = jnp.clip(d0 / jnp.where(d0 == d1, 1.0, d0 - d1), 0.0, 1.0)
    # Where along the line (0 = pin, length = committee) the path meets it.
    px, py = x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)
    along = (px - line.pin_x) * dx + (py - line.pin_y) * dy
    on_segment = (along >= 0.0) & (along <= length)
    up = (d0 <= 0.0) & (d1 > 0.0) & on_segment
    down = (d0 > 0.0) & (d1 <= 0.0) & on_segment
    return up, down, frac


def init_status(shape=()) -> StartStatus:
    return StartStatus(
        started=jnp.zeros(shape, bool), ocs=jnp.zeros(shape, bool), start_time=jnp.full(shape, jnp.inf)
    )


def update_status(status: StartStatus, line: Line, bow_path, t0, t1) -> StartStatus:
    """Advance the start state machine over one step whose clock goes t0 -> t1 (gun at 0).

    Order within a step: the OCS check happens at the gun (end of the step reaching t >= 0);
    an OCS boat clears when its bow is back on the pre-start side; only then can it start.
    """
    x0, y0, x1, y1 = bow_path
    up, _, frac = crossing(line, x0, y0, x1, y1)
    d1 = signed_distance(line, x1, y1)

    gun_this_step = (t0 < 0.0) & (t1 >= 0.0)
    after_gun = t0 >= 0.0

    ocs = status.ocs | (gun_this_step & (d1 > 0.0))
    ocs = ocs & ~(after_gun & (d1 <= 0.0))  # dipped back: recall cleared

    starts_now = after_gun & up & ~status.ocs & ~status.started
    return StartStatus(
        started=status.started | starts_now,
        ocs=ocs,
        start_time=jnp.where(starts_now, t0 + frac * (t1 - t0), status.start_time),
    )
