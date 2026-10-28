"""Boat kinematics: actions -> target heading, then sub-stepped turning, speed lag and motion.

One decision per `params.dt` (1 s), integrated in `params.n_substeps` sub-steps (0.25 s each).
All functions are elementwise, so they work unchanged on a single boat or on stacked boats
(leading agent axis) and under jit/vmap.

Actions (v1, discrete):
    0 HOLD     keep the current target heading (an unfinished turn carries on)
    1 UP5      target = heading + 5 deg towards the wind
    2 DOWN5    target = heading + 5 deg away from the wind
    3 UP15     same, 15 deg
    4 DOWN15   same, 15 deg
    5 TACK     tack or gybe: target = heading mirrored across the wind; speed loss; locks actions
    6 LUFF     hold course with sails eased: target speed = luff_factor x polar for this step
"""

import jax
import jax.numpy as jnp
from flax import struct

from sail_env.config import EnvParams, deg
from sail_env.geometry import heading_vector, signed_twa, wrap_angle
from sail_env.polar import Polar, polar_speed

HOLD, UP5, DOWN5, UP15, DOWN15, TACK, LUFF = range(7)
N_ACTIONS = 7

# Heading change per action, in "towards the wind" units (positive = head up).
_ACTION_TURN = jnp.array([0.0, deg(5), -deg(5), deg(15), -deg(15), 0.0, 0.0])

TURN_DONE_TOL = deg(0.5)  # a manoeuvre is finished once heading is this close to the target


@struct.dataclass
class BoatState:
    x: jax.Array  # m, east (centre of the boat)
    y: jax.Array  # m, north
    heading: jax.Array  # rad, compass
    speed: jax.Array  # m/s through the water
    target_heading: jax.Array  # rad, where the helm is steering to
    manoeuvring: jax.Array  # bool, tack/gybe in progress (new actions ignored)


def turn_rate(speed: jax.Array, params: EnvParams) -> jax.Array:
    """Max turn rate (rad/s): 0 below min_steer_speed, rising linearly to max_turn_rate at full_steer_speed."""
    steer = (speed - params.min_steer_speed) / (params.full_steer_speed - params.min_steer_speed)
    return params.max_turn_rate * jnp.clip(steer, 0.0, 1.0)


def relax_speed(speed, target, dt, params: EnvParams):
    """Exact first-order lag over dt: v <- v + (v* - v)(1 - exp(-dt / tau)), tau depends on direction."""
    tau = jnp.where(target > speed, params.tau_accel, params.tau_decel)
    return speed + (target - speed) * (1.0 - jnp.exp(-dt / tau))


def drag_force(speed, twa, tws, params: EnvParams):
    """Resistance along the boat's heading (N, > 0 slows it): water drag + windage.

    Windage uses the wind speed along the bow, u = TWS cos(TWA) + v: head to wind it is TWS + v and
    opposes the boat; downwind u < 0 and the windage pushes it forward.
    """
    u = tws * jnp.cos(twa) + speed
    return params.water_drag_k * speed * jnp.abs(speed) + 0.5 * params.rho_air * params.windage_cda * u * jnp.abs(u)


def update_speed(speed, target, twa, tws, dt, params: EnvParams):
    """Speed after dt.

    Speeding up (target > speed): first-order lag with tau_accel, as before.
    Slowing down: force balance m dv/dt = F_drive - drag(v), where F_drive = drag(target) is the drive
    that would hold the target speed (0 drive inside the no-go zone). Never below the target or 0.
    """
    if not params.drag_decel:  # static flag
        return relax_speed(speed, target, dt, params)
    accel = relax_speed(speed, target, dt, params)
    drive = jnp.where(target > 0.0, drag_force(target, twa, tws, params), 0.0)
    drive = jnp.maximum(drive, 0.0)
    decel = speed + dt * (drive - drag_force(speed, twa, tws, params)) / params.boat_mass
    decel = jnp.maximum(decel, jnp.maximum(target, 0.0))
    return jnp.where(target > speed, accel, decel)


def apply_action(boat: BoatState, action: jax.Array, wind_dir: jax.Array, params: EnvParams, extra_loss=None):
    """Turn a discrete action into a new target heading (and speed loss for a tack/gybe).

    Returns (boat, luffing). While a manoeuvre is in progress the action is ignored.
    extra_loss: additional fraction of speed lost on a tack or gybe (waves), or None.
    """
    twa = signed_twa(boat.heading, wind_dir)
    up = jnp.sign(twa)  # heading change that turns towards the wind: +1 on starboard, -1 on port
    up = jnp.where(up == 0, 1.0, up)  # exactly head to wind: pick a side

    # Steering actions: relative to where the bow points now.
    steer_target = wrap_angle(boat.heading + up * _ACTION_TURN[action])
    is_steer = (action >= UP5) & (action <= DOWN15)

    # Tack/gybe: mirror across the wind, new TWA = -TWA, i.e. heading = wind_dir + TWA.
    # Turning the short way to it goes through the wind when |TWA| < 90 deg (tack) and through
    # dead downwind when |TWA| > 90 deg (gybe), so the same rule covers both.
    is_tack = action == TACK
    tack_target = wrap_angle(wind_dir + twa)
    is_gybe = jnp.abs(twa) > jnp.pi / 2
    loss = jnp.where(is_gybe, params.gybe_speed_loss, params.tack_speed_loss)
    if extra_loss is not None:  # static
        loss = jnp.minimum(loss + extra_loss, 1.0)

    free = ~boat.manoeuvring
    target = jnp.where(free & is_steer, steer_target, boat.target_heading)
    target = jnp.where(free & is_tack, tack_target, target)
    speed = jnp.where(free & is_tack, boat.speed * (1.0 - loss), boat.speed)
    manoeuvring = boat.manoeuvring | (free & is_tack)
    luffing = free & (action == LUFF)
    return boat.replace(target_heading=target, speed=speed, manoeuvring=manoeuvring), luffing


def substep(boat: BoatState, polar: Polar, wind_dir, tws, luffing, dt, params: EnvParams,
            current=None, waves=None) -> BoatState:
    """Advance one physics sub-step: turn towards target, relax speed towards polar, move.

    Optional weather (weather.py): current = (cx, cy) water velocity, m/s; waves = (height, direction
    from). wind_dir and tws are then the wind over the water; speed is through the water.
    """
    # 1. Turn: the short way towards the target, limited by the speed-dependent turn rate.
    #    Bearing away (turning away from the wind) is always possible at irons_bear_away_rate, even
    #    when stopped: the crew backs the jib / pushes the boom out, so a boat can get out of irons.
    err = wrap_angle(boat.target_heading - boat.heading)
    up = jnp.sign(signed_twa(boat.heading, wind_dir))  # heading change that turns towards the wind
    bearing_away = (err * up) < 0
    rate = turn_rate(boat.speed, params)
    rate = jnp.where(bearing_away, jnp.maximum(rate, params.irons_bear_away_rate), rate)
    max_turn = rate * dt
    heading = wrap_angle(boat.heading + jnp.clip(err, -max_turn, max_turn))
    done = jnp.abs(wrap_angle(boat.target_heading - heading)) < TURN_DONE_TOL
    # A tack that runs out of speed head to wind is abandoned, so the helm is free to bear away.
    stalled = boat.speed < params.min_steer_speed
    manoeuvring = boat.manoeuvring & ~done & ~stalled

    # 2. Speed: polar target at the new TWA (0 inside the no-go zone), reduced when luffing.
    target = polar_speed(polar, signed_twa(heading, wind_dir), tws)
    if waves is not None:  # static
        from sail_env.weather import wave_speed_factor

        target = target * wave_speed_factor(heading, waves, params)
    target = jnp.where(luffing, params.luff_factor * target, target)
    speed = update_speed(boat.speed, target, signed_twa(heading, wind_dir), tws, dt, params)

    # 3. Move along the heading (through the water), plus the current (over the ground).
    ex, ny = heading_vector(heading)
    vx, vy = speed * ex, speed * ny
    if current is not None:  # static
        vx, vy = vx + current[0], vy + current[1]
    return boat.replace(
        x=boat.x + vx * dt,
        y=boat.y + vy * dt,
        heading=heading,
        speed=speed,
        manoeuvring=manoeuvring,
    )


def physics_step(boat: BoatState, action, polar: Polar, wind_dir, tws, params: EnvParams, current=None, waves=None):
    """One decision: apply the action, then integrate n_substeps sub-steps.

    Wind is held fixed over the decision. Returns (new_boat, bow_path) where bow_path is
    (x0, y0, x1, y1) of the bow over the step, for line-crossing detection. current and waves: see substep.
    """
    extra_loss = None if waves is None else params.wave_manoeuvre_loss * waves[0]
    boat, luffing = apply_action(boat, action, wind_dir, params, extra_loss)
    sub_dt = params.dt / params.n_substeps
    bx0, by0 = bow_position(boat, params)

    def body(b, _):
        return substep(b, polar, wind_dir, tws, luffing, sub_dt, params, current, waves), None

    boat, _ = jax.lax.scan(body, boat, None, length=params.n_substeps)
    bx1, by1 = bow_position(boat, params)
    return boat, (bx0, by0, bx1, by1)


def bow_position(boat: BoatState, params: EnvParams):
    """Bow = centre + half a boat length along the heading."""
    ex, ny = heading_vector(boat.heading)
    half = 0.5 * params.boat_length
    return boat.x + half * ex, boat.y + half * ny
