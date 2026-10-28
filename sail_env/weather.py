"""Weather on top of the base wind: shifts and pressure that move down the course, current and waves.

Every part has its own static switch in EnvParams and is off by default; with all of them off the
simulator is exactly the one of sail_env 0.1 (the base wind of wind.py, uniform in space).

    weather_shifts     persistent drift + random shifts + a geographic bend across the course
    weather_pressure   puffs and lulls (with their own small shifts) + a pressure gradient across the course
    water_current      a uniform current: boats sail through the water, the water moves over the ground
    waves              slower into the waves, a little faster with them, dearer tacks and gybes

Moving field. Shifts and puffs are carried downwind at U = weather_advection x the base wind speed,
so what is happening upwind reaches a boat a/U seconds later. In the mean-wind frame (a = distance
upwind of the line centre, c = distance to the right looking upwind) every travelling feature is a
function of the "arrival time" tau = t + a / U, or equivalently of the air coordinate xi = U tau:

    direction(t, x, y) = base wind direction(tau)                      (wind.py's oscillation, in tau)
                       + drift x tau                                   (persistent shift, rad/s)
                       + sum_k A_k sin(2 pi tau / P_k + phi_k)         (random shifts)
                       + bend x c / 100 m                              (geographic: steady across the course)
                       + sum_p dir_p g_p                               (shift inside each puff)
    speed(t, x, y)     = TWS (1 + grad x c / 100 m) (1 + sum_p amp_p g_p),   at least tws_floor_frac x TWS
    g_p                = exp(-(d_a / 1.5 r_p)^2 / 2 - (d_c / r_p)^2 / 2)    (a puff, longer along the wind)

with d_a, d_c the distances from puff p's centre, taken in a periodic cell (puff_cell_length along the
wind x puff_cell_width across) so the supply of puffs never runs out.

Current. The water moves at C (m/s, over the ground). The sails feel the wind over the water,
W - C (W = air velocity), so a current against the wind makes more wind; the boat moves at its polar
speed through the water plus C over the ground. Marks and the line stay fixed to the ground.

Waves. Height h (m) from direction wave_dir (from the wind, give or take wave_dir_spread). The polar
target speed is multiplied by 1 - wave_upwind_loss h max(cos b, 0) + wave_downwind_gain h max(-cos b, 0),
b = angle between the heading and the direction the waves come from; tacks and gybes lose an extra
wave_manoeuvre_loss x h of speed. A polar model can only fake waves; this is the simplest version.
"""

import jax
import jax.numpy as jnp
from flax import struct

from sail_env.config import EnvParams
from sail_env.geometry import wrap_angle
from sail_env.wind import Wind, wind_at

# What a boat can observe about the weather (observe_weather), appended to the observation:
HISTORY_LAGS = (10, 20, 30, 40, 50, 60)  # s: what the boat's instruments read this long ago (its wind log)
LOG_SECONDS = max(HISTORY_LAGS)  # each boat logs the wind it sailed in, once a second, this far back
PROBES = ((100.0, -150.0), (100.0, 0.0), (100.0, 150.0), (250.0, -150.0), (250.0, 0.0), (250.0, 150.0))  # (upwind, right) m
WEATHER_FEATURES = 2 * len(HISTORY_LAGS) + len(PROBES) + 2 + 2  # history, look upwind, current, waves = 22
DIR_SCALE = 0.25  # rad (~14 deg): direction offsets are divided by this in the observation


@struct.dataclass
class Weather:
    drift: jax.Array  # rad/s, persistent shift (> 0 veers, i.e. clockwise)
    shift_amp: jax.Array  # (K,) rad, random shift modes
    shift_period: jax.Array  # (K,) s
    shift_phase: jax.Array  # (K,) rad
    bend: jax.Array  # rad per 100 m to the right (looking upwind)
    grad: jax.Array  # fraction of TWS per 100 m to the right
    puff_xi: jax.Array  # (P,) m, puff centre along the wind in the periodic cell (air coordinate)
    puff_c: jax.Array  # (P,) m, puff centre across the wind
    puff_amp: jax.Array  # (P,) fraction of TWS (> 0 puff, < 0 lull)
    puff_dir: jax.Array  # (P,) rad, shift at the puff's centre
    puff_r: jax.Array  # (P,) m, puff radius (one standard deviation across the wind)
    current_x: jax.Array  # m/s, water velocity east
    current_y: jax.Array  # m/s, water velocity north
    wave_h: jax.Array  # m
    wave_dir: jax.Array  # rad, direction the waves come FROM


def any_weather(params: EnvParams) -> bool:
    """True if any weather feature is on (static: decided when the environment is compiled)."""
    return params.weather_shifts or params.weather_pressure or params.water_current or params.waves


def sample_weather(key: jax.Array, base_dir, params: EnvParams, tws=None) -> Weather:
    """One episode's weather. Always sampled (fixed shapes); a switched-off part is simply not used.
    tws (the race's wind speed) is needed only for params.current_max_frac."""
    ks = jax.random.split(key, 16)
    k_modes, n_puffs = params.n_shift_modes, params.n_puffs
    u = jax.random.uniform
    # Random shifts: K sinusoids with log-uniform periods; equal amplitudes give the requested std
    # (each sinusoid has variance A^2 / 2).
    log_p = u(ks[0], (k_modes,), minval=jnp.log(params.random_shift_period_min),
              maxval=jnp.log(params.random_shift_period_max))
    amp = params.random_shift_std * jnp.sqrt(2.0 / k_modes) * jnp.ones(k_modes)
    cur_max = params.current_max
    if tws is not None:  # race committee: the current at most current_max_frac x the wind speed
        cur_max = jnp.where(params.current_max_frac > 0.0, jnp.minimum(cur_max, params.current_max_frac * tws), cur_max)
    current_speed = u(ks[11], maxval=cur_max)
    current_to = u(ks[12], maxval=2 * jnp.pi)  # direction the water flows TO, compass
    return Weather(
        drift=u(ks[1], minval=-params.drift_max, maxval=params.drift_max),
        shift_amp=amp,
        shift_period=jnp.exp(log_p),
        shift_phase=u(ks[2], (k_modes,), maxval=2 * jnp.pi),
        bend=u(ks[3], minval=-params.bend_max, maxval=params.bend_max),
        grad=u(ks[4], minval=-params.pressure_grad_max, maxval=params.pressure_grad_max),
        puff_xi=u(ks[5], (n_puffs,), maxval=params.puff_cell_length),
        puff_c=u(ks[6], (n_puffs,), minval=-0.5 * params.puff_cell_width, maxval=0.5 * params.puff_cell_width),
        puff_amp=u(ks[7], (n_puffs,), minval=-params.lull_amp_max, maxval=params.puff_amp_max),
        puff_dir=u(ks[8], (n_puffs,), minval=-params.puff_shift_max, maxval=params.puff_shift_max),
        puff_r=u(ks[9], (n_puffs,), minval=params.puff_radius_min, maxval=params.puff_radius_max),
        current_x=current_speed * jnp.sin(current_to),
        current_y=current_speed * jnp.cos(current_to),
        wave_h=u(ks[13], maxval=params.wave_height_max),
        wave_dir=base_dir + u(ks[14], minval=-params.wave_dir_spread, maxval=params.wave_dir_spread),
    )


def wind_frame(base_dir, x, y):
    """(a, c): distance upwind of the origin (the line centre) and to the right, in the mean-wind frame."""
    return x * jnp.sin(base_dir) + y * jnp.cos(base_dir), x * jnp.cos(base_dir) - y * jnp.sin(base_dir)


def _wrap(d, period):
    """Signed distance in a periodic cell: d mapped to [-period / 2, period / 2)."""
    return d - period * jnp.floor(d / period + 0.5)


def puffs(weather: Weather, xi, c, params: EnvParams):
    """(speed factor - 1, direction offset) from all puffs and lulls at air coordinate xi, cross position c."""
    d_a = _wrap(xi[..., None] - weather.puff_xi, params.puff_cell_length)
    d_c = _wrap(c[..., None] - weather.puff_c, params.puff_cell_width)
    g = jnp.exp(-0.5 * (d_a / (1.5 * weather.puff_r)) ** 2 - 0.5 * (d_c / weather.puff_r) ** 2)
    return (weather.puff_amp * g).sum(-1), (weather.puff_dir * g).sum(-1)


def wind_at_point(wind: Wind, weather: Weather, t, x, y, params: EnvParams):
    """(direction, speed) of the wind over the ground at clock t and position (x, y)."""
    if not (params.weather_shifts or params.weather_pressure):  # static: the uniform base wind, untouched
        return wind_at(wind, t)
    x, y, t = jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.float32), jnp.asarray(t, jnp.float32)
    a, c = wind_frame(wind.base_dir, x, y)
    u = jnp.maximum(params.weather_advection * wind.tws, 0.5)  # m/s, how fast the field moves downwind
    tau = t + a / u  # when what is here now reaches the line centre's distance upwind
    direction = wind_at(wind, tau if params.weather_shifts else t)[0]
    speed = wind.tws * jnp.ones_like(a)
    if params.weather_shifts:  # static
        modes = (weather.shift_amp * jnp.sin(2 * jnp.pi * tau[..., None] / weather.shift_period
                                             + weather.shift_phase)).sum(-1)
        direction = direction + weather.drift * tau + modes + weather.bend * c / 100.0
    if params.weather_pressure:  # static
        p_speed, p_dir = puffs(weather, u * tau, c, params)
        direction = direction + p_dir
        speed = speed * (1.0 + weather.grad * c / 100.0) * (1.0 + p_speed)
        speed = jnp.maximum(speed, params.tws_floor_frac * wind.tws)
    return direction, speed


def water_wind(wind: Wind, weather: Weather, t, x, y, params: EnvParams):
    """(direction, speed) of the wind over the water at (x, y): what the sails and instruments feel."""
    d, s = wind_at_point(wind, weather, t, x, y, params)
    if not params.water_current:  # static
        return d, s
    # Air velocity (where the wind blows TO) minus water velocity.
    vx = -s * jnp.sin(d) - weather.current_x
    vy = -s * jnp.cos(d) - weather.current_y
    return jnp.arctan2(-vx, -vy), jnp.hypot(vx, vy)


def reference_wind(wind: Wind, weather: Weather, t, params: EnvParams):
    """Wind over the ground at the line centre (the origin): the 'wind of the race', for display and ranking."""
    if not any_weather(params):  # static
        return wind_at(wind, t)
    return wind_at_point(wind, weather, t, jnp.float32(0.0), jnp.float32(0.0), params)


def current_velocity(weather: Weather, params: EnvParams):
    """(cx, cy) water velocity in m/s, or None without current (static)."""
    return (weather.current_x, weather.current_y) if params.water_current else None


def wave_state(weather: Weather, params: EnvParams):
    """(height, direction from) of the waves, or None without waves (static)."""
    return (weather.wave_h, weather.wave_dir) if params.waves else None


def wave_speed_factor(heading, waves, params: EnvParams):
    """Multiplier on the polar target speed: < 1 heading into the waves, > 1 running with them."""
    h, from_dir = waves
    cos_b = jnp.cos(wrap_angle(heading - from_dir))  # 1 = bow into the waves
    return 1.0 - params.wave_upwind_loss * h * jnp.maximum(cos_b, 0.0) + params.wave_downwind_gain * h * jnp.maximum(-cos_b, 0.0)


def log_size(params: EnvParams) -> int:
    """Length of the wind log in EnvState (static): LOG_SECONDS with observe_weather, else 0 (no cost)."""
    return LOG_SECONDS if params.observe_weather else 0


def initial_log(t, x, y, params: EnvParams):
    """Empty wind log at reset: ((log_size, 2) zeros, (3,) = where and when the episode began: x, y, t).

    The episode starts in the middle of the boat's day: it was already sailing in the start area. Where
    exactly is unknown, so readings from before the episode are taken as if the boat had been at its
    starting position (computed in observe, only for the lags it reads). That is the one approximation;
    every reading from the episode itself is what the boat sailed in, where it was.
    """
    n = log_size(params)
    origin = jnp.stack([x, y, t]).astype(jnp.float32) if n else jnp.zeros((0,), jnp.float32)
    return jnp.zeros((n, 2), jnp.float32), origin


def push_log(log, wind_dir, tws):
    """Add this second's reading (the wind the boat sailed in) at the front; the oldest drops off."""
    if log.shape[0] == 0:  # static
        return log
    return jnp.concatenate([jnp.stack([wind_dir, tws])[None].astype(log.dtype), log[:-1]], axis=0)


def observe(wind: Wind, weather: Weather, wind_log, log_origin, t, x, y, heading, params: EnvParams):
    """WEATHER_FEATURES numbers for one boat (observe_weather), roughly in [-1, 1]:

    - history: for each lag in HISTORY_LAGS, what the boat's instruments read that long ago, from its own
      wind log (the wind over the water where the boat was then): direction offset from the mean wind
      (the course axis) / DIR_SCALE, speed / 8 m/s                     (a sailor watching the compass)
    - look upwind: wind speed / 8 m/s over the ground at PROBES, (upwind, right) of the boat in the
      mean-wind frame                                                  (dark patches on the water)
    - current in the boat's frame (forward, starboard), m/s / 0.5     (from the anchored committee boat)
    - waves: height / 1 m, height x cos(angle between bow and waves)
    """
    hist = []
    logged = jnp.round((t - log_origin[2]) / params.dt)  # readings taken since the episode began
    for lag in HISTORY_LAGS:
        d_before, s_before = water_wind(wind, weather, t - lag, log_origin[0], log_origin[1], params)
        d = jnp.where(lag <= logged, wind_log[lag - 1, 0], d_before)
        s = jnp.where(lag <= logged, wind_log[lag - 1, 1], s_before)
        hist += [wrap_angle(d - wind.base_dir) / DIR_SCALE, s / 8.0]
    ux, uy = jnp.sin(wind.base_dir), jnp.cos(wind.base_dir)  # upwind
    rx, ry = uy, -ux  # right, looking upwind
    probes = []
    for up, right in PROBES:
        _, s = wind_at_point(wind, weather, t, x + up * ux + right * rx, y + up * uy + right * ry, params)
        probes.append(s / 8.0)
    fx, fy = jnp.sin(heading), jnp.cos(heading)
    cx, cy = (weather.current_x, weather.current_y) if params.water_current else (0.0, 0.0)
    cur = [(cx * fx + cy * fy) / 0.5, (cx * fy - cy * fx) / 0.5]
    h = weather.wave_h if params.waves else jnp.float32(0.0)
    waves = [h, h * jnp.cos(wrap_angle(heading - weather.wave_dir))]
    return [*hist, *probes, *cur, *waves]
