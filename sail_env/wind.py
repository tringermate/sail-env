"""True wind: a base direction and speed per episode, with an optional oscillating shift.

    direction(t) = base_dir + shift_amp * sin(2 pi t / shift_period + shift_phase)

Wind is uniform in space here; wind shadows (Phase 4) multiply TWS per boat on top of this.
"""

import jax
import jax.numpy as jnp
from flax import struct

from sail_env.config import EnvParams


@struct.dataclass
class Wind:
    base_dir: jax.Array  # radians, direction the wind blows from
    tws: jax.Array  # m/s
    shift_amp: jax.Array  # radians
    shift_period: jax.Array  # seconds
    shift_phase: jax.Array  # radians


def sample_wind(key: jax.Array, params: EnvParams) -> Wind:
    """Uniform TWS in [tws_min, tws_max], shift amplitude in [0, shift_amp_max], random phase."""
    k_tws, k_amp, k_phase = jax.random.split(key, 3)
    return Wind(
        base_dir=jnp.float32(params.wind_dir),
        tws=jax.random.uniform(k_tws, minval=params.tws_min, maxval=params.tws_max),
        shift_amp=jax.random.uniform(k_amp, maxval=params.shift_amp_max),
        shift_period=jnp.float32(params.shift_period),
        shift_phase=jax.random.uniform(k_phase, maxval=2 * jnp.pi),
    )


def wind_at(wind: Wind, t: jax.Array):
    """(direction, speed) of the true wind at clock time t (seconds)."""
    direction = wind.base_dir + wind.shift_amp * jnp.sin(2 * jnp.pi * t / wind.shift_period + wind.shift_phase)
    return direction, wind.tws
