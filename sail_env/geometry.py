"""Angle conventions shared by the whole simulator.

- Position: x east, y north (metres).
- Heading: compass, 0 = north, clockwise positive (radians). Velocity = speed * (sin h, cos h).
- Wind direction: the direction the wind blows FROM, same compass convention.
- Signed true wind angle: TWA = wrap(wind_dir - heading). TWA > 0 is starboard tack (wind over the
  starboard side), TWA < 0 is port tack. |TWA| = 0 is head to wind, pi is dead downwind.
"""

import jax.numpy as jnp


def wrap_angle(a):
    """Wrap an angle into (-pi, pi]."""
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


def heading_vector(h):
    """Unit vector (east, north) for compass heading h."""
    return jnp.sin(h), jnp.cos(h)


def signed_twa(heading, wind_dir):
    """Signed true wind angle; > 0 starboard tack, < 0 port tack."""
    return wrap_angle(wind_dir - heading)
