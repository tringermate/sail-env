"""sail_env: a fast JAX simulator of J/70 sailing starts (solo and fleet) for reinforcement learning.

    import sail_env as se
    env, params = se.make("fleet")              # or "solo"
    obs, state = env.reset(key, params)
    obs, state, reward, done, info = env.step(key, state, actions, params)

See README.md for the environment, the agent interface, playing, evaluating and training.
"""

from sail_env.config import KNOT, EnvParams, deg, kn
from sail_env.fleet_env import FleetStartEnv
from sail_env.presets import PRESETS, make
from sail_env.start_env import SoloStartEnv

__version__ = "0.2.0"
__all__ = ["EnvParams", "FleetStartEnv", "SoloStartEnv", "make", "PRESETS", "KNOT", "kn", "deg", "__version__"]
