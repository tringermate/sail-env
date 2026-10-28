"""CartPole-v1 in JAX, with the same functional API as our sailing environments.

Used only as a sanity check for PPO: a correct implementation reaches a return of ~500 quickly.
Dynamics, thresholds and rewards follow the classic Gym / gymnax CartPole-v1 (Barto et al. 1983):
+1 reward per step, episode ends when the pole falls past 12 deg, the cart leaves +-2.4 m,
or after 500 steps.
"""

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class CartPoleParams:
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5  # half the pole length
    force_mag: float = 10.0
    tau: float = 0.02  # seconds per step
    theta_threshold: float = 12 * 2 * jnp.pi / 360
    x_threshold: float = 2.4
    max_steps: int = 500


@struct.dataclass
class CartPoleState:
    x: jax.Array
    x_dot: jax.Array
    theta: jax.Array
    theta_dot: jax.Array
    t: jax.Array


class CartPole:
    num_actions = 2
    obs_dim = 4

    def reset(self, key, params: CartPoleParams):
        x, x_dot, theta, theta_dot = jax.random.uniform(key, (4,), minval=-0.05, maxval=0.05)
        state = CartPoleState(x, x_dot, theta, theta_dot, jnp.int32(0))
        return self.get_obs(state), state

    def step_env(self, key, state: CartPoleState, action, params: CartPoleParams):
        p = params
        force = jnp.where(action == 1, p.force_mag, -p.force_mag)
        cos, sin = jnp.cos(state.theta), jnp.sin(state.theta)
        total_mass = p.masscart + p.masspole
        polemass_length = p.masspole * p.length
        temp = (force + polemass_length * state.theta_dot**2 * sin) / total_mass
        theta_acc = (p.gravity * sin - cos * temp) / (p.length * (4.0 / 3.0 - p.masspole * cos**2 / total_mass))
        x_acc = temp - polemass_length * theta_acc * cos / total_mass
        # Explicit Euler, same order as Gym.
        state = CartPoleState(
            x=state.x + p.tau * state.x_dot,
            x_dot=state.x_dot + p.tau * x_acc,
            theta=state.theta + p.tau * state.theta_dot,
            theta_dot=state.theta_dot + p.tau * theta_acc,
            t=state.t + 1,
        )
        done = (
            (jnp.abs(state.x) > p.x_threshold)
            | (jnp.abs(state.theta) > p.theta_threshold)
            | (state.t >= p.max_steps)
        )
        return self.get_obs(state), state, jnp.float32(1.0), done, {}

    def step(self, key, state, action, params):
        """step_env + auto-reset on done (gymnax convention)."""
        k_step, k_reset = jax.random.split(key)
        obs, state, reward, done, info = self.step_env(k_step, state, action, params)
        obs_re, state_re = self.reset(k_reset, params)
        state = jax.tree.map(lambda a, b: jnp.where(done, a, b), state_re, state)
        obs = jnp.where(done, obs_re, obs)
        return obs, state, reward, done, info

    @staticmethod
    def get_obs(state: CartPoleState):
        return jnp.stack([state.x, state.x_dot, state.theta, state.theta_dot])
