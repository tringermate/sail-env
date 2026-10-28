"""Simulator throughput: environment steps per second (jit + vmap over parallel episodes, random actions).

    python tools/benchmark_speed.py              # all presets, 1024 parallel episodes
    python tools/benchmark_speed.py --envs 4096
    python tools/benchmark_speed.py --weather    # also the -weather presets

Reports decisions per second (one decision = one simulated second for every boat in the episode) and
boat-steps per second (decisions x boats). Timing excludes compilation.
"""

import argparse
import platform
import time

import jax
import jax.numpy as jnp

import sail_env as se


def throughput(preset, n_envs, n_steps=200, repeats=3):
    env, p = se.make(preset)
    fleet = hasattr(env, "num_agents")
    n = env.num_agents if fleet else 1
    shape = (n,) if fleet else ()
    _, states = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(0), n_envs), p)

    @jax.jit
    def run(states, key):
        def body(s, k):
            k_a, k_s = jax.random.split(k)
            a = jax.random.randint(k_a, (n_envs, *shape), 0, env.num_actions)
            _, s, r, d, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.random.split(k_s, n_envs), s, a, p)
            return s, r.sum()
        return jax.lax.scan(body, states, jax.random.split(key, n_steps))

    jax.block_until_ready(run(states, jax.random.PRNGKey(1)))  # compile
    best = float("inf")
    for i in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(run(states, jax.random.PRNGKey(2 + i)))
        best = min(best, time.perf_counter() - t)
    steps = n_envs * n_steps / best
    return steps, steps * n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--weather", action="store_true", help="also time the -weather presets")
    args = ap.parse_args()
    print(f"jax {jax.__version__} on {jax.devices()[0].platform} ({platform.processor() or platform.machine()}), "
          f"{args.envs} parallel episodes")
    presets = ["solo", "fleet", "race-solo", "race"]
    for preset in presets + ([f"{p}-weather" for p in presets] if args.weather else []):
        steps, boat_steps = throughput(preset, args.envs)
        print(f"  {preset:18s} {steps:12,.0f} decisions/s   {boat_steps:12,.0f} boat-steps/s")


if __name__ == "__main__":
    main()
