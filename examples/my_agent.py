"""A template for your own agent. Copy it, change policy(), then evaluate or race it:

    python -m sail_env.evaluate --agent examples.my_agent:make_policy --mode solo
    python -m sail_env.evaluate --agent examples.my_agent:make_policy --mode fleet-vs --opponents rl
    python -m sail_env.evaluate --agent examples.my_agent:make_policy --mode race-vs --episodes 64
    python -m sail_env.play --opponents examples.my_agent:make_policy     # race against it yourself

The policy must be a pure JAX function (it is jitted and vmapped over boats and episodes).
This example only reads the observation, like a learned agent should. Observation layout
(sail_env/start_env.py, get_obs), all in the boat's own frame:

    0 speed / 5 m/s         1 sin TWA     2 cos TWA      3 tack (+1 starboard / -1 port)
    4 time to gun / 60 s    5-6 pin (forward, starboard) / 100 m       7-8 committee boat, same
    9 distance below(-) / above(+) the line / 100 m      10 line bias   11 manoeuvring (tacking)
    12 started   13 OCS     14 true wind speed / 8 m/s
    15-59 the 5 nearest other boats, 9 numbers each (zeros in solo): present, relative position
          (forward, starboard) / 100 m, sin / cos relative heading, speed / 5, tack, I have right of
          way over it, I am in its wind shadow.

In the race ("race", "race-solo") 12 more numbers follow (sail_env/race_env.py):

    60-64 current leg, one-hot: beat 1, run 1, beat 2, run 2, finished (the run shows from the moment
          the boat crosses the line straight upwind of the windward mark; the gate counts once a gate
          mark is rounded: through between the marks, then back up outside one)
    65-66 next mark (forward, starboard) / 100 m: windward mark on a beat, left gate mark on run 1,
          finish line centre on run 2      67-68 right gate mark on run 1 (else 0)
    69 race clock / 600 s   70 penalty circles still owed   71 penalty turn under way (0/1)
"""

import jax.numpy as jnp

HOLD, UP5, DOWN5, UP15, DOWN15, TACK, LUFF = range(7)
RACE_OBS = 72


def make_policy(env, params):
    race = hasattr(env, "course")  # the race env (its observation may also carry the weather block)

    def start(obs):
        time_to_gun = obs[4] * 60.0
        below = -obs[9] * 100.0
        on_starboard = obs[3] > 0
        # A deliberately naive rule: get onto starboard, luff while far ahead of time, then go.
        early = below < 3.0 * time_to_gun  # (very rough: ~3 m of upwind progress per second)
        return jnp.where(~on_starboard, TACK, jnp.where(early & (time_to_gun > 5.0), LUFF, HOLD))

    def sail_to(obs, fwd, stbd):
        # Equally naive: turn towards a point (forward, starboard), never closer than 45 degrees to
        # the wind; tack when it is behind on the windward side.
        rel = jnp.degrees(jnp.arctan2(stbd, fwd))  # > 0: the point is to starboard
        twa = jnp.degrees(jnp.arctan2(obs[1], obs[2]))  # signed: > 0 starboard tack
        windward = jnp.where(twa >= 0, rel > 0, rel < 0)  # turning towards the mark = heading up
        too_high = jnp.abs(twa) < 45.0
        turn = jnp.where(windward, jnp.where(too_high, HOLD, UP5), DOWN5)
        behind = jnp.abs(rel) > 90.0
        # Behind on the windward side: tack if beating, head up if running (a "tack" there is a gybe).
        turn = jnp.where(behind, jnp.where(windward, jnp.where(jnp.abs(twa) < 90.0, TACK, UP15), DOWN15), turn)
        return jnp.where(jnp.abs(rel) < 5.0, HOLD, turn)

    def policy(key, obs, state, boat):
        if not race:
            return start(obs)
        twa = jnp.arctan2(obs[1], obs[2])
        uf, us = jnp.cos(twa), jnp.sin(twa)  # upwind, in the boat's frame (forward, starboard)
        rf, rs = -us, uf  # right, looking upwind
        fwd, stbd = obs[65] * 100.0, obs[66] * 100.0
        # Run 1: through the gate between its marks, then round the left one (looking upwind): out past it
        # below the gate, then back up outside it. Decided from where the boat is, so no memory is needed.
        gate = obs[61] > 0.5
        gf, gs = 50.0 * (obs[65] + obs[67]), 50.0 * (obs[66] + obs[68])  # gate centre
        above = -(gf * uf + gs * us) > 0.0  # the boat is above the gate line
        outside = -(fwd * rf + stbd * rs) < -8.0  # the boat is more than 8 m left of the left gate mark
        run_f = jnp.where(above, gf - 15.0 * uf, jnp.where(outside, fwd - 10.0 * rf + 15.0 * uf, fwd - 12.0 * rf - 8.0 * uf))
        run_s = jnp.where(above, gs - 15.0 * us, jnp.where(outside, stbd - 10.0 * rs + 15.0 * us, stbd - 12.0 * rs - 8.0 * us))
        fwd, stbd = jnp.where(gate, run_f, fwd), jnp.where(gate, run_s, stbd)
        # Beats: leave the windward mark to port. Aim 10 m to its right (looking upwind), then, once
        # there, 10 m above-left of it: over the top from right to left, then down its left side, rounds it.
        beat = (obs[60] > 0.5) | (obs[62] > 0.5)
        a_f, a_s = fwd + 10.0 * rf, stbd + 10.0 * rs
        at_a = jnp.hypot(a_f, a_s) < 12.0
        c_f, c_s = fwd + 10.0 * uf - 10.0 * rf, stbd + 10.0 * us - 10.0 * rs
        tf = jnp.where(beat, jnp.where(at_a, c_f, a_f), fwd)
        ts = jnp.where(beat, jnp.where(at_a, c_s, a_s), stbd)
        mark = sail_to(obs, tf, ts)
        # Not started after the gun (late or OCS): sail to the middle of the line, crossing it.
        line = sail_to(obs, 0.5 * (obs[5] + obs[7]), 0.5 * (obs[6] + obs[8]))
        return jnp.where(obs[12] > 0.5, mark, jnp.where(obs[4] < 0.0, line, start(obs)))

    return policy
