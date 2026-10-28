# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-10-27

Learned racers, fleet and team racing, optional weather, and the rounding fixes.

### Added

- Opponent pool for fleet training (`PPOConfig.opp_frac`, `opp_learners`, `opp_snapshots`, `opp_snapshot_every`; config key `opponents:`): in a share of the envs only some boats learn, the others are sailed by frozen snapshots of the policy or fixed agents (e.g. the digital racer), drawn per race; only learner boats enter the loss (masked means; default training unchanged bit for bit). Logged: `pool_place`, `pool_snapshots_place`, `pool_fixed_place`, `pool_team_win_rate`.
- `anneal:` in training configs: env parameters moving linearly over training (`{field: [start, end, from, until]}`), e.g. phasing out `race_shaping` and `race_time_cost` so that only finishing and place pay.

- Team racing: `team_size` (e.g. `make("race", fleet_size=4, team_size=2)` for 2v2). Lower sum of places wins, a tie goes to the team without first place (RRS Appendix D); `race_rank_reward` then pays the team result. The observation gets one teammate flag per neighbour slot after the race features (racers trained without them still load and ignore them); `info["team_win"]`, `info["team_score"]`; training logs `winning_team_score`. Off by default.

- Fleet-racing measures: `info["shadow_time"]` (s sailed after the gun in another boat's wind shadow) and, in races, `info["mark1_place"]` (order at the first windward mark); logged in training as `shadow_time` and `place_changes_after_mark1`. No effect on the dynamics.

- Race against your own agent in the browser: `python -m sail_env.agents.export <run dir | bundled name> -o agent.json` (also `sail-export-agent`) writes a policy as JSON; the browser version loads it under Opponents → Your agent… (starts and races; no weather). The browser simulator now computes the race observation, so the bundled `rl-racer` is the **Trained RL boats** option in races; `web/check.js` replays a race sailed by `rl-racer` and requires the same action at every step (6450/6450).

- Bundled racers: `rl-racer` (`pretrained/racer`, the full race in steady wind, kickstarted from the digital racer, 2048 envs on a GPU; faster than the digital racer in paired races: -3 to -11 s over ~673 s) and `rl-racer-weather` (`pretrained/racer-weather`, the full race with weather, current capped at 6% of the wind; 15-19 s faster than the digital racer over ~685 s). Before a boat has started, the bundled fleet start policy sails the start.

- `current_max_frac` (race committee): caps the current at that fraction of the race's wind speed. 0 (default) = no cap, presets unchanged. Weather racers got lost in strong current in light wind (current > 8% of the wind: 24% of races unfinished).
- Kickstarting (Schmitt et al. 2018) in PPO: with `teacher: <agent>` in a training config (e.g. `digital`)
  the loss gets `kick * cross-entropy(teacher's action, policy)` on the states the policy visits itself, the
  weight decaying from `ppo.kick_coef` to 0 at `ppo.kick_until` (fraction of training). Logs `kick_coef`,
  `teacher_ce` and `teacher_agreement`. Off by default (`kick_coef: 0`), training without it is unchanged.
- Optional weather (`sail_env/weather.py`). Each part has its own switch in `EnvParams` and is off by default:
  - shifts that travel down the course: a persistent drift, random shifts, and a geographic bend across the course;
  - pressure: puffs and lulls that travel downwind, and a pressure gradient across the course;
  - a uniform current: boats sail through the water, the line and marks stay fixed to the ground, and the
    sails feel the wind over the water;
  - waves: slower into them, slightly faster with them, and dearer tacks and gybes.
- Weather observations (`observe_weather`, 22 features, appended last): the wind the boat itself sailed in
  over the last minute (a per-boat instrument log in `EnvState.wind_log`), the pressure seen upwind,
  the current and the sea state.
- Presets `solo-weather`, `fleet-weather`, `race-solo-weather` and `race-weather`, and a `--weather` flag
  for `sail_env.evaluate` and `tools/benchmark_speed.py`. The weather modes are also in `sail_env.play`.
- Browser game, helm: tap ↑/↓ to turn 1°, hold to keep turning (Shift: 10°); hold Luff to let the sails
  fly and drift to a stop (browser only); the intro card shows today's best upwind and downwind angles.
- Browser game, steering: ← / → (or A / D) turn the bow left / right, seen from the boat (tap 1°, hold to
  keep turning, Shift 10°); ↑ / ↓ still head up / bear away. Steering through head to wind tacks onto
  close-hauled on the new tack, as a helmsman would; through dead downwind it gybes. A bow carried over
  the wind by momentum or a shift also goes about properly. The Tack button (Space) stays.
- Browser game: a map of the whole area in the corner (M), a view switch in every mode (V: follow
  your boat or see the whole course / start area; click the map for the whole view), an "in irons:
  bear away" hint and a "too slow to tack" warning, and "Skip to results" once you have finished.
- The GIFs shade the pressure and draw the local wind direction when the weather is on.
  New figure: `docs/figures/weather.png`.
- 17 tests for the weather.

- Training racers: `race_time_cost` and `race_shaping` (potential-based, time-to-go) reward options;
  presets as training envs (`env: race-solo`); a start bank (race states just after the start, sailed by
  a bundled start policy) and a situation bank (states anywhere on the course) for `ConfigResetEnv`, with
  uniform / SFL / SGS sampling; race metrics and race GIFs on W&B.
- RRS 16.1 (`rrs16`, off by default): a right-of-way boat that changed course by more than `rrs16_turn`
  over the last 2 s before a contact takes the penalty instead of the keep-clear boat.
- Trained racers load as agents: a run trained from the start bank sails the start with the start policy
  it was trained behind, then races. Policies without weather observations can sail the `-weather` presets.

### Fixed

- **Windward mark must be gone round.** Crossing the half-line straight upwind of the mark (right to left) used to complete the beat, so a boat could touch that line and turn back down the right side. Now that is only the first part (`FleetState.mark_through`); the leg counts when the boat comes back down past the mark's level on its left side, and crossing the half-line back to the right undoes it. The observation shows the next leg from the first part on, as before, so trained policies see the same inputs; the reward shaping no longer routes a boat that is past the half-line back round the mark. The digital racer bears away down the mark's left side (it came back down the right side in ~9% of roundings). Python and browser; parity traces regenerated (all pass). Effect on 64 four-boat races (same seeds): digital racer and the fleet racers within ±2 s, finish rates unchanged.
- The example agent (`examples/my_agent.py`) rounds the gate (through, then out round the left mark and back up outside it, decided from its position, no memory); since the gate fix it had finished no race. `docs/baselines.json` and the README tables are regenerated (digital sailor alone 702 s, was 693 s before the rounding fixes).
- **Leeward gate must be rounded.** Passing between the gate marks going downwind used to complete the run, so boats (and the digital racer) turned upwind in the middle of the gate. Now the boat must pass through and then come back up across the gate line outside either mark (`FleetState.gate_through`); coming back up between the marks undoes the pass. The distance to go (shaping, ranking of unfinished boats) routes round a mark; the digital racer rounds the mark on its lane's side (Python and browser). Race presets change: the digital racer is ~10 s slower; policies trained before this cut the gate and mostly fail to round it. The bundled `rl-racer` and `rl-racer-weather` are retrained under the rule (20-28 s faster than the digital racer, partly its simple gate rounding).

- Race ranking compared exactly (finish time, then legs and distance to go) instead of a float32 score `1e7 - finish_time`, which only resolved whole seconds: boats finishing within ~1 s of each other tied and shared places (and the place/team reward). Affects fleet races with close finishes; solo races are unchanged. The browser version (float64) already ranked them correctly.
- The example agent (`examples/my_agent.py`) did not race with weather observations: it recognised the
  race by its observation size (72), which the weather block changes.
- Browser game: the helm stopped answering after using a dropdown (View, Opponents). The dropdown kept
  the keyboard focus, so the arrows changed it instead of steering, and the boat sailed on with nobody
  at the helm. Game keys now always reach the helm (except while typing a start number), and
  dropdowns let go of focus after a choice.
- Boats no longer get stuck head to wind. The digital sailor and racer could ask for a tack forever
  while in irons, or keep starting tacks too slowly that were then abandoned, and sit there until
  the time limit. In the browser race this looked like the game had stopped. They now bear away when
  in irons, and wait with a tack until they have steerage way (3 kn). Digital racers now start every
  race and finish 99.8% of 5-boat races (was 99.4%); `docs/baselines.json` and the README tables
  are regenerated.

### Changed
- Browser parity check (`web/check.js`): each step starts from Python's recorded boats, so a boat that
  grazes a wind-shadow cone edge by a millimetre (float32 against float64) can no longer make the two
  replays drift apart. Tolerance 1 cm per step (was 5 cm accumulated).
- With the weather off, every preset was bit-identical to 0.1.0 when the weather was added (checked on
  recorded solo, fleet and race episodes with random, RL and digital agents). The race presets have since
  changed through the fixes above (gate, windward mark, ranking).
- The bundled start policies can sail the `-weather` start presets: they read the first 60 features
  and ignore the weather block.
- The digital sailor and racer use the wind at their own boat.

## [0.1.0] - 2026-10-20

First public release.

### Added
- Presets `solo` and `fleet`: J/70 starts, solo and with 5 boats. They include:
  - an ORC polar with speed lag and drag-based deceleration;
  - start rules with OCS and individual recall;
  - right of way (RRS 10–13, 22.2), capsule contacts and solid marks;
  - penalties as 360° turns;
  - apparent-wind shadow cones, oscillating shifts and line bias.
- Presets `race` and `race-solo`: the full race. Windward-leeward course, 2 laps, leeward gate, finish
  across the start line, port roundings, simplified mark-room (RRS 18), and penalty turns taken after
  getting clear (RRS 44.2).
- Agent interface (`make_policy(env, params) -> policy(key, obs, state, boat)`).
- Built-in agents: random, hold, the digital sailor and racer, and the bundled PPO policies.
- Tools:
  - keyboard play;
  - saved runs and a leaderboard;
  - GIF replays;
  - the evaluation CLI with confidence intervals;
  - the baseline, benchmark and figure scripts.
- Reference PPO (PureJaxRL style), curricula, and the configs used for the bundled policies.
- Browser version (`web/`), checked against the Python simulator in CI.
- Documentation: `docs/ENVIRONMENT.md`.
- 148 tests, with CI on Python 3.11 and 3.12.
