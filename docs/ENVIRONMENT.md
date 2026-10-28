# sail_env: environment specification

This document specifies the simulator: conventions, dynamics, rules, the race course, observation and
action spaces, rewards, presets, and what is not modelled. Every number below is a field of
`sail_env.config.EnvParams`. The value given is the one used by the named presets (`sail_env/presets.py`).

## 1. Conventions

- **Units:** SI internally (m, s, rad). Knots and degrees appear only at the edges (`kn()`, `deg()`; 1 kn = 0.5144 m/s).
- **Frame:** x east, y north. A heading is a compass angle, with 0 = north and clockwise positive. Boat velocity is v·(sin h, cos h).
- **Wind direction:** the direction the wind blows *from*, in the same compass convention. The mean wind is from the north (`wind_dir = 0`).
- **True wind angle:** signed, TWA = wrap(wind_dir − heading). TWA > 0 means starboard tack (wind over the starboard side), TWA < 0 port tack.
- **Time:** the clock is in seconds relative to the starting signal (the gun at t = 0). There is one decision per `dt` = 1 s, integrated in 4 physics sub-steps of 0.25 s.
- **Positions:** a boat's position is its centre. The bow is half a hull length (`boat_length` = 6.9 m) ahead of it. Line crossings and roundings use the bow's path.

## 2. Wind and start line

- **True wind speed:** uniform in [6, 16] kn per episode, the same everywhere except inside wind shadows (§5.4), unless the optional weather is on (§2.1).
- **Shifts:** oscillating, θ(t) = θ₀ + A sin(2πt / T + φ). A is uniform in [0, `shift_amp_max`] and φ uniform in [0, 2π).
  - Fleet and race: `shift_amp_max` = 5°, T = 120 s.
  - Solo: steady wind.
- **Start line:** centred at the origin, square to the mean wind, rotated by a bias uniform in [−`line_bias_max`, +`line_bias_max`]. A positive bias puts the committee-boat end upwind.
  - Fleet and race: bias ±10°, line length 1.6 × 5 boats × 6.9 m = 55.2 m.
  - Solo: no bias, 1.25 × 15 × 6.9 m = 129.4 m (sized like a 15-boat line).
- **Marks:** the committee boat (starboard end, radius 4 m) and the pin (port end, radius 1 m) are solid.

### 2.1 Weather (optional)

`sail_env/weather.py` adds four parts on top of the base wind. Each has its own switch in `EnvParams`
and is off in the standard presets (with it off, the wind is exactly the base wind). The `-weather` presets
(§8) switch all four on, together with the weather observations; parts can be switched off again,
e.g. `make("race-weather", waves=False)`. Every number below is a field of `EnvParams`.

**The field moves.** Shifts and puffs are carried downwind at U = `weather_advection` × the base wind
speed (0.8). In the mean-wind frame (a = distance upwind of the line centre, c = distance to the right
looking upwind), every moving feature depends on the arrival time τ = t + a / U. What happens at the
windward mark reaches the line 400 m / U later: about 95 s in 10 kn.

| Switch | Part | Model |
|---|---|---|
| `weather_shifts` | persistent shift | the direction drifts at a rate uniform in ±`drift_max` (6° per 10 min) |
| | random shifts | 4 sinusoids (`n_shift_modes`) with log-uniform periods in [60, 600] s; standard deviation `random_shift_std` = 4°. The base oscillation of §2 also travels with the field |
| | geographic bend | the direction changes across the course: uniform in ±`bend_max` = 2° per 100 m |
| `weather_pressure` | puffs and lulls | 48 Gaussian patches (`n_puffs`) in a periodic cell, 3000 m along the wind × 1000 m across, so they never run out. Radius 40–120 m across the wind, 1.5× that along it. Amplitude +25% (puff) to −20% (lull) of TWS; a shift of up to ±5° at the centre |
| | pressure gradient | TWS × (1 + g × c / 100 m), with g uniform in ±`pressure_grad_max` = 4%: one side of the course has more wind |
| | floor | the wind never drops below 0.3 × the base speed |
| `water_current` | current | uniform over the course; speed uniform in [0, 1] kn, direction uniform |
| `waves` | waves | height uniform in [0, 0.8] m, coming from the mean wind ± 20° |

Formulas:

    direction = base(τ) + drift τ + Σ_k A_k sin(2π τ / P_k + φ_k) + bend c / 100 + Σ_p dir_p g_p
    speed     = TWS (1 + grad c / 100) (1 + Σ_p amp_p g_p),     at least 0.3 TWS
    g_p       = exp(−(d_a / 1.5 r_p)² / 2 − (d_c / r_p)² / 2)

Here d_a and d_c are the distances from puff p, taken in the periodic cell.

**Current.** The sails feel the wind over the water, W − C, where W is the air velocity and C the
water velocity. A current flowing into the wind therefore adds wind, and a cross-current swings it.
The boat sails at its polar speed through the water, and the current carries it over the ground.
The line and the marks are fixed to the ground. TWA, TWS and boat speed (in the observation, the
success test and the physics) are all relative to the water, as on a real boat's instruments.

**Waves.** Let b be the angle between the heading and the direction the waves come from. The target
speed is multiplied by

    1 − 0.10 h max(cos b, 0) + 0.03 h max(−cos b, 0)

so a boat is slower into the waves and slightly faster running with them. A tack or gybe also loses an
extra 0.10 × h of speed. In a start with waves, "at least 90% of the polar" means 90% of the
wave-reduced target. A polar model can only fake waves: this is the simplest version.

**Each boat sails in its own wind.** Right of way (which tack a boat is on), wind shadow, TWA and the
upwind efficiency all use the wind over the water at that boat. The fleet's ranking by height at T+10
uses the wind at the line centre.

![weather](figures/weather.png)

## 3. Boat model (J/70)

### 3.1 Polar

Target boat speed V*(TWA, TWS) comes from the median of the ORC VPP output over the 25 J/70 certificates of 2024 ([jieter/orc-data](https://github.com/jieter/orc-data), `sail_env/data/polar_j70_orc.csv`). The table covers TWA 52–150° and TWS 6–24 kn. Outside it:

| TWA range | Target speed |
|---|---|
| 0–30° (`no_go`) | 0: sails flapping |
| 30° to the optimal beat angle (37–43°) | linear ramp up to the beat speed (pinching) |
| 150–180° | linear down to 0.9 × the best run VMG at 180° (an assumption: an asymmetric-spinnaker boat is slow dead downwind) |
| TWS below 6 kn | linear to 0 at 0 kn, so light air inside a shadow slows the boat proportionally |

The grid has 0.25° resolution, with bilinear interpolation in (|TWA|, TWS). See `docs/figures/polar.png`.

### 3.2 Speed

When speeding up (V* > v) there is a first-order lag with τ = 4 s, integrated exactly over each sub-step.

When slowing down, the speed follows a force balance:

    m dv/dt = F_drive − k v|v| − ½ ρ C_dA u|u|,    u = TWS cos(TWA) + v

| Symbol | Value | Meaning |
|---|---|---|
| m | 1265 kg | boat 810 kg + crew 340 kg + ~10% added mass |
| k | 37 N/(m/s)² | hull drag, ~350 N at 6 kn |
| C_dA | 5 m² | windage of flogging sails, rig, hull and crew |
| ρ | 1.225 kg/m³ | air density |
| F_drive | k V*² + windage at V* | the drive that would hold the target speed; 0 in the no-go zone |

Speed never falls below max(V*, 0). Head to wind, windage dominates, so a boat stops faster in a breeze than in light air.

### 3.3 Steering

- **Turn rate:** 20°/s at or above 3 kn, falling linearly to 0 at 0.5 kn. Bearing away is always possible at ≥ 5°/s, even when stopped (backing the jib), so a boat can get out of irons.
- **Tack or gybe:** the target heading mirrors across the wind (TWA → −TWA), and further actions are ignored until the turn is complete. A gybe costs 10% of speed. A tack has no extra loss, because passing through the no-go zone already costs about a third of the speed. A tack that runs out of steerage is abandoned.
- **Luff:** for one decision, the target speed is 12% of the polar.

## 4. Start rules

- **Starting:** a boat starts when its bow crosses the line segment from the pre-start side after the gun, while not OCS.
- **OCS and individual recall:** a boat whose bow is on the course side at the gun is OCS. It must return fully to the pre-start side, and only then can it start.
- **Good start** (`good_start_on_line`), the success measure used for all results. All of the following must hold:
  1. At the gun, the bow is 0–3 m below the line and between its ends.
  2. The boat starts within 3 s of the gun.
  3. At the crossing, its speed is at least 90% of the polar target.
  4. It starts on starboard tack.
  5. Fleet only: at T+10 s it is in clear air (outside every shadow) and owes no penalty.

  There is no "time to line" or "distance to line" instrument in the observation; the agent must judge both.

## 5. Fleet rules

Implemented in `sail_env/fleet_rules.py`, `fleet_env.py` and `race_env.py`. The rules are a simplified subset of the Racing Rules of Sailing (RRS).

### 5.1 Right of way

| Rule | Implementation |
|---|---|
| RRS 10, opposite tacks | the port-tack boat keeps clear |
| RRS 11, same tack, overlapped | the windward boat keeps clear |
| RRS 12, same tack, not overlapped | the boat clear astern keeps clear. Clear astern: its bow is behind the line abeam the other boat's stern |
| RRS 13, while tacking or gybing | that boat keeps clear. If both are manoeuvring, no rule decides |
| RRS 22.2, taking a penalty | that boat keeps clear of boats that are not |
| RRS 18, mark-room (race, simplified) | see §6.3 |
| RRS 16.1, changing course (`rrs16`, off in the presets) | a right-of-way boat that turned by more than `rrs16_turn` (10°) over the last 2 s before a contact did not give room: it takes the penalty instead of the keep-clear boat. Not when it is entitled to mark-room |

### 5.2 Contact and marks

Each hull is a capsule: the centreline from stern to bow, with radius beam/2 (beam 2.25 m). Marks are circles. A new contact between two boats penalises the boat that had to keep clear, or both boats if no rule decides. Touching a mark is a penalty (RRS 31).

A contact also costs reward: every boat involved pays `contact_cost` = 2, except the right-of-way boat, which pays `contact_cost_row` = 0.5. This reflects RRS 14: avoid contact if reasonably possible.

### 5.3 Penalties

- **The turn:** a penalty is one full 360° turn at 30°/s, starting by bearing away, with actions ignored while turning. The speed loss comes from the physics: the boat gybes, goes head to wind and tacks.
- **When it starts:**
  - Start presets: the turn starts immediately.
  - Race presets (RRS 44.2, "after getting well clear"): the boat keeps control and its normal rights until no boat or mark is within 2 boat lengths, or for at most 10 s, and then turns.
- **Contacts during the turn:** a contact while turning is part of the same incident. It costs reward but adds no further penalty.
- **At the end:** a penalty still owed at the end of the episode costs `foul_penalty` = 1.

### 5.4 Wind shadow

Each boat casts a cone downstream along its apparent wind (true wind minus boat velocity), so a close-hauled boat's dirty air falls to leeward and aft. The cone is 6 boat lengths long with a 12° half-angle. Inside it, the wind speed is multiplied by a factor that rises linearly from 0.75 at the boat to 1 at the tip. A boat in several cones takes the strongest one.

## 6. The race (`race`, `race-solo`)

### 6.1 Course

The course is set to the mean wind (`docs/figures/course.png`):

- **Windward mark:** 400 m upwind of the line centre.
- **Leeward gate:** two marks 8 boat lengths (55.2 m) apart, 60 m upwind of the line centre. All course marks have radius 1 m.
- **Finish:** the start line, crossed downwind between its ends.
- **Legs:** after a legal start, beat → run (through the gate) → beat → run (to the finish), i.e. 2 laps.

Wind shifts swing the true wind around this fixed course, as on the water.

### 6.2 Roundings

| Leg | How it is completed |
|---|---|
| Windward mark (legs 1 and 3) | Must be left to port, and gone round. The bow's path crosses the half-line straight upwind of the mark, from right to left looking upwind, and then comes back down past the mark's level on its left side; the leg counts then. Crossing the half-line back to the right in between undoes the first part (`FleetState.mark_through`), so touching the half-line and turning back down the right side does not count. From the first part on, the observation already shows the next leg |
| Gate (leg 2) | The bow passes between the gate marks going downwind, then rounds either mark: the leg counts when it comes back up across the gate line outside a mark (`FleetState.gate_through`). Coming back up between the marks undoes the pass |
| Finish (leg 4) | The bow crosses the start line downwind between the pin and the committee boat. The finish time is interpolated within the second |

Finished boats leave the course (they are parked far away) and no longer interact.

### 6.3 Mark-room (simplified RRS 18)

The zone is a circle of 3 boat lengths around the mark a boat is rounding. When the first of two boats rounding the same mark enters the zone, room is assigned:

- if they are overlapped, the outside boat (farther from the mark) owes room to the inside boat;
- otherwise, the boat clear astern owes room to the boat clear ahead.

The boat owing room must keep clear, which overrides §5.1. Room lasts while at least one of the two boats is in the zone, and ends when either of them rounds the mark.

Not modelled: room for a tack at the windward mark (RRS 18.3), and changes of overlap inside the zone.

### 6.4 End and ranking

The episode ends when every boat has finished, or at T+1200 s. Boats are ranked by finishing order. Unfinished boats rank behind all finishers, ordered by legs completed and then by straight-line distance still to sail.

## 7. Spaces

### 7.1 Actions

There are 7 discrete actions, one decision per second, identical for every boat:

| # | Action | Effect |
|---|---|---|
| 0 | hold | keep the target heading (an unfinished turn carries on) |
| 1 / 2 | head up / bear away 5° | target heading ±5° towards / away from the wind |
| 3 / 4 | head up / bear away 15° | the same, 15° |
| 5 | tack or gybe | mirror the heading across the wind; locks steering until complete |
| 6 | luff | ease the sails for this second (target speed × 0.12) |

### 7.2 Observation

The observation is a float32 vector in the boat's own frame, roughly scaled to [−1, 1]. Starts have 60 features; the race has 72.
With `observe_weather` (the `-weather` presets), 22 weather features are appended last (82 and 94 in total); no other index moves.

| Index | Feature |
|---|---|
| 0 | speed / 5 m/s |
| 1, 2 | sin TWA, cos TWA |
| 3 | tack (+1 starboard, −1 port) |
| 4 | time to the gun / 60 s (negative after it) |
| 5, 6 / 7, 8 | pin / committee boat (forward, starboard) / 100 m |
| 9 | signed distance from the line / 100 m (> 0 on the course side) |
| 10 | line bias / `line_bias_max` |
| 11 | manoeuvring (tack or gybe in progress) |
| 12, 13 | started, OCS |
| 14 | true wind speed / 8 m/s |
| 15–59 | the 5 nearest other boats, 9 features each, zero-padded: present; relative position (forward, starboard) / 100 m; sin, cos of relative heading; speed / 5; tack; "I have right of way over it"; "I am in its wind shadow" |
| 60–64 | race: current leg, one-hot (beat 1, run 1, beat 2, run 2, finished) |
| 65, 66 | race: next mark (forward, starboard) / 100 m. Windward mark on a beat, left gate mark on run 1, line centre on run 2 |
| 67, 68 | race: right gate mark on run 1 (else 0) |
| 69 | race: race clock / 600 s |
| 70, 71 | race: penalty circles still owed; penalty turn under way |
| last 22 | weather (`observe_weather`, below) |

The weather block (`weather.observe`, in order):

| Features | What a sailor would use |
|---|---|
| 12: for each of 10, 20, … 60 s ago, what the boat's own instruments read then, i.e. the wind over the water where the boat was at that moment (a per-boat log, one reading a second): direction offset from the course axis / 0.25 rad, speed / 8 m/s | the wind log on the instruments, or a sailor watching the compass |
| 6: wind speed / 8 m/s at 100 m and 250 m upwind of the boat, and 150 m to either side, in the mean-wind frame | seeing dark patches (puffs) on the water upwind |
| 2: current in the boat's frame (forward, starboard), m/s / 0.5 | the current at the committee boat's anchor line |
| 2: wave height / 1 m; height × cos(angle between bow and waves) | the sea state |

The history holds only what the boat sailed in. The one assumption is for readings from before the
episode began: the boat was already in the start area, so it is taken to have been at its starting
position then.

Features 1, 2 (TWA) and 14 (wind speed) become the wind at the boat itself. The base features do not
show the mean wind direction; the history features give the shift relative to it.

The full simulator state is also passed to agents (privileged), for hand-written controllers. Learned agents are comparable only when they use the observation alone.

## 8. Rewards and presets

| | `solo` | `fleet` | `race-solo` | `race` |
|---|---|---|---|---|
| Boats | 1 | 5 | 1 | 5 |
| Clock | T−60 → T+10 s | T−90 → T+10 s | T−90 → finish, ≤ T+1200 s | T−90 → all finished, ≤ T+1200 s |
| Decisions per episode | 70 | 100 | ≤ 1290 | ≤ 1290 |
| Line | 129.4 m, square | 55.2 m, bias ±10° | 55.2 m, ±10° | 55.2 m, ±10° |
| Wind | 6–16 kn, steady | 6–16 kn, shifts ±5° | as `fleet` | as `fleet` |
| Penalty turn | – | at once | after getting clear | after getting clear |

Each preset has a `-weather` variant (`solo-weather`, `fleet-weather`, `race-solo-weather`, `race-weather`): the same
settings, plus moving shifts and pressure, current, waves and the weather observations (§2.1).

**Reward, start presets.** Paid at the end of the episode:

- **Successful start:** +1, minus 1 if it is on port tack (`port_penalty` = 1). A successful start meets conditions 1–3 and, in the fleet, 5 of §4.
- **Upwind efficiency bonus:** up to +1 after a successful start (`upwind_bonus` = 1). Efficiency is the mean over the seconds since the start of min(VMG / best polar VMG, 1), counting 10 extra seconds at the final VMG. Each second is capped at 1, so momentum tricks such as shooting head to wind do not pay.
- **OCS at the end:** −1.
- **Fleet only:**
  - −2 per contact at fault (−0.5 as the right-of-way boat), during the episode;
  - −1 for a penalty still owed at the end;
  - a zero-sum rank by height upwind at T+10, from +1.5 for the leader to −1.5 for the last boat. Only legal starters with no penalty owed are ranked.

**Reward, race presets.**

- **Finishing:** +1.
- **Place:** a zero-sum score by finishing place at the end, from +1 for the winner to −1 for the last boat.
- **Contacts and penalties:** −2 per contact at fault (−0.5 as the right-of-way boat), and −1 for a penalty still owed at the end.
- **Optional dense reward:** `race_progress_reward` pays per 100 m of distance-to-go saved; it is 0 in the presets.

The start itself is not rewarded in the race.

**API.** The environments follow the gymnax-style functional API:

- `env.reset(key, params)` returns `(obs, state)`.
- `env.step(key, state, action, params)` returns `(obs, state, reward, done, info)` and auto-resets when an episode ends.
- `env.step_env(...)` is the same step without the reset.

All of these are pure functions, compatible with `jax.jit` and `jax.vmap`. In the fleet and race environments, actions, rewards and dones have shape `(N,)`, and observations `(N, obs_dim)`.

## 9. Not modelled (limitations)

- **Water and boat attitude:** no heel, trim or crew weight. The optional current is uniform in space and time (no tidal streams or gradients). The optional waves only change the target speed and the cost of a manoeuvre. Sail trim beyond "luff" is not modelled, and the polar assumes an optimally trimmed boat.
- **Wind:** in the standard presets the wind field is uniform apart from geometric shadow cones. The optional weather (§2.1) adds shifts and puffs that travel downwind, but it is a kinematic pattern, not a simulation of the air: puffs keep their shape, there is no gust spreading and no land or thermal effects, and wake effects are not modelled.
- **Racing rules not modelled:** proper course, acquiring right of way, changing course (except the optional RRS 16.1), room at obstructions, and room to tack at a mark (RRS 15, 16.2, 17, 18.3, 19; RRS 16.1 is optional, `rrs16`). Overlaps are evaluated instantaneously.
- **Polar and physics constants:** the polar is VPP output, not measured on the water. The deceleration constants (hull drag, windage, added mass) are estimates.
- **Actions:** the action space is coarse. Headings are set in 5° or 15° steps, once per second.
- **Opponents:** the race opponents are hand-written controllers. The bundled RL policies cover the starts only.

## 10. Validation

- **Python tests:** `tests/` holds 166 tests covering:
  - physics: the polar, speed lag, drag, turning, tacks and gybes;
  - weather: off means the base wind exactly, the field moving downwind, drift, the spread of the shifts, bend and gradient, puffs and the floor, current (wind over the water, drift over the ground), waves, and the weather observations;
  - start rules: crossings, OCS and recall;
  - fleet rules: right of way, contacts, penalties and wind shadow;
  - the race: geometry, roundings, gate, finish, mark-room, ranking and penalty timing;
  - the agent interface, saved runs and replays, evaluation, and play mode.
- **Browser parity:** the browser version (`web/sim.js`) is a port of this simulator. CI replays six reference episodes from the Python simulator in it, including two full 5-boat races. Each step starts from the Python boats' state; after the step, positions must agree within 1 cm, discrete states must be identical, and the agents must choose the same action ([`web/check.js`](../web/check.js)).
