"""Environment parameters and unit conversions.

Internal units are metres, seconds and radians. Knots and degrees appear only here, at the edges,
through the helpers below. Every number is a starting value to tune.

`EnvParams` is a flax.struct.dataclass, i.e. a JAX pytree: it can be passed straight into jitted
`reset`/`step` functions (gymnax style). Fields marked `pytree_node=False` are static: they fix
array shapes or loop lengths, so changing them triggers a recompile.
"""

import math

from flax import struct

KNOT = 0.5144  # m/s per knot


def kn(x: float) -> float:
    """Knots -> m/s."""
    return x * KNOT


def deg(x: float) -> float:
    """Degrees -> radians."""
    return math.radians(x)


@struct.dataclass
class EnvParams:
    # --- Timing ---------------------------------------------------------------------------
    dt: float = struct.field(pytree_node=False, default=1.0)  # seconds per decision
    n_substeps: int = struct.field(pytree_node=False, default=4)  # physics sub-steps (0.25 s each)
    t_start: float = struct.field(pytree_node=False, default=-60.0)  # clock at reset, seconds relative to the gun (solo)
    t_end: float = struct.field(pytree_node=False, default=10.0)  # episode ends at T+10 s

    # --- Wind -----------------------------------------------------------------------------
    wind_dir: float = 0.0  # direction the wind blows FROM, radians (0 = north)
    tws_min: float = kn(6.0)
    tws_max: float = kn(16.0)
    shift_amp_max: float = deg(0.0)  # oscillating shift amplitude (off by default in v1)
    shift_period: float = 120.0  # seconds

    # --- Weather (weather.py; every part off by default, so 0.1 presets are unchanged) -----------
    weather_shifts: bool = struct.field(pytree_node=False, default=False)  # drift, random shifts, bend
    weather_pressure: bool = struct.field(pytree_node=False, default=False)  # puffs, lulls, gradient
    water_current: bool = struct.field(pytree_node=False, default=False)  # uniform current
    waves: bool = struct.field(pytree_node=False, default=False)  # heading-dependent speed, dearer manoeuvres
    observe_weather: bool = struct.field(pytree_node=False, default=False)  # + WEATHER_FEATURES observations
    weather_advection: float = 0.8  # shifts and puffs travel downwind at this x the base wind speed
    drift_max: float = deg(6.0) / 600.0  # rad/s: persistent shift up to 6 deg per 10 minutes, either way
    random_shift_std: float = deg(4.0)  # standard deviation of the random shifts
    random_shift_period_min: float = 60.0  # s, periods of the random shift modes (log-uniform)
    random_shift_period_max: float = 600.0
    n_shift_modes: int = struct.field(pytree_node=False, default=4)
    bend_max: float = deg(2.0)  # geographic shift across the course, per 100 m (either way)
    pressure_grad_max: float = 0.04  # more wind on one side: up to 4% of TWS per 100 m across the course
    n_puffs: int = struct.field(pytree_node=False, default=48)  # in a periodic cell (one every ~250 m)
    puff_cell_length: float = 3000.0  # m along the wind
    puff_cell_width: float = 1000.0  # m across the wind
    puff_amp_max: float = 0.25  # puffs up to +25% of TWS at their centre ...
    lull_amp_max: float = 0.20  # ... lulls down to -20%
    puff_radius_min: float = 40.0  # m (one standard deviation across the wind; 1.5 x along it)
    puff_radius_max: float = 120.0
    puff_shift_max: float = deg(5.0)  # shift at a puff's centre, either way
    tws_floor_frac: float = 0.3  # the wind never drops below this x the base speed
    current_max: float = kn(1.0)  # current speed uniform in [0, max], direction uniform
    # Race committee: no race when the current is too strong for the wind. > 0 caps the current at this
    # fraction of the race's wind speed (uniform in [0, min(current_max, frac x TWS)]); 0 = no cap.
    current_max_frac: float = 0.0
    wave_height_max: float = 0.8  # m, uniform in [0, max]
    wave_dir_spread: float = deg(20.0)  # waves come from the mean wind direction, give or take this
    wave_upwind_loss: float = 0.10  # target speed x (1 - this x height) with the bow into the waves
    wave_downwind_gain: float = 0.03  # ... x (1 + this x height) running with them
    wave_manoeuvre_loss: float = 0.10  # extra fraction of speed lost per tack or gybe, per metre of height

    # --- Boat (J/70) ----------------------------------------------------------------------
    boat_length: float = struct.field(pytree_node=False, default=6.9)
    boat_beam: float = struct.field(pytree_node=False, default=2.25)
    tau_accel: float = 4.0  # s, time constant when speeding up
    tau_decel: float = 8.0  # s, time constant when slowing down (only if drag_decel is False)
    # Slowing down from forces instead of a fixed time constant (drag_decel=True):
    #   m dv/dt = F_drive - k_water v|v| - 1/2 rho_air CdA u|u|,  u = wind speed along the bow = TWS cos(TWA) + v
    # with F_drive just large enough to hold the target speed (polar, luff or no-go ramp). Head to wind
    # in a breeze the windage of flogging sails dominates, so the boat stops much faster than in light air.
    # Starting values (estimates, to calibrate against on-the-water experience):
    drag_decel: bool = struct.field(pytree_node=False, default=True)
    boat_mass: float = 1265.0  # kg: J/70 810 kg + crew 340 kg (ORC), + ~10% water carried along
    water_drag_k: float = 37.0  # N/(m/s)^2: ~350 N of hull drag at 6 kn
    windage_cda: float = 5.0  # m^2: flogging sails, rig, hull and crew, head to wind
    rho_air: float = 1.225  # kg/m^3
    no_go: float = deg(30.0)  # target speed 0 below this |TWA|; ramps up to the ORC beat angle (37-43 deg)
    luff_factor: float = 0.12  # luffing: target speed = this x polar
    max_turn_rate: float = deg(20.0)  # rad/s at full steerage
    full_steer_speed: float = kn(3.0)  # turn rate reaches max at/above this speed
    min_steer_speed: float = kn(0.5)  # no steerage below this ...
    irons_bear_away_rate: float = deg(5.0)  # ... except bearing away, always possible ("backing the jib")
    # Extra one-off loss on a tack. 0 because the pass through the no-go zone already costs ~1/3 of
    # boat speed (measured: 30% extra would give a 50-56% total drop).
    tack_speed_loss: float = 0.0
    gybe_speed_loss: float = 0.10  # fraction of speed lost on a gybe

    # --- Start line -----------------------------------------------------------------------
    fleet_size: int = struct.field(pytree_node=False, default=1)  # boats in the simulation
    # The line is sized for this many boats (brief: 1.25 x boats x boat length), even in solo
    # training, so solo boats learn on the same geometry as a real fleet start (~130 m for 15).
    line_fleet_size: int = struct.field(pytree_node=False, default=15)
    line_length_factor: float = struct.field(pytree_node=False, default=1.25)  # line = factor x boats x length

    # --- Fleet (Phase 4) ------------------------------------------------------------------
    shadow_length_bl: float = 6.0  # wind shadow cone length, boat lengths
    shadow_half_angle: float = deg(12.0)
    shadow_min_factor: float = 0.75  # wind speed factor right behind a boat, rising to 1 at the tip
    # Cast the shadow along each boat's apparent wind (true wind minus boat velocity) instead of
    # straight downwind: a close-hauled boat's dirty air goes to leeward and aft.
    shadow_apparent: bool = struct.field(pytree_node=False, default=False)
    penalty_seconds: float = 15.0  # time loss of a penalty turn (sails eased, no steering)
    # Real penalty turns instead of the forced luff: the penalised boat immediately turns a full
    # circle (starting by bearing away) at penalty_turn_rate, actions ignored, keeping clear of
    # everyone (RRS 22.2); speed loss comes from the physics (gybe, head to wind, tack).
    penalty_turn: bool = struct.field(pytree_node=False, default=False)
    penalty_turns: float = 1.0  # full circles per penalty (the rule book's two-turns penalty = 2)
    penalty_turn_rate: float = deg(30.0)  # rad/s: a 360 in ~12 s
    # RRS 44.2: take the penalty "after getting well clear". With penalty_clear_distance > 0 the penalised
    # boat keeps sailing (in control, normal rights) until no other boat and no mark is within that many
    # metres, or penalty_clear_timeout seconds have passed, then turns. 0 = turn at once.
    penalty_clear_distance: float = 0.0
    penalty_clear_timeout: float = 10.0
    # RRS 16.1: a right-of-way boat that changes course must give the other boat room to keep clear.
    # With rrs16, a right-of-way boat that turned by more than rrs16_turn over the last 2 s before a
    # contact broke RRS 16: it takes the penalty instead of the keep-clear boat (not when it is entitled
    # to mark-room: rounding the mark is allowed). Off by default: the presets are unchanged.
    rrs16: bool = struct.field(pytree_node=False, default=False)
    rrs16_turn: float = deg(10.0)
    foul_penalty: float = 1.0  # terminal reward -foul_penalty with a penalty still unserved at the end
    # Avoid contact at all times (RRS 14): every new contact (with a boat or a mark) costs every boat
    # involved this much reward, on top of the time penalty for the boat at fault.
    contact_cost: float = 0.0
    # Cost for a boat that had right of way in a boat-to-boat contact (it still has to avoid contact
    # if reasonably possible, RRS 14). < 0 means the same as contact_cost.
    contact_cost_row: float = -1.0
    pin_radius: float = 1.0  # m, the pin buoy
    committee_radius: float = 4.0  # m, the committee boat (modelled as a circle)
    # Racing each other: at the end, boats are ranked by height (bow distance upwind of the line);
    # boats without a legal start or with a penalty to serve share the last places. Zero-sum reward
    # rank_reward x ((N - 1 - rank) / (N - 1) - 0.5): leader +rank_reward/2, last -rank_reward/2.
    rank_reward: float = 0.0

    # --- Full race (RaceEnv): windward-leeward, 2 laps, leeward gate, finish = start line crossed downwind
    course_beat_length: float = 400.0  # m from the line centre to the windward mark, along the mean wind
    gate_offset: float = 60.0  # m from the line centre to the leeward gate centre, upwind
    gate_width_bl: float = 8.0  # boat lengths between the two gate marks
    course_mark_radius: float = 1.0  # m, windward mark and gate marks
    zone_bl: float = 3.0  # mark-room zone radius, boat lengths (RRS 18)
    # Legs in the race, for training in stages: 1 = beat to the windward mark (rounding it finishes the
    # race), 2 = + run through the leeward gate (passing it finishes), 4 = the full race (the presets).
    race_legs: int = struct.field(pytree_node=False, default=4)
    start_reward_weight: float = 1.0  # scales the start's terminal reward (0 in the race presets)
    race_finish_reward: float = 0.0  # reward for finishing the race
    race_rank_reward: float = 0.0  # zero-sum ranking by finishing order: leader +x/2, last -x/2
    # Team racing (2 teams of team_size boats: boats 0..team_size-1 vs the rest; 0 = off, individual racing).
    # The team with the lower sum of finishing places wins; on a tie the team without first place wins
    # (RRS Appendix D). race_rank_reward then pays the team result instead of the individual place: every boat
    # of the winning team +race_rank_reward/2, of the losing team -race_rank_reward/2. The observation gets one
    # teammate flag per neighbour slot (after the race features).
    team_size: int = struct.field(pytree_node=False, default=0)
    race_progress_reward: float = 0.0  # optional dense reward per 100 m of progress along the course
    # Rewards for learning to race (0 in the presets). Summed over a finished race, progress always adds
    # up to the course length, so on its own it cannot tell a fast boat from a slow one: what makes
    # speed pay is a cost per second, and the shaping below only tells the agent earlier how it is doing.
    race_time_cost: float = 0.0  # reward -race_time_cost per second until the boat finishes
    # Potential-based shaping (Ng et al. 1999), Phi = -distance still to sail / 100 m (0 once finished), by
    # the route a port rounding needs (race_env.rounding_to_go: via the right side of the windward mark):
    # reward += race_shaping x (shaping_gamma x Phi(s') - Phi(s)). With shaping_gamma = PPO's gamma it
    # leaves the best policy unchanged; it only credits progress in the second it is made.
    race_shaping: float = 0.0
    # Bias sampled in [-max, +max] off square to the wind. 0 = line always perpendicular to the wind
    # (current choice); the brief's later setting is deg(15.0).
    line_bias_max: float = deg(0.0)

    # --- Initial state (solo, Phase 2 uniform sampling) -------------------------------------
    # A sailor stays close and uses the width of the course: up to 120 m below the line and up to
    # 100 m either side, but every start must be winnable: the distance is capped at
    # init_reach_frac x (best close-hauled VMG x time to gun), and the sideways offset at half the
    # line plus init_lateral_speed x time to gun.
    init_dist_min: float = 20.0  # m downwind of the line at reset
    init_dist_max: float = 120.0
    init_reach_frac: float = 0.75
    init_lateral_max: float = 100.0  # m either side of the line centre, measured along the line
    init_lateral_speed: float = 1.5  # m/s of sideways progress assumed for the lateral cap
    # Easier starts in the mix: begin each episode at a random time to gun in
    # [init_time_to_gun_min, -t_start] instead of always at t_start (T-60).
    random_start_time: bool = struct.field(pytree_node=False, default=False)
    init_time_to_gun_min: float = 5.0
    init_speed_frac_max: float = 1.0  # initial speed uniform in [0, this x polar]

    # --- Success --------------------------------------------------------------------------
    success_window: float = 3.0  # must cross between T=0 and T+this
    success_speed_frac: float = 0.90  # ... at >= this fraction of polar target speed
    # The full rules, fixed: used to report "strict" success even while a curriculum relaxes the
    # two values above during training.
    strict_success_window: float = struct.field(pytree_node=False, default=3.0)
    strict_success_speed_frac: float = struct.field(pytree_node=False, default=0.90)
    # On the line at the gun: success also needs the bow 0..gun_distance_max m below the line and
    # between the ends when the gun goes (a professional's definition: ~1 s at 6 kn for 3 m).
    # >= 1e6 disables the check (the default, so earlier configs keep their meaning).
    gun_distance_max: float = 1e9
    strict_gun_distance_max: float = struct.field(pytree_node=False, default=3.0)  # for reporting

    # --- Per-episode rules (joint curriculum over situations and rules) ---------------------
    # With per_episode_rules, each episode has its own strictness w in [0, 1] (EnvState.rule_w):
    # every rule value = easy + w x (full - easy), full = the values in this config. observe_rules
    # appends the episode's rule values to the observation so one policy can learn across levels.
    per_episode_rules: bool = struct.field(pytree_node=False, default=False)
    observe_rules: bool = struct.field(pytree_node=False, default=False)
    easy_success_window: float = 10.0
    easy_success_speed_frac: float = 0.0
    easy_gun_distance_max: float = 30.0
    easy_ocs_penalty: float = 0.0
    easy_port_penalty: float = 0.0
    easy_speed_bonus: float = 0.02

    # --- Reward ---------------------------------------------------------------------------
    ocs_penalty: float = 1.0  # terminal reward -ocs_penalty if still OCS at the end
    reward_shaping: float = 0.0  # coefficient of the potential-based shaping term (0 = sparse)
    shaping_gamma: float = 0.99  # must equal PPO's gamma for the policy-invariance guarantee
    # v2 potential: add the time needed to accelerate, tau_accel x (missing fraction of beat VMG).
    # False reproduces the first (position-only) shaping, which rewarded waiting and slow arrivals.
    shaping_speed_aware: bool = struct.field(pytree_node=False, default=False)
    # Speed bonus around the gun (not potential-based, so it does change what is optimal while on;
    # meant to be faded out by a curriculum): per step, speed_bonus x (VMG towards the line / best
    # close-hauled VMG, clipped to [0, 1]) while the clock is in [speed_bonus_from, speed_bonus_until].
    speed_bonus: float = 0.0
    speed_bonus_from: float = -10.0
    speed_bonus_until: float = 3.0
    # What a good start also means (Phase 2, run D):
    # - starboard: a legal start on port tack loses this share of the success reward (1 = a port
    #   start earns nothing). In a fleet, starboard has right of way; in solo it must be imposed.
    port_penalty: float = 0.0
    # - height: a successful boat earns up to this bonus at T+10, scaled by its upwind efficiency =
    #   average over the seconds after crossing of VMG / best close-hauled VMG, each second capped
    #   at 1. Sailing at the polar optimum scores ~1; pinching (slow) and footing (low) score less;
    #   the cap stops "shooting" head to wind (coasting upwind faster than the polar for a few
    #   seconds, then stalling) from paying. (Was: height at T+10, which rewarded shooting.)
    upwind_bonus: float = 0.0
    # The episode ends at T+10, but a sailor keeps sailing: count this many extra seconds at the
    # VMG fraction the boat has at the end, so being stalled at T+10 costs (0 = off).
    efficiency_tail_seconds: float = 0.0
    # - approach on starboard: per second on starboard before the legal start, starboard_bonus x w,
    #   with w rising linearly from 0 at starboard_bonus_from to 1 at the gun (and 1 after it until
    #   the start). Which tack you are on matters more and more as the gun approaches.
    starboard_bonus: float = 0.0
    starboard_bonus_from: float = -60.0

    @property
    def n_steps(self) -> int:
        """Decisions per episode (70 for solo: T-60 .. T+10)."""
        return int(round((self.t_end - self.t_start) / self.dt))

    @property
    def line_length(self) -> float:
        """line_length_factor x line_fleet_size x boat length, at least 3 boat lengths."""
        return max(self.line_length_factor * self.line_fleet_size * self.boat_length, 3.0 * self.boat_length)
