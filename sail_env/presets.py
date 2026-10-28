"""Named environment settings.

The standard presets are the settings the bundled pretrained policies were trained with, so agents
evaluated (or humans playing) on them face the same task as the reference results:

    solo   one boat, T-60 to T+10, 130 m line (sized for 15 boats), steady wind, square line.
    fleet  five boats, T-90 to T+10, 55 m line (1.6 x 5 x boat length), wind shifts up to +-5 deg,
           line bias up to +-10 deg, wind shadow along the apparent wind, contact and mark rules,
           penalties as 360-degree turns, zero-sum ranking at T+10.

    race       the full race: the fleet preset's start, then a windward-leeward course (beat 400 m, 2 laps,
               leeward gate, finish = start line downwind), up to T+1200 s. Reward: +1 for finishing and a
               zero-sum ranking by finishing order (+1 winner ... -1 last); contact and penalty costs as in
               the fleet; no reward for the start itself.
    race-solo  the same course alone (learn to sail the course: VMG, laylines, roundings).

    <name>-weather  any of the above with the full weather of weather.py: shifts and pressure that move
                    down the course, current and waves, and the weather observations (22 more features).
                    Switch parts off with overrides, e.g. make("race-weather", waves=False).

The start presets use the full rules: a good start = bow 0-3 m below the line and between the ends at the gun,
then crossing within 3 s at >= 90% of polar speed, on starboard (fleet: also clear air and no
unserved penalty). Episodes start at the preset's T-60 / T-90 (random_start_time=False).
"""

from pathlib import Path

import yaml

from sail_env.config import EnvParams

PRETRAINED = Path(__file__).resolve().parent / "pretrained"


def _from_pretrained(name: str) -> dict:
    cfg = yaml.safe_load((PRETRAINED / name / "config.yaml").read_text(encoding="utf-8"))
    env_params = dict(cfg.get("env_params") or {})
    env_params["random_start_time"] = False  # standard evaluation: always the full sequence
    return env_params


RACE_OVERRIDES = {
    "t_end": 1200.0,  # time limit, s after the gun (light-wind races take ~15 min)
    "start_reward_weight": 0.0,  # the start is not rewarded for itself in the race ...
    "rank_reward": 0.0,  # ... nor ranked at T+10
    "upwind_bonus": 0.0,
    "port_penalty": 0.0,
    "ocs_penalty": 0.0,  # an OCS boat simply cannot progress until it has restarted
    "race_finish_reward": 1.0,
    "race_rank_reward": 2.0,
    "penalty_clear_distance": 2.0 * 6.9,  # RRS 44.2: get clear (2 boat lengths) before turning
}

WEATHER_OVERRIDES = {
    "weather_shifts": True,
    "weather_pressure": True,
    "water_current": True,
    "waves": True,
    "observe_weather": True,
}

PRESETS = {
    "solo": lambda: _from_pretrained("solo"),
    "fleet": lambda: _from_pretrained("fleet"),
    "race": lambda: {**_from_pretrained("fleet"), **RACE_OVERRIDES},
    "race-solo": lambda: {**_from_pretrained("fleet"), **RACE_OVERRIDES, "fleet_size": 1},
}
PRESETS.update({f"{name}-weather": (lambda base=base: {**base(), **WEATHER_OVERRIDES}) for name, base in list(PRESETS.items())})
RACE_PRESETS = {"race", "race-solo", "race-weather", "race-solo-weather"}


def make(name: str = "fleet", **overrides):
    """(env, params) for a preset; keyword arguments override EnvParams fields."""
    from sail_env.fleet_env import FleetStartEnv
    from sail_env.race_env import RaceEnv
    from sail_env.start_env import SoloStartEnv

    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    params = EnvParams(**{**PRESETS[name](), **overrides})
    if name in RACE_PRESETS:
        return RaceEnv(params), params
    env = FleetStartEnv(params) if params.fleet_size > 1 else SoloStartEnv(params)
    return env, params
