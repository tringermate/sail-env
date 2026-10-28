"""Staged rules: start training with easy success rules, then tighten them to the full rules.

    progress in [0, 1]  (fraction of training done)
    w = 0                      while progress <= easy_until     (easy rules)
    w rises linearly to 1      until progress  = full_from      (tightening)
    w = 1                      afterwards                       (full rules)
    value = (1 - w) * easy_value + w * full_value

The full values are the environment's own parameters; `easy` gives the relaxed starting values,
e.g. a 10 s success window, no speed requirement, no OCS penalty and a speed bonus that fades out.
Only non-static EnvParams fields can be scheduled (they change inside the compiled training loop).
"""

import jax.numpy as jnp


def make_staged_params(full_params, easy: dict, easy_until: float = 0.2, full_from: float = 0.7):
    """Returns env_params_fn(progress) -> (params, {field: current value}) for make_train."""
    assert 0.0 <= easy_until < full_from <= 1.0
    fields = {k: (float(v), float(getattr(full_params, k))) for k, v in easy.items()}

    def env_params_fn(progress):
        w = jnp.clip((progress - easy_until) / (full_from - easy_until), 0.0, 1.0)
        values = {k: (1.0 - w) * e + w * f for k, (e, f) in fields.items()}
        return full_params.replace(**values), values

    return env_params_fn
