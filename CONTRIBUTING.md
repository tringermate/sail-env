# Contributing

Issues and pull requests are welcome. Bug reports, rule or physics corrections from sailors, new
agents and new tasks are all useful.

## Development setup

```sh
pip install -e . pytest
PYTHONPATH=. pytest -q              # the full suite takes a few minutes on a laptop CPU
```

## Guidelines

- **Units:** keep the simulator in SI units. Knots and degrees belong only at the edges (`kn()`, `deg()`).
- **New numbers:** every new number is an `EnvParams` field with a comment saying what it means.
- **Pure functions:** environment code must stay jit- and vmap-compatible. That means no Python
  control flow on traced values, and fixed shapes.
- **Tests:** a change to dynamics or rules needs a test that pins the new behaviour down
  (see `tests/` for the style).
- **Browser version:** if you change the dynamics, rules or agents, update `web/sim.js` to match. Then run:

  ```sh
  python tools/export_web_data.py --traces web/ref/traces.json
  node web/check.js web/ref/traces.json
  node web/build.js
  ```

  CI runs the same parity check.
- **Baselines:** if a change moves the baseline numbers, re-run `python tools/run_baselines.py`
  and update the README tables.
