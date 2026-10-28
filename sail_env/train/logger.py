"""Training logger: console summary always, Weights & Biases optionally.

W&B gets one run per experiment, with the seed mean under "mean/<metric>" and each seed under
"seed<i>/<metric>", so the dashboard shows both the average curve and the spread.
"""

import time

import numpy as np


class Logger:
    def __init__(self, name: str, config: dict, use_wandb: bool, print_every: int = 10, project: str = "sail_env"):
        self.print_every = print_every
        self.t0 = time.perf_counter()
        self.run = None
        if use_wandb:
            import wandb  # optional dependency: python -m uv sync --extra wandb

            self.run = wandb.init(project=project, name=name, config=config, group=config.get("env"))

    def log(self, update: int, env_steps: int, metrics: dict):
        # Episode metrics are NaN for seeds that finished no episode in this update: skip those.
        row = {f"mean/{k}": float(np.nanmean(v)) for k, v in metrics.items() if not np.all(np.isnan(v))}
        if self.run is not None:
            for k, v in metrics.items():
                for i, x in enumerate(np.asarray(v)):
                    if not np.isnan(x):
                        row[f"seed{i}/{k}"] = float(x)
            self.run.log({"env_steps": env_steps, **row}, step=update)
        if update % self.print_every == 0:
            elapsed = time.perf_counter() - self.t0
            extra = f"  success {row['mean/success_rate']:.2f}" if "mean/success_rate" in row else ""
            ret = row.get("mean/episode_return", float("nan"))
            print(
                f"update {update:5d}  steps {env_steps:>9,d}  return {ret:8.2f}{extra}"
                f"  entropy {row['mean/entropy']:.3f}  kl {row['mean/approx_kl']:.4f}"
                f"  clip {row['mean/clip_frac']:.3f}  ev {row['mean/explained_variance']:+.2f}  [{elapsed:5.0f}s]",
                flush=True,
            )

    def log_gif(self, update: int, path, caption: str = ""):
        if self.run is not None:
            import wandb

            self.run.log({"episode_gif": wandb.Video(str(path), format="gif", caption=caption)}, step=update)

    def finish(self):
        if self.run is not None:
            self.run.finish()
