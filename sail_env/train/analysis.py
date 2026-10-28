"""Seed statistics and training-curve plots.

With few seeds (n = 5) the sample mean's 95% confidence interval uses Student's t distribution:
    mean +- t_{0.975, n-1} * std / sqrt(n)        (std with ddof = 1)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Two-sided 95% critical values t_{0.975, df}; beyond df = 30 the normal value 1.96 is close enough.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
         10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def t975(df: int) -> float:
    if df <= 0:
        return float("nan")
    keys = [k for k in sorted(_T975) if k <= df]
    return _T975[keys[-1]] if df <= 30 else 1.96  # conservative: round df down to a table entry


def mean_ci(x: np.ndarray, axis: int = 0):
    """(mean, half_width) of the 95% CI over `axis` (the seed axis)."""
    n = x.shape[axis]
    mean = x.mean(axis=axis)
    if n < 2:
        return mean, np.zeros_like(mean)
    return mean, t975(n - 1) * x.std(axis=axis, ddof=1) / np.sqrt(n)


def ffill(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along the last axis (updates with no finished episode keep the last value)."""
    x = np.array(x, dtype=float)
    for row in x.reshape(-1, x.shape[-1]):
        last = np.nan
        for i, v in enumerate(row):
            if np.isnan(v):
                row[i] = last
            else:
                last = v
    return x


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average along the last axis (window 1 = no smoothing)."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    pad = np.concatenate([np.repeat(x[..., :1], window - 1, axis=-1), x], axis=-1)
    return np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), -1, pad)


def plot_curves(metrics: dict, steps: np.ndarray, keys, path, title="", window=5):
    """One panel per metric: mean over seeds with a 95% CI band. metrics[k] has shape (seeds, updates)."""
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.4), squeeze=False)
    n = next(iter(metrics.values())).shape[0]
    for ax, k in zip(axes[0], keys):
        m, h = mean_ci(smooth(ffill(metrics[k]), window))
        ax.plot(steps, m, lw=1.5)
        ax.fill_between(steps, m - h, m + h, alpha=0.25)
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("env steps per seed")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{title}  (mean, 95% CI over {n} seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
