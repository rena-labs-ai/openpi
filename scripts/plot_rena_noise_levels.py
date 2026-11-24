"""eval_rena_noise_levels.py's scores for one or more runs, as a figure and a table. Per run: the eps-prediction error
against the noise level on train and on holdout frames, one curve per checkpoint ("Beyond Validation Loss" figure
11 a, b), and each checkpoint's error over the run's first checkpoint's at the same t, on the same frames and noise.
Memorization shows as the holdout ratio rising above 1 at small t while the train ratio falls. A ratio at one t is
the same in every unit (v, eps or clean action), and its interval is a paired bootstrap over frames.

    uv run scripts/plot_rena_noise_levels.py noise_levels/v3 noise_levels/v3_boxes --out noise_levels.png
"""

import pathlib

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

LOW_T = 0.1  # the low-noise end: the sampler's last call is at t = 0.1 of 10 steps


def load_run(run: pathlib.Path) -> dict[int, dict[str, np.ndarray]]:
    return {int(f.stem): dict(np.load(f)) for f in sorted(run.glob("*.npz"), key=lambda f: int(f.stem))}


def ratio(later: np.ndarray, first: np.ndarray, rng: np.random.Generator, n: int = 2000) -> np.ndarray:
    """[3, times]: mean(later) / mean(first) per t, and its 95% paired-bootstrap interval; later, first [frames, times]."""
    picks = rng.integers(0, len(first), (n, len(first)))
    boot = later[picks].mean(1) / first[picks].mean(1)
    return np.stack([later.mean(0) / first.mean(0), *np.percentile(boot, [2.5, 97.5], axis=0)])


def main(runs: tyro.conf.Positional[list[pathlib.Path]], out: pathlib.Path = pathlib.Path("noise_levels.png")) -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(runs), 3, figsize=(15, 4.2 * len(runs)), squeeze=False)
    for row, run in zip(axes, runs, strict=True):
        steps = load_run(run)
        first = min(steps)
        times = steps[first]["times"]
        noisy = times < 1  # the eps error is 0 at t = 1
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(steps)))
        print(f"\n{run.name}: holdout and train error over step {first}'s, at t <= {LOW_T} (95% interval)")
        print("  step   " + "  ".join(f"t={t:<22}" for t in times[times <= LOW_T]))
        for color, (step, got) in zip(colors, steps.items(), strict=True):
            for ax, split in zip(row[:2], ("train", "holdout"), strict=True):
                eps = (1 - times) ** 2 * got[f"{split}_v"].mean(0)
                ax.plot(times[noisy], eps[noisy], color=color, marker="o", ms=3, label=f"step {step}")
            if step == first:
                continue
            for split, style in (("holdout", "-"), ("train", "--")):
                r = ratio(got[f"{split}_v"], steps[first][f"{split}_v"], rng)
                row[2].plot(times, r[0], style, color=color, label=f"{split} {step}")
                if split == "holdout":
                    row[2].fill_between(times, r[1], r[2], color=color, alpha=0.15, lw=0)
            cells = []
            for i in np.flatnonzero(times <= LOW_T):
                h = ratio(got["holdout_v"][:, [i]], steps[first]["holdout_v"][:, [i]], rng)[:, 0]
                tr = got["train_v"][:, i].mean() / steps[first]["train_v"][:, i].mean()
                cells.append(f"{h[0]:.2f} [{h[1]:.2f},{h[2]:.2f}] / {tr:.2f}")
            print(f"  {step:<6} " + "  ".join(f"{c:<24}" for c in cells))
        for ax, split in zip(row[:2], ("train", "holdout"), strict=True):
            ax.set(xscale="log", yscale="log", xlabel="noise level t (1 = pure noise)", ylabel="eps-prediction MSE")
            ax.set_title(f"{run.name}: {split} frames")
            ax.legend(fontsize=7)
        row[2].axhline(1, color="grey", lw=0.8)
        row[2].set(xscale="log", xlabel="noise level t", ylabel=f"error / step {first}'s at the same t")
        row[2].set_title(f"{run.name}: over step {first}, holdout solid, train dashed")
        row[2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\n-> {out}")


if __name__ == "__main__":
    tyro.cli(main)
