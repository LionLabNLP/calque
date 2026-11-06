import argparse
import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metric_definitions import (
    BASE_RESULT_COLUMNS,
    DEFAULT_METRIC,
    METRIC_Y_LABELS,
    SETTING_ORDER,
    setting_label,
)

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
else:
    print(f"Font not found: {FONT_PATH}. Non-Latin text in plots may not render correctly.")

plt.rcParams["axes.unicode_minus"] = False


def load_all_results(outputs_dir, model_name=None):
    outputs_dir = Path(outputs_dir)
    pattern = f"{model_name}/*.xlsx" if model_name else "*/*.xlsx"
    files = sorted(outputs_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No result xlsx files found under {outputs_dir} ({pattern})")

    frames = [pd.read_excel(f) for f in files]
    all_df = pd.concat(frames, ignore_index=True)

    missing = [c for c in BASE_RESULT_COLUMNS if c not in all_df.columns]
    if missing:
        raise ValueError(f"Result files are missing expected column(s): {missing}")

    return all_df


def plot_group(group_df, title, out_path, metric):
    group_df = group_df.copy()
    group_df["setting"] = [
        setting_label(pl, plang)
        for pl, plang in zip(group_df["prompt_length"], group_df["prompt_language"])
    ]

    present = set(group_df["setting"])
    categories = [c for c in SETTING_ORDER if c in present]
    categories += sorted(present - set(categories))

    fig, ax = plt.subplots(figsize=(max(6, len(categories) * 1.3), 5))
    rng = np.random.default_rng(0)

    for i, cat in enumerate(categories):
        ys = group_df.loc[group_df["setting"] == cat, metric].to_numpy(dtype=float)
        ys = ys[~np.isnan(ys)]
        if ys.size == 0:
            continue

        xs = np.full(ys.shape, i, dtype=float) + rng.uniform(-0.08, 0.08, size=ys.shape)
        ax.scatter(xs, ys, alpha=0.6, color="tab:blue", zorder=2, label="data point" if i == 0 else None)

        median = np.median(ys)
        ax.hlines(median, i - 0.2, i + 0.2, color="tab:red", linewidth=2.5, zorder=3,
                   label="median" if i == 0 else None)

        mean = np.mean(ys)
        ax.scatter(i, mean, marker="D", s=70, color="tab:green", edgecolor="black", zorder=4,
                   label="mean" if i == 0 else None)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel(METRIC_Y_LABELS.get(metric, metric))
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, default="./outputs")
    parser.add_argument("--plots_dir", type=str, default="./outputs/plots")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        choices=sorted(METRIC_Y_LABELS.keys()),
        help="Which positive-vs-negative metric to plot.",
    )
    args = parser.parse_args()

    all_df = load_all_results(args.outputs_dir, args.model_name)

    if args.metric not in all_df.columns:
        raise ValueError(
            f"Requested --metric {args.metric!r} is not present in the loaded result "
            f"file(s); available columns include: {sorted(all_df.columns)}"
        )

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    group_cols = ["model_name", "lang", "construction_lang", "pos_strategy", "neg_strategy"]

    n_plots = 0
    for keys, group_df in all_df.groupby(group_cols):
        model_name_g, lang, construction_lang, pos_strategy, neg_strategy = keys

        title = f"{model_name_g}: {lang} ({construction_lang}) — {pos_strategy} vs {neg_strategy} [{args.metric}]"
        out_name = f"{lang}_{construction_lang}_{pos_strategy}_vs_{neg_strategy}_{args.metric}.png".replace(" ", "_")
        out_path = plots_dir / str(model_name_g) / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        plot_group(group_df, title, out_path, args.metric)
        n_plots += 1
        print(f"Saved plot to {out_path}")

    print(f"Saved {n_plots} plot(s) to {plots_dir}")


if __name__ == "__main__":
    main()
