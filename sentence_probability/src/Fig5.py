from __future__ import annotations


import argparse

from contextlib import contextmanager

import textwrap

from pathlib import Path


import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt

import matplotlib.ticker

import numpy as np

import pandas as pd

from matplotlib import font_manager

from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = PROJECT_ROOT / "plots2" / "main_fig"

DEFAULT_OUT_DIR = PROJECT_ROOT / "Main_plot"

FIGURE = "main_regression_delta-SLOR_coefficients_standardized.pdf"

COEFFICIENTS_CSV = "main_regression_coefficients_with_fixed_effects.csv"

SUMMARY_CSV = "main_regression_family_summary.csv"

OUTCOME = "SLOR_token_logprob_diff"


TERMS = [

    ("log10(model_size_b)", "log10(model size, B)"),

    ("log10_common_crawl_size", "Pretraining Language Proportion"),

    ("instruction_tuned", "Instruction-tuned"),

    ("strong_preference", "Human preference"),

    ("divergence", "Strategy Distance"),

    ("other_language_strategy_divergence", "Other-Language Strategy Divergence"),

    ("word_order_divergence", "Word-order divergence"),

    ("non11_alignment", "Non-1:1 Alignment Rate"),

    ("frequency_contrast", "Corpus-frequency contrast"),

]


FAMILIES = [

    ("aya", "Aya", "#ffb000", "o"),

    ("eurollm", "EuroLLM", "#fe6100", "s"),

    ("gemma", "Gemma 4", "#dc267f", "^"),

    ("apertus", "Apertus", "#648fff", "D"),

]


MARKER_DIAMETER = 19.0

MARKER_AREA_FACTOR = {"o": 1.0, "s": 0.892, "^": 1.262, "D": 0.884}


PREDICTOR_SIZE, TICK_SIZE, AXIS_LABEL_SIZE, LEGEND_SIZE = 38, 36, 40, 36

LABEL_WRAP = 20

LEGEND_NCOL = 2

TERM_HEIGHT = 2.05

FAMILY_SPREAD = 0.32

CI_WIDTH = 4.0


CAP_SIZE, CAP_THICK = 9, 3.0

X_LABEL = "Standardized coefficient (β)"

TEXT_COLOR, GRID_COLOR = "#0b0b0b", "#e4e3df"

SPINE_WIDTH = 1.6

ZERO_LINE_WIDTH = 2.2


FACTOR_SEPARATOR_STYLE = {"color": "#B5B5B5", "linewidth": 0.8, "linestyle": "-", "zorder": 0.5}


FONT_CANDIDATES = ["TeX Gyre Termes", "Times New Roman", "Times", "Nimbus Roman", "Nimbus Roman No9 L",

                   "STIX Two Text", "STIXGeneral"]


def _paper_font():

    installed = {f.name for f in font_manager.fontManager.ttflist}

    return next((n for n in FONT_CANDIDATES if n in installed), "DejaVu Serif")


STYLE = {

    "pdf.fonttype": 42, "ps.fonttype": 42,

    "font.family": "serif", "font.serif": [_paper_font()] + FONT_CANDIDATES + ["DejaVu Serif"],

    "mathtext.fontset": "stix", "axes.unicode_minus": True,

    "axes.edgecolor": "black", "axes.labelcolor": TEXT_COLOR,

    "xtick.color": "black", "ytick.color": "black", "xtick.labelcolor": TEXT_COLOR, "ytick.labelcolor": TEXT_COLOR,

    "axes.spines.top": False, "axes.spines.right": False,

}


@contextmanager

def figure_style():


    with plt.rc_context():

        matplotlib.rcdefaults()

        plt.rcParams.update(STYLE)

        yield


def load_results(input_dir=DEFAULT_INPUT_DIR):


    input_dir = Path(input_dir)

    paths = [input_dir / COEFFICIENTS_CSV, input_dir / SUMMARY_CSV]

    for p in paths:

        if not p.is_file():

            raise FileNotFoundError(f"Fig5: {p} not found (run export_plots2.py --only_main first)")

    coefs, summary = (pd.read_csv(p) for p in paths)

    if set(coefs["outcome"]) != {OUTCOME} or set(summary["outcome"]) != {OUTCOME}:

        raise ValueError("Fig5: regression results are not for the saved ΔSLOR outcome")

    expected = {f for f, _l, _c, _m in FAMILIES}

    if set(summary["family"]) != expected or not set(coefs["family"]) <= expected:

        raise ValueError(f"Fig5: unexpected regression families {sorted(set(summary['family']))}")

    beta = coefs["beta"].astype(float)

    scale = (coefs["standardized_beta"].astype(float) / beta).where(beta != 0)

    for col, raw in (("standardized_ci_lower", "ci_lower"), ("standardized_ci_upper", "ci_upper")):

        if not np.allclose(coefs[col], coefs[raw] * scale, equal_nan=True, rtol=1e-9, atol=1e-12):

            raise ValueError(f"Fig5: {col} is not the rescaled raw {raw}")

    return coefs, summary


def _marker_size(marker):

    return MARKER_DIAMETER * MARKER_AREA_FACTOR.get(marker, 1.0)


def plot_coefficients(coefs, summary, out_path):


    present = set(coefs["term"])

    terms = [(t, l) for t, l in TERMS if t in present]

    fams = [f for f in FAMILIES if f[0] in set(coefs["family"]) | set(summary["family"])]

    if not terms or not fams:

        raise ValueError("Fig5: nothing to draw")

    rows = coefs.loc[coefs["term"].isin([t for t, _l in terms])]

    xmax = np.nanmax(np.abs(rows[["standardized_ci_lower", "standardized_ci_upper"]].to_numpy(dtype=float)))

    xmax = float(xmax) if np.isfinite(xmax) and xmax > 0 else 1.0

    offsets = np.linspace(-FAMILY_SPREAD, FAMILY_SPREAD, len(fams)) if len(fams) > 1 else [0.0]

    n_by = summary.set_index("family")["n"].to_dict()

    status_by = summary.set_index("family")["status"].to_dict()

    with figure_style():

        fig, ax = plt.subplots(figsize=(14.0, 0.8 * (TERM_HEIGHT * len(terms) + 2.6)))

        for fi, (fam, _label, color, marker) in enumerate(fams):

            sub = coefs.loc[coefs["family"] == fam].set_index("term")

            for ti, (term, _l) in enumerate(terms):

                b = float(sub["standardized_beta"].get(term, np.nan)) if term in sub.index else np.nan

                if not np.isfinite(b):

                    continue

                y = ti + offsets[fi]

                lo, hi = float(sub.loc[term, "standardized_ci_lower"]), float(sub.loc[term, "standardized_ci_upper"])

                ax.errorbar([b], [y], xerr=[[b - lo], [hi - b]], fmt="none", ecolor=color, capsize=CAP_SIZE,

                            linewidth=CI_WIDTH, capthick=CAP_THICK, zorder=3)

                ax.plot([b], [y], marker=marker, markersize=_marker_size(marker), color=color,

                        markeredgecolor="none", markeredgewidth=0, linestyle="none", zorder=4)

        ax.axvline(0, color=TEXT_COLOR, linestyle="-", linewidth=ZERO_LINE_WIDTH, zorder=1)

        for i in range(len(terms) - 1):

            ax.axhline(i + 0.5, **FACTOR_SEPARATOR_STYLE)

        ax.set_yticks(range(len(terms)))

        ax.set_yticklabels([textwrap.fill(l, LABEL_WRAP) for _t, l in terms], fontsize=PREDICTOR_SIZE,

                           linespacing=0.95)

        ax.set_ylim(len(terms) - 0.5, -0.5)

        ax.set_xlim(-1.08 * xmax, 1.08 * xmax)

        ax.tick_params(axis="both", labelcolor=TEXT_COLOR)

        ax.tick_params(axis="x", labelsize=TICK_SIZE)

        ax.tick_params(axis="both", length=10, width=2.0, pad=8)

        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, symmetric=True))

        ax.set_xlabel(X_LABEL, fontsize=AXIS_LABEL_SIZE, labelpad=10)

        for side in ("left", "bottom"):

            ax.spines[side].set_color("black")

            ax.spines[side].set_linewidth(SPINE_WIDTH)

        ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.6)

        ax.set_axisbelow(True)


        def legend_label(fam, label):

            status = str(status_by.get(fam, "not fitted"))

            if status != "ok":

                return f"{label} (not estimated: {status.split(' (')[0]})"

            return f"{label} (N={int(n_by.get(fam, 0)):,})"


        handles = [Line2D([0], [0], marker=m, color=c, markeredgecolor="none", markeredgewidth=0,

                          markersize=_marker_size(m), linewidth=CI_WIDTH, label=legend_label(f, l))

                   for f, l, c, m in fams]

        fig.tight_layout()


        n_rows = -(-len(handles) // LEGEND_NCOL)

        handles = [handles[r * LEGEND_NCOL + c] for c in range(LEGEND_NCOL) for r in range(n_rows)

                   if r * LEGEND_NCOL + c < len(handles)]

        fig.legend(handles=handles, frameon=False, fontsize=LEGEND_SIZE, loc="lower center",

                   bbox_to_anchor=(0.5, 1.0), ncol=LEGEND_NCOL, handletextpad=0.4, columnspacing=1.2,

                   handlelength=1.6)

        fig.savefig(out_path, format="pdf", bbox_inches="tight")

        plt.close(fig)

    return True


def generate(input_dir=DEFAULT_INPUT_DIR, out_dir=DEFAULT_OUT_DIR):

    coefs, summary = load_results(input_dir)

    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / FIGURE

    plot_coefficients(coefs, summary, path)

    print(f"[Fig5] {path} (from {Path(input_dir) / COEFFICIENTS_CSV})")

    return path


def main(argv=None):

    parser = argparse.ArgumentParser(description="Figure 5: standardized regression coefficients (standalone).")

    parser.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR),

                        help="directory with the saved main_regression_*.csv (default plots2/main_fig)")

    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="figure directory (default Main_plot/)")

    args = parser.parse_args(argv)

    generate(args.input_dir, args.out_dir)


if __name__ == "__main__":

    main()
