from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from paper_inputs import DEFAULT_MAIN_PLOT_DIR, DEFAULT_OUTPUTS_DIR, load_probability_outputs


FIGURE_A = "violin_main_a__no-mean-median.pdf"
FIGURE_B = "violin_main_b__no-sv-no-de__manaca-1b__no-mean-median.pdf"
FIGURE_JA = "delta-SLOR__japanese-control-languages__paired-ja-baseline__pref-gtgt__no-mean-median.pdf"
FIGURE_COMBINED = "violin_main_combined__no-mean-median.pdf"
FIGURES = {"a": FIGURE_A, "b": FIGURE_B, "combined": FIGURE_COMBINED, "ja": FIGURE_JA}

DEFAULT_LANGUAGES = ("de", "fr", "pt", "sv", "zh", "ja")
LANGUAGE_AXIS_ORDER = ["de", "sv", "fr", "pt", "zh", "ja"]
PANEL_B_REMOVED_LANGUAGES = ("sv",)
PREFERENCE = ">>"  # ここでどれを残すかをきめてる

SHOW_MEANS = False
SHOW_MEDIANS = False
CONNECT_MEANS = True

YELLOW, ORANGE, MAGENTA, PURPLE, BLUE = "#ffb000", "#fe6100", "#dc267f", "#785ef0", "#648fff"
MODEL_COLORS = {
    "utter-project_EuroLLM-22B-2512": ORANGE,
    "google_gemma-4-31B": MAGENTA,
    "CohereLabs_aya-expanse-32b": YELLOW,
    "swiss-ai_Apertus-70B-2509": BLUE,
    "OpenLLM-France_Lucie-7B": BLUE,
    "AIInstituteLNCC_manaca-1b-base": YELLOW,
    "hfl_chinese-llama-2-13b": MAGENTA,
    "llm-jp_llm-jp-3-13b": PURPLE,
    "Aleph-Alpha_Kolibri-1-BF16": ORANGE,
}
NON_FOCUS_GRAY = "#BDBDBD"

MULTILINGUAL_MODELS = [
    ("utter-project_EuroLLM-22B-2512", "EuroLLM 22B"),
    ("google_gemma-4-31B", "Gemma 4 31B"),
    ("CohereLabs_aya-expanse-32b", "Aya Expanse 32B"),
    ("swiss-ai_Apertus-70B-2509", "Apertus 70B"),
]
NON_ENGLISH_CENTRIC_MODELS = [
    ("Aleph-Alpha_Kolibri-1-BF16", "Kolibri"),
    ("OpenLLM-France_Lucie-7B", "Lucie 7B"),
    ("AIInstituteLNCC_manaca-1b-base", "Manacá 1B"),
    ("hfl_chinese-llama-2-13b", "Chinese-LLaMA-2 13B"),
    ("llm-jp_llm-jp-3-13b", "LLM-jp-3 13B"),
]
JAPANESE_CONTROL_MODELS = MULTILINGUAL_MODELS

BALANCED, CONTINUAL, MONOLINGUAL = "Balanced Language Models", "Continually Pretrained Model", "Monolingual Model"
TRAINING_KINDS = (BALANCED, CONTINUAL, MONOLINGUAL)
TRAINING = {
    "OpenLLM-France_Lucie-7B": (BALANCED, "En- and Fr-focused"),
    "AIInstituteLNCC_manaca-1b-base": (MONOLINGUAL, "Pt-focused"),
    "hfl_chinese-llama-2-13b": (CONTINUAL, "Zh-focused"),
    "llm-jp_llm-jp-3-13b": (BALANCED, "En- and Ja-focused"),
    "Aleph-Alpha_Kolibri-1-BF16": (BALANCED, "En- and De-focused"),
}
FOCUS_LANGUAGES = {
    "OpenLLM-France_Lucie-7B": ("fr",),
    "AIInstituteLNCC_manaca-1b-base": ("pt",),
    "hfl_chinese-llama-2-13b": ("zh",),
    "llm-jp_llm-jp-3-13b": ("ja",),
    "Aleph-Alpha_Kolibri-1-BF16": ("de", "en"),
}

FIG_WIDTH, AXES_HEIGHT = 7.5, 3.3
TICK_SIZE, LABEL_SIZE, LEGEND_SIZE = 16, 18, 11
Y_PAD_FRACTION = 0.05
LEGEND_MARGIN = 0.03
SLOT_TOTAL, SLOT_MAX, BODY_FRACTION = 0.84, 0.2, 0.9

TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
SPINE_COLOR = "black"
SPINE_WIDTH = 1.3
ZERO_LINE_COLOR, ZERO_LINE_WIDTH = "#202020", 2.4
SEPARATOR_COLOR = "#B5B5B5"
SEPARATOR_WIDTH = 0.63
ENGLISH_BOUNDARY_COLOR, ENGLISH_BOUNDARY_WIDTH = "#555555", 0.91
PAIR_GROUP_SEPARATOR_WIDTH = 0.88
WITHIN_PAIR_SEPARATOR_WIDTH = 0.5
MEAN_MARKER = {"marker": "D", "s": 18, "facecolor": "white", "edgecolor": TEXT_COLOR, "linewidth": 0.9}
MEDIAN_WIDTH = 2.0
MEAN_LINK_STYLE = {"color": "#4D4D4D", "linewidth": 0.7, "linestyle": "-", "solid_capstyle": "round"}
Z_SEPARATOR, Z_ZERO, Z_LINK, Z_MEDIAN, Z_MEAN = 0.5, 1.5, 3.5, 3.8, 4.0

for _w in (SEPARATOR_WIDTH, ENGLISH_BOUNDARY_WIDTH, PAIR_GROUP_SEPARATOR_WIDTH, WITHIN_PAIR_SEPARATOR_WIDTH):
    assert _w < SPINE_WIDTH, "category separators must be thinner than the axis spines"

FONT_CANDIDATES = ["TeX Gyre Termes", "Times New Roman", "Times", "Nimbus Roman", "Nimbus Roman No9 L",
                   "STIX Two Text", "STIXGeneral"]


def _paper_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in FONT_CANDIDATES if n in installed), "DejaVu Serif")


STYLE = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": [_paper_font()] + FONT_CANDIDATES + ["DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.unicode_minus": True,
    "axes.edgecolor": SPINE_COLOR, "axes.labelcolor": TEXT_COLOR,
    "xtick.color": SPINE_COLOR, "ytick.color": SPINE_COLOR,
    "xtick.labelcolor": TEXT_COLOR, "ytick.labelcolor": TEXT_COLOR,
    "axes.spines.top": False, "axes.spines.right": False,
}


@contextmanager
def figure_style():
    with plt.rc_context():
        matplotlib.rcdefaults()
        plt.rcParams.update(STYLE)
        yield


SLOR_COL = "SLOR_token_logprob_diff"
VALUE = "delta_SLOR"
MEANLP_COL = "token_logprob_diff"
TARGET_POS = "Proper strategy in target language"
ENGLISH_POS = "Proper English strategy"
JAPANESE_POS = "Proper strategy in Japanese"
JA_BASELINE_NEG = "Literal translation of Proper English into target language"
TARGET_PREF = "Target language preference annotation"
ENGLISH_PREF = "English preference annotation"
JAPANESE_PREF = "Japanese preference annotation"
CONTENT = "Proper English strategy"
EN = "en"
JA_CONTROL, JA_BASELINE = "JA control", "JA baseline"


def _stripped(series):
    return series.where(series.notna(), "").astype(str).str.strip()


def _canonical(code):
    return "sv" if str(code).strip() == "se" else str(code).strip()


def _scored(rows, include_identical, metric_col=SLOR_COL):
    if not include_identical and {"pos_sentence", "neg_sentence"} <= set(rows.columns):
        rows = rows.loc[rows["pos_sentence"] != rows["neg_sentence"]]
    if "prompt_language" in rows.columns:
        rows = rows.loc[rows["prompt_language"].astype(str) == "raw"]
    rows = rows.copy()
    rows[VALUE] = pd.to_numeric(rows[metric_col], errors="coerce") if metric_col in rows.columns else np.nan
    return rows.loc[rows[VALUE].notna()]


def language_order(values):
    present = {str(v) for v in values}
    rest = sorted(present - set(LANGUAGE_AXIS_ORDER) - {EN})
    return [c for c in LANGUAGE_AXIS_ORDER if c in present] + rest + ([EN] if EN in present else [])


def standard_violin_rows(df, langs, include_identical=False, metric_col=SLOR_COL):
    langs = {_canonical(l) for l in langs}
    rows = df.loc[df["experiment"] == "standard"]
    affected = (rows["pos"] == TARGET_POS) & rows["Target language"].map(_canonical).isin(langs)
    #ここでデータの>と>>だけ残してる。あとでもっと強い処理を加えてるよ
    strong = _stripped(rows[TARGET_PREF]).isin({">", ">>"}) if TARGET_PREF in rows else False
    rows = _scored(rows.loc[~affected | strong], include_identical, metric_col)
    target = rows.loc[rows["pos"] == TARGET_POS].copy()
    target["side_annotation"] = _stripped(target[TARGET_PREF]) if TARGET_PREF in target else ""
    english = rows.loc[rows["pos"] == ENGLISH_POS].copy()
    english["target_language_origin"] = english["Target language"]
    english["Target language"] = EN
    english["side_annotation"] = _stripped(english[ENGLISH_PREF]) if ENGLISH_PREF in english else ""
    out = pd.concat([target, english], ignore_index=True)
    out = out.loc[out["Target language"].isin(langs | {EN})]
    return out.loc[out["side_annotation"] == PREFERENCE]#>>のものだけ残す処理はここでやってる


def japanese_control_pairs(df, models, include_identical=False, metric_col=SLOR_COL):
    models = set(models)
    control = _scored(df.loc[(df["experiment"] == "japanese_control") & df["model_name"].isin(models)],
                      include_identical, metric_col)
    control = control.loc[(control["pos"] == JAPANESE_POS)
                          & (_stripped(control[JAPANESE_PREF]) == PREFERENCE if JAPANESE_PREF in control else False)]
    #ここでcontrolのpreferenceを>>だけにしてる
    control = control.loc[control[CONTENT].notna()].copy()
    control["original_language"] = control["Target language"].map(_canonical)
    control["content"] = control[CONTENT].astype(str)
    base = _scored(df.loc[(df["experiment"] == "standard") & df["model_name"].isin(models)], include_identical, metric_col)
    base = base.loc[(base["pos"] == TARGET_POS) & (base["neg"] == JA_BASELINE_NEG)
                    & (base["Target language"].map(_canonical) == "ja")]
    base = base.loc[_stripped(base[TARGET_PREF]) == PREFERENCE] if TARGET_PREF in base else base.iloc[0:0]
    #ここでbaseもpreferenceを>>だけにしてる
    base = base.loc[base[CONTENT].notna()].copy()
    base["content"] = base[CONTENT].astype(str)
    for rows, keys, side in ((control, ["original_language", "model_name", "content"], "Japanese-control"),
                             (base, ["model_name", "content"], "JA-baseline")):
        if rows.duplicated(keys).any():
            raise ValueError(f"{side} rows are not unique per {keys}: the pairing would not be one-to-one")
    matched = control.merge(base[["model_name", "content", VALUE]], on=["model_name", "content"],
                            suffixes=("", "_baseline"), validate="many_to_one")
    #semantic contentが一致するものだけ残してる
    keep = ["original_language", "model_name", "content"]
    long = pd.concat([matched[keep].assign(category=JA_CONTROL, **{VALUE: matched[VALUE]}),
                      matched[keep].assign(category=JA_BASELINE, **{VALUE: matched[f"{VALUE}_baseline"]})],
                     ignore_index=True)
    return long


def _violin(ax, values, pos, color, width, show_means=SHOW_MEANS):
    if values.size > 1 and np.ptp(values) > 0:
        parts = ax.violinplot([values], positions=[pos], widths=width, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(1.0)
            body.set_linewidth(0)
    mean = float(np.mean(values))
    if SHOW_MEDIANS:
        half = width * 0.25
        ax.plot([pos - half, pos + half], [np.median(values)] * 2, color=TEXT_COLOR, linewidth=MEDIAN_WIDTH,
                solid_capstyle="butt", zorder=Z_MEDIAN)
    if show_means:
        ax.scatter([pos], [mean], zorder=Z_MEAN, **MEAN_MARKER)
    return mean


def _separator(ax, x, color, width, linestyle="-"):
    ax.axvline(x, color=color, linewidth=width, linestyle=linestyle, zorder=Z_SEPARATOR)


def _style_axes(ax, y_label):
    ax.tick_params(axis="both", labelsize=TICK_SIZE, labelcolor=TEXT_COLOR, length=5, width=1.1)
    ax.tick_params(axis="x", labelsize=LABEL_SIZE, labelbottom=True)
    ax.set_ylabel(y_label, fontsize=LABEL_SIZE - 5, color=TEXT_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(SPINE_WIDTH)


def _y_limits(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return -1.0, 1.0
    lo, hi = min(float(values.min()), 0.0), max(float(values.max()), 0.0)
    pad = Y_PAD_FRACTION * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad


def _blank(label=""):
    return Patch(facecolor="none", edgecolor="none", label=label)


def _model_handle(model, label, missing, missing_text="no outputs"):
    if model in missing:
        return Patch(facecolor="white", edgecolor=MUTED_COLOR, linestyle="--", label=f"{label} ({missing_text})")
    return Patch(facecolor=MODEL_COLORS[model], edgecolor="none", label=label)


def _summary_handles(linked, show_means=SHOW_MEANS):
    handles = []
    if SHOW_MEDIANS:
        handles.append(Line2D([0], [0], color=TEXT_COLOR, linewidth=MEDIAN_WIDTH, label="median"))
    if show_means:
        handles.append(Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white",
                              markeredgecolor=TEXT_COLOR, markersize=5, label="mean"))
    if linked:
        handles.append(Line2D([0], [0], marker="D", markerfacecolor="white", markeredgecolor=TEXT_COLOR,
                              markersize=4, label="paired means", **{k: v for k, v in MEAN_LINK_STYLE.items()
                                                                     if k != "solid_capstyle"}))
    return handles


def _legend_entries(models, missing, grouped=False, linked=False, missing_text="no outputs"):
    summary = _summary_handles(linked)
    if grouped:
        kinds = [k for k in TRAINING_KINDS if any(TRAINING[m][0] == k for m, _l in models)]
        groups = [[_blank(k)] + [_model_handle(m, f"{l} ({TRAINING[m][1]})", missing, missing_text)
                                 for m, l in models if TRAINING[m][0] == k] for k in kinds]
        columns = groups[:2]
        for extra in groups[2:]:
            columns[-1] = columns[-1] + extra
        rows = max(len(c) for c in columns)
        free = sum(rows - len(c) for c in columns)
        if len(summary) > free:
            rows += 1
        pending = list(summary)
        for i, col in enumerate(columns):
            while len(col) < rows:
                col.append(pending.pop(0) if pending else _blank())
            columns[i] = col
        return [h for c in columns for h in c], len(columns), set(kinds)
    handles = [_model_handle(m, l, missing, missing_text) for m, l in models]
    rows = 3 if len(handles) > 4 else 2 if len(handles) > 2 else max(len(handles), 2)
    rows = max(rows, len(summary))
    ncol = -(-len(handles) // rows)
    handles += [_blank()] * (ncol * rows - len(handles))
    if not summary:
        return handles, ncol, set()
    return handles + summary + [_blank()] * (rows - len(summary)), ncol + 1, set()


def _place_legend(ax, handles, ncol, bold, above=False):
    legend = ax.legend(handles=handles, frameon=not above, framealpha=0.92, facecolor="white", edgecolor="none",
                       fontsize=LEGEND_SIZE, loc="lower left" if above else "upper left",
                       bbox_to_anchor=(0.0, 1.0), ncol=ncol,
                       columnspacing=0.6, handlelength=1.1, handleheight=0.8, handletextpad=0.35,
                       labelspacing=0.25, borderpad=0.25, borderaxespad=0.2)
    for text in legend.get_texts():
        if text.get_text() in bold:
            text.set_fontweight("bold")
    return legend


def _legend_headroom(ax, legend, category_values, lo, hi):
    fig = ax.figure
    fig.canvas.draw()
    box = legend.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
    x_lo, x_hi = ax.get_xlim()
    left, right = x_lo + box.x0 * (x_hi - x_lo), x_lo + box.x1 * (x_hi - x_lo)
    covered = [v for i, v in enumerate(category_values) if i + 0.5 > left and i - 0.5 < right]
    values = np.concatenate([np.asarray(v, dtype=float) for v in covered]) if covered else np.array([])
    values = values[np.isfinite(values)]
    frac = box.y0 - LEGEND_MARGIN
    if not values.size or frac <= 0.2:
        return hi
    return max(hi, lo + (float(values.max()) - lo) / frac)


def _per_model_n_label(rows, model_col):
    counts = rows.groupby(model_col).size()
    if counts.empty:
        return "N=0"
    low, high = int(counts.min()), int(counts.max())
    return f"{low}" if low == high else f"{low}–{high}"


def _model_offsets(models):
    n = max(len(models), 1)
    slot = min(SLOT_TOTAL / n, SLOT_MAX)
    return {m: (i - (n - 1) / 2) * slot for i, m in enumerate(models)}, slot * BODY_FRACTION


def _draw_language_axes(ax, rows, models, color_at, y_label):
    present = set(rows["model_name"])
    names = [m for m, _l in models]
    missing = [m for m in names if m not in present]
    drawn = [m for m in names if m in present]
    languages = language_order(rows["Target language"].unique())
    offsets, width = _model_offsets(drawn)
    for i in range(1, len(languages)):
        if (languages[i] == EN) != (languages[i - 1] == EN):
            _separator(ax, i - 0.5, ENGLISH_BOUNDARY_COLOR, ENGLISH_BOUNDARY_WIDTH)
        else:
            _separator(ax, i - 0.5, SEPARATOR_COLOR, SEPARATOR_WIDTH)
    means, per_category = {}, []
    for li, lang in enumerate(languages):
        sub = rows.loc[rows["Target language"] == lang]
        per_category.append(sub[VALUE].to_numpy(dtype=float))
        for m in drawn:
            values = sub.loc[sub["model_name"] == m, VALUE].to_numpy(dtype=float)
            if values.size:
                x = li + offsets[m]
                means[(lang, m)] = (x, _violin(ax, values, x, color_at(m, lang), width), int(values.size))
    ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=ZERO_LINE_WIDTH, zorder=Z_ZERO)
    ax.set_xticks(range(len(languages)))
    ax.set_xticklabels([
        f"{lang} ({_per_model_n_label(rows.loc[rows['Target language'] == lang], 'model_name')})"
        for lang in languages
    ])
    ax.set_xlim(-0.5, len(languages) - 0.5)
    _style_axes(ax, y_label)
    ax.tick_params(axis="x", labelsize=15)
    return means, per_category, missing


def plot_language_panel(rows, models, out_path, color_at, grouped=False, y_label="ΔSLOR"):
    with figure_style():
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, AXES_HEIGHT))
        means, per_category, missing = _draw_language_axes(ax, rows, models, color_at, y_label)
        lo, hi = _y_limits(rows[VALUE])
        ax.set_ylim(lo, hi)
        legend = _place_legend(ax, *_legend_entries(models, missing, grouped))
        ax.set_ylim(lo, _legend_headroom(ax, legend, per_category, lo, hi))
        fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    return means


def plot_combined_panels(rows_a, rows_b, out_path, y_label="ΔSLOR"):
    n_a, n_b = (len(language_order(r["Target language"].unique())) for r in (rows_a, rows_b))
    with figure_style():
        fig, (ax_a, ax_b) = plt.subplots(1, 2, sharey=True, figsize=(2 * FIG_WIDTH, AXES_HEIGHT),
                                         gridspec_kw={"width_ratios": [n_a, n_b], "wspace": 0.03})
        means_a, per_a, missing_a = _draw_language_axes(ax_a, rows_a, MULTILINGUAL_MODELS, _model_color, y_label)
        means_b, per_b, missing_b = _draw_language_axes(ax_b, rows_b, NON_ENGLISH_CENTRIC_MODELS, _focus_color, "")
        ax_b.tick_params(axis="y", left=False, labelleft=False)
        ax_b.spines["left"].set_visible(False)
        lo, hi = _y_limits(pd.concat([rows_a[VALUE], rows_b[VALUE]]))
        ax_a.set_ylim(lo, hi)
        legend_a = _place_legend(ax_a, *_legend_entries(MULTILINGUAL_MODELS, missing_a))
        legend_b = _place_legend(ax_b, *_legend_entries(NON_ENGLISH_CENTRIC_MODELS, missing_b, True))
        shared_hi = max(hi,
                        _legend_headroom(ax_a, legend_a, per_a, lo, hi),
                        _legend_headroom(ax_b, legend_b, per_b, lo, hi))
        ax_a.set_ylim(lo, shared_hi)
        fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    return {**{("a",) + k: v for k, v in means_a.items()}, **{("b",) + k: v for k, v in means_b.items()}}


def plot_japanese_control(pairs, models, out_path, connect_means=CONNECT_MEANS, y_label="ΔSLOR"):
    languages = language_order(pairs["original_language"].unique())
    present = set(pairs["model_name"])
    names = [m for m, _l in models]
    missing = [m for m in names if m not in present]
    pair_slot = 0.84 / max(len(names), 1)
    violin_width = min(pair_slot * 0.40, 0.10)
    with figure_style():
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, AXES_HEIGHT))
        for i in range(1, len(languages)):
            _separator(ax, i - 0.5, SEPARATOR_COLOR, PAIR_GROUP_SEPARATOR_WIDTH)
        means, per_category = {}, []
        for li, lang in enumerate(languages):
            sub = pairs.loc[pairs["original_language"] == lang]
            per_category.append(sub[VALUE].to_numpy(dtype=float))
            for mi, m in enumerate(names):
                center = li + (mi - (len(names) - 1) / 2) * pair_slot
                for cat, direction in ((JA_CONTROL, -1), (JA_BASELINE, 1)):
                    values = sub.loc[(sub["model_name"] == m) & (sub["category"] == cat), VALUE].to_numpy(dtype=float)
                    if values.size:
                        x = center + direction * pair_slot * 0.24
                        base_color = MODEL_COLORS[m]
                        color = base_color if cat == JA_CONTROL else tuple(0.78 * v + 0.22 for v in to_rgb(base_color))
                        means[(lang, cat, m)] = (x, _violin(ax, values, x, color, violin_width, show_means=True), int(values.size))
        linked = True
        if linked:
            for lang in languages:
                for m in names:
                    a = means.get((lang, JA_CONTROL, m))
                    b = means.get((lang, JA_BASELINE, m))
                    if a is not None and b is not None:
                        ax.plot([a[0], b[0]], [a[1], b[1]], zorder=Z_LINK, **MEAN_LINK_STYLE)
        ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=ZERO_LINE_WIDTH, zorder=Z_ZERO)
        ax.set_xticks(range(len(languages)))
        ax.set_xticklabels([f"{lang} ({_per_model_n_label(pairs.loc[(pairs['original_language'] == lang) & (pairs['category'] == JA_CONTROL)], 'model_name')})" for lang in languages])
        ax.set_xlim(-0.5, len(languages) - 0.5)
        _style_axes(ax, y_label)
        lo, hi = _y_limits(pairs[VALUE])
        ax.set_ylim(lo, hi)
        handles = [_model_handle(m, label, missing, "no paired data") for m, label in models]
        handles += [_blank("Target-language strategy: full color"),
                    _blank("English strategy: lighter shade")]
        handles += _summary_handles(linked, show_means=True)
        legend = _place_legend(ax, handles, 2, set())
        ax.set_ylim(lo, _legend_headroom(ax, legend, per_category, lo, hi))
        fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    return means


def _model_color(model, _lang):
    return MODEL_COLORS[model]


def _focus_color(model, lang):
    if lang != EN and lang not in FOCUS_LANGUAGES.get(model, (lang,)):
        return NON_FOCUS_GRAY
    return MODEL_COLORS[model]


def generate(df=None, outputs_dir=DEFAULT_OUTPUTS_DIR, out_dir=DEFAULT_MAIN_PLOT_DIR, langs=DEFAULT_LANGUAGES,
             include_identical=False, connect_means=CONNECT_MEANS, only=None):
    if df is None:
        df, warnings = load_probability_outputs(outputs_dir)
        for w in warnings:
            print(f"[Fig2_4] warning: {w}")
    if df.empty:
        raise ValueError("Fig2_4: no probability outputs loaded")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(only or FIGURES)
    out = {}
    for metric, metric_col in (("slor", SLOR_COL), ("meanlp", MEANLP_COL)):
        if metric_col not in df.columns:
            raise KeyError(f"Required {metric} column {metric_col!r} missing from loaded outputs2")
        suffix = "" if metric == "slor" else "__delta-meanLP"
        label = "ΔSLOR" if metric == "slor" else "ΔmeanLP"
        def path_for(key):
            name = FIGURES[key]
            return out_dir / (name[:-4] + suffix + ".pdf")
        if {"a", "b", "combined"} & set(keys):
            rows = standard_violin_rows(df, langs, include_identical, metric_col)
            sel_a = rows.loc[rows["model_name"].isin([m for m, _l in MULTILINGUAL_MODELS])]
            sel_b = rows.loc[rows["model_name"].isin([m for m, _l in NON_ENGLISH_CENTRIC_MODELS])
                             & ~rows["Target language"].isin(PANEL_B_REMOVED_LANGUAGES)]
            if "a" in keys and len(sel_a):
                path = path_for("a")
                out[f"a_{metric}"] = (path, plot_language_panel(sel_a, MULTILINGUAL_MODELS, path, _model_color, y_label=label))
            if "b" in keys and len(sel_b):
                path = path_for("b")
                out[f"b_{metric}"] = (path, plot_language_panel(sel_b, NON_ENGLISH_CENTRIC_MODELS, path, _focus_color, grouped=True, y_label=label))
            if "combined" in keys and len(sel_a) and len(sel_b):
                path = path_for("combined")
                out[f"combined_{metric}"] = (path, plot_combined_panels(sel_a, sel_b, path, y_label=label))
        if "ja" in keys:
            pairs = japanese_control_pairs(df, [m for m, _l in JAPANESE_CONTROL_MODELS], include_identical, metric_col)
            if len(pairs):
                path = path_for("ja")
                out[f"ja_{metric}"] = (path, plot_japanese_control(pairs, JAPANESE_CONTROL_MODELS, path, connect_means, y_label=label))
    for key, (path, means) in out.items():
        print(f"[Fig2_4] {path} ({len(means)} violins)")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Figures 2 and 4: main ΔSLOR violin plots (standalone).")
    parser.add_argument("--outputs_dir", default=str(DEFAULT_OUTPUTS_DIR), help="saved outputs2 directory")
    parser.add_argument("--out_dir", default=str(DEFAULT_MAIN_PLOT_DIR), help="figure directory (default Main_plot/)")
    parser.add_argument("--use-preference-annotation", dest="langs", nargs="+", default=list(DEFAULT_LANGUAGES),
                        metavar="LANG", help="languages of the target preference filter (default: de fr pt sv zh ja)")
    parser.add_argument("--include_identical", action="store_true", help="keep identical pos/neg pairs")
    parser.add_argument("--no-connect-means", dest="connect_means", action="store_false",
                        help="do not join paired means in Figure 4")
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES), default=None, help="figures to draw")
    args = parser.parse_args(argv)
    generate(outputs_dir=args.outputs_dir, out_dir=args.out_dir, langs=args.langs,
             include_identical=args.include_identical, connect_means=args.connect_means, only=args.only)


if __name__ == "__main__":
    main()
