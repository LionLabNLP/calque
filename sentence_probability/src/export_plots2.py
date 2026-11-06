import argparse
import math
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import Fig2_4
import Fig3
import Fig5
import Fig6
import ja_control_lmm
import main_inference
import rq1_lmm
import spearman_groups
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cross_model_consistency import (
    ICC_FORMULA,
    MIN_SHARED_PAIRS_SPEARMAN,
    PAIR_ID_COL,
    POOLED as POOLED_FIT,
    SentencePairIdentityError,
    language_summary as consistency_language_summary,
    run_consistency_analysis,
    sentence_pair_frame,
    sign_agreement_and_kappa,
    spearman_matrix,
)
from model_metadata2 import METADATA_FIELDS
from viz_data2 import (
    ALL_GROUPS,
    COMMON_CRAWL_CAVEAT,
    COMMON_CRAWL_LABEL,
    COMMON_CRAWL_TERM,
    MAIN_FIG_TERM_LABELS,
    MAIN_SPEARMAN_MODELS,
    MAIN_VIOLIN_PREFERENCE_SUBSETS,
    MAIN_VIOLIN_SETS,
    REGRESSION_FAMILIES,
    filter_side_preference,
    resolve_main_models,
    run_family_regressions,
    CONTROL_CATEGORY_COL,
    CONTROL_ENGLISH,
    CONTROL_JA_BASELINE,
    CONTROL_JAPANESE,
    DELTA_MEANLP,
    DELTA_SLOR,
    JAPANESE_CONTROL_PREFERENCE_SUBSET,
    ORIGINAL_TARGET_LANGUAGE_COL,
    PAIRED_PREFERENCE_SUBSET_COL,
    SEMANTIC_CONTENT_KEY_COL,
    SLOR_AXIS_LABEL,
    EXCLUSION_REASONS,
    canonical_language_code,
    pair_japanese_baseline,
    paired_baseline_exclusions,
    paired_baseline_inputs,
    paired_category_label,
    paired_category_order,
    DIFF_CONDITION_COL,
    DIFF_CONDITIONS,
    DIFF_VALUE_COL,
    ENGLISH_COMPARISON_POS,
    ENGLISH_CONTROL_CATEGORY,
    SCOPE_KIND_ALL,
    SCOPE_KIND_MODEL,
    SCOPE_KIND_PICKED,
    SCORE_METRICS,
    SLOR_COL,
    TARGET_WITH_EN_SOURCE_COL,
    analysis_rows,
    diff_comparison_frame,
    language_axis_order,
    filter_by_scope,
    model_color,
    plot_scopes,
    score_metric,
    target_language_with_english_frame,
    FAILURE_MATRIX_COLUMNS,
    FAMILY_SCOPE_ALL,
    FAMILY_SCOPE_LABELS,
    FAMILY_SCOPES,
    GROUP_PALETTE,
    MEANLP_AXIS_LABEL,
    METADATA_FEATURE_COLUMNS,
    METADATA_FEATURES,
    SOURCE_ROW_ID_COL,
    STRONG_PREF_TERM,
    TARGET_COMPARISON_POS,
    TERM_LABELS,
    LANGUAGE_FE_PREFIX,
    add_source_row_id,
    compute_failure_rate_matrix,
    family_colors,
    family_scope_of,
    model_size_win_rate_frame,
    filter_by_family_scope,
    filter_target_preference,
    gloss_feature_inventory,
    gloss_pair_counts,
    gloss_regression_frame,
    japanese_control_frame,
    load_metadata_table,
    models_in,
    parse_preference_languages,
    preference_filter_label,
    run_gloss_feature_regressions,
    run_main_regression,
    short_model_label,
    EXPERIMENT_JAPANESE_CONTROL,
    EXPERIMENT_STANDARD,
    EXPERIMENTS,
    GLOSS_MIN_CONTRAST,
    GLOSS_OUTCOME_LABEL,
    GROUP_COL,
    IT_TERM,
    LOG_SIZE_TERM,
    MEANLP_COL,
    METRIC_Y_LABELS,
    MODEL_VIEW_ALL,
    MODEL_VIEW_LABELS,
    MODEL_VIEWS,
    PREFERENCE_ALL,
    PREFERENCE_COL,
    PREFERENCE_SUBSET_LABELS,
    PREFERENCE_SUBSETS,
    VIOLIN_METRICS,
    XAXIS_MODES,
    apply_filters2,
    category_labels,
    compute_failure_rates,
    compute_translation_preference,
    experiment_title,
    filter_by_experiment,
    filter_by_group,
    group_choices,
    group_color,
    groups_to_display,
    load_all_data2,
    load_all_translation_data2,
    metric_available,
    metric_distributions,
    metric_slug,
    model_metadata_frame,
    model_view_panels,
    sanitize_filename,
    summarize_distributions,
    unresolved_model_names,
    FAILURE_BAR_LABELS,
    FAILURE_BAR_MODELS,
    FAILURE_BAR_MODEL_COUNT,
    FAILURE_BAR_SECTIONS,
    LANGUAGE_COLORS,
    LANGUAGE_NAMES,
    TRANSFERRED_PROBABILITY_COL,
    TRANSLATION_STATUS_NO_MODEL,
    TRANSLATION_STATUS_OK,
    compute_transferred_probability,
    failure_workbook_model_key,
    family_color,
    paper_family,
    paper_model_style,
    violin_model_style,
    STRICT_PALETTE,
    VIOLIN_MODEL_COLORS,
    MAIN_MONOLINGUAL_SET_A,
    MAIN_MONOLINGUAL_SET_A_MANACA,
    MAIN_MANACA_MODEL,
    MAIN_SABIA_MODEL,
    MAIN_SPEARMAN_MODELS_MANACA,
    MAIN_ALL_MODELS,
    STAT_MODEL_COUNT,
    STAT_MODEL_SECTIONS,
    english_side_origin_frame,
    ENGLISH_COMPARISON_NEG,
    ENGLISH_PREFERENCE_COL,
    RQ1_ORIGINS,
    RQ1_PREFERENCE,
    STRATEGY_ORIGIN_COL,
    failure_rate_observations,
)
from model_metadata2 import lookup_model

DPI = 300

FIG_EXT = ".pdf"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
SPINE_COLOR = "black"

PAPER_FONT_CANDIDATES = ["TeX Gyre Termes", "Times New Roman", "Times", "Nimbus Roman", "Nimbus Roman No9 L",
                         "STIX Two Text", "STIXGeneral"]


def resolve_paper_font():
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in PAPER_FONT_CANDIDATES:
        if name in installed:
            return name, not name.startswith("STIX")
    return "DejaVu Serif", False


PAPER_FONT, PAPER_FONT_IS_TIMES = resolve_paper_font()

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": [PAPER_FONT] + [f for f in PAPER_FONT_CANDIDATES if f != PAPER_FONT] + ["DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": True,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": SPINE_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": SPINE_COLOR,
    "ytick.color": SPINE_COLOR,
    "xtick.labelcolor": TEXT_COLOR,
    "ytick.labelcolor": TEXT_COLOR,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": DPI,
})


LANGUAGE_SEPARATOR_COLOR = "#B5B5B5"
LANGUAGE_SEPARATOR_WIDTH = 0.45
ENGLISH_BOUNDARY_COLOR = "#555555"
ENGLISH_BOUNDARY_WIDTH = 0.65
ZERO_LINE_COLOR = "#202020"
ZERO_LINE_WIDTH = 1.8
CONNECT_PAIRED_MEANS = True
MEAN_LINK_STYLE = {"color": "#4D4D4D", "linewidth": 0.7, "linestyle": "-", "solid_capstyle": "round", "zorder": 3.5}


def draw_zero_line(ax, width=ZERO_LINE_WIDTH):
    ax.axhline(0, color=ZERO_LINE_COLOR, linestyle="-", linewidth=width, zorder=1.5)


def draw_language_separators(ax, categories, scale=1.0, english_boundary=True):
    drawn = []
    cats = [str(c) for c in categories]
    for i in range(1, len(cats)):
        is_english = english_boundary and (
            (cats[i] == ENGLISH_CONTROL_CATEGORY) != (cats[i - 1] == ENGLISH_CONTROL_CATEGORY))
        if is_english:
            ax.axvline(i - 0.5, color=ENGLISH_BOUNDARY_COLOR, linewidth=ENGLISH_BOUNDARY_WIDTH * scale, zorder=0.6)
        else:
            ax.axvline(i - 0.5, color=LANGUAGE_SEPARATOR_COLOR, linewidth=LANGUAGE_SEPARATOR_WIDTH * scale,
                       zorder=0.5)
        drawn.append((i - 0.5, "english" if is_english else "language"))
    return drawn


def draw_language_group_separators(ax, group_of_category, scale=1.0, within_group=True):
    drawn = []
    for i in range(1, len(group_of_category)):
        if group_of_category[i] != group_of_category[i - 1]:
            ax.axvline(i - 0.5, color=LANGUAGE_SEPARATOR_COLOR, linewidth=LANGUAGE_SEPARATOR_WIDTH * 1.4 * scale,
                       zorder=0.5)
            drawn.append((i - 0.5, "language"))
        elif within_group:
            ax.axvline(i - 0.5, color=LANGUAGE_SEPARATOR_COLOR, linewidth=LANGUAGE_SEPARATOR_WIDTH * 0.8 * scale,
                       linestyle=(0, (2, 3)), zorder=0.5)
            drawn.append((i - 0.5, "within"))
    return drawn


def vector_heatmap(ax, values, cmap, vmin, vmax):
    k_rows, k_cols = values.shape
    ax.set_facecolor(cmap.get_bad())
    mesh = ax.pcolormesh(np.arange(k_cols + 1) - 0.5, np.arange(k_rows + 1) - 0.5, np.ma.masked_invalid(values),
                         cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="none", linewidth=0, antialiased=False)
    ax.set_xlim(-0.5, k_cols - 0.5)
    ax.set_ylim(k_rows - 0.5, -0.5)
    ax.set_aspect("equal")
    return mesh


def black_spines(ax, width=None):
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE_COLOR)
        if width is not None:
            ax.spines[side].set_linewidth(width)


def _group_suffix(group):
    return "all" if group == ALL_GROUPS else sanitize_filename(group)


def _group_title(group):
    return "all construction groups" if group == ALL_GROUPS else f"group: {group}"


def _setup_axes(ax, categories, x_col, x_title):
    ax.set_xticks(range(len(categories)))
    labels = category_labels(categories, x_col)
    rotate = x_col == "model_name" or max((len(s) for s in labels), default=0) > 8
    ax.set_xticklabels(labels, rotation=30 if rotate else 0, ha="right" if rotate else "center")
    ax.set_xlim(-0.6, len(categories) - 0.4)
    ax.set_xlabel(x_title)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)


def _figure(n_categories, n_panels=1):
    width = max(5.5, 1.15 * n_categories + 2.5)
    return plt.subplots(1, n_panels, figsize=(width * n_panels if n_panels > 1 else width, 4.4), sharey=True, squeeze=False)


def _view_title(view):
    return "" if view in (None, MODEL_VIEW_ALL) else f" — {MODEL_VIEW_LABELS[view]}"


def _pref_title(subset):
    return "" if subset in (None, PREFERENCE_ALL) else f" — preference {PREFERENCE_SUBSET_LABELS[subset]}"


def _draw_violins(ax, df, metric, x_col, groups, all_groups):
    categories, dists = metric_distributions(df, metric, x_col, groups)
    if not dists:
        return categories
    n_groups = len(groups)
    slot = 0.8 / n_groups
    cat_index = {c: i for i, c in enumerate(categories)}
    for cat, g, values in dists:
        pos = cat_index[cat] + (groups.index(g) - (n_groups - 1) / 2) * slot
        color = group_color(g, all_groups)
        if values.size > 1 and np.ptp(values) > 0:
            parts = ax.violinplot([values], positions=[pos], widths=slot * 0.92, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("none")
                body.set_alpha(1.0)
                body.set_linewidth(0)
        half = slot * 0.22
        ax.plot([pos - half, pos + half], [np.median(values)] * 2, color=TEXT_COLOR, linewidth=2, solid_capstyle="butt", zorder=3)
        ax.scatter([pos], [np.mean(values)], marker="D", s=22, facecolor="white", edgecolor=TEXT_COLOR, linewidth=0.9, zorder=4)
    return categories


def plot_probability_violins(df, metric, xaxis_mode, group, all_groups, out_path, title_suffix="", view=MODEL_VIEW_ALL,
                             x_title=None):
    x_col, default_x_title = XAXIS_MODES[xaxis_mode]
    x_title = x_title or default_x_title
    groups = groups_to_display(df, group)
    panels = model_view_panels(df, view)
    if not any(metric_distributions(sub, metric, x_col, groups)[1] for _label, sub in panels):
        return False

    n_cats = max(len(metric_distributions(sub, metric, x_col, groups)[0]) for _label, sub in panels)
    fig, axes = _figure(n_cats, len(panels))
    for ax, (label, sub) in zip(axes[0], panels):
        categories = _draw_violins(ax, sub, metric, x_col, groups, all_groups)
        draw_zero_line(ax, 1.2)
        if x_col == "Target language":
            draw_language_separators(ax, categories)
        _setup_axes(ax, categories, x_col, x_title)
        if label is not None:
            ax.set_title(label if categories else f"{label} (no data)", color=TEXT_COLOR, fontsize=10)
    axes[0][0].set_ylabel(METRIC_Y_LABELS.get(metric, metric))
    fig.suptitle(f"{_group_title(group)}{title_suffix}", color=TEXT_COLOR, x=0.01, ha="left", fontsize=11)

    handles = []
    if group == ALL_GROUPS:
        handles += [Patch(facecolor=group_color(g, all_groups), edgecolor="none", label=g) for g in groups]
    handles += [
        Line2D([0], [0], color=TEXT_COLOR, linewidth=2, label="median"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor=TEXT_COLOR, markersize=5, label="mean"),
    ]
    axes[0][-1].legend(handles=handles, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    fig.savefig(out_path, format="pdf", dpi=DPI)
    plt.close(fig)
    return True


def _draw_translation_bars(ax, pref_df, x_col, groups, all_groups):
    categories = sorted(pref_df[x_col].dropna().unique().tolist()) if not pref_df.empty else []
    n_groups = len(groups)
    slot = 0.8 / max(n_groups, 1)
    indexed = pref_df.set_index([x_col, GROUP_COL]) if not pref_df.empty else None
    for gi, g in enumerate(groups):
        color = group_color(g, all_groups)
        for ci, cat in enumerate(categories):
            if (cat, g) not in indexed.index:
                continue
            row = indexed.loc[(cat, g)]
            pos = ci + (gi - (n_groups - 1) / 2) * slot
            p = row[PREFERENCE_COL]
            n = int(row["N_proper_or_literal"])
            if pd.isna(p):
                ax.text(pos, 0.01, "n=0", ha="center", va="bottom", fontsize=6, color=MUTED_COLOR, rotation=90)
                continue
            ax.bar(pos, p, width=slot * 0.9, color=color, edgecolor="white", linewidth=0.6, zorder=2)
            ax.text(
                pos, p + 0.01, f"n={n}", ha="center", va="bottom", fontsize=6, color=MUTED_COLOR, rotation=90,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=3,
            )
    return categories


def plot_translation_preference(translation_df, xaxis_mode, group, all_groups, out_path, view=MODEL_VIEW_ALL):
    x_col, x_title = XAXIS_MODES[xaxis_mode]
    panels = [(label, compute_translation_preference(sub, x_col)) for label, sub in model_view_panels(translation_df, view)]
    if all(pref.empty for _label, pref in panels):
        return False
    groups = groups_to_display(filter_by_group(translation_df, group), group)

    n_cats = max(pref[x_col].nunique() if not pref.empty else 0 for _label, pref in panels)
    fig, axes = _figure(n_cats, len(panels))
    for ax, (label, pref) in zip(axes[0], panels):
        categories = _draw_translation_bars(ax, pref, x_col, groups, all_groups)
        ax.axhline(0.5, color=MUTED_COLOR, linestyle=":", linewidth=0.8, zorder=1)
        _setup_axes(ax, categories, x_col, x_title)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        if label is not None:
            ax.set_title(label if categories else f"{label} (no data)", color=TEXT_COLOR, fontsize=10)
    axes[0][0].set_ylabel("P(proper | proper or literal)")
    fig.suptitle(
        f"{_group_title(group)} — conditional on discriminative exact matches (n = N_proper + N_literal)",
        color=TEXT_COLOR, x=0.01, ha="left", fontsize=9,
    )
    if group == ALL_GROUPS:
        handles = [Patch(facecolor=group_color(g, all_groups), label=g) for g in groups]
        axes[0][-1].legend(handles=handles, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    fig.savefig(out_path, format="pdf", dpi=DPI)
    plt.close(fig)
    return True


def _comparisons(df):
    out = [(None, df, "")]
    if "pos" in df.columns:
        for pos in sorted(df["pos"].dropna().unique().tolist()):
            sub = df.loc[df["pos"] == pos]
            negs = sorted(sub["neg"].dropna().unique().tolist()) if "neg" in sub.columns else []
            title = f" — {pos} vs {' / '.join(negs)}" if negs else f" — {pos}"
            out.append((sanitize_filename(pos), sub, title))
    return out


def _record(written, path, ok=True):
    if ok:
        written.append(path)
        print(path)
    else:
        print(f"[skipped: no data] {path}")


GRID_METRICS = [m for m in VIOLIN_METRICS if m not in {c for c, _l in DIFF_CONDITIONS}]


def _variants(metric):
    if metric in SCORE_METRICS:
        return [(p, v) for p, _pl, _acc in PREFERENCE_SUBSETS for v, _vl in MODEL_VIEWS]
    return [(PREFERENCE_ALL, MODEL_VIEW_ALL)]


def experiment_x_title(xaxis_mode, experiment):
    if experiment == EXPERIMENT_JAPANESE_CONTROL and XAXIS_MODES[xaxis_mode][0] == "Target language":
        return "Original target language (strategy origin)"
    return XAXIS_MODES[xaxis_mode][1]


def export_probability_plots(df, plots_dir, all_groups, include_identical=False, experiment=EXPERIMENT_STANDARD,
                             preference_languages=None):
    out_dir = Path(plots_dir) / "probability"
    out_dir.mkdir(parents=True, exist_ok=True)
    written, summaries = [], []
    base = apply_filters2(df, {}, include_identical=include_identical, experiment=experiment,
                          preference_languages=preference_languages)
    group_options = [ALL_GROUPS] + group_choices(base)

    for metric in GRID_METRICS:
        for pref, view in _variants(metric):
            selected = apply_filters2(
                df, {}, include_identical=include_identical, preference_subset=pref, experiment=experiment,
                preference_languages=preference_languages,
            )
            for cmp_slug, cmp_df, cmp_title in _comparisons(selected):
                if not metric_available(cmp_df, metric):
                    continue
                title = f"{experiment_title(experiment)}{cmp_title}{_pref_title(pref)}{_view_title(view)}"
                for xaxis_mode, (x_col, _x_title) in XAXIS_MODES.items():
                    for group in group_options:
                        sub = filter_by_group(cmp_df, group)
                        name = f"probability_{metric_slug(metric)}"
                        if cmp_slug:
                            name += f"__cmp-{cmp_slug}"
                        name += f"__x-{xaxis_mode}__group-{_group_suffix(group)}__pref-{pref}__view-{view}{FIG_EXT}"
                        path = out_dir / name
                        _record(written, path, plot_probability_violins(
                            sub, metric, xaxis_mode, group, all_groups, path, title_suffix=title, view=view,
                            x_title=experiment_x_title(xaxis_mode, experiment),
                        ))
                    for panel, panel_df in model_view_panels(cmp_df, view):
                        s = summarize_distributions(panel_df, metric, x_col, group_choices(panel_df))
                        s = s.rename(columns={x_col: "x"})
                        s.insert(0, "model_split", panel or "all")
                        s.insert(0, "model_view", view)
                        s.insert(0, "preference_subset", PREFERENCE_SUBSET_LABELS[pref])
                        s.insert(0, "x_axis", xaxis_mode)
                        s.insert(0, "metric", metric_slug(metric))
                        s.insert(0, "comparison", cmp_slug or "all")
                        s.insert(0, "experiment", experiment)
                        summaries.append(s)

    summaries = [s for s in summaries if not s.empty]
    if summaries:
        csv_path = out_dir / "probability_summary.csv"
        pd.concat(summaries, ignore_index=True).to_csv(csv_path, index=False)
        _record(written, csv_path)
    return written


def export_translation_plots(translation_df, plots_dir, all_groups):
    out_dir = Path(plots_dir) / "translation"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if translation_df.empty:
        print("[skipped] no translation-generation data")
        return written

    for view, _view_label in MODEL_VIEWS:
        for xaxis_mode, (x_col, _x_title) in XAXIS_MODES.items():
            tables = []
            for panel, panel_df in model_view_panels(translation_df, view):
                pref = compute_translation_preference(panel_df, x_col)
                pref.insert(0, "model_split", panel or "all")
                tables.append(pref)
            csv_path = out_dir / f"translation_preference__x-{xaxis_mode}__view-{view}.csv"
            pd.concat(tables, ignore_index=True).to_csv(csv_path, index=False)
            _record(written, csv_path)

            for group in [ALL_GROUPS] + group_choices(translation_df):
                path = out_dir / f"translation_preference__x-{xaxis_mode}__group-{_group_suffix(group)}__view-{view}{FIG_EXT}"
                _record(written, path, plot_translation_preference(
                    filter_by_group(translation_df, group), xaxis_mode, group, all_groups, path, view=view,
                ))
    return written


def export_failure_rates(df, plots_dir, include_identical=False, experiment=EXPERIMENT_STANDARD,
                         preference_languages=None, metric=MEANLP_COL):
    spec = score_metric(metric)
    out_dir = Path(plots_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec['table']}_failure_rates.xlsx"
    tables = {
        pref: compute_failure_rates(df, pref, include_identical=include_identical, experiment=experiment,
                                    preference_languages=preference_languages, metric=metric)
        for pref, _l, _a in PREFERENCE_SUBSETS
    }
    matrix = (compute_failure_rate_matrix(df, preference_languages, include_identical=include_identical, metric=metric)
              if experiment == EXPERIMENT_STANDARD else None)
    label = spec["label"]
    definition = pd.DataFrame({"item": ["experiment", "metric", "failure", "tie", "N", "failure_rate", "rows", "comparison"], "definition": [
        f"{experiment} only (standard comparisons and Japanese control are never pooled)",
        f"{spec['axis']} (saved column {metric})",
        f"{label} < 0",
        f"{label} = 0 is a tie, not a failure (counted in N_tie)",
        f"number of non-missing {label} scores; NaN scores are excluded from the denominator",
        "N_failure / N",
        "no-context (raw) condition only; identical pos/neg pairs excluded" if not include_identical
        else "no-context (raw) condition only; identical pos/neg pairs included",
        "'All comparisons' pools every comparison; other rows are one positive column (pos)",
    ]})
    with pd.ExcelWriter(path) as writer:
        if matrix is not None:
            for sheet in ("failure_rate", "N_failure", "N_total", "N_tie"):
                matrix[sheet].reset_index().to_excel(writer, sheet_name=sheet, index=False)
            failure_matrix_definitions(preference_languages, include_identical, metric).to_excel(
                writer, sheet_name="matrix_definitions", index=False)
        definition.to_excel(writer, sheet_name="definitions", index=False)
        non_empty = [t for t in tables.values() if not t.empty]
        long_table = pd.concat(non_empty, ignore_index=True) if non_empty else next(iter(tables.values()))
        long_table.to_excel(writer, sheet_name="all_subsets", index=False)
        for pref, table in tables.items():
            table.to_excel(writer, sheet_name=f"pref_{pref}", index=False)
    written = []
    _record(written, path)
    return written


def export_model_metadata(df, translation_df, plots_dir):
    out_dir = Path(plots_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = set()
    for d in (df, translation_df):
        if not d.empty and "model_name" in d.columns:
            names.update(d["model_name"].dropna().astype(str).unique())
    meta = model_metadata_frame(names).reset_index()
    meta["metadata_resolved"] = meta["model_id"].notna()
    meta["family_scope"] = meta["family"].map(family_scope_of)
    meta["plot_color"] = meta["model_name"].map(model_color)
    path = out_dir / "model_metadata.csv"
    meta[["model_name", "metadata_resolved"] + METADATA_FIELDS + ["family_scope", "plot_color"]].to_csv(path, index=False)
    written = []
    _record(written, path)
    return written


CONTROL_COLORS = {CONTROL_JAPANESE: GROUP_PALETTE[0], CONTROL_ENGLISH: GROUP_PALETTE[1]}
DIFF_CONDITION_STYLE = {
    "diff-token": {"color": GROUP_PALETTE[0], "hatch": None},
    "after-diff-token": {"color": GROUP_PALETTE[1], "hatch": "///"},
}
COMPARISON_SLUGS = {
    ENGLISH_COMPARISON_POS: "cmp-proper-english-strategy",
    TARGET_COMPARISON_POS: "cmp-proper-strategy-in-target-language",
}
COMPARISON_TITLES = {
    ENGLISH_COMPARISON_POS: "Proper English strategy vs Literal English translation of target language strategy "
                            "(panel = target-language origin)",
    TARGET_COMPARISON_POS: "Proper strategy in target language vs Literal translation of Proper English into target language",
}


def _ordered_languages(values):
    return language_axis_order(values)


def _ordered_models(model_names):
    meta = model_metadata_frame(model_names)
    scope_rank = {slug: i for i, (slug, _l, _f) in enumerate(FAMILY_SCOPES)}

    def key(name):
        r = meta.loc[str(name)]
        scope = family_scope_of(r["family"])
        size = r["model_size_b"] if r["model_size_b"] is not None and pd.notna(r["model_size_b"]) else float("inf")
        return (scope_rank.get(scope, len(scope_rank)), size, bool(r["instruction_tuned"]), str(name))

    return sorted({str(m) for m in model_names}, key=key)


def _model_labels(models):
    meta = model_metadata_frame(models)
    return [meta.loc[m, "display_name"] or short_model_label(m) for m in models]


def _violin(ax, values, pos, color, width, hatch=None, show_mean_median=True):
    if values.size > 1 and np.ptp(values) > 0:
        parts = ax.violinplot([values], positions=[pos], widths=width, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(1.0)
            body.set_linewidth(0)
    mean = float(np.mean(values))
    if show_mean_median:
        half = width * 0.25
        ax.plot([pos - half, pos + half], [np.median(values)] * 2, color=TEXT_COLOR, linewidth=2,
                solid_capstyle="butt", zorder=3.8)
    ax.scatter([pos], [mean], marker="D", s=18, facecolor="white", edgecolor=TEXT_COLOR, linewidth=0.9, zorder=4)
    return mean


def _summary_handles(show_median=True, linked=False):
    handles = [Line2D([0], [0], color=TEXT_COLOR, linewidth=2, label="median")] if show_median else []
    handles.append(Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white",
                          markeredgecolor=TEXT_COLOR, markersize=5, label="mean"))
    if linked:
        handles.append(Line2D([0], [0], marker="D", markerfacecolor="white", markeredgecolor=TEXT_COLOR,
                              markersize=4, label="paired means",
                              **{k: v for k, v in MEAN_LINK_STYLE.items() if k not in ("zorder", "solid_capstyle")}))
    return handles


def draw_paired_mean_links(ax, means, pairs):
    n = 0
    for a, b in pairs:
        if a in means and b in means:
            ax.plot([means[a][0], means[b][0]], [means[a][1], means[b][1]], **MEAN_LINK_STYLE)
            n += 1
    return n


def _finish_language_axes(ax, languages, y_label, english_boundary=True):
    draw_zero_line(ax, 1.2)
    draw_language_separators(ax, languages, english_boundary=english_boundary)
    ax.set_xticks(range(len(languages)))
    ax.set_xticklabels(languages)
    ax.set_xlim(-0.65, len(languages) - 0.35)
    ax.set_xlabel("Language")
    ax.set_ylabel(y_label, fontsize=8)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_language_model_violins(rows, out_path, title, value_col=DELTA_MEANLP, y_label=MEANLP_AXIS_LABEL):
    if rows.empty:
        return False
    is_en = (rows[TARGET_WITH_EN_SOURCE_COL] == ENGLISH_COMPARISON_POS
             if TARGET_WITH_EN_SOURCE_COL in rows.columns else pd.Series(False, index=rows.index))
    languages = _ordered_languages(rows["Target language"].unique())
    models = _ordered_models(rows["model_name"].unique())
    n_models = max(len(models), 1)
    slot = min(0.8 / n_models, 0.16)
    fig_w = max(7.0, 1.55 * len(languages) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))

    for li, lang in enumerate(languages):
        sub = rows.loc[rows["Target language"] == lang]
        for mi, m in enumerate(models):
            values = sub.loc[sub["model_name"] == m, value_col].to_numpy(dtype=float)
            if not values.size:
                continue
            pos = li + (mi - (n_models - 1) / 2) * slot
            _violin(ax, values, pos, model_color(m), slot * 0.9)

    _finish_language_axes(ax, languages, y_label, english_boundary=bool(is_en.any()))
    handles = [Patch(facecolor=model_color(m), edgecolor="none", label=short_model_label(m)) for m in models]
    handles += _summary_handles()
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title, color=TEXT_COLOR, loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=DPI)
    plt.close(fig)
    return True


def plot_diff_comparison(frame, out_path, title):
    if frame.empty:
        return False
    languages = _ordered_languages(frame["Target language"].unique())
    conditions = [label for _c, label in DIFF_CONDITIONS]
    fig, ax = plt.subplots(figsize=(max(6.5, 1.35 * len(languages) + 3.0), 4.6))
    offset = 0.2
    for li, lang in enumerate(languages):
        sub = frame.loc[frame["Target language"] == lang]
        for ci, cond in enumerate(conditions):
            values = sub.loc[sub[DIFF_CONDITION_COL] == cond, DIFF_VALUE_COL].to_numpy(dtype=float)
            if not values.size:
                continue
            pos = li + (ci - 0.5) * 2 * offset
            style = DIFF_CONDITION_STYLE[cond]
            _violin(ax, values, pos, style["color"], 0.36, hatch=style["hatch"])
            ax.text(pos, 0.0, f"n={values.size}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=5.5, color=MUTED_COLOR)
    _finish_language_axes(ax, languages, "token-wise log-probability difference (positive − negative)")
    handles = [Patch(facecolor=DIFF_CONDITION_STYLE[c]["color"], edgecolor="none", label=c) for c in conditions]
    handles += _summary_handles()
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title, color=TEXT_COLOR, loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=DPI)
    plt.close(fig)
    return True


def plot_japanese_control(frame, out_path, title, value_col=DELTA_MEANLP, y_label=MEANLP_AXIS_LABEL):
    if frame.empty:
        return False
    languages = _ordered_languages(frame["Target language"].unique())
    fig, ax = plt.subplots(figsize=(max(6.0, 1.25 * len(languages) + 2.5), 4.6))
    for i, lang in enumerate(languages):
        values = frame.loc[frame["Target language"] == lang, value_col].to_numpy(dtype=float)
        if not values.size:
            continue
        cat = CONTROL_ENGLISH if lang == "en" else CONTROL_JAPANESE
        _violin(ax, values, i, CONTROL_COLORS[cat], 0.72)
        ax.text(i, 0.0, f"n={values.size}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=6, color=MUTED_COLOR)
    _finish_language_axes(ax, languages, y_label, english_boundary=False)
    ax.set_xlim(-0.6, len(languages) - 0.4)
    ax.legend(handles=_summary_handles(), frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title, color=TEXT_COLOR, loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=DPI)
    plt.close(fig)
    return True


PAIRED_BASELINE_HATCH = "////"


def plot_japanese_control_paired_baseline(frame, out_path, title, value_col=DELTA_SLOR, y_label=SLOR_AXIS_LABEL):
    if frame.empty:
        return False
    categories = [c for c in paired_category_order(frame[ORIGINAL_TARGET_LANGUAGE_COL].unique())
                  if ((frame[ORIGINAL_TARGET_LANGUAGE_COL] == c[0]) & (frame[CONTROL_CATEGORY_COL] == c[1])).any()]
    models = _ordered_models(frame["model_name"].unique())
    n_models = max(len(models), 1)
    slot = min(0.8 / n_models, 0.16)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.05 * len(categories) + 3.5), 5.2))
    draw_language_group_separators(ax, [lang for lang, _c in categories])
    means = {}
    for ci, (lang, cat) in enumerate(categories):
        sub = frame.loc[(frame[ORIGINAL_TARGET_LANGUAGE_COL] == lang) & (frame[CONTROL_CATEGORY_COL] == cat)]
        for mi, m in enumerate(models):
            values = sub.loc[sub["model_name"] == m, value_col].to_numpy(dtype=float)
            if not values.size:
                continue
            pos = ci + (mi - (n_models - 1) / 2) * slot
            means[(lang, cat, m)] = (pos, _violin(ax, values, pos, model_color(m), slot * 0.9,
                                                  hatch=PAIRED_BASELINE_HATCH if cat == CONTROL_JA_BASELINE else None))
        ax.text(ci, 0.0, f"n={len(sub)}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=6, color=MUTED_COLOR)
    linked = CONNECT_PAIRED_MEANS and bool(draw_paired_mean_links(
        ax, means, [((l, CONTROL_JAPANESE, m), (l, CONTROL_JA_BASELINE, m)) for l in dict.fromkeys(l for l, _c in categories)
                    for m in models]))
    draw_zero_line(ax, 1.2)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([paired_category_label(l, c, sep="\n") for l, c in categories], fontsize=7)
    ax.set_xlim(-0.65, len(categories) - 0.35)
    ax.set_xlabel("Original target language (strategy origin). All scored sentences are Japanese: "
                  "JA control = proper Japanese vs target-language strategy in Japanese;\n"
                  "JA baseline = proper Japanese vs English strategy in Japanese, same semantic contents and model",
                  fontsize=7)
    ax.set_ylabel(y_label, fontsize=8)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=model_color(m), edgecolor="none", label=short_model_label(m)) for m in models]
    handles += _summary_handles(linked=linked)
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title, color=TEXT_COLOR, loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=DPI)
    plt.close(fig)
    return True


PAIRED_SUMMARY_KEYS = [ORIGINAL_TARGET_LANGUAGE_COL, CONTROL_CATEGORY_COL, PAIRED_PREFERENCE_SUBSET_COL, "model_name"]


def paired_baseline_summary(frame, value_col):
    columns = PAIRED_SUMMARY_KEYS + ["n", "n_source_contrasts", "mean", "median", "std"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for k, g in frame.groupby(PAIRED_SUMMARY_KEYS, sort=True):
        y = g[value_col].to_numpy(dtype=float)
        rows.append({**dict(zip(PAIRED_SUMMARY_KEYS, k)), "n": int(y.size),
                     "n_source_contrasts": int(g[SEMANTIC_CONTENT_KEY_COL].nunique()),
                     "mean": float(np.mean(y)), "median": float(np.median(y)),
                     "std": float(np.std(y, ddof=1)) if y.size > 1 else float("nan")})
    out = pd.DataFrame(rows, columns=columns)
    order = {c: i for i, c in enumerate(paired_category_order(out[ORIGINAL_TARGET_LANGUAGE_COL].unique()))}
    out["_o"] = [order[(l, c)] for l, c in zip(out[ORIGINAL_TARGET_LANGUAGE_COL], out[CONTROL_CATEGORY_COL])]
    return out.sort_values(["_o", "model_name"], kind="stable").drop(columns="_o").reset_index(drop=True)


def _distribution_summary(rows, keys, value_col=DELTA_MEANLP):
    if rows.empty:
        return pd.DataFrame(columns=keys + ["n", "n_source_contrasts", "n_models", "mean", "median", "N_failure", "failure_rate"])
    out = []
    for k, g in rows.groupby(keys, sort=True):
        k = k if isinstance(k, tuple) else (k,)
        y = g[value_col].to_numpy(dtype=float)
        out.append({**dict(zip(keys, k)), "n": int(y.size),
                    "n_source_contrasts": int(g[SOURCE_ROW_ID_COL].nunique()) if SOURCE_ROW_ID_COL in g else None,
                    "n_models": int(g["model_name"].nunique()), "mean": float(np.mean(y)), "median": float(np.median(y)),
                    "N_failure": int((y < 0).sum()), "failure_rate": float((y < 0).mean())})
    return pd.DataFrame(out)


def _scope_title(kind, label):
    return {SCOPE_KIND_ALL: label, SCOPE_KIND_PICKED: label, SCOPE_KIND_MODEL: f"Model: {label}"}.get(
        kind, f"Family: {label}")


def _write_csv(written, path, tables):
    non_empty = [t for t in tables if not t.empty]
    (pd.concat(non_empty, ignore_index=True) if non_empty else tables[0]).to_csv(path, index=False)
    _record(written, path)


def export_scoped_probability_plots(df, plots_dir, preference_languages, include_identical, audit):
    written = []
    scores = {m: add_source_row_id(analysis_rows(df, m, include_identical=include_identical,
                                                 preference_languages=preference_languages))
              for m in SCORE_METRICS}
    with_en = target_language_with_english_frame(df, SLOR_COL, preference_languages, include_identical)
    diffs = diff_comparison_frame(df, preference_languages, include_identical)
    pref_label = preference_filter_label(preference_languages)
    for kind, key, label, rel_dir in plot_scopes(df):
        out_dir = Path(plots_dir) / "probability" / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        scope_title = _scope_title(kind, label)
        for metric, spec in SCORE_METRICS.items():
            scoped = filter_by_scope(scores[metric], kind, key)
            summaries = []
            for pos, slug in COMPARISON_SLUGS.items():
                sub = scoped.loc[scoped["pos"] == pos] if "pos" in scoped.columns else scoped.iloc[0:0]
                pref = pref_label if pos == TARGET_COMPARISON_POS else "none (English comparison)"
                extra = ""
                if metric == SLOR_COL and pos == TARGET_COMPARISON_POS:
                    sub = filter_by_scope(with_en, kind, key)
                    extra = ("\nrightmost en = English comparison (Proper English strategy vs Literal English "
                             "translation of target language strategy), every origin, one violin per model")
                title = (f"{scope_title} — {spec['label']}, {COMPARISON_TITLES[pos]}\nconstruction groups pooled; "
                         f"preference filter: {pref}{extra}")
                path = out_dir / f"{spec['slug']}__{slug}{FIG_EXT}"
                _record(written, path, plot_language_model_violins(sub, path, title, spec["delta"], spec["axis"]))
                s = _distribution_summary(sub, ["Target language", "model_name"], spec["delta"])
                s.insert(0, "comparison", np.where(s["Target language"] == ENGLISH_CONTROL_CATEGORY,
                                                   ENGLISH_COMPARISON_POS + " (en control)", pos) if len(s) else pos)
                summaries.append(s)
                if metric == SLOR_COL and pos == TARGET_COMPARISON_POS:
                    for pref_slug, annotation in (("gt", ">"), ("gtgt", ">>")):
                        subset = filter_side_preference(sub, annotation)
                        variant_path = out_dir / f"{spec['slug']}__{slug}__pref-{pref_slug}{FIG_EXT}"
                        variant_title = title + f"\ncomparison-specific preference annotation = {annotation} only"
                        _record(written, variant_path, plot_language_model_violins(
                            subset, variant_path, variant_title, spec["delta"], spec["axis"]))
                        variant_summary = _distribution_summary(
                            subset, ["Target language", "model_name"], spec["delta"])
                        _write_table(written, out_dir / f"{spec['slug']}__{slug}__pref-{pref_slug}_summary.csv",
                                     variant_summary)
                audit.append(_audit_row(f"probability violins ({spec['label']})", key, sub, pref, comparison=pos))
            _write_csv(written, out_dir / f"{spec['slug']}_summary.csv", summaries)

        scoped_diff = filter_by_scope(diffs, kind, key)
        diff_summaries = []
        for pos, slug in COMPARISON_SLUGS.items():
            sub = scoped_diff.loc[scoped_diff["pos"] == pos] if "pos" in scoped_diff.columns else scoped_diff.iloc[0:0]
            pref = pref_label if pos == TARGET_COMPARISON_POS else "none (English comparison)"
            n_models = sub["model_name"].nunique() if len(sub) else 0
            title = (f"{scope_title} — diff-token vs after-diff-token, {COMPARISON_TITLES[pos]}\n"
                     f"{n_models} model(s) and construction groups pooled; preference filter: {pref}")
            path = out_dir / f"diff-vs-after-diff-token__{slug}{FIG_EXT}"
            _record(written, path, plot_diff_comparison(sub, path, title))
            s = _distribution_summary(sub, ["Target language", DIFF_CONDITION_COL], DIFF_VALUE_COL)
            s.insert(0, "comparison", pos)
            diff_summaries.append(s)
        _write_csv(written, out_dir / "diff-vs-after-diff-token_summary.csv", diff_summaries)
    return written


def export_japanese_control_plots(df, plots_dir, include_identical, audit):
    written = []
    for metric, spec in SCORE_METRICS.items():
        frame = japanese_control_frame(df, include_identical=include_identical, metric=metric)
        for kind, key, label, rel_dir in plot_scopes(df):
            out_dir = Path(plots_dir) / "japanese_control" / rel_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            sub = filter_by_scope(frame, kind, key)
            n_models = sub["model_name"].nunique() if not sub.empty else 0
            title = (f"{_scope_title(kind, label)} — Japanese control, {spec['label']} pooled over {n_models} model(s)\n"
                     "de/sv/fr/pt/zh/ja: Japanese preference annotation in {>, >>}; "
                     "en: ordinary Japanese target-language strategy vs English-strategy transfer")
            path = out_dir / f"{spec['slug']}__japanese-control-languages{FIG_EXT}"
            _record(written, path, plot_japanese_control(sub, path, title, spec["delta"], spec["axis"]))
            csv_path = out_dir / f"{spec['slug']}__japanese-control-languages_summary.csv"
            _distribution_summary(sub, ["Target language"], spec["delta"]).to_csv(csv_path, index=False)
            _record(written, csv_path)
            if metric != MEANLP_COL:
                continue
            for lang in _ordered_languages(sub["Target language"].unique()) if not sub.empty else []:
                desc = ("ordinary target-language comparison for Japanese; semantic contents selected from Japanese-control sample"
                        if lang == "en" else "Japanese preference annotation in {>, >>}")
                audit.append(_audit_row(f"japanese control ({lang})", key,
                                        sub.loc[sub["Target language"] == lang], desc))
    return written


PAIRED_BASELINE_VARIANTS = [
    (JAPANESE_CONTROL_PREFERENCE_SUBSET, "", "Japanese preference annotation in {>, >>}"),
    ("gt", "__pref-gt", "Japanese and JA-baseline target preference annotations > only"),
    ("gtgt", "__pref-gtgt", "Japanese and JA-baseline target preference annotations >> only"),
]
PAIRED_BASELINE_SLUG = "japanese-control-languages__paired-ja-baseline"


def export_japanese_control_paired_baseline_plots(df, plots_dir, include_identical, audit):
    written = []
    for metric, spec in SCORE_METRICS.items():
        for pref, suffix, pref_title in PAIRED_BASELINE_VARIANTS:
            japanese, baseline = paired_baseline_inputs(df, metric, include_identical, pref)
            exclusions = paired_baseline_exclusions(df, japanese, baseline, pref, include_identical)
            for kind, key, label, rel_dir in plot_scopes(df):
                out_dir = Path(plots_dir) / "japanese_control" / rel_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                frame = pair_japanese_baseline(filter_by_scope(japanese, kind, key),
                                               filter_by_scope(baseline, kind, key), pref)
                n_models = frame["model_name"].nunique() if len(frame) else 0
                title = (f"{_scope_title(kind, label)} — Japanese control vs paired JA baseline, {spec['label']}, "
                         f"{n_models} model(s) shown separately\n{pref_title}, then inner "
                         "join on (model, Proper English strategy): per language and model both categories hold "
                         "exactly the same contents\n(JA baseline: Proper strategy in target language vs Literal "
                         "translation of Proper English into target language, Target language = ja; contents "
                         "without a valid baseline are dropped from both sides)")
                name = f"{spec['slug']}__{PAIRED_BASELINE_SLUG}{suffix}"
                path = out_dir / f"{name}{FIG_EXT}"
                _record(written, path, plot_japanese_control_paired_baseline(frame, path, title, spec["delta"],
                                                                              spec["axis"]))
                _write_table(written, out_dir / f"{name}_summary.csv", paired_baseline_summary(frame, spec["delta"]))
                scope_exclusions = filter_by_scope(exclusions, kind, key)
                _write_table(written, out_dir / f"{name}_exclusions.csv", scope_exclusions)
                if metric != SLOR_COL:
                    continue
                for lang in language_axis_order(frame[ORIGINAL_TARGET_LANGUAGE_COL].unique()) if len(frame) else []:
                    for cat in (CONTROL_JAPANESE, CONTROL_JA_BASELINE):
                        rows = frame.loc[(frame[ORIGINAL_TARGET_LANGUAGE_COL] == lang) & (frame[CONTROL_CATEGORY_COL] == cat)]
                        lang_excl = scope_exclusions.loc[scope_exclusions[ORIGINAL_TARGET_LANGUAGE_COL] == lang]
                        reasons = ", ".join(f"{r}={int(lang_excl[r].sum())}" for r, _d in EXCLUSION_REASONS
                                            if lang_excl[r].sum())
                        audit.append(_audit_row(
                            f"japanese control paired-ja-baseline ({lang}, {cat})", key, rows,
                            f"{pref_title} on the Japanese-control side; inner join with the standard ja comparison "
                            f"on (model, Proper English strategy); {int(lang_excl['n_control_excluded'].sum())} "
                            f"control observations without a valid baseline dropped from both sides"
                            + (f" ({reasons})" if reasons else "")))
    return written


def export_scoped_translation_plots(translation_df, plots_dir, all_groups, preference_languages, audit):
    written = []
    if translation_df.empty:
        return written
    for kind, key, _label, rel_dir in plot_scopes(translation_df):
        out_dir = Path(plots_dir) / "translation" / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        sub = filter_by_scope(translation_df, kind, key)
        for xaxis_mode, (x_col, _x_title) in XAXIS_MODES.items():
            csv_path = out_dir / f"translation_preference__x-{xaxis_mode}.csv"
            compute_translation_preference(sub, x_col).to_csv(csv_path, index=False)
            _record(written, csv_path)
            path = out_dir / f"translation_preference__x-{xaxis_mode}{FIG_EXT}"
            _record(written, path, plot_translation_preference(sub, xaxis_mode, ALL_GROUPS, all_groups, path))
        audit.append(_audit_row("translation preference", key, add_source_row_id(sub),
                                preference_filter_label(preference_languages)))
    return written


WIN_RATE_LINESTYLES = {False: "-", True: "--"}


def draw_model_size_win_rate(ax, table):
    colors = family_colors()
    series = []
    plotted = table.loc[table["win_rate_percent"].notna()]
    for (family, it), g in plotted.groupby(["family", "instruction_tuned"], sort=True):
        g = g.sort_values(["model_size", "model_name"], kind="stable")
        x, y = g["model_size"].to_numpy(dtype=float), g["win_rate_percent"].to_numpy(dtype=float)
        style = WIN_RATE_LINESTYLES[bool(it)] if len(g) > 1 else "none"
        color = colors.get(family, MUTED_COLOR)
        ax.plot(x, y, linestyle=style, linewidth=1.6, color=color, marker="o", markersize=6,
                markerfacecolor="white" if it else color, markeredgecolor=color, markeredgewidth=1.4,
                label=f"{family} ({'IT' if it else 'base'})", gid=f"{family}|{bool(it)}", zorder=3)
        for xi, yi, name in zip(x, y, g["display_name"]):
            ax.annotate(name, (xi, yi), xytext=(4, 3), textcoords="offset points", fontsize=5.5, color=MUTED_COLOR)
        series.append({"family": family, "instruction_tuned": bool(it), "models": list(g["model_name"]),
                       "x": list(x), "y": list(y), "linestyle": style})
    ax.set_xscale("log")
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200) if len(plotted)
             and plotted["model_size"].min() / 1.5 <= t <= plotted["model_size"].max() * 1.5]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t}B" for t in ticks]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(100))
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Model size (total parameters, billions; log scale)")
    ax.set_ylabel("ΔSLOR win rate: N(ΔSLOR > 0) / N(valid ΔSLOR)")
    families = [f for f in colors if f in set(plotted["family"])]
    handles = [Patch(facecolor=colors[f], label=f) for f in families]
    handles += [Line2D([0], [0], color=TEXT_COLOR, linestyle="-", marker="o", label="non-IT / base (solid)"),
                Line2D([0], [0], color=TEXT_COLOR, linestyle="--", marker="o", markerfacecolor="white",
                       label="instruction-tuned (dashed)")]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return series


def plot_model_size_win_rate(table, out_path, title):
    if table["win_rate_percent"].notna().sum() == 0:
        return False
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    draw_model_size_win_rate(ax, table)
    ax.set_title(title, color=TEXT_COLOR, loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=DPI)
    plt.close(fig)
    return True


def export_model_size_win_rate(df, plots_dir, preference_languages, include_identical):
    written = []
    out_dir = Path(plots_dir) / "probability"
    out_dir.mkdir(parents=True, exist_ok=True)
    table, unresolved = model_size_win_rate_frame(df, preference_languages, include_identical)
    if unresolved:
        print(f"[warning] model-size win rate: no registry size/family/IT, not plotted: {unresolved}")
    _write_table(written, out_dir / "model-size-vs-SLOR-win-rate.csv", table)
    path = out_dir / f"model-size-vs-SLOR-win-rate{FIG_EXT}"
    title = ("ΔSLOR win rate by model size — target-language comparison (proper vs literal translation of "
             "Proper English)\none point per model, pooled over target languages and construction groups; "
             "win = ΔSLOR > 0 (ties are not wins)\n"
             f"preference filter: {preference_filter_label(preference_languages)}")
    _record(written, path, plot_model_size_win_rate(table, path, title))
    return written


DIST_DIR = "dist"
HEATMAP_SPECS = {
    "spearman": ("spearman_model_correlation", "RdBu_r", -1.0, 1.0, "Spearman's ρ (shared sentence pairs)"),
    "sign_agreement": ("sign_agreement", "viridis", 0.0, 1.0, "sign agreement (shared non-tied pairs)"),
    "cohens_kappa": ("cohens_kappa", "RdBu_r", -1.0, 1.0, "Cohen's κ (same pairs as sign agreement)"),
}
VC_COLORS = {"sentence pair": GROUP_PALETTE[0], "model": GROUP_PALETTE[1], "residual": "#b8b6b0"}


def plot_model_heatmap(matrix, out_path, title, cmap, vmin, vmax, label):
    models = list(matrix.index)
    names = _model_labels(models)
    n = len(models)
    size = max(5.0, 0.36 * n + 2.6)
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    values = matrix.to_numpy(dtype=float)
    cm = matplotlib.colormaps[cmap].copy()
    cm.set_bad("#e4e3df")
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=6.5)
    ax.set_yticklabels(names, fontsize=6.5)
    if n <= 30:
        for i in range(n):
            for j in range(n):
                if np.isfinite(values[i, j]):
                    v = values[i, j]
                    dark = abs(v) > 0.6 if vmin < 0 else v < 0.35
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.5,
                            color="white" if dark else TEXT_COLOR)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label(label, fontsize=8)
    ax.set_title(title, loc="left", fontsize=8.5, color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=DPI)
    plt.close(fig)
    return True


def plot_variance_components(icc_row, out_path, title):
    parts = [("sentence pair", icc_row["ICC_pair"]), ("model", icc_row["proportion_model"]),
             ("residual", icc_row["proportion_residual"])]
    if not all(np.isfinite(float(v)) for _l, v in parts):
        return False
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    left = 0.0
    for name, v in parts:
        ax.barh([0], [v], left=left, color=VC_COLORS[name], edgecolor="white", label=f"{name}: {v:.1%}")
        if v > 0.04:
            ax.text(left + v / 2, 0, f"{name}\n{v:.1%}", ha="center", va="center", fontsize=8,
                    color="white" if name != "residual" else TEXT_COLOR)
        left += v
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_yticks([])
    ax.set_xlabel("share of total ΔSLOR variance")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title, loc="left", fontsize=8.5, color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=DPI)
    plt.close(fig)
    return True


def plot_icc_by_language(icc, out_path, title):
    langs = icc.loc[(icc["fit_id"] != POOLED_FIT) & icc["ICC_pair"].notna()]
    if langs.empty:
        return False
    fig, ax = plt.subplots(figsize=(max(5.5, 0.9 * len(langs) + 2.5), 4.2))
    x = np.arange(len(langs))
    ax.bar(x, langs["ICC_pair"], color=VC_COLORS["sentence pair"], width=0.6, zorder=2)
    for xi, (_i, r) in zip(x, langs.iterrows()):
        ax.text(xi, r["ICC_pair"] - 0.015, f"{r['ICC_pair']:.2f}\nn={int(r['n_obs'])}\n{int(r['n_models'])} models",
                ha="center", va="top", fontsize=6.5, color="white", zorder=3)
    pooled = icc.loc[icc["fit_id"] == POOLED_FIT, "ICC_pair"]
    if len(pooled) and np.isfinite(pooled.iloc[0]):
        ax.axhline(pooled.iloc[0], color=MUTED_COLOR, linestyle="--", linewidth=0.9,
                   label=f"pooled: {pooled.iloc[0]:.2f}")
        ax.legend(frameon=False, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(langs["fit_id"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Target language")
    ax.set_ylabel("ICC_pair = σ²_pair / (σ²_pair + σ²_model + σ²_residual)", fontsize=8)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=8.5, color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=DPI)
    plt.close(fig)
    return True


SUMMARY_SERIES = [
    ("mean_spearman_rho", "mean pairwise Spearman ρ", GROUP_PALETTE[0], "o"),
    ("mean_cohens_kappa", "mean pairwise Cohen's κ", GROUP_PALETTE[1], "s"),
    ("ICC_pair", "sentence-pair ICC", GROUP_PALETTE[2], "D"),
]


def plot_consistency_by_language(summary, out_path, title):
    if summary.empty:
        return False
    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(summary) + 3.0), 4.2))
    x = np.arange(len(summary))
    for col, label, color, marker in SUMMARY_SERIES:
        ax.plot(x, summary[col].to_numpy(dtype=float), color=color, marker=marker, markersize=6, linewidth=1.6,
                label=label, zorder=3)
    lo = np.nanmin(summary[[c for c, *_rest in SUMMARY_SERIES]].to_numpy(dtype=float))
    ax.set_ylim(min(0.0, lo - 0.05), 1.0)
    ax.axhline(0, color=MUTED_COLOR, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["target_language"])
    ax.set_xlim(-0.5, len(summary) - 0.5)
    ax.set_xlabel("Target language")
    ax.set_ylabel("cross-model consistency of ΔSLOR")
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title, loc="left", fontsize=8.5, color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=DPI)
    plt.close(fig)
    return True


def export_consistency_language_summary(out_dir, matrices, icc):
    written = []
    out_dir = Path(out_dir)
    summary = consistency_language_summary(matrices, icc)
    _write_table(written, out_dir / "cross_model_consistency_by_language.csv", summary)
    path = out_dir / f"cross_model_consistency_by_language{FIG_EXT}"
    _record(written, path, plot_consistency_by_language(
        summary, path, "Cross-model consistency of ΔSLOR by target language\n"
                       "means over estimable off-diagonal model pairs (diagonal and missing cells excluded); "
                       f"ICC from {ICC_FORMULA}"))
    return written


def export_cross_model_consistency(df, plots_dir, preference_languages, include_identical):
    written = []
    out_dir = Path(plots_dir) / DIST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pref = preference_filter_label(preference_languages)
    try:
        frame = sentence_pair_frame(df, preference_languages, include_identical)
    except SentencePairIdentityError as e:
        _write_table(written, out_dir / "cross_model_sentence_pair_duplicates.csv", e.diagnostic)
        print(f"[error] cross-model consistency skipped: {e}")
        return written
    if frame.empty:
        print("[skipped] cross-model consistency: no target-language ΔSLOR observations")
        return written
    models = _ordered_models(frame["model_name"].unique())
    res = run_consistency_analysis(df, preference_languages, include_identical, models=models)
    _write_table(written, out_dir / "cross_model_sentence_pair_data.csv", res["data"])
    _write_table(written, out_dir / "cross_model_sentence_pair_coverage.csv", res["coverage"])
    _write_table(written, out_dir / "cross_model_sentence_pair_identity_variants.csv", res["identity_variants"])
    for subset, mats in res["matrices"].items():
        suffix = "" if subset == POOLED_FIT else f"__{subset}"
        where = "pooled over target languages" if subset == POOLED_FIT else f"target language {subset}"
        for key, (stem, cmap, vmin, vmax, label) in HEATMAP_SPECS.items():
            mats[key].to_csv(out_dir / f"{stem}{suffix}.csv")
            _record(written, out_dir / f"{stem}{suffix}.csv")
            mats[f"{key}_n"].to_csv(out_dir / f"{stem}{suffix}_n.csv")
            _record(written, out_dir / f"{stem}{suffix}_n.csv")
            path = out_dir / f"{stem}{suffix}{FIG_EXT}"
            title = (f"Cross-model ΔSLOR consistency — {label}\n{COMPARISON_TITLES[TARGET_COMPARISON_POS]}; {where}; "
                     f"preference filter: {pref}\nsame exact sentence pair across models; missing never imputed; "
                     "grey = no / too few shared pairs")
            _record(written, path, plot_model_heatmap(mats[key], path, title, cmap, vmin, vmax, label))
    icc = res["icc"]
    _write_table(written, out_dir / "icc_variance_components.csv", icc.loc[icc["fit_id"] == POOLED_FIT])
    _write_table(written, out_dir / "icc_variance_components_by_language.csv", icc.loc[icc["fit_id"] != POOLED_FIT])
    summary_path = out_dir / "mixed_effects_summary.txt"
    header = (f"Null crossed random-effects model (variance decomposition only; NOT the main regression)\n"
              f"formula: {ICC_FORMULA}; lme4::lmer, REML, fitted via Rscript from export_plots2.py\n"
              f"outcome: saved {SLOR_COL} (ΔSLOR); {COMPARISON_TITLES[TARGET_COMPARISON_POS]}; preference filter: {pref}\n"
              "ICC_pair = var_sentence_pair / (var_sentence_pair + var_model + var_residual)\n\n"
              + icc.drop(columns=["formula", "estimation"]).to_string(index=False) + "\n\n")
    summary_path.write_text(header + (res["summary"] or "(no model fitted)\n"), encoding="utf-8")
    _record(written, summary_path)
    pooled = icc.loc[icc["fit_id"] == POOLED_FIT]
    for _i, r in icc.loc[~icc["status"].astype(str).str.startswith("ok")].iterrows():
        print(f"[warning] ICC fit {r['fit_id']}: {r['status']} {r['message'] if pd.notna(r['message']) else ''}")
    path = out_dir / f"icc_variance_components{FIG_EXT}"
    if len(pooled):
        r = pooled.iloc[0]
        _record(written, path, plot_variance_components(
            r, path, f"ΔSLOR variance decomposition, pooled over target languages ({ICC_FORMULA}; REML)\n"
                     f"n={int(r['n_obs'])} observations, {int(r['n_sentence_pairs'])} sentence pairs, "
                     f"{int(r['n_models'])} models; ICC_pair = {r['ICC_pair']:.3f}"))
    written += export_consistency_language_summary(out_dir, res["matrices"], icc)
    path = out_dir / f"icc_by_language{FIG_EXT}"
    _record(written, path, plot_icc_by_language(
        icc, path, f"Sentence-pair ICC of ΔSLOR by target language ({ICC_FORMULA}; REML)\n"
                   "share of ΔSLOR variance from persistent sentence-pair differences across models"))
    return written


def failure_matrix_definitions(preference_languages, include_identical, metric=MEANLP_COL):
    spec = score_metric(metric)
    return pd.DataFrame({"item": [
        "rows", "columns", "en", "de .. zh", "preference filter", "failure", "tie", "N_total", "failure_rate", "rows used",
    ], "definition": [
        "one row per model (outputs2 directory name)",
        ", ".join(FAILURE_MATRIX_COLUMNS),
        "Proper English strategy vs Literal English translation of target language strategy, pooled over every "
        "target-language origin (no preference filter)",
        "Proper strategy in target language vs Literal translation of Proper English into target language",
        preference_filter_label(preference_languages) + " (target-language columns only)",
        f"{spec['axis']} < 0 (positive = proper, negative = transferred; saved column {metric})",
        f"{spec['label']} = 0 is a tie, not a failure (sheet N_tie)",
        f"number of valid, non-missing {spec['label']} observations (denominator)",
        "proportion N_failure / N_total in [0, 1]; empty when N_total = 0",
        "standard experiment, no-context (raw) condition; identical proper/transferred pairs "
        + ("included" if include_identical else "excluded (as in every other analysis)"),
    ]})


SPEC_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

REGRESSION_DIR = "regression"
MAIN_REGRESSION_DIR = "main"
GLOSS_REGRESSION_DIR = "gloss"
MAIN_REGRESSION_TABLES = ["regression_coefficients.csv", "regression_model_summary.csv", "regression_frame.csv"]
GLOSS_REGRESSION_TABLES = ["gloss_feature_inventory.csv", "gloss_regression_pairs.csv",
                           "gloss_regression_pair_counts.csv", "gloss_feature_regressions.csv"]
REGRESSION_MANIFEST_TABLE = "regression_analyses.csv"
REGRESSION_AUDIT_TABLE = "regression_audit.csv"
REGRESSION_TABLES = MAIN_REGRESSION_TABLES + GLOSS_REGRESSION_TABLES + [REGRESSION_MANIFEST_TABLE,
                                                                         REGRESSION_AUDIT_TABLE]
LEGACY_REGRESSION_PATTERN = "meanLP_ols_coefficients*"
REGRESSION_AUDIT_PREFIXES = ("regression:", "main regression:", "gloss regression:")

MAIN_OUTCOME_TEXT = f"ΔSLOR = SLOR(positive/proper) − SLOR(negative/transferred) (saved column {SLOR_COL})"
MAIN_COEF_XLABEL = "Raw coefficient (β)"
MAIN_STD_COEF_XLABEL = "Standardized coefficient (β)"
MAIN_COEF_FIGURES = [
    ("raw", "beta", ("ci_lower", "ci_upper"), MAIN_COEF_XLABEL,
     "raw coefficients β with 95% CI (units: ΔSLOR per predictor unit)"),
    ("standardized", "standardized_beta", ("standardized_ci_lower", "standardized_ci_upper"), MAIN_STD_COEF_XLABEL,
     "standardized coefficients β·sd(x)/sd(ΔSLOR) with 95% CI scaled by the same factor"),
]


def clean_regression_outputs(plots_dir):
    plots_dir = Path(plots_dir)
    removed = []
    for exp_dir in [plots_dir] + [plots_dir / exp for exp, _l in EXPERIMENTS if exp != EXPERIMENT_STANDARD]:
        reg_dir = exp_dir / REGRESSION_DIR
        if reg_dir.is_dir():
            removed += sorted(p for p in reg_dir.rglob("*") if p.is_file())
            shutil.rmtree(reg_dir)
    for name in REGRESSION_TABLES:
        path = plots_dir / "tables" / name
        if path.is_file():
            path.unlink()
            removed.append(path)
    if plots_dir.is_dir():
        for path in sorted(plots_dir.rglob(LEGACY_REGRESSION_PATTERN)):
            if path.is_file():
                path.unlink()
                removed.append(path)
    for path in removed:
        print(f"[removed stale regression artifact] {path}")
    return removed


def standardized_coefficient_intervals(coefs):
    out = coefs.copy()
    beta = out["beta"].astype(float)
    scale = (out["standardized_beta"].astype(float) / beta).where(beta != 0)
    out["standardized_ci_lower"] = out["ci_lower"].astype(float) * scale
    out["standardized_ci_upper"] = out["ci_upper"].astype(float) * scale
    return out


REG_FACTOR_SEPARATOR_STYLE = {"color": "#B5B5B5", "linewidth": 0.8, "linestyle": "-", "zorder": 0.5}


def draw_factor_separators(ax, n_factors):
    return [ax.axhline(i + 0.5, **REG_FACTOR_SEPARATOR_STYLE) for i in range(n_factors - 1)]


def _plot_main_full_model(rows, out_path, title, xlabel=MAIN_COEF_XLABEL, value_col="beta",
                          ci_cols=("ci_lower", "ci_upper")):
    rows = rows.loc[rows[value_col].notna() & rows["term"].isin(TERM_ORDER)].copy()
    rows["_order"] = rows["term"].map({t: i for i, t in enumerate(TERM_ORDER)})
    rows = rows.sort_values("_order")
    if rows.empty:
        return False
    fig_h = max(4.0, 0.55 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    y = np.arange(len(rows))
    beta = rows[value_col].to_numpy(dtype=float)
    lo = rows[ci_cols[0]].to_numpy(dtype=float)
    hi = rows[ci_cols[1]].to_numpy(dtype=float)
    ax.errorbar(beta, y, xerr=np.vstack([beta - lo, hi - beta]), fmt="o",
                capsize=3, markersize=5, linewidth=1.2)
    ax.axvline(0, color=MUTED_COLOR, linestyle="--", linewidth=0.9, zorder=0)
    draw_factor_separators(ax, len(rows))
    ax.set_yticks(y)
    ax.set_yticklabels([TERM_LABELS.get(t, t) for t in rows["term"]], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=9, color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=DPI)
    plt.close(fig)
    return True


TERM_ORDER = [LOG_SIZE_TERM, COMMON_CRAWL_TERM, IT_TERM, STRONG_PREF_TERM] + [t for t, _c, _l in METADATA_FEATURES]


def _write_table(written, path, table):
    table.to_csv(path, index=False)
    _record(written, path)

#この関数が論文中のsec5.2 (fig5)の回帰処理を開始
def export_main_regression(df, plots_dir, langs, metadata, include_identical, audit, exclude_freq=False,
                           glotcc=False):
    written = []
    tables_dir = Path(plots_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    res = run_main_regression(df, langs, metadata if not metadata.empty else None,
                              include_identical=include_identical, exclude_freq=exclude_freq, glotcc=glotcc)
    frame, coefs, summary, report = res["frame"], res["coefficients"], res["summary"], res["report"]
    assert res["outcome"] == SLOR_COL and (frame["outcome"] == SLOR_COL).all()
    assert np.array_equal(frame["y"].to_numpy(dtype=float), frame[SLOR_COL].to_numpy(dtype=float), equal_nan=True)
    if len(summary) != 1 or summary.iloc[0]["scope"] != "pooled":
        raise AssertionError("active main regression must contain exactly one pooled specification")

    r = summary.iloc[0]
    print(f"[main regression] ONE full pooled model; outcome = {res['outcome']} ({res['outcome_label']}); "
          f"complete-case N = {int(r['n'])}; languages = {r['languages']}")
    print(f"[main regression] predictors = {r['predictors']}; language fixed effects = {r['language_fixed_effects']}")
    if res["features_excluded"]:
        print(f"[main regression] excluded by --exclude_freq: {res['features_excluded']}")
    if glotcc:
        print(f"[warning] --glotcc: {COMMON_CRAWL_CAVEAT}")
    if res["features_missing"]:
        print(f"[warning] metadata features unavailable: {res['features_missing']}")
    omitted = report.get("constant_unidentifiable_in_joint_sample", [])
    if omitted:
        print(f"[warning] unidentifiable in joint complete-case sample, omitted from the ONE full model: {omitted}")
    for status, n in report.get("metadata_join_status", {}).items():
        if status != "ok":
            print(f"[warning] {n} regression rows have metadata_join_status={status} "
                  f"(by model: {report.get('metadata_join_not_ok_by_model')})")

    _write_table(written, tables_dir / "regression_coefficients.csv", coefs)
    _write_table(written, tables_dir / "regression_model_summary.csv", summary)
    _write_table(written, tables_dir / "regression_frame.csv",
                 frame.drop(columns=["pos_sentence", "neg_sentence"], errors="ignore"))

    main_dir = Path(plots_dir) / REGRESSION_DIR / MAIN_REGRESSION_DIR
    main_dir.mkdir(parents=True, exist_ok=True)
    omitted_note = f"\nUnidentifiable in joint sample: {'; '.join(omitted)}" if omitted else ""
    if res["features_excluded"]:
        omitted_note += f"\nExcluded by --exclude_freq: {', '.join(res['features_excluded'])}"
    plot_coefs = standardized_coefficient_intervals(coefs)
    fe_note = (f"{COMMON_CRAWL_LABEL} instead of language fixed effects" if glotcc
               else "language fixed effects included as controls")
    for suffix, value_col, ci_cols, xlabel, note in MAIN_COEF_FIGURES:
        path = main_dir / f"main_regression_delta-SLOR_coefficients__{suffix}{FIG_EXT}"
        _record(written, path, _plot_main_full_model(
            plot_coefs, path,
            f"Main multivariable regression predicting ΔSLOR — {note}\n"
            f"all displayed coefficients from the same full model; {fe_note}; "
            f"N={int(r['n'])}{omitted_note}", xlabel=xlabel, value_col=value_col, ci_cols=ci_cols))

    audit.append({
        "analysis": "main regression: full multivariable model", "scope": "pooled", "languages": r["languages"],
        "comparison": TARGET_COMPARISON_POS, "outcome": r["outcome"],
        "preference_filter": preference_filter_label(langs),
        "n_source_contrasts": r["n_source_contrasts"], "n_observations": r["n"],
        "models_included": r["models"],
        "models_excluded_registry": ",".join(report.get("models_excluded_unresolved_registry", [])),
        "family_scope": FAMILY_SCOPE_ALL,
        "metadata_features_used": ",".join(t for t in r["predictors"].split(" + ") if t in METADATA_FEATURE_COLUMNS),
        "metadata_features_missing": ",".join(res["features_missing"]),
        "rows_dropped_missing_metadata": r["n_dropped_missing_predictor"],
        "notes": "; ".join(n for n in [
            (f"features omitted in joint complete-case sample: {r['features_omitted_in_scope']}"
             if r["features_omitted_in_scope"] else ""),
            (f"excluded by --exclude_freq: {','.join(res['features_excluded'])}" if res["features_excluded"] else ""),
            (f"--glotcc: {COMMON_CRAWL_CAVEAT}" if glotcc else ""),
        ] if n),
    })
    return written, res


def export_gloss_regression(df, plots_dir, langs, metadata, include_identical, audit):
    written = []
    tables_dir = Path(plots_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    gloss_dir = Path(plots_dir) / REGRESSION_DIR / GLOSS_REGRESSION_DIR
    gloss_dir.mkdir(parents=True, exist_ok=True)
    inventory = gloss_feature_inventory(metadata) if not metadata.empty else pd.DataFrame()
    if len(inventory):
        inventory = inventory.loc[inventory["language"].isin(langs)]
    _write_table(written, tables_dir / "gloss_feature_inventory.csv",
                 inventory if len(inventory) else pd.DataFrame(columns=["language", "gloss_feature"]))
    gloss_frame = gloss_regression_frame(df, langs, metadata if not metadata.empty else None,
                                         include_identical=include_identical)
    _write_table(written, tables_dir / "gloss_regression_pairs.csv", gloss_frame)
    counts = gloss_pair_counts(gloss_frame)
    _write_table(written, tables_dir / "gloss_regression_pair_counts.csv", counts)
    gloss_table, gloss_status = run_gloss_feature_regressions(gloss_frame, languages=langs,
                                                              inventory=inventory if len(inventory) else None)
    _write_table(written, tables_dir / "gloss_feature_regressions.csv", gloss_table)
    print(f"[gloss regressions] outcome = {GLOSS_OUTCOME_LABEL} (model-mean {SLOR_COL}); {gloss_status}")
    estimable = gloss_table.loc[gloss_table["status"] == "ok"] if not gloss_table.empty else gloss_table
    if not estimable.empty:
        for lang, g in estimable.groupby("language"):
            fig, ax = plt.subplots(figsize=(6, 0.35 * len(g) + 1.4))
            ax.barh(range(len(g)), g["standardized_beta"], color=SPEC_COLORS[0])
            ax.set_yticks(range(len(g)))
            ax.set_yticklabels(g["gloss_feature"], fontsize=7)
            ax.invert_yaxis()
            ax.axvline(0, color=MUTED_COLOR, linestyle="--", linewidth=0.9)
            ax.set_xlabel("standardized β of univariate OLS: pair-mean ΔSLOR (across models) ~ 1 + G_j\n"
                          "(no controls; one observation per unique sentence pair)", fontsize=7)
            ax.set_title(f"Gloss-feature regressions on ΔSLOR — {lang} "
                         f"(features with N_contrast ≥ {GLOSS_MIN_CONTRAST})", fontsize=9, loc="left")
            p = gloss_dir / f"gloss_regression_delta-SLOR_feature_ranking__language-{lang}{FIG_EXT}"
            fig.savefig(p, format="pdf", dpi=DPI)
            plt.close(fig)
            _record(written, p)
    status_path = gloss_dir / "gloss_feature_regressions_status.txt"
    status_path.write_text(f"{gloss_status}\noutcome: {GLOSS_OUTCOME_LABEL} = mean over models of saved "
                           f"{SLOR_COL} (ΔSLOR) per unique sentence pair\n", encoding="utf-8")
    _record(written, status_path)

    n_pairs = counts.set_index("language") if len(counts) else pd.DataFrame()
    for lang in langs:
        t = gloss_table.loc[gloss_table["language"] == lang] if not gloss_table.empty else gloss_table
        n_status = t["status"].value_counts().to_dict() if len(t) else {}
        audit.append({
            "analysis": "gloss regression: pair-mean ΔSLOR ~ 1 + G_j", "scope": f"language={lang}",
            "family_scope": FAMILY_SCOPE_ALL, "comparison": TARGET_COMPARISON_POS, "languages": lang,
            "outcome": f"{GLOSS_OUTCOME_LABEL} ({SLOR_COL})", "preference_filter": preference_filter_label(langs),
            "n_source_contrasts": int(n_pairs.loc[lang, "n_pairs_valid_gloss"]) if lang in n_pairs.index else 0,
            "n_observations": int(n_pairs.loc[lang, "n_pairs_valid_gloss"]) if lang in n_pairs.index else 0,
            "models_included": "", "models_excluded_registry": "", "metadata_features_used": "",
            "metadata_features_missing": "", "rows_dropped_missing_metadata": "",
            "notes": (f"unique pairs available: {int(n_pairs.loc[lang, 'n_pairs']) if lang in n_pairs.index else 0}; "
                      "observations = pairs with valid gloss contrasts"
                      + ("; features: " + ", ".join(f"{k}={v}" for k, v in sorted(n_status.items())) if n_status else "")),
        })
    return written, {"frame": gloss_frame, "counts": counts, "table": gloss_table, "status": gloss_status}


def regression_manifest(main_res, gloss_res, langs):
    frame, summary = main_res["frame"], main_res["summary"]
    predictors = sorted({t for p in summary["predictors"] for t in p.split(" + ") if t}, key=TERM_ORDER.index)
    rows = [{
        "analysis": "main_regression", "outcome": main_res["outcome"], "outcome_label": main_res["outcome_label"],
        "sign_convention": "SLOR(positive/proper) - SLOR(negative/transferred)",
        "observation_unit": "one (model, sentence pair) row",
        "scope": ("one pooled full model with language FE" if main_res.get("language_fe", True)
                  else f"one pooled full model with {COMMON_CRAWL_LABEL} instead of language FE"),
        "languages": ",".join(langs), "predictors": " + ".join(predictors),
        "n_observations": int(len(frame)), "n_unique_pairs": int(frame[SOURCE_ROW_ID_COL].nunique()) if len(frame) else 0,
        "n_models": int(frame["model_name"].nunique()) if len(frame) else 0,
        "tables": ",".join(MAIN_REGRESSION_TABLES),
    }]
    if main_res.get("features_excluded"):
        rows[0]["excluded_predictors"] = ",".join(main_res["features_excluded"]) + " (--exclude_freq)"
    counts = gloss_res["counts"].set_index("language") if len(gloss_res["counts"]) else pd.DataFrame()
    table = gloss_res["table"]
    for lang in langs:
        t = table.loc[table["language"] == lang] if not table.empty else table
        rows.append({
            "analysis": "gloss_feature_regression", "outcome": SLOR_COL, "outcome_label": GLOSS_OUTCOME_LABEL,
            "sign_convention": "mean over models of SLOR(positive/proper) - SLOR(negative/transferred)",
            "observation_unit": "one unique sentence pair", "scope": f"language={lang}", "languages": lang,
            "predictors": f"1 + G_j (one gloss feature per fit; N_contrast >= {GLOSS_MIN_CONTRAST})",
            "n_observations": int(counts.loc[lang, "n_pairs_valid_gloss"]) if lang in counts.index else 0,
            "n_unique_pairs": int(counts.loc[lang, "n_pairs_valid_gloss"]) if lang in counts.index else 0,
            "n_unique_pairs_available": int(counts.loc[lang, "n_pairs"]) if lang in counts.index else 0,
            "n_models": int(gloss_res["frame"].loc[gloss_res["frame"]["language"] == lang, "n_models"].max())
            if lang in counts.index else 0,
            "tables": ",".join(GLOSS_REGRESSION_TABLES),
            "n_features_fitted": int((t["status"] == "ok").sum()) if len(t) else 0,
            "n_features_insufficient_contrast": int((t["status"] == "insufficient_contrast").sum()) if len(t) else 0,
            "n_features_no_variation": int((t["status"] == "no_variation").sum()) if len(t) else 0,
        })
    return pd.DataFrame(rows)


def export_regressions(df, plots_dir, preference_languages, metadata_path, include_identical, audit,
                       exclude_freq=False, glotcc=False):
    clean_regression_outputs(plots_dir)
    written = []
    langs = parse_preference_languages(preference_languages)
    if not langs:
        print("[skipped] regressions: no --use-preference-annotation languages given")
        return written
    metadata = load_metadata_table(metadata_path) if metadata_path else pd.DataFrame()
    if metadata.empty:
        print(f"[warning] metadata workbook not found ({metadata_path}); regression uses model-level predictors only")
    reg_audit = []
    w, main_res = export_main_regression(df, plots_dir, langs, metadata, include_identical, reg_audit,
                                         exclude_freq=exclude_freq, glotcc=glotcc)
    written += w
    w, gloss_res = export_gloss_regression(df, plots_dir, langs, metadata, include_identical, reg_audit)
    written += w
    tables_dir = Path(plots_dir) / "tables"
    _write_table(written, tables_dir / REGRESSION_MANIFEST_TABLE, regression_manifest(main_res, gloss_res, langs))
    _write_table(written, tables_dir / REGRESSION_AUDIT_TABLE, pd.DataFrame(reg_audit, columns=AUDIT_COLUMNS))
    audit += reg_audit
    return written


AUDIT_COLUMNS = [
    "analysis", "outcome", "scope", "family_scope", "comparison", "languages", "preference_filter", "n_source_contrasts",
    "n_observations", "models_included", "models_excluded_registry", "metadata_features_used",
    "metadata_features_missing", "rows_dropped_missing_metadata", "notes",
]


def _audit_row(analysis, scope, rows, preference_filter, comparison=""):
    langs = sorted(rows["Target language"].astype(str).unique()) if "Target language" in rows.columns and len(rows) else []
    models = models_in(rows)
    return {
        "analysis": analysis, "scope": scope, "family_scope": scope, "comparison": comparison,
        "languages": ",".join(langs), "preference_filter": preference_filter,
        "n_source_contrasts": int(rows[SOURCE_ROW_ID_COL].nunique()) if SOURCE_ROW_ID_COL in rows.columns else None,
        "n_observations": int(len(rows)), "models_included": ",".join(models),
        "models_excluded_registry": ",".join(unresolved_model_names(rows)) if scope == FAMILY_SCOPE_ALL else "",
        "metadata_features_used": "", "metadata_features_missing": "", "rows_dropped_missing_metadata": "",
        "notes": "",
    }


def export_failure_matrix_audit(df, preference_languages, include_identical, audit):
    m = compute_failure_rate_matrix(df, preference_languages, include_identical=include_identical)
    for col in FAILURE_MATRIX_COLUMNS:
        audit.append({
            "analysis": "failure-rate matrix", "scope": col, "family_scope": FAMILY_SCOPE_ALL,
            "comparison": ENGLISH_COMPARISON_POS if col == "en" else TARGET_COMPARISON_POS, "languages": col,
            "preference_filter": (preference_filter_label(preference_languages)
                                  if col in parse_preference_languages(preference_languages) else "none"),
            "n_source_contrasts": None, "n_observations": int(m["N_total"][col].sum()),
            "models_included": ",".join(m["N_total"].index[m["N_total"][col] > 0]),
            "models_excluded_registry": "", "metadata_features_used": "", "metadata_features_missing": "",
            "rows_dropped_missing_metadata": "", "notes": f"N_failure total = {int(m['N_failure'][col].sum())}",
        })


def export_audit(audit, plots_dir):
    out_dir = Path(plots_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "analysis_audit.csv"
    pd.DataFrame(audit, columns=AUDIT_COLUMNS).to_csv(path, index=False)
    written = []
    _record(written, path)
    return written


def merge_regression_audit(regression_audit, plots_dir):
    path = Path(plots_dir) / "tables" / "analysis_audit.csv"
    written = []
    if not path.is_file():
        return written
    old = pd.read_csv(path, keep_default_na=False, dtype=str)
    keep = old.loc[~old["analysis"].astype(str).str.startswith(REGRESSION_AUDIT_PREFIXES)]
    merged = pd.concat([keep, pd.DataFrame(regression_audit, columns=AUDIT_COLUMNS)], ignore_index=True)
    merged.reindex(columns=AUDIT_COLUMNS).to_csv(path, index=False)
    _record(written, path)
    return written


def _checked_dirs(outputs_dir, plots_dir):
    outputs_dir, plots_dir = Path(outputs_dir), Path(plots_dir)
    if plots_dir.resolve() == outputs_dir.resolve() or outputs_dir.resolve() in plots_dir.resolve().parents:
        raise ValueError("--plots_dir must not be inside --outputs_dir (experimental results are never overwritten)")
    return outputs_dir, plots_dir


def export_regressions_only(outputs_dir, plots_dir, include_identical=False, preference_languages=None,
                            metadata_path=None, exclude_freq=False, glotcc=False):
    outputs_dir, plots_dir = _checked_dirs(outputs_dir, plots_dir)
    preference_languages = parse_preference_languages(preference_languages)
    if not preference_languages:
        raise ValueError("--regression-only requires --use-preference-annotation LANG ... (the regression languages)")
    print(f"[regression-only] languages: {', '.join(preference_languages)}; "
          f"preference filter: {preference_filter_label(preference_languages)}")
    df, warnings = load_all_data2(outputs_dir)
    for w in warnings:
        print(f"[warning] {w}")
    if df.empty:
        raise ValueError(f"no probability outputs found in {outputs_dir}")
    unresolved = unresolved_model_names(df)
    if unresolved:
        print(f"[warning] no registry metadata (excluded from the main regression): {unresolved}")
    audit = []
    written = export_regressions(df, plots_dir, preference_languages, metadata_path, include_identical, audit,
                                 exclude_freq=exclude_freq, glotcc=glotcc)
    written += merge_regression_audit(audit, plots_dir)
    return written


def export_all(outputs_dir, plots_dir, include_identical=False, preference_languages=None, metadata_path=None,
               run_regression=False, exclude_freq=False, glotcc=False):
    outputs_dir, plots_dir = _checked_dirs(outputs_dir, plots_dir)
    preference_languages = parse_preference_languages(preference_languages)
    print(f"[preference filter] {preference_filter_label(preference_languages)}")

    df, warnings = load_all_data2(outputs_dir)
    translation_df, translation_warnings = load_all_translation_data2(outputs_dir)
    for w in warnings + translation_warnings:
        print(f"[warning] {w}")

    unresolved = unresolved_model_names(df, translation_df)
    if unresolved:
        print(f"[warning] no registry metadata (excluded from model-split views and OLS): {unresolved}")
    translation_df = filter_target_preference(translation_df, preference_languages)

    all_groups = group_choices(df, translation_df)
    written = []
    for experiment, _label in EXPERIMENTS:
        exp_df = filter_by_experiment(df, experiment) if not df.empty else df
        if exp_df.empty:
            if experiment != EXPERIMENT_STANDARD:
                print(f"[skipped] no {experiment} results")
                continue
        exp_dir = plots_dir if experiment == EXPERIMENT_STANDARD else plots_dir / experiment
        if not exp_df.empty:
            written += export_probability_plots(df, exp_dir, all_groups, include_identical=include_identical,
                                                experiment=experiment, preference_languages=preference_languages)
        if experiment == EXPERIMENT_STANDARD:
            written += export_translation_plots(translation_df, plots_dir, all_groups)
        if not exp_df.empty:
            for metric in SCORE_METRICS:
                written += export_failure_rates(df, exp_dir, include_identical=include_identical,
                                                experiment=experiment, preference_languages=preference_languages,
                                                metric=metric)
    written += export_model_metadata(df, translation_df, plots_dir)

    audit = []
    if not df.empty:
        written += export_scoped_probability_plots(df, plots_dir, preference_languages, include_identical, audit)
        written += export_japanese_control_plots(df, plots_dir, include_identical, audit)
        written += export_japanese_control_paired_baseline_plots(df, plots_dir, include_identical, audit)
        written += export_model_size_win_rate(df, plots_dir, preference_languages, include_identical)
        written += export_cross_model_consistency(df, plots_dir, preference_languages, include_identical)
        export_failure_matrix_audit(df, preference_languages, include_identical, audit)
        if run_regression:
            written += export_regressions(df, plots_dir, preference_languages, metadata_path, include_identical, audit,
                                          exclude_freq=exclude_freq, glotcc=glotcc)
    written += export_scoped_translation_plots(translation_df, plots_dir, all_groups, preference_languages, audit)
    written += export_audit(audit, plots_dir)
    return written


MAIN_FIG_DIR = "main_fig"
MAIN_FIG_EXT = FIG_EXT
MAIN_COEF_FIGURE = "main_regression_delta-SLOR_coefficients_standardized" + MAIN_FIG_EXT
MAIN_FAILURE_TABLE = "SLOR_failure_rates.xlsx"
MAIN_FAILURE_BAR_FIGURE = "SLOR_failure_rates_bar" + MAIN_FIG_EXT
MAIN_FAILURE_BAR_TABLE = "SLOR_failure_rates_bar.csv"
MAIN_FAILURE_LINE_FIGURE = "SLOR_failure_rates_line" + MAIN_FIG_EXT
MAIN_FAILURE_LINE_TABLE = "SLOR_failure_rates_line.csv"
MAIN_TRANSLATION_STEM = "translation_preference__x-model__transferred_line"
MAIN_SPEARMAN_FIGURE = "Spearman_model_correlation" + MAIN_FIG_EXT
MAIN_SPEARMAN_MANACA_STEM = "Spearman_model_correlation_manaca"
MAIN_VIOLIN_STEM = "delta-SLOR__cmp-proper-strategy-in-target-language"

MAIN_PLOT_DIR_NAME = "Main_plot"
PAPER_FIGURE_SCRIPTS = {
    Fig2_4.FIGURE_A: "src/Fig2_4.py",
    Fig2_4.FIGURE_B: "src/Fig2_4.py",
    Fig2_4.FIGURE_COMBINED: "src/Fig2_4.py",
    Fig2_4.FIGURE_JA: "src/Fig2_4.py",
    Fig3.FIGURE: "src/Fig3.py",
    Fig5.FIGURE: "src/Fig5.py",
    Fig6.FIGURE: "src/Fig6.py",
}
_DELEGATED_DIRS = []


def is_delegated(out_path):
    p = Path(out_path)
    return bool(_DELEGATED_DIRS) and p.name in PAPER_FIGURE_SCRIPTS and p.parent.resolve() == _DELEGATED_DIRS[-1]


def default_main_plot_dir(plots_dir, main_dir=None):
    return Path(main_dir) / MAIN_PLOT_DIR_NAME if main_dir else Path(plots_dir).resolve().parent / MAIN_PLOT_DIR_NAME


def export_paper_figures(df, main_dir, main_plot_dir, langs, include_identical, manifest):
    written, produced = [], {}
    main_plot_dir = Path(main_plot_dir)
    for key, (path, means) in Fig2_4.generate(df=df, out_dir=main_plot_dir, langs=langs,
                                              include_identical=include_identical).items():
        produced[path.name] = (path, f"{len(means)} violins (means shown; Figure 4 paired means joined)")
    path, table = Fig3.generate(workbook=Path(main_dir) / MAIN_FAILURE_TABLE, out_dir=main_plot_dir)
    produced[path.name] = (path, f"{table['order'].nunique()} models from {MAIN_FAILURE_TABLE}")
    path = Fig5.generate(input_dir=main_dir, out_dir=main_plot_dir)
    produced[path.name] = (path, f"from {Fig5.COEFFICIENTS_CSV} / {Fig5.SUMMARY_CSV}")
    res = Fig6.generate(df=df, out_dir=main_plot_dir, langs=langs, include_identical=include_identical)
    if res is not None:
        produced[res[0].name] = (res[0], f"{len(res[1])} models")
    for name, (path, note) in produced.items():
        copy = Path(main_dir) / name
        shutil.copyfile(path, copy)
        _record(written, path)
        _manifest_row(manifest, f"paper figure: {name}", path, "", f"rendered by {PAPER_FIGURE_SCRIPTS[name]} "
                      "(standalone; selection documented in its docstring)",
                      notes=f"{note}; identical copy at {copy}")
    return written, set(produced)


def export_main_ja_control_lmm(df, out_path, include_identical, manifest):
    res = ja_control_lmm.run_ja_control_lmm(df, include_identical)
    path = ja_control_lmm.write_workbook(res, out_path, include_identical)
    lc = res["language_contrasts"]
    for _i, r in (lc.loc[lc["specification"] == "primary"].iterrows() if len(lc) else []):
        print(f"[ja-control LMM] {r['target_language']}: diff = {r['estimated_difference']:.3f} "
              f"(SE {r['std_error']:.3f}; descriptive, not tested)")
    pooled = res["pooled"]
    if len(pooled):
        r = pooled.loc[pooled["specification"] == "primary"].iloc[0]
        print(f"[ja-control LMM] pooled (equal weights): diff = {r['estimated_difference']:.3f} "
              f"[{r['ci_lower_95']:.3f}, {r['ci_upper_95']:.3f}], t({r['df']:.2f}) = {r['wald_statistic']:.2f}, "
              f"p = {r['p_value']:.3g}")
    else:
        print("[warning] ja-control LMM: no valid paired observations (workbook states it; nothing estimated)")
    _manifest_row(manifest, "Japanese-control paired LMM (English-centric models)", path, int(len(res["pairs"])),
                  "Figure 4 pairing (Japanese annotation >> / ja target annotation >>), all English-centric "
                  "statistical models", notes="sheets " + ", ".join(ja_control_lmm.SHEETS))
    return [path], res


MAIN_VIOLIN_METRICS = [SLOR_COL, MEANLP_COL]
MAIN_METRIC_Y_LABELS = {SLOR_COL: "ΔSLOR", MEANLP_COL: "ΔmeanLP"}


def _metric_tag(metric):
    return "" if metric == SLOR_COL else f" {score_metric(metric)['label']}"


def main_violin_stem(metric):
    return f"{score_metric(metric)['slug']}__cmp-proper-strategy-in-target-language"


MAIN_FAMILY_STYLE = {
    "aya": {"color": family_color("Aya"), "marker": "o"},
    "eurollm": {"color": family_color("EuroLLM"), "marker": "s"},
    "gemma": {"color": family_color("Gemma 4"), "marker": "^"},
    "apertus": {"color": family_color("Apertus"), "marker": "D"},
}
MAIN_TICK_SIZE = 14
MAIN_LABEL_SIZE = 17
MAIN_LEGEND_SIZE = 13

MAIN_MERGED_VIOLIN_STEM = "delta-SLOR__multilingual-monolingual"
MAIN_MERGED_VIOLIN_PANELS = [
    ("a", "multilingual"),
    ("b", "monolingual-setA"),
]
MAIN_PANEL_TITLES = {"multilingual": "Multilingual models", "monolingual-setA": "Non-English-centric models"}


def main_merged_violin_stem(metric):
    return f"{score_metric(metric)['slug']}__multilingual-monolingual"


MAIN_SPLIT_VIOLIN_STEM = "violin_main"


def main_split_violin_name(letter, metric, pref_slug, ablation=""):
    name = f"{MAIN_SPLIT_VIOLIN_STEM}_{letter}"
    if metric != SLOR_COL:
        name += f"__{score_metric(metric)['slug']}"
    if pref_slug != "gtgt":
        name += f"__pref-{pref_slug}"
    if ablation:
        name += f"__{ablation}"
    return name + MAIN_FIG_EXT


MAIN_PANEL_B_ABLATIONS = [("no-sv", ("sv",)), ("no-sv-no-de", ("sv", "de"))]


def ablate_languages(rows, drop):
    if not len(rows):
        return rows
    return rows.loc[~rows["Target language"].isin(set(drop))]


MAIN_SEPARATE_VIOLIN_SETS = ("monolingual-setB",)
MONOLINGUAL_BALANCED = "Balanced Language Models"
MONOLINGUAL_CONTINUAL = "Continually Pretrained Model"
MONOLINGUAL_PURE = "Monolingual Model"
MONOLINGUAL_KINDS = (MONOLINGUAL_BALANCED, MONOLINGUAL_CONTINUAL, MONOLINGUAL_PURE)
MAIN_MONOLINGUAL_TRAINING = {
    "OpenLLM-France_Lucie-7B": (MONOLINGUAL_BALANCED, "En- and Fr-focused"),
    "maritaca-ai_sabia-7b": (MONOLINGUAL_CONTINUAL, "Pt-focused"),
    MAIN_MANACA_MODEL: (MONOLINGUAL_PURE, "Pt-focused"),
    "hfl_chinese-llama-2-13b": (MONOLINGUAL_CONTINUAL, "Zh-focused"),
    "llm-jp_llm-jp-3-13b": (MONOLINGUAL_BALANCED, "En- and Ja-focused"),
}
_MONOLINGUAL_NAMES = {"OpenLLM-France_Lucie-7B": "Lucie 7B", "maritaca-ai_sabia-7b": "Sabiá 7B",
                      MAIN_MANACA_MODEL: "Manacá 1B",
                      "hfl_chinese-llama-2-13b": "Chinese-LLaMA-2 13B", "llm-jp_llm-jp-3-13b": "LLM-jp-3 13B"}
MAIN_MONOLINGUAL_LEGEND_LABELS = {m: f"{_MONOLINGUAL_NAMES[m]} ({kind}; {shares})"
                                  for m, (kind, shares) in MAIN_MONOLINGUAL_TRAINING.items()}
MAIN_MONOLINGUAL_TARGET_LANGUAGE = {
    "OpenLLM-France_Lucie-7B": "fr",
    "maritaca-ai_sabia-7b": "pt",
    MAIN_MANACA_MODEL: "pt",
    "hfl_chinese-llama-2-13b": "zh",
    "llm-jp_llm-jp-3-13b": "ja",
}
MUTED_NEUTRAL = "#D0D0D0"
NON_FOCUS_GRAY = "#BDBDBD"
MAIN_PANEL_LABEL_SIZE = 20

MAIN_JAPANESE_CONTROL_MODELS = [
    ("utter-project_EuroLLM-22B-2512", "EuroLLM 22B"),
    ("google_gemma-4-31B", "Gemma 4 31B"),
    ("CohereLabs_aya-expanse-32b", "Aya Expanse 32B"),
    ("swiss-ai_Apertus-70B-2509", "Apertus 70B"),
]
MAIN_JAPANESE_CONTROL_PREFERENCE = "gtgt"
MAIN_JAPANESE_CONTROL_STEM = f"delta-SLOR__{PAIRED_BASELINE_SLUG}__pref-{MAIN_JAPANESE_CONTROL_PREFERENCE}"


def main_japanese_control_stem(metric):
    return f"{score_metric(metric)['slug']}__{PAIRED_BASELINE_SLUG}__pref-{MAIN_JAPANESE_CONTROL_PREFERENCE}"


MAIN_MIN_PAIRED_OBSERVATIONS = 2

MAIN_WIN_RATE_STEM = "model-size-vs-SLOR-win-rate"
MAIN_WIN_RATE_YLIM = (20, 70)
MAIN_WIN_RATE_FAMILIES = ("aya", "eurollm", "apertus")

def main_model_style(model):
    color, _hatch = violin_model_style(model)
    return color, None


MAIN_MODEL_STYLE = {m: main_model_style(m) for m in dict.fromkeys(
    [m for _s, models in MAIN_VIOLIN_SETS for m, _l in models] + [m for m, _l in MAIN_MONOLINGUAL_SET_A_MANACA]
    + [m for m, _l in MAIN_JAPANESE_CONTROL_MODELS])}
NO_MEAN_MEDIAN_SUFFIX = "__no-mean-median"
NO_MEDIAN_TWIN_NOTE = "without the median bar and its legend key; mean diamonds kept (file name kept for compatibility although means are shown, CLAUDE.md 2026-10-13 §3.1)"


def no_mean_median_path(path):
    path = Path(path)
    return path.with_name(f"{path.stem}{NO_MEAN_MEDIAN_SUFFIX}{path.suffix}")


def write_violin_pair(written, manifest, path, draw, artifact, n_rows, selection, requested="", missing="",
                      notes=""):
    ylim = draw(path, True, None)
    _record(written, path, bool(ylim))
    twin = no_mean_median_path(path)
    ylim2 = draw(twin, False, tuple(ylim) if ylim else None) if ylim else False
    _record(written, twin, bool(ylim2))
    _manifest_row(manifest, artifact, path, n_rows, selection, requested, missing, notes)
    _manifest_row(manifest, f"{artifact} [no mean/median]", twin, n_rows, selection, requested, missing,
                  notes=f"same as {Path(path).name} {NO_MEDIAN_TWIN_NOTE} "
                        "(identical rows, KDE, colours, order, legend position and y limits); " + notes)
    return ylim


def check_violin_colors(models, context):
    names = [m for m, _l in models]
    unassigned = [m for m in names if m.replace("_", "/", 1) not in VIOLIN_MODEL_COLORS]
    if unassigned:
        print(f"[warning] main_fig {context}: no fixed checkpoint colour for {unassigned} (grey)")
    colors = [main_model_style(m)[0] for m in names if m not in unassigned]
    off = [c for c in colors if c not in set(STRICT_PALETTE.values())]
    if off:
        raise AssertionError(f"main_fig {context}: colours outside the strict palette: {off}")
    same = [(a, b) for a, b in zip(names, names[1:]) if a not in unassigned and b not in unassigned
            and main_model_style(a)[0] == main_model_style(b)[0]]
    if same:
        print(f"[warning] main_fig {context}: adjacent models share a violin colour: {same}")


SPLIT_VIOLIN_WIDTH = 7.5
SPLIT_VIOLIN_AXES_HEIGHT = 3.3
SPLIT_TICK_SIZE = 16
SPLIT_LABEL_SIZE = 18
SPLIT_LEGEND_SIZE = 14
SPLIT_Y_PAD_FRACTION = 0.05


def muted_model_color(original_color, strength=0.75):
    original = np.asarray(matplotlib.colors.to_rgb(original_color))
    neutral = np.asarray(matplotlib.colors.to_rgb(MUTED_NEUTRAL))
    return tuple((1 - strength) * original + strength * neutral)


def _main_dir(plots_dir, main_dir=None):
    d = Path(main_dir) if main_dir else Path(plots_dir) / MAIN_FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_row(manifest, artifact, path, n_rows, selection, requested="", missing="", notes=""):
    manifest.append({"artifact": artifact, "path": str(path), "n_rows": n_rows, "selection": selection,
                     "models_requested": requested, "models_missing": missing, "notes": notes})
    print(f"[main_fig] {path} (rows={n_rows}{'; MISSING models: ' + missing if missing else ''})")


def main_violin_frame(df, langs, include_identical, metric=SLOR_COL):
    rows = target_language_with_english_frame(df, metric, langs, include_identical)
    if rows.empty:
        return rows
    keep = rows["Target language"].isin(set(langs) | {ENGLISH_CONTROL_CATEGORY})
    return rows.loc[keep]


def violin_y_limits(values, pad_fraction=SPLIT_Y_PAD_FRACTION):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return -1.0, 1.0
    lo, hi = min(float(values.min()), 0.0), max(float(values.max()), 0.0)
    pad = pad_fraction * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad


def _draw_main_violin_panel(ax, rows, models, missing, languages, color_at, value_col, y_label,
                            tick_size=SPLIT_TICK_SIZE, label_size=SPLIT_LABEL_SIZE, show_mean_median=True,
                            english_boundary=True):
    names = [m for m, _l in models]
    drawn = [m for m in names if m not in set(missing)]
    n = max(len(drawn), 1)
    slot = min(0.84 / n, 0.2)
    draw_language_separators(ax, languages, scale=1.4, english_boundary=english_boundary)
    for li, lang in enumerate(languages):
        sub = rows.loc[rows["Target language"] == lang] if len(rows) else rows
        for mi, m in enumerate(drawn):
            values = sub.loc[sub["model_name"] == m, value_col].to_numpy(dtype=float) if len(sub) else np.array([])
            if not values.size:
                continue
            _color, hatch = main_model_style(m)
            _violin(ax, values, li + (mi - (n - 1) / 2) * slot, color_at(m, lang), slot * 0.9, hatch=hatch,
                    show_mean_median=show_mean_median)
    draw_zero_line(ax, 2.4)
    ax.set_xticks(range(len(languages)))
    ax.set_xticklabels(languages)
    ax.set_xlim(-0.5, len(languages) - 0.5)
    _style_main_violin_axes(ax, y_label, tick_size, label_size)


def _style_main_violin_axes(ax, y_label, tick_size=SPLIT_TICK_SIZE, label_size=SPLIT_LABEL_SIZE):
    ax.tick_params(axis="both", labelsize=tick_size, labelcolor=TEXT_COLOR, length=5, width=1.1)
    ax.tick_params(axis="x", labelsize=label_size, labelbottom=True)
    ax.set_ylabel(y_label, fontsize=max(7, label_size - 5), color=TEXT_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    black_spines(ax, 1.3)


def _model_legend_handle(model, label, missing):
    color, hatch = main_model_style(model)
    if model in set(missing):
        return Patch(facecolor="white", edgecolor=MUTED_COLOR, linestyle="--", label=f"{label} (no outputs)")
    return Patch(facecolor=color, edgecolor="none", label=label)


def _blank_handle(label=""):
    return Patch(facecolor="none", edgecolor="none", label=label)


def main_violin_legend_entries(models, missing, legend_labels, grouped_training=False, show_mean_median=True,
                               linked=False):
    labels = dict(models)
    summary = _summary_handles(show_median=show_mean_median, linked=linked)
    if grouped_training and all(m in MAIN_MONOLINGUAL_TRAINING for m, _l in models):
        kinds = [k for k in MONOLINGUAL_KINDS if any(MAIN_MONOLINGUAL_TRAINING[m][0] == k for m, _l in models)]
        groups = []
        for kind in kinds:
            col = [_blank_handle(kind)]
            col += [_model_legend_handle(m, f"{_MONOLINGUAL_NAMES.get(m, labels[m])} "
                                            f"({MAIN_MONOLINGUAL_TRAINING[m][1]})", missing)
                    for m, _l in models if MAIN_MONOLINGUAL_TRAINING[m][0] == kind]
            groups.append(col)
        columns = groups[:2]
        for extra in groups[2:]:
            columns[-1] = columns[-1] + extra
        rows = max(len(c) for c in columns)
        columns = [c + [_blank_handle()] * (rows - len(c)) for c in columns]
        if summary:
            for i, col in enumerate(columns):
                col.append(summary[i] if i < len(summary) else _blank_handle())
        return [h for c in columns for h in c], len(columns), set(kinds)
    model_handles = [_model_legend_handle(m, legend_labels.get(m, labels[m]), missing) for m, _l in models]
    rows = 3 if len(model_handles) > 4 else 2 if len(model_handles) > 2 else max(len(model_handles), 2)
    rows = max(rows, len(summary))
    n_model_cols = -(-len(model_handles) // rows)
    model_handles += [_blank_handle()] * (n_model_cols * rows - len(model_handles))
    if not summary:
        return model_handles, n_model_cols, set()
    return model_handles + summary + [_blank_handle()] * (rows - len(summary)), n_model_cols + 1, set()


MAIN_VIOLIN_LEGEND_SIZE = 11
MAIN_VIOLIN_LEGEND_MARGIN = 0.03


def _place_main_violin_legend(ax, handles, ncol, bold_labels, fontsize=MAIN_VIOLIN_LEGEND_SIZE):
    legend = ax.legend(handles=handles, frameon=True, framealpha=0.92, facecolor="white", edgecolor="none",
                       fontsize=fontsize, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=ncol,
                       columnspacing=0.6, handlelength=1.1, handleheight=0.8, handletextpad=0.35, labelspacing=0.25,
                       borderpad=0.25, borderaxespad=0.2)
    for text in legend.get_texts():
        if text.get_text() in bold_labels:
            text.set_fontweight("bold")
    return legend


def fit_legend_headroom(ax, legend, rows, languages, value_col, lo, hi, margin=MAIN_VIOLIN_LEGEND_MARGIN,
                        category_values=None):
    fig = ax.figure
    fig.canvas.draw()
    box = legend.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
    x_lo, x_hi = ax.get_xlim()
    left, right = x_lo + box.x0 * (x_hi - x_lo), x_lo + box.x1 * (x_hi - x_lo)
    covered = [i for i in range(len(category_values if category_values is not None else languages))
               if i + 0.5 > left and i - 0.5 < right]
    if category_values is not None:
        values = np.concatenate([np.asarray(category_values[i], dtype=float) for i in covered]) if covered else []
    else:
        langs = [languages[i] for i in covered]
        values = rows.loc[rows["Target language"].isin(langs), value_col].to_numpy(dtype=float) if len(rows) else []
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    frac = box.y0 - margin
    if not values.size or frac <= 0.2:
        return hi
    need = lo + (float(values.max()) - lo) / frac
    return max(hi, need)


def _finish_main_violin_axes(ax, panel, languages, value_col, show_mean_median=True, ylim=None):
    rows = panel["rows"]
    lo, hi = violin_y_limits(rows[value_col] if len(rows) else [])
    ax.set_ylim(lo, hi)
    handles, ncol, bold = main_violin_legend_entries(panel["models"], panel["missing"], panel["legend_labels"],
                                                     panel.get("grouped_training", False), show_mean_median)
    legend = _place_main_violin_legend(ax, handles, ncol, bold)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(lo, fit_legend_headroom(ax, legend, rows, languages, value_col, lo, hi))
    return ax.get_ylim()


def _panel_languages(panel):
    rows = panel["rows"]
    present = set(rows["Target language"].unique()) if len(rows) else set()
    if panel.get("languages"):
        return [l for l in panel["languages"] if l in present]
    return _ordered_languages(present)


def plot_main_violin_panel(panel, out_path, value_col, y_label, letter_title=False, show_mean_median=True,
                           ylim=None):
    rows = panel["rows"]
    languages = _panel_languages(panel)
    if not languages:
        return False
    if is_delegated(out_path):
        return tuple(ylim) if ylim is not None else violin_y_limits(rows[value_col])
    fig, ax = plt.subplots(figsize=(SPLIT_VIOLIN_WIDTH, SPLIT_VIOLIN_AXES_HEIGHT))
    _draw_main_violin_panel(ax, rows, panel["models"], panel["missing"], languages, panel["color_at"], value_col,
                            y_label, show_mean_median=show_mean_median,
                            english_boundary=panel.get("english_boundary", True))
    if panel.get("x_label"):
        ax.set_xlabel(panel["x_label"], fontsize=SPLIT_TICK_SIZE, color=TEXT_COLOR, labelpad=3)
    ylim = _finish_main_violin_axes(ax, panel, languages, value_col, show_mean_median, ylim)
    if letter_title and panel.get("letter"):
        ax.set_title(f"({panel['letter']}) {panel['title']}", loc="left", fontsize=SPLIT_LABEL_SIZE,
                     fontweight="bold", color=TEXT_COLOR, pad=6)
    fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return ylim


def plot_main_violins(rows, models, out_path, missing=(), value_col=DELTA_SLOR, y_label="ΔSLOR",
                      show_mean_median=True, ylim=None):
    return plot_main_violin_panel({"rows": rows, "models": models, "missing": list(missing),
                                   "color_at": _original_color_at, "legend_labels": {}}, out_path, value_col,
                                  y_label, show_mean_median=show_mean_median, ylim=ylim)


def plot_main_violins_merged(panels, out_path, value_col=DELTA_SLOR, y_label="ΔSLOR", show_mean_median=True,
                             ylims=None):
    all_rows = [p["rows"] for p in panels if len(p["rows"])]
    if not all_rows:
        return False
    languages = _ordered_languages(pd.concat(all_rows)["Target language"].unique())
    fig, axes = plt.subplots(len(panels), 1, figsize=(SPLIT_VIOLIN_WIDTH, (SPLIT_VIOLIN_AXES_HEIGHT + 0.5) * len(panels)))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        _draw_main_violin_panel(ax, p["rows"], p["models"], p["missing"], languages, p["color_at"], value_col,
                                y_label, show_mean_median=show_mean_median)
        ax.set_title(f"({p['letter']}) {p['title']}", loc="left", fontsize=SPLIT_LABEL_SIZE, fontweight="bold",
                     color=TEXT_COLOR, pad=6)
    fig.tight_layout(h_pad=1.2)
    out = []
    for i, (ax, p) in enumerate(zip(axes, panels)):
        out.append(_finish_main_violin_axes(ax, p, languages, value_col, show_mean_median,
                                            ylims[i] if ylims else None))
    fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out


def _violin_summary(subset, annotation, value_col=DELTA_SLOR):
    summary = _distribution_summary(subset, ["Target language", "model_name"], value_col)
    if len(summary):
        summary.insert(0, "annotation_column", np.where(
            summary["Target language"] == ENGLISH_CONTROL_CATEGORY, "English preference annotation",
            "Target language preference annotation"))
        summary.insert(0, "annotation_value", annotation)
    return summary


def _monolingual_color_at(model, lang):
    color, _h = main_model_style(model)
    target = MAIN_MONOLINGUAL_TARGET_LANGUAGE.get(model)
    if target is not None and lang != ENGLISH_CONTROL_CATEGORY and lang != target:
        return NON_FOCUS_GRAY
    return color


def _original_color_at(model, _lang):
    return main_model_style(model)[0]


MAIN_MANACA_SUFFIX = "manaca-1b"
MAIN_MANACA_ABLATION = "no-sv-no-de"


def _panel_b_extras(base, df, metric, pref_slug, annotation, p):
    extras = []
    for suffix, drop in MAIN_PANEL_B_ABLATIONS:
        extras.append((suffix, drop, dict(p, rows=ablate_languages(p["rows"], drop)), ""))
        if suffix != MAIN_MANACA_ABLATION:
            continue
        present, missing = resolve_main_models(df, MAIN_MONOLINGUAL_SET_A_MANACA)
        selected = base.loc[base["model_name"].isin(present)] if len(base) else base
        subset = filter_side_preference(selected, annotation) if len(selected) else selected
        extras.append((f"{suffix}__{MAIN_MANACA_SUFFIX}", drop,
                       dict(p, rows=ablate_languages(subset, drop), models=MAIN_MONOLINGUAL_SET_A_MANACA,
                            missing=missing),
                       "Manacá 1B (monolingual Portuguese, 100% Portuguese pretraining; not BLM) in Sabiá 7B's slot; "
                       "every other model, row and setting as in the Sabiá version; "))
    return extras


def export_main_violins(df, main_dir, langs, include_identical, manifest):
    written = []
    sets = dict(MAIN_VIOLIN_SETS)
    for slug, models in MAIN_VIOLIN_SETS + [("monolingual-setA-manaca", MAIN_MONOLINGUAL_SET_A_MANACA)]:
        check_violin_colors(models, f"violins {slug}")
    for metric in MAIN_VIOLIN_METRICS:
        spec = score_metric(metric)
        value_col, y_label, stem = spec["delta"], MAIN_METRIC_Y_LABELS[metric], main_violin_stem(metric)
        base = main_violin_frame(df, langs, include_identical, metric)
        subsets = {}
        for slug, models in MAIN_VIOLIN_SETS:
            present, missing = resolve_main_models(df, models)
            if missing and metric == SLOR_COL:
                print(f"[warning] main_fig violins {slug}: no outputs for {missing} (shown as 'no outputs', "
                      "not replaced)")
            selected = base.loc[base["model_name"].isin(present)] if len(base) else base
            for pref_slug, annotation in MAIN_VIOLIN_PREFERENCE_SUBSETS:
                subset = filter_side_preference(selected, annotation) if len(selected) else selected
                subsets[(slug, pref_slug)] = (subset, models, missing)
                spath = main_dir / f"{stem}__{slug}__pref-{pref_slug}_summary.csv"
                _write_table(written, spath, _violin_summary(subset, annotation, value_col))
                if slug not in MAIN_SEPARATE_VIOLIN_SETS:
                    continue
                path = main_dir / f"{stem}__{slug}__pref-{pref_slug}{MAIN_FIG_EXT}"
                write_violin_pair(
                    written, manifest, path,
                    lambda pth, show, ylim, subset=subset, models=models, missing=missing: plot_main_violins(
                        subset, models, pth, missing=missing, value_col=value_col, y_label=y_label,
                        show_mean_median=show, ylim=ylim),
                    f"violin {slug} pref-{pref_slug}{_metric_tag(metric)} (supplementary)", int(len(subset)),
                    f"{spec['label']} (saved {metric}); own-side preference annotation exactly {annotation}; "
                    f"languages {','.join(langs)} + en", ",".join(m for m, _l in models), ",".join(missing),
                    notes="fixed IBM Carbon checkpoint colours")
        for pref_slug, annotation in MAIN_VIOLIN_PREFERENCE_SUBSETS:
            panels, summaries, requested, missing_all = [], [], [], []
            for letter, slug in MAIN_MERGED_VIOLIN_PANELS:
                subset, models, missing = subsets[(slug, pref_slug)]
                mono = slug.startswith("monolingual")
                panels.append({"letter": letter, "title": MAIN_PANEL_TITLES.get(slug, slug), "slug": slug,
                               "rows": subset, "models": models, "missing": missing,
                               "color_at": _monolingual_color_at if mono else _original_color_at,
                               "legend_labels": MAIN_MONOLINGUAL_LEGEND_LABELS if mono else {},
                               "grouped_training": mono})
                s = _violin_summary(subset, annotation, value_col)
                s.insert(0, "panel_set", slug)
                s.insert(0, "panel", letter)
                summaries.append(s)
                requested += [m for m, _l in models]
                missing_all += list(missing)
            for p in panels:
                path = main_dir / main_split_violin_name(p["letter"], metric, pref_slug)
                lim = violin_y_limits(p["rows"][value_col]) if len(p["rows"]) else (np.nan, np.nan)
                shown = plot_main_violin_panel(p, path, value_col, y_label)
                _record(written, path, bool(shown))
                twin = no_mean_median_path(path)
                _record(written, twin, bool(shown) and bool(plot_main_violin_panel(
                    p, twin, value_col, y_label, show_mean_median=False, ylim=shown)))
                top = shown[1] if shown else np.nan
                artifact = f"violin ({p['letter']}) {p['title']} pref-{pref_slug}{_metric_tag(metric)}"
                selection = (f"{spec['label']} (saved {metric}); own-side preference annotation exactly "
                             f"{annotation}; languages {','.join(langs)} + en; "
                             + ",".join(m for m, _l in sets[p["slug"]]))
                notes = (f"own y range {lim[0]:.3f} .. {lim[1]:.3f} (data min/max + 5% padding; nothing "
                         f"clipped; displayed top {top:.3f} for the in-axes upper-left legend); grey "
                         "separators between languages, black separator at the English boundary; fixed IBM Carbon "
                         "checkpoint colours; rows also in "
                         f"{main_merged_violin_stem(metric)}__pref-{pref_slug}_summary.csv (panel {p['letter']})"
                         + ("; target language and en in full colour, other languages one neutral grey "
                            "(display only)" if p["slug"].startswith("monolingual") else ""))
                _manifest_row(manifest, artifact, path, int(len(p["rows"])), selection,
                              ",".join(m for m, _l in p["models"]), ",".join(p["missing"]), notes=notes)
                _manifest_row(manifest, f"{artifact} [no mean/median]", twin, int(len(p["rows"])), selection,
                              ",".join(m for m, _l in p["models"]), ",".join(p["missing"]),
                              notes=f"same as {path.name} {NO_MEDIAN_TWIN_NOTE} "
                                    "(identical rows, KDE, colours, order, legend position and y limits)")
            for p in panels:
                if p["letter"] != "b":
                    continue
                for suffix, drop, ablated, extra_note in _panel_b_extras(base, df, metric, pref_slug, annotation, p):
                    path = main_dir / main_split_violin_name("b", metric, pref_slug, suffix)
                    rows_b = ablated["rows"]
                    langs_b = _ordered_languages(rows_b["Target language"].unique()) if len(rows_b) else []
                    lim = violin_y_limits(rows_b[value_col]) if len(rows_b) else (np.nan, np.nan)
                    counts = rows_b.groupby("Target language").size() if len(rows_b) else pd.Series(dtype=int)
                    s_ab = _violin_summary(rows_b, annotation, value_col)
                    s_ab.insert(0, "languages_removed", ",".join(drop))
                    s_ab.insert(0, "panel_models", ",".join(m for m, _l in ablated["models"]))
                    _write_table(written, path.with_name(path.stem + "_summary.csv"), s_ab)
                    shown = write_violin_pair(
                        written, manifest, path,
                        lambda pth, show, ylim, ablated=ablated: plot_main_violin_panel(
                            ablated, pth, value_col, y_label, show_mean_median=show, ylim=ylim),
                        f"violin (b) {p['title']} pref-{pref_slug}{_metric_tag(metric)} without {'+'.join(drop)}"
                        + (" (Manacá 1B instead of Sabiá 7B)" if MAIN_MANACA_SUFFIX in suffix else ""),
                        int(len(rows_b)),
                        f"{spec['label']} (saved {metric}); own-side preference annotation exactly "
                        f"{annotation}; languages {','.join(l for l in langs if l not in drop)} + en "
                        f"(removed: {','.join(drop)}); " + ",".join(m for m, _l in ablated["models"]),
                        ",".join(m for m, _l in ablated["models"]), ",".join(ablated["missing"]),
                        notes=extra_note + f"x categories {','.join(langs_b)} (recomputed from the filtered rows); "
                              "rows per category " + ", ".join(f"{k}={int(v)}" for k, v in counts.items())
                              + f"; own y range {lim[0]:.3f} .. {lim[1]:.3f}"
                              + "; `en` unchanged (English comparison pooled over all origins); same "
                                "styles and legend layout as the full panel (b)")
                    if not shown:
                        print(f"[warning] main_fig: {path.name} not written (no rows)")
            path = main_dir / f"{main_merged_violin_stem(metric)}__pref-{pref_slug}{MAIN_FIG_EXT}"
            _write_csv(written, main_dir / f"{main_merged_violin_stem(metric)}__pref-{pref_slug}_summary.csv",
                       summaries)
            n_rows = int(sum(len(p["rows"]) for p in panels))
            write_violin_pair(
                written, manifest, path,
                lambda pth, show, ylim, panels=panels: plot_main_violins_merged(
                    panels, pth, value_col, y_label, show_mean_median=show,
                    ylims=list(ylim) if ylim else None),
                f"violin multilingual (a) + monolingual-setA (b) pref-{pref_slug}{_metric_tag(metric)} "
                "(combined, backward compatibility)", n_rows,
                f"{spec['label']} (saved {metric}); own-side preference annotation exactly {annotation}; "
                f"languages {','.join(langs)} + en; (a) " + ",".join(m for m, _l in sets["multilingual"])
                + "; (b) " + ",".join(m for m, _l in sets["monolingual-setA"]),
                ",".join(requested), ",".join(missing_all),
                notes="separate paper panels: " + ", ".join(
                    main_split_violin_name(l, metric, pref_slug) for l, _s in MAIN_MERGED_VIOLIN_PANELS)
                      + "; per-panel rows in the _summary.csv `panel` column")
    return written


def export_main_failure_rates(df, main_dir, langs, include_identical, manifest):
    written = []
    subset = "gt-or-gtgt"
    matrix = compute_failure_rate_matrix(df, langs, include_identical=include_identical, metric=SLOR_COL,
                                         preference_subset=subset)
    long = compute_failure_rates(df, subset, include_identical=include_identical, preference_languages=langs,
                                 metric=SLOR_COL)
    long = long.loc[long["comparison"].isin([TARGET_COMPARISON_POS, ENGLISH_COMPARISON_POS])]
    obs = failure_rate_observations(df, langs, include_identical)
    trace = pd.DataFrame({
        "model_name": obs["model_name"].astype(str),
        "matrix_column": np.where(obs["pos"] == ENGLISH_COMPARISON_POS, "en", obs["Target language"].astype(str)),
        "comparison": obs["pos"], "target_language_origin": obs["Target language"],
        SOURCE_ROW_ID_COL: obs[SOURCE_ROW_ID_COL],
        "English preference annotation": obs.get("English preference annotation"),
        "Target language preference annotation": obs.get("Target language preference annotation"),
        SLOR_COL: obs[DELTA_SLOR], "failure": obs[DELTA_SLOR] < 0, "tie": obs[DELTA_SLOR] == 0,
    }) if len(obs) else pd.DataFrame()
    definitions = failure_matrix_definitions(langs, include_identical, SLOR_COL)
    definitions.loc[definitions["item"] == "en", "definition"] = (
        "Proper English strategy vs Literal English translation of target language strategy, pooled over every "
        "target-language origin; English preference annotation in {>, >>}")
    definitions.loc[definitions["item"] == "preference filter", "definition"] = (
        "side-specific: de .. zh rows need Target language preference annotation in {>, >>}; en rows need English "
        "preference annotation in {>, >>}; the other annotation of the same source row is not required")
    path = main_dir / MAIN_FAILURE_TABLE
    with pd.ExcelWriter(path) as writer:
        for sheet in ("failure_rate", "N_failure", "N_total", "N_tie"):
            matrix[sheet].reset_index().to_excel(writer, sheet_name=sheet, index=False)
        definitions.to_excel(writer, sheet_name="matrix_definitions", index=False)
        long.to_excel(writer, sheet_name="by_comparison", index=False)
    _record(written, path)
    trace_path = main_dir / "SLOR_failure_rates_observations.csv.gz"
    trace.to_csv(trace_path, index=False)
    _record(written, trace_path)
    if len(trace):
        n_trace = trace.groupby(["model_name", "matrix_column"]).size().unstack().reindex(
            index=matrix["N_total"].index, columns=FAILURE_MATRIX_COLUMNS).fillna(0).astype(int)
        if not n_trace.equals(matrix["N_total"]):
            raise AssertionError("failure-rate trace does not reproduce N_total")
    _manifest_row(manifest, "SLOR failure rates", path, int(matrix["N_total"].to_numpy().sum()),
                  "side-specific preference filter (target: target annotation in {>,>>}; en: English annotation "
                  "in {>,>>})", notes=f"{len(matrix['N_total'])} models; traced in {trace_path.name}")
    return written


FAILURE_BAR_STATUS_OK = "ok"
FAILURE_BAR_STATUS_MISSING = "no data (model not in workbook)"
FAILURE_BAR_STATUS_EMPTY = "no observations (N_total = 0)"


def load_failure_rate_workbook(path):
    sheets = pd.read_excel(path, sheet_name=["failure_rate", "N_failure", "N_total"])
    return tuple(sheets[s].set_index("model_name") for s in ("failure_rate", "N_failure", "N_total"))


def failure_bar_table(rates, n_failure, n_total, languages=None):
    languages = languages or language_axis_order([c for c in rates.columns])
    keyed = {failure_workbook_model_key(m): m for m in rates.index}
    out = []
    order = 0
    for section, family, models in FAILURE_BAR_SECTIONS:
        for model_id in models:
            key = failure_workbook_model_key(model_id)
            in_book = key in keyed
            for lang in languages:
                rate = nf = nt = np.nan
                if in_book:
                    row = keyed[key]
                    rate, nf, nt = (float(t.loc[row, lang]) if lang in t.columns else np.nan
                                    for t in (rates, n_failure, n_total))
                status = (FAILURE_BAR_STATUS_MISSING if not in_book
                          else FAILURE_BAR_STATUS_EMPTY if not (np.isfinite(nt) and nt > 0) else FAILURE_BAR_STATUS_OK)
                if status == FAILURE_BAR_STATUS_OK:
                    if not np.isfinite(rate) or abs(rate - nf / nt) > 1e-9:
                        raise AssertionError(f"failure workbook inconsistent for {key} {lang}: rate={rate}, "
                                             f"N_failure={nf}, N_total={nt}")
                elif in_book and np.isfinite(rate):
                    raise AssertionError(f"failure workbook has a rate without observations: {key} {lang}")
                out.append({"order": order, "section": section, "paper_family": family or paper_family(model_id),
                            "model_id": model_id, "model_name": key, "label": FAILURE_BAR_LABELS.get(model_id, key),
                            "registered": lookup_model(model_id) is not None, "language": lang, "status": status,
                            "failure_rate": rate if status == FAILURE_BAR_STATUS_OK else np.nan,
                            "failure_rate_percent": 100 * rate if status == FAILURE_BAR_STATUS_OK else np.nan,
                            "N_failure": nf, "N_total": nt})
            order += 1
    return pd.DataFrame(out)


FAILURE_BAR_WIDTH = 6.3
FAILURE_BAR_ROW_HEIGHT = 1.75
FAILURE_BAR_TICK_SIZE = 7
FAILURE_BAR_LABEL_SIZE = 8
FAILURE_BAR_HEADER_SIZE = 7.5
FAILURE_BAR_LEGEND_SIZE = 7
FAILURE_BAR_SECTION_GAP = 0.7
FAILURE_BAR_ROWS = [("EuroLLM", "Gemma 4", "Apertus"),
                    ("Tiny Aya", "Aya Expanse", "Other non-English-centric models")]
SECTION_RULE_COLOR = "#6E6E6E"
SECTION_RULE_WIDTH = 0.45
SECTION_SEPARATOR_COLOR = "#D9D9D9"
SECTION_SEPARATOR_WIDTH = 0.45
LINE_PLOT_MARKER_SIZE = 3.0
LINE_PLOT_LINE_WIDTH = 0.7


def _section_rows(models):
    sections = list(dict.fromkeys(models["section"]))
    rows_spec = [[s for s in r if s in sections] for r in FAILURE_BAR_ROWS]
    rows_spec = [r for r in rows_spec if r] + [[s for s in sections if not any(s in r for r in FAILURE_BAR_ROWS)]]
    return [r for r in rows_spec if r]


def plot_model_language_figure(table, out_path, kind, value_col, y_label, y_max, percent, ok_status,
                               missing_status, empty_label, legend_extra=()):
    languages = list(dict.fromkeys(table["language"]))
    models = table.drop_duplicates("order").set_index("order")
    rows_spec = _section_rows(models)

    def row_width(secs):
        n = sum((models["section"] == s).sum() for s in secs)
        return n + FAILURE_BAR_SECTION_GAP * (len(secs) - 1)

    width_units = max(row_width(r) for r in rows_spec)
    fig, axes = plt.subplots(len(rows_spec), 1, figsize=(FAILURE_BAR_WIDTH, FAILURE_BAR_ROW_HEIGHT * len(rows_spec) + 1.0),
                             squeeze=False)
    n_lang = len(languages)
    bar_w = 0.86 / n_lang
    any_empty = False
    head_y, rule_y = 1.07 * y_max, 1.04 * y_max
    for ax, secs in zip(axes[:, 0], rows_spec):
        x, ticks, tick_labels, tick_missing = 0.0, [], [], []
        for si, sec in enumerate(secs):
            sec_models = models.loc[models["section"] == sec]
            start = x
            series = {lang: [] for lang in languages}
            for order, m in sec_models.iterrows():
                sub = table.loc[table["order"] == order].set_index("language")
                missing = (sub["status"] == missing_status).all()
                if missing:
                    ax.text(x, y_max / 2, "no data", rotation=90, ha="center", va="center",
                            fontsize=FAILURE_BAR_TICK_SIZE - 0.5, color=MUTED_COLOR, style="italic")
                for li, lang in enumerate(languages):
                    r = sub.loc[lang]
                    ok = r["status"] == ok_status and np.isfinite(r[value_col])
                    pos = x + (li - (n_lang - 1) / 2) * bar_w if kind == "bar" else x
                    series[lang].append((x, r[value_col] if ok else np.nan))
                    if ok and kind == "bar":
                        ax.bar(pos, r[value_col], width=bar_w, color=LANGUAGE_COLORS.get(lang, MUTED_COLOR),
                               edgecolor="none", linewidth=0, zorder=2)
                    elif not ok and not missing:
                        any_empty = True
                        xpos = pos if kind == "bar" else x + (li - (n_lang - 1) / 2) * 0.1
                        ax.plot([xpos], [0.025 * y_max], marker="x", markersize=2.6,
                                color=LANGUAGE_COLORS.get(lang, MUTED_COLOR), markeredgewidth=0.7, zorder=3,
                                clip_on=False)
                ticks.append(x)
                label = str(m["label"])
                if sec == "Other non-English-centric models":
                    label = __import__("re").sub(r"(?<![\w.])(\d+(?:\.\d+)?B)$",
                        lambda match: r"$_{" + match.group(1) + r"}$", label)
                tick_labels.append(label)
                tick_missing.append(missing)
                x += 1
            if kind == "line":
                for lang, pts in series.items():
                    color = LANGUAGE_COLORS.get(lang, MUTED_COLOR)
                    xs = np.array([p[0] for p in pts], dtype=float)
                    ys = np.array([p[1] for p in pts], dtype=float)
                    valid = np.isfinite(ys)
                    for i in range(len(xs) - 1):
                        if valid[i] and valid[i + 1]:
                            ax.plot(xs[i:i + 2], ys[i:i + 2], color=color, linewidth=LINE_PLOT_LINE_WIDTH,
                                    alpha=0.85, zorder=2, solid_capstyle="round")
                    ax.plot(xs[valid], ys[valid], linestyle="none", marker="o", markersize=LINE_PLOT_MARKER_SIZE,
                            color=color, markeredgecolor="white", markeredgewidth=0.3, zorder=3)
            end = x - 1
            ax.plot([start - 0.4, end + 0.4], [rule_y, rule_y], color=SECTION_RULE_COLOR,
                    linewidth=SECTION_RULE_WIDTH, clip_on=False, solid_capstyle="butt")
            ax.text((start + end) / 2, head_y, sec, ha="center", va="bottom", fontsize=FAILURE_BAR_HEADER_SIZE,
                    color=TEXT_COLOR, clip_on=False)
            if si < len(secs) - 1:
                ax.axvline(x - 0.5 + FAILURE_BAR_SECTION_GAP / 2, color=SECTION_SEPARATOR_COLOR,
                           linewidth=SECTION_SEPARATOR_WIDTH, zorder=0.5)
            x += FAILURE_BAR_SECTION_GAP
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", rotation_mode="anchor",
                           fontsize=FAILURE_BAR_TICK_SIZE, color=TEXT_COLOR)
        for lbl, miss in zip(ax.get_xticklabels(), tick_missing):
            if miss:
                lbl.set_color(MUTED_COLOR)
                lbl.set_fontstyle("italic")
        ax.set_xlim(-0.6, width_units - 0.4)
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.linspace(0, y_max, 5))
        if percent:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(y_max, decimals=0))
        ax.tick_params(axis="y", labelsize=FAILURE_BAR_TICK_SIZE, length=3, width=0.8)
        ax.tick_params(axis="x", length=2, width=0.8, pad=1.5)
        ax.set_ylabel(y_label, fontsize=FAILURE_BAR_LABEL_SIZE)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
        ax.set_axisbelow(True)
        black_spines(ax, 0.8)
    if kind == "bar":
        handles = [Patch(facecolor=LANGUAGE_COLORS.get(l, MUTED_COLOR), label=LANGUAGE_NAMES.get(l, l))
                   for l in languages]
    else:
        handles = [Line2D([0], [0], color=LANGUAGE_COLORS.get(l, MUTED_COLOR), linewidth=LINE_PLOT_LINE_WIDTH + 0.3,
                          marker="o", markersize=LINE_PLOT_MARKER_SIZE + 0.5, label=LANGUAGE_NAMES.get(l, l))
                   for l in languages]
    if any_empty:
        handles.append(Line2D([0], [0], marker="x", linestyle="none", color=MUTED_COLOR, markersize=3.5,
                              label=empty_label))
    handles += list(legend_extra)
    free_rows = [(width_units - row_width(r), i) for i, r in enumerate(rows_spec)]
    free, ri = max(free_rows)
    if free >= 3:
        x_left = (width_units - free - 0.4 + 0.25 + 0.6) / width_units
        axes[ri, 0].legend(handles=handles, frameon=False, fontsize=FAILURE_BAR_LEGEND_SIZE, ncol=1,
                           loc="center left", bbox_to_anchor=(x_left, 0.5), handlelength=1.4,
                           handletextpad=0.5, labelspacing=0.3, borderaxespad=0.0)
    else:
        axes[0, 0].legend(handles=handles, frameon=True, framealpha=0.9, edgecolor="none",
                          fontsize=FAILURE_BAR_LEGEND_SIZE, ncol=1, loc="upper right", handlelength=1.4,
                          labelspacing=0.3)
    fig.tight_layout(h_pad=0.8)
    fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return True


def plot_failure_rate_bars(table, out_path):
    return plot_model_language_figure(table, out_path, "bar", "failure_rate_percent", "Failure rate", 100, True,
                                      FAILURE_BAR_STATUS_OK, FAILURE_BAR_STATUS_MISSING, "N = 0")


def plot_failure_rate_lines(table, out_path):
    if is_delegated(out_path):
        return True
    return plot_model_language_figure(table, out_path, "line", "failure_rate_percent", "Failure rate", 100, True,
                                      FAILURE_BAR_STATUS_OK, FAILURE_BAR_STATUS_MISSING, "N = 0")


def export_main_failure_rate_bars(main_dir, manifest, workbook=None):
    written = []
    workbook = Path(workbook) if workbook else Path(main_dir) / MAIN_FAILURE_TABLE
    if not workbook.is_file():
        print(f"[warning] main_fig failure-rate bars: workbook not found: {workbook} (figure not written)")
        return written
    rates, n_failure, n_total = load_failure_rate_workbook(workbook)
    table = failure_bar_table(rates, n_failure, n_total)
    models = table.drop_duplicates("order")
    missing = models.loc[(table.groupby("order")["status"].first() == FAILURE_BAR_STATUS_MISSING).to_numpy(),
                         "model_name"].tolist()
    unregistered = models.loc[~models["registered"], "model_id"].tolist()
    empty = table.loc[table["status"] == FAILURE_BAR_STATUS_EMPTY]
    keys = set(models["model_name"])
    unrequested = sorted(failure_workbook_model_key(m) for m in rates.index
                         if failure_workbook_model_key(m) not in keys)
    n_requested = len(FAILURE_BAR_MODELS)
    print(f"[main_fig] failure-rate bars: {n_requested} requested checkpoints in fixed order "
          f"({n_requested - len(missing)} with workbook data)")
    if n_requested != FAILURE_BAR_MODEL_COUNT:
        raise AssertionError(f"failure-rate model list has {n_requested} IDs, expected {FAILURE_BAR_MODEL_COUNT}")
    if missing:
        print(f"[warning] main_fig failure-rate bars: no workbook data, shown as 'no data': {missing}")
    if unregistered:
        print(f"[note] main_fig failure-rate bars: not in model_metadata2 (family from the explicit "
              f"UNREGISTERED_MODEL_FAMILIES table): {unregistered}")
    if len(empty):
        print("[warning] main_fig failure-rate bars: N_total = 0 (no bar): "
              + ", ".join(f"{m} {l}" for m, l in zip(empty["model_name"], empty["language"])))
    if unrequested:
        print(f"[note] main_fig failure-rate bars: workbook models not in the requested list (not plotted): "
              f"{unrequested}")
    _write_table(written, Path(main_dir) / MAIN_FAILURE_BAR_TABLE, table)
    path = Path(main_dir) / MAIN_FAILURE_BAR_FIGURE
    _record(written, path, plot_failure_rate_bars(table, path))
    _manifest_row(manifest, "SLOR failure rates bar plot", path, int(np.nansum(table["N_total"])),
                  f"read from {workbook.name} (sheets failure_rate / N_failure / N_total; side-specific > or >> "
                  "filters; failure = ΔSLOR < 0, ties not failures); rates re-validated as N_failure / N_total",
                  ",".join(models["model_name"]), ",".join(missing),
                  notes=(f"{n_requested} requested model IDs in fixed order (Baichuan2-13B-Base removed; "
                         "Swallow-70b-hf replaced by Swallow-13b-hf, authorized; Manacá 1B "
                         "AIInstituteLNCC/manaca-1b-base between Lucie 7B and Sabiá 7B); bars coloured by language only, "
                         "neutral section headings; missing models keep their slot labelled 'no data', never zero; "
                         + (f"N_total = 0 cells (no bar): {len(empty)}; " if len(empty) else "")
                         + (f"not in model registry: {','.join(unregistered)}; " if unregistered else "")
                         + (f"workbook models not requested: {','.join(unrequested)}; " if unrequested else "")
                         + f"plotted values in {MAIN_FAILURE_BAR_TABLE}"))
    _write_table(written, Path(main_dir) / MAIN_FAILURE_LINE_TABLE, table)
    path = Path(main_dir) / MAIN_FAILURE_LINE_FIGURE
    _record(written, path, plot_failure_rate_lines(table, path))
    _manifest_row(manifest, "SLOR failure rates line plot", path, int(np.nansum(table["N_total"])),
                  f"read from {workbook.name} (identical values to the bar plot); rates re-validated as "
                  "N_failure / N_total",
                  ",".join(models["model_name"]), ",".join(missing),
                  notes=f"{n_requested} models on x in fixed order; one colour per language (full names in a "
                        "single vertical legend column); dots at valid rates, thin lines only between adjacent "
                        "valid models of the same section (no interpolation across missing checkpoints or "
                        "N = 0 cells); neutral headings with hairline rules; "
                        + (f"N_total = 0 cells marked ×: {len(empty)}; " if len(empty) else "")
                        + f"plotted values in {MAIN_FAILURE_LINE_TABLE}")
    return written


TRANSLATION_Y_LABEL = "P(transferred |\nnative or transferred)"
TRANSLATION_EMPTY_LABEL = "no native/transferred match"


def main_translation_table(translation_df, languages, group=None):
    keys = [failure_workbook_model_key(m) for m in FAILURE_BAR_MODELS]
    probs = compute_transferred_probability(translation_df, keys, languages, group=group).set_index(
        ["model_name", "language"])
    out, order = [], 0
    for section, _family, ids in FAILURE_BAR_SECTIONS:
        for model_id in ids:
            key = failure_workbook_model_key(model_id)
            for lang in languages:
                r = probs.loc[(key, lang)]
                out.append({"order": order, "section": section, "model_id": model_id, "model_name": key,
                            "label": FAILURE_BAR_LABELS.get(model_id, key), "language": lang,
                            "language_name": LANGUAGE_NAMES.get(lang, lang), "group": r["group"],
                            "N_native": int(r["N_native"]), "N_transferred": int(r["N_transferred"]),
                            "N_both": int(r["N_both"]), "N_neither": int(r["N_neither"]),
                            "N_total": int(r["N_total"]), "denominator": int(r["denominator"]),
                            "probability": r[TRANSFERRED_PROBABILITY_COL], "status": r["status"]})
            order += 1
    return pd.DataFrame(out)


def plot_translation_transferred_lines(table, out_path):
    return plot_model_language_figure(table, out_path, "line", "probability", TRANSLATION_Y_LABEL, 1.0, False,
                                      TRANSLATION_STATUS_OK, TRANSLATION_STATUS_NO_MODEL, TRANSLATION_EMPTY_LABEL)


def export_main_translation_preference(outputs_dir, main_dir, langs, manifest):
    written = []
    translation_df, warnings = load_all_translation_data2(outputs_dir)
    for w in warnings:
        print(f"[warning] main_fig translation: {w}")
    if translation_df.empty:
        print("[warning] main_fig translation: no *__translation_generation.xlsx files (figure not written)")
        return written
    translation_df = filter_target_preference(translation_df, langs)
    translation_df = translation_df.loc[translation_df["Target language"].isin(set(langs))]
    languages = language_axis_order(langs)
    views = [(None, "")] + [(g, f"__group-{sanitize_filename(g)}") for g in group_choices(translation_df)]
    for group, suffix in views:
        table = main_translation_table(translation_df, languages, group)
        if group is None:
            per_group = [main_translation_table(translation_df, languages, g) for g, _s in views[1:]]
            if per_group:
                summed = sum(t[["N_native", "N_transferred", "denominator"]].to_numpy() for t in per_group)
                if not np.array_equal(summed, table[["N_native", "N_transferred", "denominator"]].to_numpy()):
                    print("[note] main_fig translation: some rows have no construction group; pooled counts "
                          "include them")
        stem = MAIN_TRANSLATION_STEM + suffix
        _write_table(written, Path(main_dir) / f"{stem}.csv", table)
        path = Path(main_dir) / f"{stem}{MAIN_FIG_EXT}"
        _record(written, path, plot_translation_transferred_lines(table, path))
        models = table.drop_duplicates("order")
        missing = models.loc[(table.groupby("order")["status"].first() == TRANSLATION_STATUS_NO_MODEL).to_numpy(),
                             "model_name"].tolist()
        invalid = table.loc[(table["status"] != TRANSLATION_STATUS_OK) & ~table["model_name"].isin(missing)]
        if missing and group is None:
            print(f"[warning] main_fig translation: no translation outputs, shown as 'no data': {missing}")
        if len(invalid):
            print(f"[warning] main_fig translation ({group or 'all groups'}): {len(invalid)} model/language cells "
                  "without a native/transferred match (missing, marked ×): "
                  + ", ".join(f"{m} {l}" for m, l in zip(invalid["model_name"], invalid["language"])))
        _manifest_row(manifest, f"translation preference P(transferred | native or transferred) line "
                                f"({group or 'all groups'})", path, int(table["N_total"].sum()),
                      "saved translation_match (proper = native, literal = transferred) of "
                      "*__translation_generation.xlsx; target-language preference annotation in {>, >>} for "
                      f"{','.join(langs)}; raw counts per (model, target language)"
                      + (f" within group {group!r}" if group else ", all construction groups pooled as raw counts"),
                      ",".join(models["model_name"]), ",".join(missing),
                      notes="P = N_transferred / (N_native + N_transferred); both / neither excluded from the "
                            "denominator; zero denominator = missing (×), never 0; "
                            f"{int((table['status'] == TRANSLATION_STATUS_OK).sum())} valid cells, {len(invalid)} "
                            "cells with zero denominator; standard task only (no Japanese-control translation "
                            f"outputs); values in {stem}.csv")
    return written


SPEARMAN_CELL_SIZE = 48
SPEARMAN_TEXT_CELL_FRACTION = 0.64
SPEARMAN_TICK_SIZE = 35
SPEARMAN_CBAR_LABEL_SIZE = 35
SPEARMAN_CBAR_TICK_SIZE = 30
SPEARMAN_CELL_INCHES = 2.3


def _two_line_label(label):
    head, _sep, tail = str(label).rpartition(" ")
    return f"{head}\n{tail}" if head else str(label)


def plot_main_spearman(rho, n, labels, out_path):
    if is_delegated(out_path):
        return True
    names = [_two_line_label(labels[m]) for m in rho.index]
    k = len(names)
    side = SPEARMAN_CELL_INCHES * k
    fig = plt.figure(figsize=(side + 6.2, side + 3.0))
    left, bottom = 3.6, 2.6
    W, H = fig.get_size_inches()
    ax = fig.add_axes([left / W, bottom / H, side / W, side / H])
    cax = fig.add_axes([(left + side + 0.35) / W, bottom / H, 0.45 / W, side / H])
    values = rho.to_numpy(dtype=float)
    cm = matplotlib.colormaps["RdBu_r"].copy()
    cm.set_bad("#e4e3df")
    im = vector_heatmap(ax, values, cm, -1.0, 1.0)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([labels[m] for m in rho.index], fontsize=SPEARMAN_TICK_SIZE, rotation=40, ha="right",
                       rotation_mode="anchor")
    ax.set_yticklabels(names, fontsize=SPEARMAN_TICK_SIZE, linespacing=0.95)
    texts = []
    for i in range(k):
        for j in range(k):
            v = values[i, j]
            dark = np.isfinite(v) and abs(v) > 0.6
            texts.append(ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "n/a", ha="center", va="center",
                                 fontsize=SPEARMAN_CELL_SIZE, color="white" if dark else TEXT_COLOR))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    cell_px = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    widest = max(t.get_window_extent(renderer).width for t in texts)
    size = SPEARMAN_CELL_SIZE * SPEARMAN_TEXT_CELL_FRACTION * cell_px / widest
    for t in texts:
        t.set_fontsize(size)
    cb = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.solids.set_rasterized(False)
    cb.set_label("Spearman's ρ", fontsize=SPEARMAN_CBAR_LABEL_SIZE, labelpad=12)
    cb.ax.tick_params(labelsize=SPEARMAN_CBAR_TICK_SIZE, labelcolor=TEXT_COLOR, length=8, width=1.5)
    ax.tick_params(labelcolor=TEXT_COLOR, length=0, pad=10)
    ax.spines[:].set_visible(False)
    fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return True


def spearman_long_table(frame, rho, n, labels):
    observed = set(frame["model_name"]) if len(frame) else set()
    return main_inference.spearman_long(rho, n, labels, observed)


def export_main_spearman(df, main_dir, langs, include_identical, manifest, models=None, stem=None, artifact=None):
    written = []
    models = MAIN_SPEARMAN_MODELS if models is None else models
    stem = stem or Path(MAIN_SPEARMAN_FIGURE).stem
    artifact = artifact or "Spearman correlation"
    present, missing = resolve_main_models(df, models)
    if missing:
        print(f"[warning] main_fig {artifact}: no outputs for {missing}")
    frame = sentence_pair_frame(df.loc[df["model_name"].isin(present)], langs, include_identical)
    rho, n = spearman_matrix(frame, present)
    labels = dict(models)
    rho.to_csv(main_dir / f"{stem}.csv")
    _record(written, main_dir / f"{stem}.csv")
    n.to_csv(main_dir / f"{stem}_n.csv")
    _record(written, main_dir / f"{stem}_n.csv")
    path = main_dir / f"{stem}{MAIN_FIG_EXT}"
    _record(written, path, plot_main_spearman(rho, n, labels, path) if len(present) else False)
    off = n.to_numpy()[~np.eye(len(present), dtype=bool)] if len(present) > 1 else np.array([0])
    for m in present:
        others = [o for o in present if o != m and n.loc[m, o] < MIN_SHARED_PAIRS_SPEARMAN]
        if others:
            print(f"[warning] main_fig {artifact}: {m} shares < {MIN_SHARED_PAIRS_SPEARMAN} exact sentence pairs "
                  f"with {others} (rho shown as n/a; check that its outputs were scored on the current data version)")
    selection = f"target-language comparison, {preference_filter_label(langs)}; pooled over languages"
    long = spearman_long_table(frame, rho, n, labels)
    _write_table(written, main_dir / f"{stem}_long.csv", long)
    _manifest_row(manifest, f"{artifact}: long table (rho, N, status)", main_dir / f"{stem}_long.csv",
                  int(len(frame)), selection, ",".join(m for m, _l in models), ",".join(missing),
                  notes="descriptive Spearman rho per unique model pair (no p-values, CIs or multiplicity correction)")
    for p_, what in ((path, artifact), (main_dir / f"{stem}.csv", f"{artifact}: rho matrix"),
                     (main_dir / f"{stem}_n.csv", f"{artifact}: shared sentence-pair counts")):
        _manifest_row(manifest, what, p_, int(len(frame)), selection, ",".join(m for m, _l in models),
                      ",".join(missing), notes=f"models in order: {', '.join(l for _m, l in models)}; shared pairs "
                                               f"per model pair: min {int(off.min())}, max {int(off.max())}")
    return written


SPEARMAN_ALL_STEM = "Spearman_model_correlation_ALL"
KAPPA_ALL_STEM = "cohens_kappa_ALL"
SPEARMAN_ALL_TICK_SIZE = 5.6
SPEARMAN_ALL_SIZE = 6.3


def all_model_labels():
    labels, sections = {}, {}
    for section, _family, ids in STAT_MODEL_SECTIONS:
        for model_id in ids:
            key, short = failure_workbook_model_key(model_id), FAILURE_BAR_LABELS.get(model_id, model_id)
            labels[key] = short if section.startswith("Other") else f"{section} {short}"
            sections[key] = section
    return labels, sections


def plot_main_spearman_all(rho, labels, sections, out_path):
    names = list(rho.index)
    k = len(names)
    fig = plt.figure(figsize=(SPEARMAN_ALL_SIZE, SPEARMAN_ALL_SIZE * 0.93))
    ax = fig.add_axes([0.2, 0.2, 0.68, 0.73])
    cax = fig.add_axes([0.9, 0.2, 0.018, 0.73])
    values = rho.to_numpy(dtype=float)
    cm = matplotlib.colormaps["RdBu_r"].copy()
    cm.set_bad("#e4e3df")
    im = vector_heatmap(ax, values, cm, -1.0, 1.0)
    bad = np.argwhere(~np.isfinite(values))
    if len(bad):
        ax.scatter(bad[:, 1], bad[:, 0], marker="x", s=4, linewidths=0.4, color=MUTED_COLOR, zorder=3)
    for i in range(1, k):
        if sections[names[i]] != sections[names[i - 1]]:
            ax.axhline(i - 0.5, color="white", linewidth=1.2)
            ax.axvline(i - 0.5, color="white", linewidth=1.2)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([labels[m] for m in names], fontsize=SPEARMAN_ALL_TICK_SIZE, rotation=90)
    ax.set_yticklabels([labels[m] for m in names], fontsize=SPEARMAN_ALL_TICK_SIZE)
    ax.tick_params(labelcolor=TEXT_COLOR, length=0, pad=2)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.solids.set_rasterized(False)
    cb.set_label("Spearman's ρ", fontsize=7.5, labelpad=4)
    cb.ax.tick_params(labelsize=6.5, labelcolor=TEXT_COLOR, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return True


def _pair_long(matrix_a, n_a, labels, value_name, n_name, extra=None):
    rows = []
    names = list(matrix_a.index)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            row = {"model_a": a, "model_b": b, "label_a": labels[a], "label_b": labels[b],
                   value_name: matrix_a.loc[a, b], n_name: int(n_a.loc[a, b])}
            for name, table in (extra or {}).items():
                row[name] = table.loc[a, b]
            rows.append(row)
    return pd.DataFrame(rows)


def export_main_consistency_all(df, main_dir, langs, include_identical, manifest):
    written = []
    keys = [k for k, _hf in MAIN_ALL_MODELS]
    labels, sections = all_model_labels()
    present, missing = resolve_main_models(df, MAIN_ALL_MODELS)
    if missing:
        print(f"[warning] main_fig Spearman/kappa ALL: no outputs for {missing} (all-NaN rows, never 0)")
    try:
        frame = sentence_pair_frame(df.loc[df["model_name"].isin(present)], langs, include_identical)
    except SentencePairIdentityError as err:
        diag = main_dir / f"{SPEARMAN_ALL_STEM}_identity_diagnostic.csv"
        err.diagnostic.to_csv(diag, index=False)
        _record(written, diag)
        print(f"[warning] main_fig Spearman/kappa ALL not computed: {err} (see {diag.name})")
        return written
    rho, n = spearman_matrix(frame, keys)
    agree, kappa, n_k = sign_agreement_and_kappa(frame, keys)
    for t in (rho, n, agree, kappa, n_k):
        t.index.name = "model_name"
        t.columns.name = None
    assert list(rho.index) == list(kappa.index) == keys
    assert (n_k.to_numpy() <= n.to_numpy()).all()
    observed = set(frame["model_name"])
    eye = np.eye(len(keys), dtype=bool)

    def spearman_status(a, b):
        if a not in observed or b not in observed:
            return "missing model (no rows)"
        if n.loc[a, b] == 0:
            return "no shared sentence pairs"
        if n.loc[a, b] < MIN_SHARED_PAIRS_SPEARMAN:
            return f"insufficient overlap (< {MIN_SHARED_PAIRS_SPEARMAN} shared pairs)"
        return "ok" if np.isfinite(rho.loc[a, b]) else "undefined (constant ranks)"

    def kappa_status(a, b):
        if a not in observed or b not in observed:
            return "missing model (no rows)"
        if n_k.loc[a, b] == 0:
            return "undefined: no shared non-tied pairs"
        return "ok" if np.isfinite(kappa.loc[a, b]) else "undefined: constant labels (p_e = 1)"

    status_s = pd.DataFrame([[spearman_status(a, b) for b in keys] for a in keys], index=keys, columns=keys)
    status_k = pd.DataFrame([[kappa_status(a, b) for b in keys] for a in keys], index=keys, columns=keys)
    for name, table in ((f"{SPEARMAN_ALL_STEM}.csv", rho), (f"{SPEARMAN_ALL_STEM}_n.csv", n),
                        (f"{KAPPA_ALL_STEM}.csv", kappa), (f"{KAPPA_ALL_STEM}_n.csv", n_k)):
        table.to_csv(main_dir / name)
        _record(written, main_dir / name)
    s_long = _pair_long(rho, n, labels, "spearman_rho", "n_shared_pairs", {"status": status_s})
    desc_long = spearman_long_table(frame, rho, n, labels)
    s_long = s_long.merge(desc_long[["model_a", "model_b", "rho"]], on=["model_a", "model_b"], how="left",
                          validate="one_to_one")
    k_long = _pair_long(kappa, n_k, labels, "cohens_kappa", "n_shared_non_tied_pairs",
                        {"sign_agreement": agree, "n_shared_pairs_spearman": n, "status": status_k})
    _write_table(written, main_dir / f"{SPEARMAN_ALL_STEM}_long.csv", s_long)
    _write_table(written, main_dir / f"{KAPPA_ALL_STEM}_long.csv", k_long)
    path = main_dir / f"{SPEARMAN_ALL_STEM}{MAIN_FIG_EXT}"
    _record(written, path, plot_main_spearman_all(rho, labels, sections, path))
    bad_s = s_long.loc[s_long["status"] != "ok"]
    bad_k = k_long.loc[k_long["status"] != "ok"]
    ok_n = s_long.loc[s_long["status"] == "ok", "n_shared_pairs"]
    for kind, bad in (("Spearman", bad_s), ("kappa", bad_k)):
        if len(bad):
            print(f"[warning] main_fig {kind} ALL: {len(bad)} of {len(s_long)} model pairs undefined: "
                  + "; ".join(f"{s}: {c}" for s, c in bad["status"].value_counts().items()))
    selection = (f"target-language comparison, {preference_filter_label(langs)}; pooled over languages; exact "
                 "sentence-pair identity (sentence_pair_frame); saved ΔSLOR; same frame as the 6-model "
                 f"{MAIN_SPEARMAN_FIGURE}, only the model set differs ({len(keys)} statistical models)")
    coverage = (f"{len(keys)} models ({len(present)} with rows); {len(s_long)} model pairs, "
                f"{int((s_long['status'] == 'ok').sum())} with a defined rho"
                + (f" (shared pairs min {int(ok_n.min())}, median {int(ok_n.median())}, max {int(ok_n.max())})"
                   if len(ok_n) else ""))
    _manifest_row(manifest, f"Spearman correlation ALL ({len(keys)} statistical models; heatmap)", path, int(len(frame)),
                  selection, ",".join(keys), ",".join(missing),
                  notes=coverage + "; no cell numbers (values in the CSVs); grey × = undefined; white rules "
                        f"between model sections; rho {SPEARMAN_ALL_STEM}.csv, N {SPEARMAN_ALL_STEM}_n.csv, "
                        f"long with status {SPEARMAN_ALL_STEM}_long.csv")
    for name, what in ((f"{SPEARMAN_ALL_STEM}.csv", "pairwise Spearman rho matrix (NaN = undefined)"),
                       (f"{SPEARMAN_ALL_STEM}_n.csv", "pairwise shared sentence-pair counts"),
                       (f"{SPEARMAN_ALL_STEM}_long.csv", "long: rho, N, status per model pair")):
        _manifest_row(manifest, f"Spearman correlation ALL: {what}", main_dir / name, int(len(frame)), selection,
                      ",".join(keys), ",".join(missing), notes=coverage)
    k_ok = k_long.loc[k_long["status"] == "ok", "n_shared_non_tied_pairs"]
    k_cov = (f"{len(k_long)} model pairs, {int((k_long['status'] == 'ok').sum())} with a defined kappa"
             + (f" (shared non-tied pairs min {int(k_ok.min())}, max {int(k_ok.max())})" if len(k_ok) else "")
             + "; kappa uses the shared pairs of the Spearman ALL frame minus pairs where either model has "
               "ΔSLOR = 0 (cross_model_consistency.sign_agreement_and_kappa; positive / negative sign)")
    for name, what in ((f"{KAPPA_ALL_STEM}.csv", "pairwise Cohen's kappa matrix (NaN = undefined)"),
                       (f"{KAPPA_ALL_STEM}_n.csv", "pairwise shared non-tied pair counts"),
                       (f"{KAPPA_ALL_STEM}_long.csv", "long: kappa, sign agreement, N, status per model pair")):
        _manifest_row(manifest, f"Cohen's kappa ALL: {what}", main_dir / name, int(len(frame)), selection,
                      ",".join(keys), ",".join(missing), notes=k_cov)
    return written


SG_STEM = "Spearman_group_comparison"
SG_FONT = 7.5
SG_GROUP_COLORS = {"EC–EC": "#0F62FE", "NEC–NEC": "#D12771", "EC–NEC": "#8D8D8D"}
SG_VARIANTS = (
    ("ALL", f"{STAT_MODEL_COUNT} statistical models (Spearman correlation ALL)", None, f"{SPEARMAN_ALL_STEM}.csv", True),
    ("ALL_manaca_not_lucie", "ALL with Manacá 1B instead of Lucie 7B (Lucie excluded)",
     "drop:OpenLLM-France_Lucie-7B", f"{SPEARMAN_ALL_STEM}.csv", False),
    ("ALL_without_manaca", "ALL without Manacá 1B", f"drop:{MAIN_MANACA_MODEL}",
     f"{SPEARMAN_ALL_STEM}.csv", False),
    ("ALL_without_gemma4_it", "ALL without the five Gemma 4 instruction-tuned checkpoints (POST HOC, "
     "chosen after inspecting the pair distribution)", "drop_gemma4_it", f"{SPEARMAN_ALL_STEM}.csv", False),
    ("selected", "6 selected models of Spearman_model_correlation.pdf", "selected", f"{Path(MAIN_SPEARMAN_FIGURE).stem}.csv",
     False),
    ("selected_manaca", "6 selected models, Manacá 1B instead of Lucie 7B (Spearman_model_correlation_manaca.pdf)",
     "selected_manaca", f"{MAIN_SPEARMAN_MANACA_STEM}.csv", False),
)


def _sg_models(spec):
    if spec is None:
        return list(MAIN_ALL_MODELS)
    if spec == "selected":
        return list(MAIN_SPEARMAN_MODELS)
    if spec == "selected_manaca":
        return list(MAIN_SPEARMAN_MODELS_MANACA)
    if spec == "drop_gemma4_it":
        def gemma_it(k):
            meta = lookup_model(k)
            return meta is not None and meta.family == "Gemma 4" and meta.instruction_tuned
        return [(k, h) for k, h in MAIN_ALL_MODELS if not gemma_it(k)]
    drop = spec.split(":", 1)[1]
    return [(k, h) for k, h in MAIN_ALL_MODELS if k != drop]


def _registry_family(model_name):
    meta = lookup_model(model_name)
    return meta.family if meta is not None else paper_family(model_name)


def spearman_group_inputs(df, langs, include_identical, models, saved_csv=None):
    present, missing = resolve_main_models(df, models)
    frame = sentence_pair_frame(df.loc[df["model_name"].isin(present)], langs, include_identical)
    rho, n = spearman_matrix(frame, present)
    if saved_csv is not None and Path(saved_csv).is_file():
        saved = pd.read_csv(saved_csv, index_col=0)
        ref = saved.loc[present, present].to_numpy(dtype=float)
        if not np.allclose(rho.to_numpy(dtype=float), ref, equal_nan=True, atol=1e-12):
            raise AssertionError(f"group comparison rho differs from the saved {Path(saved_csv).name}")
    return rho, n, frame, present, missing


def spearman_group_tables(variant, desc, res, n_shared):
    ec_mask = res["ec_mask"]
    base = {"variant": variant, "model_set": desc, "n_models_total": len(ec_mask), "n_ec": int(ec_mask.sum()),
            "n_nec": int((~ec_mask).sum())}
    summary = res["summary"].assign(**base)
    summary = summary[list(base) + [c for c in summary.columns if c not in base]]
    summary["delta_median_rho_ec_minus_nec"] = res["observed"]
    rows = []
    m_ = res["summary"].set_index("group")["mean"]
    rows.append({**base, "analysis": "difference in means (descriptive)", "role": "descriptive",
                 "statistic": "mean(EC–EC) − mean(NEC–NEC)", "estimate": float(m_["EC–EC"] - m_["NEC–NEC"]),
                 "note": "descriptive only; sensitive to low-correlation tails, hence the median is the primary "
                         "effect size"})
    rows.append({**base, "analysis": "cross-family pairs only", "role": "family-aware descriptive",
                 "statistic": "median(EC–EC cross-family) − median(NEC–NEC cross-family)",
                 "estimate": res["cross_family_delta"],
                 "note": "same-family model pairs removed before taking medians; descriptive"})
    fbd, n_e, n_n, m_e, m_n = res["family_balanced"]
    rows.append({**base, "analysis": "family-balanced contrast", "role": "family-aware descriptive",
                 "statistic": "median over family pairs of the mean rho (EC) − same (NEC)", "estimate": fbd,
                 "note": f"one value per pair of distinct families: {n_e} EC family pairs (median {m_e:.3f}), "
                         f"{n_n} NEC family pairs (median {m_n:.3f}); large families and same-family pairs get no "
                         "extra weight; descriptive"})
    for key, what in (("lofo", "leave-one-family-out"), ("lomo", "leave-one-model-out")):
        t = res[key]
        d = t["delta_median_rho"]
        lo_row, hi_row = t.loc[d.idxmin()], t.loc[d.idxmax()]
        rows.append({**base, "analysis": what, "role": "family-aware descriptive" if key == "lofo" else "descriptive",
                     "statistic": "Δmedianρ", "estimate": res["observed"], "range_min": float(d.min()),
                     "range_max": float(d.max()), "n_resamples": len(t),
                     "note": f"smallest without {lo_row['dropped']} ({lo_row['delta_median_rho']:.3f}), largest "
                             f"without {hi_row['dropped']} ({hi_row['delta_median_rho']:.3f}); all {len(t)} values "
                             f"{'positive' if (d > 0).all() else 'NOT all positive'}"})
    alt = res.get("alt_family")
    if alt:
        rows.append({**base, "analysis": "registry families (Tiny Aya and Aya Expanse separate)",
                     "role": "family-definition sensitivity (descriptive)", "statistic": "cross-family Δmedianρ",
                     "estimate": alt["cross_family_delta"],
                     "note": f"{alt['n_families']} families; family-balanced Δ = {alt['family_balanced'][0]:.3f}"})
    for r in rows:
        r.setdefault("n_shared_sentence_pairs_min", int(n_shared.min()) if len(n_shared) else 0)
    cross = res["cross"].assign(family_definition="paper family (Tiny Aya + Aya Expanse = Aya)", **base)
    if alt:
        cross = pd.concat([cross, alt["cross"].assign(family_definition="registry family (Tiny Aya, Aya Expanse "
                                                                         "separate)", **base)], ignore_index=True)
    front = list(base) + ["family_definition", "category"]
    cross = cross[front + [c for c in cross.columns if c not in front]]
    return summary, pd.DataFrame(rows), cross


def plot_spearman_group_distribution(pairs, summary, out_path, title_note):
    fig, ax = plt.subplots(figsize=(3.15, 2.45))
    rng = np.random.default_rng(0)
    med = summary.set_index("group")["median"]
    for x, g in enumerate(spearman_groups.GROUPS):
        sub = pairs.loc[pairs["group"] == g]
        color = SG_GROUP_COLORS[g]
        for same, face, z in ((False, "white", 2), (True, color, 3)):
            s = sub.loc[sub["same_family"] == same, "rho"].dropna()
            if s.empty:
                continue
            jit = rng.uniform(-0.28, 0.28, len(s))
            ax.scatter(x + jit, s, s=5.5, facecolors=face, edgecolors=color, linewidths=0.45, alpha=0.75, zorder=z)
        if np.isfinite(med.get(g, np.nan)):
            ax.plot([x - 0.38, x + 0.38], [med[g]] * 2, color=TEXT_COLOR, linewidth=1.6, zorder=4,
                    solid_capstyle="butt")
            ax.text(x + 0.41, med[g], f"{med[g]:.2f}", va="center", ha="left", fontsize=SG_FONT - 0.5,
                    color=TEXT_COLOR, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.6), zorder=5)
    n = summary.set_index("group")["n_pairs"]
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{g}\n({int(n[g])} pairs)" + ("\nreference" if g == "EC–NEC" else "")
                        for g in spearman_groups.GROUPS], fontsize=SG_FONT)
    ax.set_xlim(-0.55, 2.75)
    lo = float(np.nanmin(pairs["rho"])) if pairs["rho"].notna().any() else 0.0
    ax.set_ylim(min(-0.05, lo - 0.05), 1.0)
    ax.axhline(0, color=TEXT_COLOR, linewidth=0.7, zorder=1)
    ax.set_ylabel("Spearman ρ per model pair", fontsize=SG_FONT + 0.5, color=TEXT_COLOR)
    ax.tick_params(axis="both", labelsize=SG_FONT, labelcolor=TEXT_COLOR, length=2.5, width=0.6, pad=2)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    black_spines(ax, 0.7)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=3.2, markerfacecolor=TEXT_COLOR,
                      markeredgecolor=TEXT_COLOR, label="same family"),
               Line2D([0], [0], marker="o", linestyle="none", markersize=3.2, markerfacecolor="white",
                      markeredgecolor=TEXT_COLOR, label="different families"),
               Line2D([0], [0], color=TEXT_COLOR, linewidth=1.6, label="group median")]
    ax.legend(handles=handles, loc="upper right", fontsize=SG_FONT - 1.2, frameon=True, framealpha=0.92,
              edgecolor="none", handlelength=1.3, borderpad=0.3, labelspacing=0.2, borderaxespad=0.3)
    ax.set_title(f"Descriptive ({title_note}); model pairs are not independent",
                 fontsize=SG_FONT - 1, color=TEXT_COLOR, loc="left", pad=3)
    fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return True


def spearman_group_text(results, inference, cross):
    L = ["Within-group Spearman correlations: English-centric (EC) vs non-English-centric (NEC) models", "=" * 78,
         "Question: do EC models agree more with each other in their sentence-pair-level ΔSLOR than NEC models do?",
         "rho: the existing pairwise Spearman correlation (target-language comparison, exact sentence-pair identity, "
         "saved ΔSLOR, per-language preference filter). Each unordered model pair once; no self-correlations.",
         "EC / NEC: rq1_lmm.model_classification (registry english_centric; unregistered models from their explicit "
         "line-plot section). Families: viz_data2.paper_family (Tiny Aya + Aya Expanse = Aya).",
         "Descriptive only: no hypothesis tests, permutation tests or bootstrap intervals.", ""]
    for variant, desc, res in results:
        s = res["summary"].set_index("group")
        inf = inference.loc[inference["variant"] == variant].set_index("analysis")
        L += [f"[{variant}] {desc}", "-" * 78,
              f"models: {len(res['ec_mask'])} ({int(res['ec_mask'].sum())} EC, {int((~res['ec_mask']).sum())} NEC)"]
        for g in spearman_groups.GROUPS:
            r = s.loc[g]
            L.append(f"  {g:8s} {int(r['n_pairs']):4d} pairs ({int(r['n_pairs_missing'])} undefined): median "
                     f"{r['median']:.3f}, mean {r['mean']:.3f}, SD {r['sd']:.3f}, IQR {r['iqr']:.3f} "
                     f"[{r['q1']:.3f}, {r['q3']:.3f}], range {r['min']:.3f} to {r['max']:.3f}"
                     if np.isfinite(r["median"]) else f"  {g:8s} {int(r['n_pairs'])} pairs: undefined")
        L.append(f"  Δmedianρ (EC–EC − NEC–NEC) = {res['observed']:.3f}")
        for name, row in inf.iterrows():
            parts = []
            if np.isfinite(row.get("estimate", np.nan)):
                parts.append(f"estimate {row['estimate']:.3f}")
            if np.isfinite(row.get("range_min", np.nan)):
                parts.append(f"range {row['range_min']:.3f} to {row['range_max']:.3f}")
            if np.isfinite(row.get("n_resamples", np.nan)) and row.get("n_resamples", 0):
                parts.append(f"n = {int(row['n_resamples'])}")
            L.append(f"  {name}: " + ("; ".join(parts) if parts else "") + f"  ({row['note']})"
                     if name == "family-balanced contrast" else f"  {name}: " + "; ".join(parts))
        c = cross.loc[(cross["variant"] == variant) & cross["family_definition"].str.startswith("paper")]
        for _i, r in c.iterrows():
            if r["n_pairs"]:
                L.append(f"  {r['category']}: {int(r['n_pairs'])} pairs, median {r['median']:.3f}, mean "
                         f"{r['mean']:.3f}, IQR {r['iqr']:.3f}")
            else:
                L.append(f"  {r['category']}: not available (0 pairs)")
        if int((~res["ec_mask"]).sum()) < 3:
            L.append(f"  CAUTION: only {int((~res['ec_mask']).sum())} NEC models -> "
                     f"{int(s.loc['NEC–NEC', 'n_pairs'])} NEC–NEC pair(s); the 'median' is a single correlation and "
                     "this set is not a group comparison. It is not equivalent to the ALL analysis.")
        L.append("")
    variant, desc, res = results[0]
    s = res["summary"].set_index("group")
    inf = inference.loc[inference["variant"] == variant].set_index("analysis")
    c = cross.loc[(cross["variant"] == variant) & cross["family_definition"].str.startswith("paper")].set_index(
        "category")
    ec_cross = c.loc["EC–EC, different families"]
    n_ne_fam = len(set(res["family"][~res["ec_mask"]]))
    n_ec_fam = len(set(res["family"][res["ec_mask"]]))
    L += ["Caveats", "-" * 78,
          "* Model pairs are not independent: every model enters many pairs, and models of one family share "
          "architecture, data and recipe. The " + f"{int(s.loc['EC–EC', 'n_pairs'])} EC–EC and "
          f"{int(s.loc['NEC–NEC', 'n_pairs'])} NEC–NEC correlations are summarised descriptively and never treated "
          "as independent observations.",
          f"* Every family is entirely EC or entirely NEC ({n_ec_fam} EC families, {n_ne_fam} NEC families, each NEC "
          "family a single checkpoint), so EC/NEC is confounded with family.",
          "* NEC models differ in focus language (de, fr, pt, zh, ja) and training type (continual pretraining, "
          "fine-tuning, from scratch); lower NEC–NEC agreement may reflect this heterogeneity rather than the "
          "absence of English-centric training as such.",
          "* Correlation of preferences is not correctness: two models can agree on wrong preferences.", "",
          "ACL-ready paragraph (descriptive)", "-" * 78]
    ec_same = c.loc["EC–EC, same family"]
    fbal = inf.loc["family-balanced contrast"]
    d = res["observed"]
    size_word = "slightly " if abs(d) < 0.1 else ""
    direction = f"{size_word}higher than" if d > 0 else (f"{size_word}lower than" if d < 0 else "equal to")
    mean_diff = s.loc["EC–EC", "mean"] - s.loc["NEC–NEC", "mean"]
    para = (
        "To describe whether English-centric (EC) models agree more with each other than non-English-centric (NEC) "
        "models do, we grouped the pairwise Spearman correlations between the models' sentence-pair-level ΔSLOR "
        f"into EC–EC ({int(s.loc['EC–EC', 'n_models'])} models, {int(s.loc['EC–EC', 'n_pairs'])} pairs) and NEC–NEC "
        f"({int(s.loc['NEC–NEC', 'n_models'])} models, {int(s.loc['NEC–NEC', 'n_pairs'])} pairs) model pairs. "
        f"The median EC–EC correlation (ρ = {s.loc['EC–EC', 'median']:.2f}, IQR {s.loc['EC–EC', 'q1']:.2f}–"
        f"{s.loc['EC–EC', 'q3']:.2f}) was {direction} the median NEC–NEC correlation (ρ = "
        f"{s.loc['NEC–NEC', 'median']:.2f}, IQR {s.loc['NEC–NEC', 'q1']:.2f}–{s.loc['NEC–NEC', 'q3']:.2f}; EC–NEC "
        f"pairs: {s.loc['EC–NEC', 'median']:.2f}), Δmedianρ = {d:.2f}"
        + (f" (the means went the other way, {s.loc['EC–EC', 'mean']:.2f} vs. {s.loc['NEC–NEC', 'mean']:.2f}, "
           "because of a low-agreement EC tail)" if np.sign(mean_diff) != np.sign(d) else "") + ". "
        + (f"EC models from different families correlated with a median ρ = {ec_cross['median']:.2f} "
           f"({int(ec_cross['n_pairs'])} pairs; same-family EC pairs: {ec_same['median']:.2f}), so agreement among "
           "EC models is not confined to models of one family. ")
        + (f"Giving every pair of families equal weight changed the difference to {fbal['estimate']:.2f}. ")
        + (f"With only {n_ec_fam} EC families and {n_ne_fam} single-checkpoint NEC models, these correlations "
           "describe the evaluated models and do not show that English-centric training causes higher agreement."))
    L += textwrap.wrap(para, 110) + [""]
    if len(results) > 1:
        L += ["Sensitivity model sets (different models; NOT equivalent to the ALL analysis; ALL_without_gemma4_it is "
              "post hoc, chosen after seeing that the five Gemma 4 IT checkpoints form the low EC–EC tail, and must "
              "be reported as exploratory):"]
        for variant, desc, r in results[1:]:
            ss = r["summary"].set_index("group")
            L.append(f"  {variant} ({int(r['ec_mask'].sum())} EC / {int((~r['ec_mask']).sum())} NEC): EC–EC "
                     f"{ss.loc['EC–EC', 'median']:.3f}, NEC–NEC {ss.loc['NEC–NEC', 'median']:.3f} "
                     f"({int(ss.loc['NEC–NEC', 'n_pairs'])} pairs), Δ = {r['observed']:.3f}")
    return "\n".join(L) + "\n"


def export_main_spearman_groups(df, main_dir, langs, include_identical, manifest, dist_dir=None):
    written = []
    ec_nec_inputs = None
    cls = rq1_lmm.model_classification().set_index("model_name")
    labels_all, _sections = all_model_labels()
    results, summaries, inferences, crosses, pair_tables, loo, skipped = [], [], [], [], [], [], []
    primary = None
    selection = f"target-language comparison, {preference_filter_label(langs)}; pooled over languages"
    for variant, desc, spec, saved, _primary in SG_VARIANTS:
        models = _sg_models(spec)
        rho, n, frame, present, missing = spearman_group_inputs(df, langs, include_identical, models,
                                                                Path(main_dir) / saved)
        unknown = [m for m in present if m not in cls.index]
        if unknown:
            raise AssertionError(f"no EC/NEC classification for {unknown}")
        ec = {m: bool(cls.loc[m, "english_centric"]) for m in present}
        n_ec = sum(ec.values())
        if n_ec < 2 or len(present) - n_ec < 2:
            print(f"[warning] main_fig Spearman group comparison {variant}: {n_ec} EC / {len(present) - n_ec} NEC "
                  "models with rows; a group needs >= 2 models -> not computed")
            skipped.append({"variant": variant, "model_set": desc, "n_ec": n_ec, "n_nec": len(present) - n_ec,
                            "analysis": "not computed", "note": "fewer than 2 models with rows in a group; missing: "
                                                                + ",".join(missing)})
            continue
        family = {m: paper_family(m) for m in present}
        family_alt = {m: _registry_family(m) for m in present}
        labels = {**labels_all, **dict(models)}
        res = spearman_groups.run_group_comparison(rho, ec, family, labels, family_alt)
        if variant == "ALL":
            ec_nec_inputs = {"rho": rho, "n": n, "frame": frame, "present": present, "missing": missing, "ec": ec,
                             "family": family, "labels": labels, "res": res, "selection": selection,
                             "include_identical": include_identical}
        off = n.to_numpy()[~np.eye(len(present), dtype=bool)]
        s, inf, cr = spearman_group_tables(variant, desc, res, off)
        if missing:
            inf["note"] = inf["note"] + f"; missing models (no rows, excluded): {','.join(missing)}"
        results.append((variant, desc, res))
        summaries.append(s)
        inferences.append(inf)
        crosses.append(cr)
        pair_tables.append(res["pairs"].assign(variant=variant))
        for key, what in (("lofo", "family"), ("lomo", "model")):
            loo.append(res[key].assign(variant=variant, left_out=what))
        if variant == "ALL":
            saved_long = Path(main_dir) / f"{SPEARMAN_ALL_STEM}_long.csv"
            if saved_long.is_file():
                sl = pd.read_csv(saved_long)
                merged = res["pairs"].merge(sl[["model_a", "model_b", "spearman_rho"]], on=["model_a", "model_b"],
                                            how="inner", validate="one_to_one")
                assert len(merged) == len(res["pairs"]), "group pairs not found in the saved ALL long table"
                assert np.allclose(merged["rho"], merged["spearman_rho"], equal_nan=True)
            primary = (variant, res, rho)
    if primary is None:
        path = Path(main_dir) / f"{SG_STEM}_summary.txt"
        path.write_text("Spearman group comparison not computed: the ALL model set has fewer than 2 EC or 2 NEC "
                        "models with rows.\n" + "".join(f"{r['variant']}: {r['note']}\n" for r in skipped),
                        encoding="utf-8")
        _record(written, path)
        _manifest_row(manifest, "Spearman group comparison: not computed", path, "", selection, "", "")
        return written
    summary = pd.concat(summaries, ignore_index=True)
    inference = pd.concat(inferences + [pd.DataFrame(skipped)], ignore_index=True)
    cross = pd.concat(crosses, ignore_index=True)
    req = ",".join(k for k, _h in MAIN_ALL_MODELS)
    note = ("rho = existing Spearman matrices (verified against the saved CSVs); EC/NEC from "
            "rq1_lmm.model_classification; families from viz_data2.paper_family; variants: "
            + "; ".join(f"{v} = {d}" for v, d, *_r in SG_VARIANTS))
    files = (
        (f"{SG_STEM}_summary.csv", summary, "EC–EC / NEC–NEC / EC–NEC descriptives per variant"),
        (f"{SG_STEM}_inference.csv", inference,
         "Δmedianρ descriptive contrasts: difference in means, cross-family, family-balanced, leave-one-out"),
        (f"{SG_STEM}_cross_family.csv", cross, "same- vs cross-family pairs per group (two family definitions)"),
        (f"{SG_STEM}_pairs.csv", pd.concat(pair_tables, ignore_index=True),
         "one row per unordered model pair with group, families and rho"),
        (f"{SG_STEM}_leave_one_out.csv", pd.concat(loo, ignore_index=True), "leave-one-family/model-out Δmedianρ"),
    )
    for name, table, what in files:
        _write_table(written, Path(main_dir) / name, table)
        _manifest_row(manifest, f"Spearman group comparison: {what}", Path(main_dir) / name, "", selection, req, "",
                      notes=note)
    _v, res, _rho = primary
    path = Path(main_dir) / f"{SG_STEM}_distribution{MAIN_FIG_EXT}"
    _record(written, path, plot_spearman_group_distribution(res["pairs"], res["summary"], path,
                                                            f"{len(res['ec_mask'])} models"))
    _manifest_row(manifest, "Spearman group comparison: model-pair rho by group (ALL; descriptive)", path, "",
                  selection, req, "", notes="descriptive only; filled = same-family pair, open = different families; "
                                            "black bar = group median; EC–NEC is a reference group")
    path = Path(main_dir) / f"{SG_STEM}_summary.txt"
    path.write_text(spearman_group_text(results, inference, cross), encoding="utf-8")
    _record(written, path)
    _manifest_row(manifest, "Spearman group comparison: text summary with caveats and ACL paragraph", path, "",
                  selection, req, "", notes=note)
    if ec_nec_inputs is not None:
        written += export_spearmann_bootstrap(ec_nec_inputs, dist_dir or Path(main_dir).parent / SPEARMANN_BOOTSTRAP_DIR,
                                              manifest, Path(main_dir))
    return written


SPEARMANN_BOOTSTRAP_DIR = "dist"
SPEARMANN_BOOTSTRAP_FILE = "spearmann_bootstrap.xlsx"
SPEARMANN_BOOTSTRAP_SHEETS = ("Summary", "Model_Pairs", "Sensitivity", "Metadata")


def export_spearmann_bootstrap(inp, out_dir, manifest, main_dir=None):
    written = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rho, ec, family, labels = inp["rho"], inp["ec"], inp["family"], inp["labels"]
    names = list(rho.index)
    r = rho.to_numpy(dtype=float)
    ec_mask = np.array([bool(ec[m]) for m in names])
    fam = np.array([family[m] for m in names], dtype=object)
    n_ec, n_nec = int(ec_mask.sum()), int((~ec_mask).sum())
    cross_mask = fam[:, None] != fam[None, :]

    grp = inp["res"]["summary"].set_index("group")
    ma, mb, delta, n_ee, n_en = spearman_groups.delta_ec_vs_ecnec(r, ec_mask)
    assert math.isclose(ma, grp.loc["EC–EC", "median"], abs_tol=1e-12)
    assert math.isclose(mb, grp.loc["EC–NEC", "median"], abs_tol=1e-12)
    pairs = inp["res"]["pairs"]
    assert len(pairs) == len(names) * (len(names) - 1) // 2
    assert not pairs.duplicated(["model_a", "model_b"]).any()
    assert not (pairs["model_a"] == pairs["model_b"]).any()
    assert len({frozenset(p) for p in zip(pairs["model_a"], pairs["model_b"])}) == len(pairs)

    def desc_row(variant, comparison, obs, n_pairs, note):
        return {"Analysis variant": variant, "Comparison": comparison, "Number of EC models": n_ec,
                "Number of NEC models": n_nec, "Number of valid EC–EC model pairs": n_pairs[0],
                "Number of valid EC–NEC model pairs": n_pairs[1], "Median EC–EC Spearman rho": obs[0],
                "Median EC–NEC Spearman rho": obs[1], "Difference in median rho": obs[2], "Note": note}

    comparison = "median(rho EC–EC) − median(rho EC–NEC)"
    summary = [desc_row("primary: all statistical models", comparison, (ma, mb, delta), (n_ee, n_en),
                        "descriptive: medians over the model pairs with a defined rho")]
    cma, cmb, cdelta, cn_ee, cn_en = spearman_groups.delta_ec_vs_ecnec(r, ec_mask, cross_mask)
    summary.append(desc_row("sensitivity: cross-family pairs only", comparison + " (same-family pairs excluded)",
                            (cma, cmb, cdelta), (cn_ee, cn_en),
                            "same-family model pairs (viz_data2.paper_family) removed before taking medians; "
                            "descriptive"))
    summary = pd.DataFrame(summary)
    frame = inp["frame"]

    n = inp["n"]
    cls = {m: ("EC" if ec[m] else "NEC") for m in names}
    model_pairs = pd.DataFrame({
        "model_a": pairs["model_a"], "model_b": pairs["model_b"], "label_a": pairs["label_a"],
        "label_b": pairs["label_b"], "class_a": pairs["model_a"].map(cls), "class_b": pairs["model_b"].map(cls),
        "pair_group": pairs["group"], "family_a": pairs["family_a"], "family_b": pairs["family_b"],
        "same_family": pairs["same_family"], "spearman_rho": pairs["rho"],
        "n_shared_sentence_pairs": [int(n.loc[a, b]) for a, b in zip(pairs["model_a"], pairs["model_b"])],
        "in_ec_ec_vs_ec_nec": pairs["group"].isin(["EC–EC", "EC–NEC"]) & np.isfinite(pairs["rho"])})

    sens = []
    for _i, row in summary.iterrows():
        if row["Analysis variant"].startswith("sensitivity"):
            sens.append({"analysis": row["Analysis variant"], "dropped": "",
                         "uncertainty_source": "none (descriptive point estimate)",
                         "n_ec": n_ec, "n_nec": n_nec, "n_ec_ec_pairs": row["Number of valid EC–EC model pairs"],
                         "n_ec_nec_pairs": row["Number of valid EC–NEC model pairs"],
                         "median_rho_ec_ec": row["Median EC–EC Spearman rho"],
                         "median_rho_ec_nec": row["Median EC–NEC Spearman rho"],
                         "delta_median_rho": row["Difference in median rho"], "note": row["Note"]})
    _u, fam_codes = np.unique(fam.astype(str), return_inverse=True)
    lomo = spearman_groups.ec_vs_ecnec_loo(r, ec_mask, [np.arange(len(names)) == i for i in range(len(names))],
                                           [labels.get(m, m) for m in names])
    lofo = spearman_groups.ec_vs_ecnec_loo(r, ec_mask, [fam_codes == i for i in range(len(_u))], list(_u))
    for what, t in (("leave-one-model-out", lomo), ("leave-one-family-out", lofo)):
        t = t.assign(analysis=what, uncertainty_source="model / family selection (point estimates; no resampling)")
        sens += t.to_dict("records")
    sens = pd.DataFrame(sens)
    sens = sens[["analysis"] + [c for c in sens.columns if c != "analysis"]]

    req = ",".join(names)
    meta = [
        ("Primary statistic", "Δ = median(rho EC–EC) − median(rho EC–NEC) (descriptive)"),
        ("Inference", "none: descriptive comparison (no bootstrap, no p-value)"),
        ("Correlation", "Spearman rho per model pair on the sentence pairs valid for both models "
                        "(cross_model_consistency.spearman_matrix; pairwise complete; average ranks), at least "
                        f"{MIN_SHARED_PAIRS_SPEARMAN} shared pairs"),
        ("Model pairs", "unordered pairs counted once, self-correlations excluded; eligible pairs = pairs with a "
                        "defined observed rho"),
        ("Input filters", inp["selection"] + "; include_identical=" + str(inp["include_identical"])),
        ("ΔSLOR", "persisted values from outputs2 (not recomputed)"),
        ("Model selection", f"viz_data2.STAT_MODELS ({len(names)} models: {n_ec} EC, {n_nec} NEC)"),
        ("Models", req),
        ("Models missing (no rows)", ",".join(inp["missing"])),
        ("EC / NEC classification", "rq1_lmm.model_classification (registry english_centric)"),
        ("Model families", "viz_data2.paper_family"),
        ("Number of sentence pairs", int(frame[PAIR_ID_COL].nunique())),
        ("Source files", "src/export_plots2.py (export_spearmann_bootstrap), src/spearman_groups.py, "
                         "src/cross_model_consistency.py, src/viz_data2.py, src/rq1_lmm.py; "
                         "observed rho verified against plots2/main_fig/" + f"{SPEARMAN_ALL_STEM}.csv"),
        ("Limitation: dependence", "model pairs share models, so the pair correlations are not independent"),
        ("Limitation: confounding", "EC / NEC status coincides with family, size and training data; the comparison "
                                    "is observational and does not show that English-centric training causes "
                                    "stronger agreement"),
        ("Unchanged analyses", "EC–EC vs NEC–NEC comparison (Spearman_group_comparison_*), heatmaps and all other "
                               "outputs are not modified by this analysis"),
    ]
    metadata = pd.DataFrame(meta, columns=["Field", "Value"])
    metadata["Value"] = metadata["Value"].astype(str)

    path = out_dir / SPEARMANN_BOOTSTRAP_FILE
    sheets = dict(zip(SPEARMANN_BOOTSTRAP_SHEETS, (summary, model_pairs, sens, metadata)))
    with pd.ExcelWriter(path) as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name, index=False)
    _record(written, path)
    p = summary.iloc[0]
    print(f"[spearmann_bootstrap] Δmedianρ(EC–EC − EC–NEC) = {p['Difference in median rho']:.4f} (descriptive)")
    _manifest_row(manifest, "Spearman EC–EC vs EC–NEC descriptive workbook", path, len(model_pairs),
                  inp["selection"], req, ",".join(inp["missing"]),
                  notes=f"Δ = {p['Difference in median rho']:.4f}; descriptive (no resampling)")
    return written


RQ1_STEM = "RQ1_violin_main_b_no-sv-no-de"
RQ1_X_LABEL = "Origin of the transferred strategy (all sentences in English)"


def rq1_violin_name(metric, manaca=False):
    name = RQ1_STEM
    if metric != SLOR_COL:
        name += f"__{score_metric(metric)['slug']}"
    if manaca:
        name += f"__{MAIN_MANACA_SUFFIX}"
    return name + MAIN_FIG_EXT


def rq1_summary(rows, value_col, metric):
    if not len(rows):
        return pd.DataFrame()
    g = rows.groupby(["model_name", STRATEGY_ORIGIN_COL], sort=False)[value_col]
    out = g.agg(N="size", mean="mean", median="median").reset_index()
    out[STRATEGY_ORIGIN_COL] = pd.Categorical(out[STRATEGY_ORIGIN_COL], RQ1_ORIGINS, ordered=True)
    out = out.sort_values(["model_name", STRATEGY_ORIGIN_COL]).reset_index(drop=True)
    out.insert(2, "scored_language", "en")
    out["metric_column"] = metric
    out["sign"] = "native English (Proper English strategy) minus transferred English (Literal English " \
                  "translation of target language strategy)"
    out["preference_filter"] = f"{ENGLISH_PREFERENCE_COL} == '{RQ1_PREFERENCE}' (only criterion)"
    out["target_language_preference_filter"] = "not applied"
    return out


def export_main_rq1_violins(df, main_dir, include_identical, manifest):
    written = []
    for metric in MAIN_VIOLIN_METRICS:
        spec = score_metric(metric)
        value_col, y_label = spec["delta"], MAIN_METRIC_Y_LABELS[metric]
        frame = english_side_origin_frame(df, metric, RQ1_ORIGINS, RQ1_PREFERENCE, include_identical)
        for manaca, models in ((False, MAIN_MONOLINGUAL_SET_A), (True, MAIN_MONOLINGUAL_SET_A_MANACA)):
            present, missing = resolve_main_models(df, models)
            if missing and metric == SLOR_COL:
                print(f"[warning] main_fig RQ1: no outputs for {missing} (shown as 'no outputs')")
            rows = frame.loc[frame["model_name"].isin(present)] if len(frame) else frame
            panel = {"rows": rows, "models": models, "missing": missing, "color_at": _monolingual_color_at,
                     "legend_labels": MAIN_MONOLINGUAL_LEGEND_LABELS, "grouped_training": True,
                     "languages": RQ1_ORIGINS, "english_boundary": False, "x_label": RQ1_X_LABEL}
            path = main_dir / rq1_violin_name(metric, manaca)
            _write_table(written, path.with_name(path.stem + "_summary.csv"), rq1_summary(rows, value_col, metric))
            counts = rows.groupby(STRATEGY_ORIGIN_COL).size() if len(rows) else pd.Series(dtype=int)
            lim = violin_y_limits(rows[value_col]) if len(rows) else (np.nan, np.nan)
            write_violin_pair(
                written, manifest, path,
                lambda pth, show, ylim, panel=panel: plot_main_violin_panel(
                    panel, pth, value_col, y_label, show_mean_median=show, ylim=ylim),
                f"RQ1 English-side violin (b) {spec['label']}" + (" (Manacá 1B instead of Sabiá 7B)" if manaca else ""),
                int(len(rows)),
                f"{spec['label']} (saved {metric}) of the ENGLISH comparison: y = native English (Proper English "
                "strategy) minus transferred English (Literal English translation of target language strategy); "
                f"x = STRATEGY ORIGIN (original target language of each row: {','.join(RQ1_ORIGINS)}; sv, de "
                f"excluded; no en category); {ENGLISH_PREFERENCE_COL} exactly {RQ1_PREFERENCE} (target-language "
                "preference annotation NOT used); " + ",".join(m for m, _l in models),
                ",".join(m for m, _l in models), ",".join(missing),
                notes="NOT the original panel (b): every category is scored in English; no black English "
                      "boundary; rows per origin " + ", ".join(f"{k}={int(v)}" for k, v in counts.items())
                      + f"; own y range {lim[0]:.3f} .. {lim[1]:.3f}; model colour at its own language, grey "
                        "elsewhere (as panel b); per-model/origin N, mean, median in "
                      + path.stem + "_summary.csv")
    return written


REG_PREDICTOR_SIZE = 38
REG_TICK_SIZE = 36
REG_AXIS_LABEL_SIZE = 40
REG_LEGEND_SIZE = 36
REG_ANNOTATION_SIZE = 32
REG_LABEL_WRAP = 20
REG_LEGEND_NCOL = 2
REG_TERM_HEIGHT = 2.05
REG_FAMILY_SPREAD = 0.32
REG_CI_WIDTH = 4.0
REG_CI_TOP_WIDTH = 1.3
REG_CAP_SIZE = 9

REG_IBM_FAMILY_COLORS = {
    "aya": STRICT_PALETTE["yellow"],
    "eurollm": STRICT_PALETTE["orange"],
    "gemma": STRICT_PALETTE["magenta"],
    "apertus": STRICT_PALETTE["blue"],
}
REG_MARKER_DIAMETER = 19.0
REG_MARKER_AREA_FACTOR = {"o": 1.0, "s": 0.892, "^": 1.262, "D": 0.884}


def reg_marker_size(marker):
    return REG_MARKER_DIAMETER * REG_MARKER_AREA_FACTOR.get(marker, 1.0)


def reg_family_style(family):
    marker = MAIN_FAMILY_STYLE[family]["marker"]
    return REG_IBM_FAMILY_COLORS[family], marker, reg_marker_size(marker)


def plot_family_coefficients(coefs, summary, out_path):
    terms = [t for t in TERM_ORDER if t in set(coefs["term"])]
    fams = [f for f, _l, _r in REGRESSION_FAMILIES if f in set(coefs["family"]) or f in set(summary["family"])]
    if not terms or not fams:
        return False
    if is_delegated(out_path):
        return True
    fig, ax = plt.subplots(figsize=(14.0, REG_TERM_HEIGHT * len(terms) + 2.6))
    offsets = np.linspace(-REG_FAMILY_SPREAD, REG_FAMILY_SPREAD, len(fams)) if len(fams) > 1 else [0.0]
    labels = dict((f, l) for f, l, _r in REGRESSION_FAMILIES)
    n_by = summary.set_index("family")["n"].to_dict() if len(summary) else {}
    status_by = summary.set_index("family")["status"].to_dict() if len(summary) else {}
    xmax = np.nanmax(np.abs(coefs.loc[coefs["term"].isin(terms), ["standardized_ci_lower", "standardized_ci_upper"]]
                            .to_numpy(dtype=float))) if len(coefs) else 1.0
    xmax = float(xmax) if np.isfinite(xmax) and xmax > 0 else 1.0
    for fi, fam in enumerate(fams):
        color, marker, msize = reg_family_style(fam)
        sub = coefs.loc[coefs["family"] == fam].set_index("term")
        for ti, term in enumerate(terms):
            yv = ti + offsets[fi]
            b = sub["standardized_beta"].get(term, np.nan) if term in sub.index else np.nan
            if not np.isfinite(b):
                continue
            lo, hi = sub.loc[term, "standardized_ci_lower"], sub.loc[term, "standardized_ci_upper"]
            ax.errorbar([b], [yv], xerr=[[b - lo], [hi - b]], fmt="none", ecolor=color, capsize=REG_CAP_SIZE,
                        linewidth=REG_CI_WIDTH, capthick=3.0, zorder=3)
            ax.plot([b], [yv], marker=marker, markersize=msize, color=color, markeredgecolor=TEXT_COLOR,
                    markeredgewidth=1.0, linestyle="none", zorder=4)
            ax.plot([lo, hi], [yv, yv], color=TEXT_COLOR, linewidth=REG_CI_TOP_WIDTH,
                    solid_capstyle="butt", zorder=5)
    ax.axvline(0, color=TEXT_COLOR, linestyle="-", linewidth=2.2, zorder=1)
    draw_factor_separators(ax, len(terms))
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels([textwrap.fill(MAIN_FIG_TERM_LABELS.get(t, TERM_LABELS.get(t, t)), REG_LABEL_WRAP)
                        for t in terms], fontsize=REG_PREDICTOR_SIZE, linespacing=0.95)
    ax.set_ylim(len(terms) - 0.5, -0.5)
    ax.set_xlim(-1.08 * xmax, 1.08 * xmax)
    ax.tick_params(axis="both", labelcolor=TEXT_COLOR)
    ax.tick_params(axis="x", labelsize=REG_TICK_SIZE)
    ax.tick_params(axis="both", length=10, width=2.0, pad=8)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, symmetric=True))
    ax.set_xlabel(MAIN_STD_COEF_XLABEL, fontsize=REG_AXIS_LABEL_SIZE, labelpad=10)
    black_spines(ax, 1.6)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)

    def legend_label(f):
        status = str(status_by.get(f, "not fitted"))
        if status != "ok":
            return f"{labels[f]} (not estimated: {status.split(' (')[0]})"
        return f"{labels[f]} (N={int(n_by.get(f, 0)):,})"

    handles = [Line2D([0], [0], marker=reg_family_style(f)[1], color=reg_family_style(f)[0],
                      markeredgecolor=TEXT_COLOR, markeredgewidth=1.0, markersize=reg_family_style(f)[2],
                      linewidth=REG_CI_WIDTH, label=legend_label(f)) for f in fams]
    fig.tight_layout()
    rows = -(-len(handles) // REG_LEGEND_NCOL)
    handles = [handles[r * REG_LEGEND_NCOL + c] for c in range(REG_LEGEND_NCOL) for r in range(rows)
               if r * REG_LEGEND_NCOL + c < len(handles)]
    fig.legend(handles=handles, frameon=False, fontsize=REG_LEGEND_SIZE, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=REG_LEGEND_NCOL, handletextpad=0.4, columnspacing=1.2,
               handlelength=1.6)
    fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return True


def export_main_family_regressions(df, main_dir, langs, metadata, include_identical, manifest, exclude_freq=True,
                                   glotcc=False):
    written = []
    res = run_family_regressions(df, langs, metadata if metadata is not None and not metadata.empty else None,
                                 include_identical=include_identical, exclude_freq=exclude_freq, glotcc=glotcc)
    frame, coefs, summary = res["frame"], res["coefficients"], res["summary"]
    if len(frame):
        assert (frame["outcome"] == SLOR_COL).all()
        assert np.array_equal(frame["y"].to_numpy(dtype=float), frame[SLOR_COL].to_numpy(dtype=float), equal_nan=True)
        assert (frame["comparison"] == TARGET_COMPARISON_POS).all()
        assert set(frame["language"]) <= set(langs)
    if summary["n"].sum() == 0:
        raise ValueError("main regression: zero eligible observations in every family "
                         "(check --metadata, --outputs_dir and --use-preference-annotation)")
    plot_coefs = standardized_coefficient_intervals(coefs)
    _write_table(written, main_dir / "main_regression_coefficients.csv",
                 plot_coefs.loc[~plot_coefs["term"].astype(str).str.startswith("language=")].assign(
                     label=lambda t: t["term"].map(lambda x: MAIN_FIG_TERM_LABELS.get(x, x))))
    _write_table(written, main_dir / "main_regression_coefficients_with_fixed_effects.csv", plot_coefs)
    _write_table(written, main_dir / "main_regression_family_summary.csv", summary)
    _write_table(written, main_dir / "main_regression_family_membership.csv", res["membership"])
    _write_table(written, main_dir / "main_regression_frame.csv",
                 frame.drop(columns=["pos_sentence", "neg_sentence"], errors="ignore"))
    path = main_dir / MAIN_COEF_FIGURE
    _record(written, path, plot_family_coefficients(plot_coefs, summary, path))
    for _i, r in summary.iterrows():
        print(f"[main regression] {r['family_label']}: status={r['status']}; models={r['models']}; N={r['n']}; "
              f"unique pairs={r['n_unique_sentence_pairs']}; languages={r['languages']}")
        print(f"    included: {r['predictors_included']}")
        if r["predictors_omitted"]:
            print(f"    omitted:  {r['predictors_omitted']}")
        if r["registered_models_without_rows"]:
            print(f"    [warning] registered but no regression rows: {r['registered_models_without_rows']}")
        print(f"    missing values by predictor (before complete-case): {r['missing_by_predictor']}")
        if r["status"] != "ok":
            print(f"    [warning] {r['family_label']} regression not estimated: {r['status']} (no coefficients shown)")
    for status, k in res["report"].get("metadata_join_status", {}).items():
        if status != "ok":
            print(f"[warning] {k} regression rows have metadata_join_status={status}")
    if glotcc:
        print(f"[warning] --glotcc: {COMMON_CRAWL_CAVEAT}")
    _manifest_row(manifest, f"main regression ({len(summary)} family fits)", path, int(summary["n"].sum()),
                  f"target-language comparison; {preference_filter_label(langs)}; identical pairs "
                  f"{'included' if include_identical else 'excluded'}; frequency "
                  f"{'excluded' if exclude_freq else 'included'}"
                  + (f"; {COMMON_CRAWL_LABEL} instead of language fixed effects" if glotcc else ""),
                  notes="; ".join(f"{r['family']}: N={r['n']}, status={r['status']}" for _i, r in summary.iterrows()))
    return written, res


def _darker(color, factor=0.7):
    return tuple(factor * np.asarray(matplotlib.colors.to_rgb(color)))


def main_japanese_control_display_label(language, control_category):
    return ENGLISH_CONTROL_CATEGORY if control_category == CONTROL_JA_BASELINE else str(language)


def _paired_categories(frame):
    return [c for c in paired_category_order(frame[ORIGINAL_TARGET_LANGUAGE_COL].unique())
            if ((frame[ORIGINAL_TARGET_LANGUAGE_COL] == c[0]) & (frame[CONTROL_CATEGORY_COL] == c[1])).any()]


def plot_main_japanese_control(frame, models, out_path, value_col=DELTA_SLOR, y_label="ΔSLOR", show_mean_median=True,
                               ylim=None):
    if frame.empty:
        return False
    if is_delegated(out_path):
        return tuple(ylim) if ylim is not None else violin_y_limits(frame[value_col])
    categories = _paired_categories(frame)
    names = [m for m, _l in models]
    n_models = max(len(names), 1)
    slot = min(0.84 / n_models, 0.2)
    fig, ax = plt.subplots(figsize=(SPLIT_VIOLIN_WIDTH, SPLIT_VIOLIN_AXES_HEIGHT))
    draw_language_group_separators(ax, [lang for lang, _c in categories], scale=1.4)
    per_category, means = [], {}
    for ci, (lang, cat) in enumerate(categories):
        sub = frame.loc[(frame[ORIGINAL_TARGET_LANGUAGE_COL] == lang) & (frame[CONTROL_CATEGORY_COL] == cat)]
        per_category.append(sub[value_col].to_numpy(dtype=float))
        for mi, m in enumerate(names):
            values = sub.loc[sub["model_name"] == m, value_col].to_numpy(dtype=float)
            if not values.size:
                continue
            color, hatch = main_model_style(m)
            pos = ci + (mi - (n_models - 1) / 2) * slot
            means[(lang, cat, m)] = (pos, _violin(ax, values, pos, color, slot * 0.9, hatch=hatch,
                                                  show_mean_median=show_mean_median))
    linked = CONNECT_PAIRED_MEANS and bool(draw_paired_mean_links(
        ax, means, [((l, CONTROL_JAPANESE, m), (l, CONTROL_JA_BASELINE, m))
                    for l in dict.fromkeys(l for l, _c in categories) for m in names]))
    draw_zero_line(ax, 2.4)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([main_japanese_control_display_label(l, c) for l, c in categories])
    ax.set_xlim(-0.5, len(categories) - 0.5)
    _style_main_violin_axes(ax, y_label)
    paired_models = set(frame["model_name"])
    missing = [m for m in names if m not in paired_models]
    handles, ncol, bold = main_violin_legend_entries(models, missing, {}, False, show_mean_median, linked=linked)
    handles = [Patch(facecolor="white", edgecolor=MUTED_COLOR, linestyle="--",
                     label=h.get_label().replace("(no outputs)", "(no paired data)"))
               if h.get_label().endswith("(no outputs)") else h for h in handles]
    legend = _place_main_violin_legend(ax, handles, ncol, bold)
    lo, hi = violin_y_limits(frame[value_col])
    ax.set_ylim(lo, hi)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(lo, fit_legend_headroom(ax, legend, None, None, value_col, lo, hi,
                                            category_values=per_category))
    out = ax.get_ylim()
    fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out


def export_main_japanese_control(df, main_dir, include_identical, manifest):
    written = []
    pref = MAIN_JAPANESE_CONTROL_PREFERENCE
    present, missing = resolve_main_models(df, MAIN_JAPANESE_CONTROL_MODELS)
    if missing:
        print(f"[warning] main_fig Japanese control: no outputs for {missing} (left out, not replaced)")
    keep = set(present)
    models = [(m, l) for m, l in MAIN_JAPANESE_CONTROL_MODELS if m in keep]
    check_violin_colors(MAIN_JAPANESE_CONTROL_MODELS, "Japanese control")
    pairing = {}
    for metric in MAIN_VIOLIN_METRICS:
        spec = score_metric(metric)
        value_col, stem = spec["delta"], main_japanese_control_stem(metric)
        all_japanese, all_baseline = paired_baseline_inputs(df, metric, include_identical, pref)
        japanese = all_japanese.loc[all_japanese["model_name"].isin(keep)] if len(all_japanese) else all_japanese
        baseline = all_baseline.loc[all_baseline["model_name"].isin(keep)] if len(all_baseline) else all_baseline
        frame = pair_japanese_baseline(japanese, baseline, pref)
        path = main_dir / f"{stem}{MAIN_FIG_EXT}"
        drawn = plot_main_japanese_control(frame, models, path, value_col, MAIN_METRIC_Y_LABELS[metric])
        _record(written, path, bool(drawn))
        twin = no_mean_median_path(path)
        _record(written, twin, bool(drawn) and bool(plot_main_japanese_control(
            frame, models, twin, value_col, MAIN_METRIC_Y_LABELS[metric], show_mean_median=False, ylim=drawn)))
        summary = paired_baseline_summary(frame, value_col)
        if len(summary):
            summary.insert(2, "display_label", [main_japanese_control_display_label(l, c) for l, c in
                                                zip(summary[ORIGINAL_TARGET_LANGUAGE_COL],
                                                    summary[CONTROL_CATEGORY_COL])])
        _write_table(written, main_dir / f"{stem}_summary.csv", summary)
        exclusions = paired_baseline_exclusions(df, all_japanese, all_baseline, pref, include_identical)
        exclusions = exclusions.loc[exclusions["model_name"].isin(keep)] if len(exclusions) else exclusions
        _write_table(written, main_dir / f"{stem}_exclusions.csv", exclusions)
        if len(frame):
            pairing[metric] = frame.groupby([ORIGINAL_TARGET_LANGUAGE_COL, CONTROL_CATEGORY_COL, "model_name"]).size()
        langs = language_axis_order(frame[ORIGINAL_TARGET_LANGUAGE_COL].unique()) if len(frame) else []
        sparse = []
        for lang in langs:
            for cat in (CONTROL_JAPANESE, CONTROL_JA_BASELINE):
                for m in present:
                    k = int(((frame[ORIGINAL_TARGET_LANGUAGE_COL] == lang) & (frame[CONTROL_CATEGORY_COL] == cat)
                             & (frame["model_name"] == m)).sum())
                    if k < MAIN_MIN_PAIRED_OBSERVATIONS:
                        sparse.append(f"{m} {lang}/{main_japanese_control_display_label(lang, cat)}: n={k}")
        candidate_langs = (set(japanese["Target language"].map(canonical_language_code)) if len(japanese) else set())
        for lang in language_axis_order(candidate_langs - set(langs)):
            sparse.append(f"{lang}: Japanese-control candidates but no matched baseline for any selected model, n=0")
        for s in sparse:
            print(f"[warning] main_fig Japanese control ({spec['label']}): insufficient paired observations ({s}; "
                  f"< {MAIN_MIN_PAIRED_OBSERVATIONS}, no violin drawn)")
        if frame.empty:
            print(f"[warning] main_fig Japanese control ({spec['label']}): no paired observations at all "
                  "(figure not written)")
        _manifest_row(manifest, f"Japanese control paired JA baseline pref-gtgt{_metric_tag(metric)}", path, int(len(frame)),
                      f"{spec['label']} (saved {metric}); Japanese-control rows with Japanese preference annotation "
                      "exactly >> inner-joined one-to-one with the standard ja comparison (target preference "
                      "annotation exactly >>) on (model, Proper English strategy) per original language; x labels: "
                      "original language (JA control) and `en` (paired JA baseline: English strategy transferred "
                      "into Japanese)",
                      ",".join(m for m, _l in MAIN_JAPANESE_CONTROL_MODELS), ",".join(missing),
                      notes=("FIGURE NOT WRITTEN (no paired observations); " if not drawn else "")
                            + f"languages {','.join(langs)}; grey separators between language pairs; no black English "
                              "boundary (no English-language category); "
                            + ("cells with < 2 paired observations: " + "; ".join(sparse) if sparse
                               else "every model/category has >= 2 paired observations")
                            + "; same design as the standard main violins (fixed IBM Carbon checkpoint colours)")
        if drawn:
            _manifest_row(manifest, f"Japanese control paired JA baseline pref-gtgt {spec['label']} [no mean/median]",
                          twin, int(len(frame)), "as " + path.name,
                          ",".join(m for m, _l in MAIN_JAPANESE_CONTROL_MODELS), ",".join(missing),
                          notes=f"same as {path.name} {NO_MEDIAN_TWIN_NOTE} "
                                "(identical paired rows, colours, order, legend position and y limits; paired "
                                "JA-control / JA-baseline means of each model joined by a thin line)")
    if len(pairing) == 2 and not pairing[SLOR_COL].equals(pairing[MEANLP_COL]):
        print("[warning] main_fig Japanese control: ΔSLOR and ΔmeanLP pairings differ in size per cell "
              "(metric-specific missing scores; see the per-metric _summary.csv / _exclusions.csv)")
    return written


def main_family_of(registry_family):
    for slug, _label, registry in REGRESSION_FAMILIES:
        if registry_family in registry:
            return slug
    return None


def plot_main_model_size_win_rate(table, out_path):
    plotted = table.loc[table["win_rate_percent"].notna()]
    if plotted.empty:
        return False
    labels = {f: l for f, l, _r in REGRESSION_FAMILIES}
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    for (fam, family, it), g in plotted.groupby(["main_family", "family", "instruction_tuned"], sort=True):
        g = g.sort_values(["model_size", "model_name"], kind="stable")
        style = MAIN_FAMILY_STYLE[fam]
        x, y = g["model_size"].to_numpy(dtype=float), g["win_rate_percent"].to_numpy(dtype=float)
        connect = len(g) > 1 and len(np.unique(x)) == len(x)
        ax.plot(x, y, linestyle=WIN_RATE_LINESTYLES[bool(it)] if connect else "none", linewidth=2.4,
                color=style["color"], marker=style["marker"], markersize=12,
                markerfacecolor="white" if it else style["color"], markeredgecolor=_darker(style["color"], 0.75)
                if not it else style["color"], markeredgewidth=2.0, zorder=3)
    ax.set_xscale("log")
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200)
             if plotted["model_size"].min() / 1.5 <= t <= plotted["model_size"].max() * 1.5]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t}B" for t in ticks]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(plotted["model_size"].min() / 1.4, plotted["model_size"].max() * 1.4)
    ax.set_ylim(*MAIN_WIN_RATE_YLIM)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(100, decimals=0))
    ax.axhline(50, color=MUTED_COLOR, linestyle=":", linewidth=1.0, zorder=1)
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=MAIN_TICK_SIZE + 4, labelcolor=TEXT_COLOR, length=6)
    ax.set_xlabel("Model size", fontsize=MAIN_LABEL_SIZE + 4, color=TEXT_COLOR)
    ax.set_ylabel("Accuracy", fontsize=MAIN_LABEL_SIZE + 4, color=TEXT_COLOR)
    black_spines(ax, 1.3)
    fams = [f for f, _l, _r in REGRESSION_FAMILIES if f in set(plotted["main_family"])]
    handles = [Line2D([0], [0], color=MAIN_FAMILY_STYLE[f]["color"], marker=MAIN_FAMILY_STYLE[f]["marker"],
                      linestyle="none", markersize=12, markeredgecolor=_darker(MAIN_FAMILY_STYLE[f]["color"], 0.75),
                      markeredgewidth=2.0, label=labels[f]) for f in fams]
    handles += [Line2D([0], [0], color=TEXT_COLOR, linestyle="-", linewidth=2.2, marker="o", markersize=9,
                       label="base"),
                Line2D([0], [0], color=TEXT_COLOR, linestyle="--", linewidth=2.2, marker="o", markersize=9,
                       markerfacecolor="white", label="instruction-tuned")]
    ax.legend(handles=handles, frameon=True, framealpha=0.92, edgecolor="none", facecolor="white",
              fontsize=MAIN_LEGEND_SIZE + 4, loc="lower right", ncol=2, columnspacing=1.2, handlelength=2.2,
              borderaxespad=0.4)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return True


def export_main_model_size_win_rate(df, main_dir, langs, include_identical, manifest):
    written = []
    table, unresolved = model_size_win_rate_frame(df, langs, include_identical)
    if unresolved:
        print(f"[warning] main_fig model-size win rate: no registry size/family/IT, not plotted: {unresolved}")
    table = table.assign(main_family=table["family"].map(main_family_of))
    table = table.loc[table["main_family"].isin(MAIN_WIN_RATE_FAMILIES)].reset_index(drop=True)
    csv_path = main_dir / f"{MAIN_WIN_RATE_STEM}.csv"
    _write_table(written, csv_path, table)
    path = main_dir / f"{MAIN_WIN_RATE_STEM}{MAIN_FIG_EXT}"
    _record(written, path, plot_main_model_size_win_rate(table, path))
    lo, hi = MAIN_WIN_RATE_YLIM
    outside = table.loc[(table["win_rate_percent"] < lo) | (table["win_rate_percent"] > hi), "model_name"].tolist()
    if outside:
        print(f"[warning] main_fig model-size win rate: outside the {lo}-{hi}% display range (kept in CSV): {outside}")
    _manifest_row(manifest, "model size vs accuracy (ΔSLOR win rate)", path, int(table["N_total"].sum()),
                  f"target-language comparison, {preference_filter_label(langs)}; registry families Aya Expanse, "
                  "Tiny Aya, EuroLLM, Apertus; accuracy = 100 x N(ΔSLOR > 0) / N(valid ΔSLOR)",
                  ",".join(table["model_name"]),
                  notes=f"{len(table)} models; y axis shown {lo}-{hi}%"
                        + (f"; outside display range: {','.join(outside)}" if outside else "; none outside display range")
                        + f"; values in {csv_path.name}")
    return written


RQ1_LMM_PREFIX = "RQ1_LMM_"
RQ1_LMM_SPEC_STYLE = {
    "ec_model_language": ("+ model×language intercepts (Satterthwaite t)", "s", False),
    "nec_model_language": ("+ model×language intercepts (Satterthwaite t)", "s", False),
}
RQ1_LMM_FONT = 7.5


def _lmm_axes(ax, xlabel):
    ax.axvline(0, color=TEXT_COLOR, linewidth=0.8, zorder=1)
    ax.set_xlabel(xlabel, fontsize=RQ1_LMM_FONT + 0.5, color=TEXT_COLOR)
    ax.tick_params(axis="both", labelsize=RQ1_LMM_FONT, labelcolor=TEXT_COLOR, length=2.5, width=0.6, pad=2)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    black_spines(ax, 0.7)


def _lmm_point(ax, x, y, lo, hi, color, marker, filled, size=4.2):
    ax.plot([lo, hi], [y, y], color=color, linewidth=1.0, solid_capstyle="butt", zorder=2)
    ax.plot([x], [y], marker=marker, markersize=size, linestyle="none", color=color,
            markerfacecolor=color if filled else "white", markeredgewidth=0.9, zorder=3)


def plot_rq1_ec_contrasts(contrasts, out_path):
    pooled = contrasts.loc[contrasts["scope"] == "pooled LMM"]
    rows = [rq1_lmm.PLANNED_CONTRAST]
    specs = [s for s in ("ec_model_language",) if s in set(pooled["specification"])]
    fig, ax = plt.subplots(figsize=(3.15, 2.55))
    offsets = np.linspace(-0.22, 0.22, len(specs)) if len(specs) > 1 else [0.0]
    for yi, contrast in enumerate(rows):
        lang = contrast.rsplit(" ", 1)[-1]
        color = LANGUAGE_COLORS.get(lang, TEXT_COLOR)
        for off, spec in zip(offsets, specs):
            r = pooled.loc[(pooled["specification"] == spec) & (pooled["contrast"] == contrast)]
            if r.empty:
                continue
            r = r.iloc[0]
            _label, marker, filled = RQ1_LMM_SPEC_STYLE[spec]
            _lmm_point(ax, r["estimate"], yi + off, r["ci_lower"], r["ci_upper"], color, marker, filled)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["English − mean of six\nnon-English (planned)"], fontsize=RQ1_LMM_FONT)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    hi = float(np.nanmax(pooled["ci_upper"])) if len(pooled) else 1.0
    ax.set_xlim(min(-0.1, float(np.nanmin(pooled["ci_lower"])) - 0.1), hi * 1.06)
    _lmm_axes(ax, "Difference in mean ΔSLOR (95% CI)")
    handles = [Line2D([0], [0], marker=RQ1_LMM_SPEC_STYLE[s][1], color=TEXT_COLOR, linestyle="-", linewidth=1.0,
                      markersize=4, markerfacecolor=TEXT_COLOR if RQ1_LMM_SPEC_STYLE[s][2] else "white",
                      label=RQ1_LMM_SPEC_STYLE[s][0]) for s in specs]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.32, -0.2), fontsize=RQ1_LMM_FONT - 1.5, frameon=True, framealpha=0.92,
              edgecolor="none", handlelength=1.6, borderpad=0.3, labelspacing=0.25, borderaxespad=0.2)
    fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return True


def plot_rq1_ec_model_contrasts(per_model, out_path, sections):
    fig, ax = plt.subplots(figsize=(3.15, 3.9))
    for yi, (_i, r) in enumerate(per_model.iterrows()):
        ax.plot([r["estimate"]], [yi], marker="o", markersize=3.4, linestyle="none", color=TEXT_COLOR,
                markerfacecolor=TEXT_COLOR, markeredgewidth=0.9, zorder=3)
    secs = [sections.get(m) for m in per_model["model"]]
    for i in range(1, len(secs)):
        if secs[i] != secs[i - 1]:
            ax.axhline(i - 0.5, color=SECTION_SEPARATOR_COLOR, linewidth=0.6)
    ax.set_yticks(range(len(per_model)))
    ax.set_yticklabels(per_model["label"], fontsize=RQ1_LMM_FONT - 0.5)
    ax.set_ylim(len(per_model) - 0.5, -0.5)
    ax.set_xlim(min(-0.1, float(per_model["estimate"].min()) - 0.1), float(per_model["estimate"].max()) * 1.05)
    _lmm_axes(ax, "English − mean(non-English) ΔSLOR")
    fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return True


def plot_rq1_nec_coefficients(coefs, model_effects, out_path):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.3, 2.75), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.75})
    focus_specs = [s for s in ("nec_model_language",)
                   if not coefs.loc[(coefs["specification"] == s) & (coefs["term"] == "is_focus")].empty]
    labels, y = [], 0
    for s in focus_specs:
        r = coefs.loc[(coefs["specification"] == s) & (coefs["term"] == "is_focus")].iloc[0]
        _label, marker, filled = RQ1_LMM_SPEC_STYLE[s]
        _lmm_point(ax, r["estimate"], y, r["ci_lower"], r["ci_upper"], TEXT_COLOR, marker, filled)
        labels.append(("Focus language: " if y == 0 else "") + RQ1_LMM_SPEC_STYLE[s][0].replace(" (Wald z)", ""))
        y += 1
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=RQ1_LMM_FONT - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    _lmm_axes(ax, "Coefficient (ΔSLOR, 95% CI)")
    ax.set_title("(a) Focus-language effect", loc="left", fontsize=RQ1_LMM_FONT + 0.5, fontweight="bold", color=TEXT_COLOR)
    me = model_effects.reset_index(drop=True)
    for yi, r in me.iterrows():
        color = LANGUAGE_COLORS.get(r["focus_languages"].split(",")[0], TEXT_COLOR)
        bx.plot([r["raw_focus_minus_nonfocus"]], [yi - 0.13], marker="o", markersize=3.6, linestyle="none",
                color=color, markerfacecolor="white", markeredgewidth=0.9, zorder=3)
        bx.plot([r["language_adjusted_difference"]], [yi + 0.13], marker="D", markersize=3.6, linestyle="none",
                color=color, zorder=3)
    bx.set_yticks(range(len(me)))
    bx.set_yticklabels([f"{l} ({f})" for l, f in zip(me["label"], me["focus_languages"])], fontsize=RQ1_LMM_FONT - 0.5)
    bx.set_ylim(len(me) + 0.6, -0.5)
    _lmm_axes(bx, "Focus − non-focus ΔSLOR")
    bx.set_title("(b) Per model", loc="left", fontsize=RQ1_LMM_FONT + 0.5, fontweight="bold", color=TEXT_COLOR)
    bx.legend(handles=[Line2D([0], [0], marker="o", color=TEXT_COLOR, markerfacecolor="white", markersize=3.6,
                              linestyle="none", label="raw"),
                       Line2D([0], [0], marker="D", color=TEXT_COLOR, markersize=3.6, linestyle="none",
                              label="language-adjusted")],
              loc="lower right", fontsize=RQ1_LMM_FONT - 1.5, frameon=True, framealpha=0.92, edgecolor="none",
              borderpad=0.3, labelspacing=0.25, borderaxespad=0.2)
    fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return True


def _fmt_p(p):
    return "n/a" if not np.isfinite(p) else ("< 1e-15" if p < 1e-15 else f"{p:.2g}")


def rq1_lmm_summary_text(res, langs):
    cls, ec, ne = res["classification"], res["ec"], res["ne"]
    con, coef, diag = res["ec_contrasts"], res["ne_coefficients"], res["diagnostics"].set_index("fit_id")
    L = []
    w = L.append
    w("RQ1 linear mixed-effects analyses of saved ΔSLOR (SLOR_token_logprob_diff)")
    w("=" * 78)
    w(f"Models: the {len(cls)} statistical checkpoints (viz_data2.STAT_MODEL_SECTIONS; {int((cls['english_centric']).sum())} "
      f"English-centric, {int((~cls['english_centric']).sum())} non-English-centric incl. Manacá 1B).")
    w("Rows: exactly the failure-rate observations (viz_data2.failure_rate_observations): standard native-vs-"
      "transferred comparisons only (no Japanese control), no-context, identical pairs excluded, side-specific "
      f"preference filter (target rows: Target language preference annotation in {{>, >>}} for {', '.join(langs)}; "
      "en rows: English preference annotation in {>, >>}). Counts equal the N_total of SLOR_failure_rates.xlsx.")
    ident = res["identity"]
    w(f"Sentence-pair identity: source_row_id x comparison side; {ident['items']} items, "
      f"{ident['duplicate_model_item_rows']} duplicated (model, item) rows, {ident['items_with_sentence_variants']} "
      "items with sentence variants across models (no ambiguity).")
    w("Inference: R lme4 (REML, bobyqa). ec_model_language / nec_model_language (paper-primary): contrast-specific "
      "Satterthwaite t (lmerTest::contest1D) for the planned English contrast and for is_focus only; other "
      "coefficients are estimates with SEs (not tested). The tests compare MEAN ΔSLOR; neither tests equality of "
      "whole distributions, and a non-significant difference is not evidence of equality.")
    w("")
    w("Analysis 1 -- English-centric models")
    w("-" * 78)
    w(f"dslor ~ language (ref = en) + (1 | model) + (1 | model_language) + (1 | item); N = {len(ec):,} observations, "
      f"{ec['model'].nunique()} models, {ec['item'].nunique():,} sentence pairs.")
    for spec in ("ec_model_language",):
        sub = con.loc[(con["scope"] == "pooled LMM") & (con["specification"] == spec)]
        if sub.empty:
            continue
        w(f"[{spec}] {sub['description'].iloc[0]}")
        for _i, r in sub.iterrows():
            w(f"  {r['contrast']:<24} {r['estimate']:+.3f} (SE {r['std_error']:.3f}; 95% CI {r['ci_lower']:+.3f} .. "
              f"{r['ci_upper']:+.3f}); {r['inference']}; p = {_fmt_p(r['p_value'])} (planned)")
    pm = res["ec_per_model"]
    low = pm.nsmallest(3, "estimate")
    w(f"Per model (descriptive): {int((pm['estimate'] > 0).sum())} of {len(pm)} models have a positive difference; "
      "smallest: " + ", ".join(f"{l} {e:+.2f}" for l, e in zip(low["label"], low["estimate"])) + ".")
    w("")
    w("Analysis 2 -- non-English-centric models (English rows excluded)")
    w("-" * 78)
    ni = res["ne_identifiability"]
    w("Focus languages (documented training-language metadata, see RQ1_LMM_model_classification.csv): "
      + "; ".join(f"{cls.set_index('model_name').loc[m, 'label']} = {','.join(f)}" for m, f in res["focus"].items()))
    amb = cls.loc[(~cls["english_centric"]) & (cls["ambiguity"] != ""), ["label", "ambiguity"]]
    for _i, r in amb.iterrows():
        w(f"  ambiguity: {r['label']}: {r['ambiguity']}")
    w(f"Identifiability: design [1, language (ref de), is_focus] rank {ni['design_rank']}/{ni['design_columns']}; "
      f"{ni['focus_cells']} focus cells of {ni['model_language_cells']} model x language cells; focus and non-focus "
      f"models both present for {ni['languages_with_focus_and_nonfocus_models']}; is_focus varies within every "
      f"model -> {'identifiable' if ni['identifiable'] else 'NOT identifiable (not fitted)'}.")
    w(f"dslor ~ is_focus + language (ref = de) + (1 | model) + (1 | model_language) + (1 | item); N = {len(ne):,} "
      f"observations, {ne['model'].nunique()} models, {ne['item'].nunique():,} sentence pairs.")
    foc = coef.loc[coef["term"] == "is_focus"]
    for _i, r in foc.iterrows():
        w(f"  [{r['specification']}] is_focus {r['estimate']:+.3f}"
          + (f" (SE {r['std_error']:.3f})" if np.isfinite(r.get("std_error", np.nan)) else "")
          + f"; 95% CI {r['ci_lower']:+.3f} .. {r['ci_upper']:+.3f}"
          + (f"; p = {_fmt_p(r['p_value'])} ({r['inference']})" if np.isfinite(r.get("p_value", np.nan)) else
             f" ({r['inference']})") + f"; N = {int(r['n_obs']):,}, models = {int(r['n_models'])}")
    me = res["ne_model_effects"]
    if len(me):
        w("Per model (focus - mean of other non-English languages, raw | language-adjusted):")
        for _i, r in me.iterrows():
            w(f"  {r['label']:<22} {r['focus_languages']}: raw {r['raw_focus_minus_nonfocus']:+.3f} "
              f"| adjusted {r['language_adjusted_difference']:+.3f}")
    w("")
    w("Diagnostics")
    w("-" * 78)
    for fid, r in diag.iterrows():
        w(f"  {fid}: status {r['status']}, converged {r.get('converged')}, singular {r.get('singular')}; "
          + ", ".join(f"{c.split(':', 1)[1]} {r[c]:.3f}" for c in diag.columns if c.startswith("var:") and
                      np.isfinite(r[c])))
    return "\n".join(L) + "\n"


def export_main_rq1_lmm(df, main_dir, langs, include_identical, manifest):
    written = []
    res = rq1_lmm.run_rq1_lmm(df, langs, include_identical)
    frame = res["frame"]
    workbook = Path(main_dir) / MAIN_FAILURE_TABLE
    if workbook.is_file():
        _rates, _nf, n_total = load_failure_rate_workbook(workbook)
        counts = frame.groupby(["model", "language"]).size()
        not_plotted = sorted(set(frame["model"]) - set(n_total.index))
        if not_plotted:
            print(f"[only_main] RQ1 LMM: {len(not_plotted)} statistical models not in the plotting workbook "
                  f"(same failure_rate_observations selection, not cross-checked): {not_plotted}")
        for (m, l), k in counts.items():
            if m in not_plotted:
                continue
            if int(n_total.loc[m, l]) != int(k):
                raise AssertionError(f"RQ1 LMM rows differ from the failure-rate workbook: {m} {l} {k} vs "
                                     f"{n_total.loc[m, l]}")
    sections = dict(zip(res["classification"]["model_name"], res["classification"]["line_plot_section"]))
    order = {m: i for i, m in enumerate(res["classification"]["model_name"])}
    res["ec_per_model"] = res["ec_per_model"].sort_values("model", key=lambda c: c.map(order)).reset_index(drop=True)
    tables = {
        "english_centric_coefficients.csv": res["ec_coefficients"],
        "english_centric_contrasts.csv": res["ec_contrasts"],
        "non_english_centric_coefficients.csv": res["ne_coefficients"],
        "non_english_centric_model_effects.csv": res["ne_model_effects"],
        "model_classification.csv": res["classification"],
        "diagnostics.csv": res["diagnostics"],
    }
    selection = (f"failure-rate observations of the {STAT_MODEL_COUNT} statistical models (standard comparisons only; side-specific preference filter, "
                 f"{preference_filter_label(langs)}; en rows English preference in {{>, >>}}); saved ΔSLOR")
    ec, ne = res["ec"], res["ne"]
    for name, table in tables.items():
        path = Path(main_dir) / f"{RQ1_LMM_PREFIX}{name}"
        _write_table(written, path, table)
        n = len(ec) if "english_centric" in name and "non" not in name else len(ne) if "non_english" in name else \
            len(frame)
        _manifest_row(manifest, f"RQ1 LMM: {name[:-4].replace('_', ' ')}", path, int(n), selection,
                      ",".join(res["classification"]["model_name"]),
                      notes="see RQ1_LMM_summary.txt for specifications, inference and caveats")
    path = Path(main_dir) / f"{RQ1_LMM_PREFIX}summary.txt"
    path.write_text(rq1_lmm_summary_text(res, langs), encoding="utf-8")
    _record(written, path)
    _manifest_row(manifest, "RQ1 LMM: summary", path, int(len(frame)), selection,
                  ",".join(res["classification"]["model_name"]))
    figures = [
        ("english_centric_contrasts" + MAIN_FIG_EXT, "RQ1 LMM English-centric: planned English - mean(non-English) "
         "contrast (95% CI)",
         lambda p: plot_rq1_ec_contrasts(res["ec_contrasts"], p), len(ec)),
        ("english_centric_model_contrasts" + MAIN_FIG_EXT, "RQ1 LMM English-centric: per-model English - "
         "mean(non-English) difference (descriptive)",
         lambda p: plot_rq1_ec_model_contrasts(res["ec_per_model"], p, sections), len(ec)),
        ("non_english_centric_coefficients" + MAIN_FIG_EXT, "RQ1 LMM non-English-centric: is_focus (95% CI) and "
         "per-model focus differences (descriptive)",
         lambda p: plot_rq1_nec_coefficients(res["ne_coefficients"], res["ne_model_effects"], p), len(ne)),
    ]
    for name, artifact, draw, n in figures:
        path = Path(main_dir) / f"{RQ1_LMM_PREFIX}{name}"
        ok = bool(draw(path)) if not res["ne_coefficients"].empty or "non_english" not in name else False
        _record(written, path, ok)
        _manifest_row(manifest, artifact, path, int(n), selection, ",".join(res["classification"]["model_name"]),
                      notes="values in the RQ1_LMM_*.csv tables; mean differences only (not distributional "
                            "equality)" + ("" if ok else "; FIGURE NOT WRITTEN"))
    bad = res["diagnostics"].loc[res["diagnostics"]["status"] != "ok", "fit_id"].tolist()
    if bad:
        print(f"[warning] RQ1 LMM: fits failed: {bad} (see {RQ1_LMM_PREFIX}diagnostics.csv)")
    sing = res["diagnostics"].loc[res["diagnostics"].get("singular", pd.Series(dtype=str)).astype(str) == "TRUE",
                                  "fit_id"].tolist()
    if sing:
        print(f"[note] RQ1 LMM: singular fits (a variance component at 0): {sing}")
    return written, res


CR_COEF_TABLE = "main_regression_coefficients_cluster_robust.csv"
CR_COEF_FE_TABLE = "main_regression_coefficients_with_fixed_effects_cluster_robust.csv"
CR_DIAG_TABLE = "main_regression_cluster_robust_diagnostics.csv"

#これが論文sec5.2で使われてるtwo way cluster robust t testの関数
def export_main_regression_cluster_robust(res, main_dir, manifest):
    written, rows, diags = [], [], []
    frame, coefs, summary = res["frame"], res["coefficients"], res["summary"]
    for _i, s in summary.iterrows():
        fam = s["family"]
        if s["status"] != "ok":
            diags.append({"family": fam, "status": f"not estimated: {s['status']}"})
            continue
        preds = [t for t in str(s["predictors_included"]).split(" + ") if t]
        r, d = main_inference.cluster_robust_family(frame.loc[frame["family"] == fam], preds,
                                                    coefs.loc[coefs["family"] == fam], fam,
                                                    language_fe=res.get("language_fe", True))
        rows += r
        diags.append({**d, "predictors_included": s["predictors_included"],
                      "n_dropped_missing_predictor": s["n_dropped_missing_predictor"],
                      "predictors_omitted": s["predictors_omitted"]})
    table = pd.DataFrame(rows)
    if len(table):
        labels = dict((f, l) for f, l, _r in REGRESSION_FAMILIES)
        table.insert(1, "family_label", table["family"].map(labels))
        table.insert(3, "label", table["term"].map(lambda t: MAIN_FIG_TERM_LABELS.get(t, TERM_LABELS.get(t, t))))
    diag = pd.DataFrame(diags)
    note = ("original OLS beta / classical SE unchanged; robust: two-way CGM (model x sentence pair = language + "
            "source_row_id; CR1 per dimension, PSD fix), t with min(G) - 1 df, for every coefficient (cr_twoway_*)")
    for name, tab, what in ((CR_COEF_FE_TABLE, table, "all terms incl. language fixed effects"),
                            (CR_COEF_TABLE, table.loc[~table["term"].astype(str).str.startswith(LANGUAGE_FE_PREFIX)]
                             if len(table) else table, "predictors (no language fixed effects)"),
                            (CR_DIAG_TABLE, diag, "diagnostics: clusters, rank, VIF, PSD fix, estimators")):
        path = Path(main_dir) / name
        _write_table(written, path, tab)
        _manifest_row(manifest, f"main regression cluster-robust inference: {what}", path,
                      int(summary["n"].sum()), "same family regressions, rows and predictors as "
                      "main_regression_coefficients*.csv", notes=note)
    return written, table, diag


def _fmt_ci(lo, hi, digits=3):
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]" if np.isfinite(lo) and np.isfinite(hi) else "n/a"


def _fmt_p2(p):
    return "n/a" if p is None or not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def rq1_violin_statistical_summary(res):
    con, coef, cls, ident = res["ec_contrasts"], res["ne_coefficients"], res["classification"], res["identity"]
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    ec_models = set(cls.loc[cls["english_centric"], "model_name"])
    nec_models = set(cls.loc[~cls["english_centric"], "model_name"])
    check(f"English-centric analysis uses all {len(ec_models)} English-centric statistical models",
          set(res["ec"]["model"]) == ec_models and len(ec_models) + len(nec_models) == STAT_MODEL_COUNT,
          f"{len(set(res['ec']['model']))} models")
    check(f"non-English-centric analysis uses all {len(nec_models)} non-English-centric statistical models "
          "(incl. Manacá 1B)",
          set(res["ne"]["model"]) == nec_models and "AIInstituteLNCC_manaca-1b-base" in nec_models,
          f"{len(set(res['ne']['model']))} models")
    check("non-English-centric analysis excludes English rows", "en" not in set(res["ne"]["language"]), "")
    check("no duplicated (model, sentence pair) observation", ident["duplicate_model_item_rows"] == 0, "")
    fr = res["frame"]
    en_src = fr.loc[(fr["comparison"] == "en") & fr["model"].isin(ec_models)]
    en_used = res["ec"].loc[res["ec"]["language"] == "en"]
    check("each English source row enters the English category exactly once (not once per target language)",
          len(en_used) == len(en_src) and not en_used.duplicated(["model", "item"]).any(),
          f"{len(en_used):,} English observations = {len(en_src):,} English source rows of the English-centric "
          "models")
    check("English sentence pairs shared by several origin languages",
          ident["en_pairs_under_several_origins"] == 0,
          f"{ident['en_pairs_under_several_origins']} model x English sentence pairs occur under >= 2 origin "
          f"languages ({ident['en_rows_in_such_pairs']} of {ident['en_rows']} English rows); "
          f"{ident['en_shared_pairs_with_differing_scores']} of them have different saved ΔSLOR")
    bad = 0
    for _i, r in con.loc[con["scope"].str.startswith("pooled LMM")].iterrows():
        df = None if r["inference"] == "Wald z" else float(r["df"])
        _s, p, lo, hi = rq1_lmm.wald(r["estimate"], r["std_error"], df)
        bad += int(not (np.isclose(lo, r["ci_lower"]) and np.isclose(hi, r["ci_upper"]) and np.isclose(p, r["p_value"])))
    for _i, r in coef.loc[coef["inference"].astype(str).str.startswith(("Wald", "Satterthwaite"))].iterrows():
        df = None if r["inference"] == "Wald z" else float(r["df"])
        _s, p, lo, hi = rq1_lmm.wald(r["estimate"], r["std_error"], df)
        bad += int(not (np.isclose(lo, r["ci_lower"]) and np.isclose(hi, r["ci_upper"]) and np.isclose(p, r["p_value"])))
    check("every LMM CI / p-value equals estimate ± crit x SE under its stated inference", bad == 0,
          f"{bad} mismatches")

    rows = []

    def add(analysis, estimand, role, r, p_adj=np.nan, adjustment="", n_items=None):
        rows.append({"analysis": analysis, "estimand": estimand, "role": role, "specification": r["specification"],
                     "estimate": r["estimate"], "std_error": r.get("std_error", np.nan),
                     "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"], "p_value": r.get("p_value", np.nan),
                     "p_adjusted": p_adj, "adjustment": adjustment, "inference": r["inference"],
                     "n_obs": r["n_obs"], "n_models": r["n_models"], "n_items": r.get("n_items", n_items),
                     "source_file": "RQ1_LMM_english_centric_contrasts.csv" if analysis.startswith("A")
                     else "RQ1_LMM_non_english_centric_coefficients.csv"})

    A = "A: English-centric (violin_main_a counterpart)"
    pooled = con.loc[con["scope"] == "pooled LMM"]
    sub = pooled.loc[pooled["contrast"] == rq1_lmm.PLANNED_CONTRAST].set_index("specification")
    if "ec_model_language" in sub.index:
        r = sub.loc["ec_model_language"]
        add(A, rq1_lmm.PLANNED_CONTRAST, "paper-primary (model x language intercepts; Satterthwaite t)",
            {**r, "specification": "ec_model_language"})
    B = "B: non-English-centric (violin_main_b counterpart)"
    foc = coef.loc[coef["term"] == "is_focus"].set_index("specification")
    if "nec_model_language" in foc.index:
        r = foc.loc["nec_model_language"]
        sing = res["diagnostics"].set_index("fit_id").get("singular", pd.Series(dtype=str)).get("nec_model_language")
        role = "paper-primary (model x language intercepts; Satterthwaite t" + ("; singular fit)" if str(sing) == "TRUE"
                                                                                else ")")
        add(B, "focus - other non-English languages (is_focus)", role, {**r, "specification": "nec_model_language"})
    return pd.DataFrame(rows), pd.DataFrame(checks)


def rq1_violin_summary_text(table, checks, res):
    L = ["RQ1 violin analyses: statistical summary (from the existing RQ1_LMM_* results)", "=" * 78,
         "Scope: these LMMs are the inferential counterparts of violin_main_a (English-centric models: English vs "
         "non-English languages) and violin_main_b (non-English-centric models: focus vs other languages). They use "
         f"ALL {STAT_MODEL_COUNT} statistical models (viz_data2.STAT_MODEL_SECTIONS; a superset of the plotted line-plot models) and the failure-rate rows (side-specific > / >> filter), not "
         "only the few models drawn in the violins, so they generalise over a larger model set than the figures "
         "show. They test differences in MEAN ΔSLOR, not whole distributions.", ""]
    for analysis, sub in table.groupby("analysis", sort=False):
        L.append(analysis)
        L.append("-" * 78)
        for _i, r in sub.iterrows():
            if not np.isfinite(r.get("estimate", np.nan)):
                L.append(f"  {r['estimand']}\n      robustness: {r['specification']}: {r['ci_lower']:+.3f} .. "
                         f"{r['ci_upper']:+.3f}")
                continue
            role = r["role"] if r["role"] != "robustness" else f"robustness: {r['specification']}"
            L.append(f"  {r['estimand']}\n      {role}: {r['estimate']:+.3f} {_fmt_ci(r['ci_lower'], r['ci_upper'])}"
                     + (f" p = {_fmt_p2(r['p_value'])}" if np.isfinite(r.get("p_value", np.nan)) else "")
                     + (f", {r['adjustment']} p = {_fmt_p2(r['p_adjusted'])}" if isinstance(r.get("adjustment"), str)
                        and r["adjustment"] else "")
                     + f"; {r['inference']}; N = {int(r['n_obs']):,}, models = {int(r['n_models'])}")
        L.append("")
    L.append("Verification checks")
    L.append("-" * 78)
    for _i, c in checks.iterrows():
        L.append(f"  [{'ok' if c['passed'] else 'ATTENTION'}] {c['check']}" + (f": {c['detail']}" if c["detail"] else ""))
    return "\n".join(L) + "\n"


def export_rq1_violin_summary(res, main_dir, manifest):
    written = []
    table, checks = rq1_violin_statistical_summary(res)
    path = Path(main_dir) / "RQ1_violin_statistical_summary.csv"
    _write_table(written, path, table)
    _manifest_row(manifest, "RQ1 violin A/B statistical summary (table)", path, int(len(res["frame"])),
                  "references RQ1_LMM_english_centric_contrasts.csv / RQ1_LMM_non_english_centric_coefficients.csv",
                  notes="the two paper LMM tests (planned English contrast, is_focus); nothing re-estimated")
    tpath = Path(main_dir) / "RQ1_violin_statistical_summary.txt"
    tpath.write_text(rq1_violin_summary_text(table, checks, res), encoding="utf-8")
    _record(written, tpath)
    _manifest_row(manifest, "RQ1 violin A/B statistical summary (text, incl. verification checks)", tpath,
                  int(len(res["frame"])), "as RQ1_violin_statistical_summary.csv")
    return written, table, checks


def main_statistical_inference_text(spearman_tables, reg_table, reg_diag, rq1_table, rq1_checks, rq1_res):
    L = ["Main-figure statistical inference summary", "=" * 78, ""]
    L += ["1. Spearman correlations between models (per-item ΔSLOR, target-language comparison; descriptive)",
          "-" * 78]
    for name, long in spearman_tables.items():
        ok = long.loc[long["status"] == "ok"]
        if ok.empty:
            continue
        L.append(f"{name}: {len(ok)} defined model pairs; rho median {ok['rho'].median():.2f} (range "
                 f"{ok['rho'].min():.2f} .. {ok['rho'].max():.2f}); shared sentence pairs {int(ok['n_shared_pairs'].min())}"
                 f"..{int(ok['n_shared_pairs'].max())}.")
        weak = ok.loc[ok["rho"].abs() < 0.3]
        if len(weak):
            L.append(f"  weak correlations (|rho| < 0.3): {len(weak)} pairs.")
    L += ["", "Method (paper): We report Spearman's ρ over the sentence pairs scored by both models as a descriptive "
          "measure of shared item-level preferences; they do not identify a causal source.", ""]
    L += ["2. Family-specific multivariable regressions: cluster-robust inference", "-" * 78]
    for _i, d in reg_diag.iterrows():
        L.append(f"{d['family']}: N = {int(d.get('n', 0) or 0):,}, model clusters = {d.get('n_model_clusters')}, "
                 f"sentence-pair clusters = {d.get('n_sentence_pair_clusters')}, max VIF = "
                 f"{d.get('max_vif', np.nan):.2f}" + (f"; {d['two_way_psd_fix']}" if isinstance(d.get("two_way_psd_fix"), str)
                                                      and d["two_way_psd_fix"] else ""))
    L.append("")
    if len(reg_table):
        t = reg_table.loc[~reg_table["term"].astype(str).str.startswith(LANGUAGE_FE_PREFIX)
                          & (reg_table["term"] != "intercept")]
        for _i, r in t.iterrows():
            L.append(f"  {r['family']:<8} {r['label'][:38]:<38} beta {r['beta']:+.3f}; classical p {_fmt_p2(r['classical_p_value'])}; "
                     f"robust: two-way CGM CR1 (model x sentence pair), t({r['cr_twoway_df']}): "
                     + (f"{_fmt_ci(r['cr_twoway_ci_lower'], r['cr_twoway_ci_upper'])} " if np.isfinite(
                         r["cr_twoway_ci_lower"]) else "")
                     + f"p {_fmt_p2(r['cr_twoway_p_value'])}"
                     + ("" if r["inference_flag"] == "ok" else f"  [{r['inference_flag']}]"))
    L += ["", "Method (paper): Standard errors are clustered two-way by model and sentence pair (Cameron, Gelbach "
          "and Miller, 2011; CR1 small-sample factors, t with G - 1 degrees of freedom for the smaller dimension; "
          "negative eigenvalues of the two-way covariance set to zero).", ""]
    L += ["3. RQ1 violin analyses (LMM)", "-" * 78]
    for _i, r in rq1_table.iterrows():
        if not np.isfinite(r.get("estimate", np.nan)):
            continue
        L.append(f"  {r['analysis'][:2]} {r['estimand']} | {r['role']}: {r['estimate']:+.3f} "
                 f"{_fmt_ci(r['ci_lower'], r['ci_upper'])}" + (f" p {_fmt_p2(r['p_value'])}" if np.isfinite(r.get("p_value", np.nan)) else ""))
    att = rq1_checks.loc[~rq1_checks["passed"]]
    for _i, c in att.iterrows():
        L.append(f"  ATTENTION: {c['check']}: {c['detail']}")
    a = rq1_table.loc[rq1_table["estimand"] == rq1_lmm.PLANNED_CONTRAST]
    b = rq1_table.loc[rq1_table["estimand"].str.startswith("focus")]
    L += ["", "Results (paper):"]
    pm = rq1_res["ec_per_model"]
    me = rq1_res["ne_model_effects"]
    no_adv = me.loc[me["language_adjusted_difference"] <= 0, "label"].tolist() if len(me) else []
    if len(a):
        a = a.iloc[0]
        L.append(f"  English-centric models assign a higher mean ΔSLOR to native strategies in English than in the "
                 f"six other languages (difference {a['estimate']:.2f}, 95% CI {_fmt_ci(a['ci_lower'], a['ci_upper'], 2)}, "
                 f"{a['inference']}; {int(a['n_models'])} models, {int(a['n_obs']):,} observations). The per-model "
                 f"difference is positive for {int((pm['estimate'] > 0).sum())} of {len(pm)} models (descriptive).")
    if len(b):
        b = b.iloc[0]
        L.append(f"  Non-English-centric models show a higher mean ΔSLOR in their focus language than in other "
                 f"non-English languages, controlling for language (β = {b['estimate']:.2f}, 95% CI "
                 f"{_fmt_ci(b['ci_lower'], b['ci_upper'], 2)}, {b['inference']}; "
                 f"{int(b['n_models'])} models). With {int(b['n_models'])} models this generalises only cautiously across "
                 "model families"
                 + (f"; no language-adjusted focus advantage for {', '.join(no_adv)}." if no_adv else "."))
    L += ["", "Caveats", "-" * 78,
          "- Significance vs effect size: with thousands of sentence pairs, very small effects become significant; "
          "read the estimates and CIs, not only p.",
          "- Mean vs distribution: all tests concern mean ΔSLOR; equal means do not imply equal distributions and "
          "non-significant differences are not evidence of equality.",
          "- Correlation vs causation: regression coefficients and Spearman correlations are associations.",
          f"- English rows: {rq1_res['identity']['en_pairs_under_several_origins']} model x English sentence pairs "
          "occur under several origin languages in the failure-rate selection (counted once per origin there); "
          f"{rq1_res['identity']['en_shared_pairs_with_differing_scores']} of them have different saved scores for "
          "identical sentences."]
    return "\n".join(L) + "\n"


def add_companion_manifest_rows(manifest, written):
    listed = {str(r["path"]) for r in manifest}
    by_path = {str(r["path"]): r for r in manifest}
    for path in written:
        path = Path(path)
        if str(path) in listed or path.suffix == FIG_EXT:
            continue
        stem = path.name
        for tag in ("_summary.csv", "_exclusions.csv", ".csv.gz", ".csv", ".xlsx"):
            if stem.endswith(tag):
                stem = stem[: -len(tag)]
                break
        figure = path.with_name(stem + MAIN_FIG_EXT)
        parent = by_path.get(str(figure))
        kind = ("per-category summary (N, mean, median, ...)" if path.name.endswith("_summary.csv") else
                "exclusion accounting" if path.name.endswith("_exclusions.csv") else "data table")
        manifest.append({"artifact": f"table: {kind}" + (f" of {figure.name}" if parent else ""), "path": str(path),
                         "n_rows": parent["n_rows"] if parent else "",
                         "selection": parent["selection"] if parent else "see the figure / analysis of the same stem",
                         "models_requested": parent["models_requested"] if parent else "",
                         "models_missing": parent["models_missing"] if parent else "",
                         "notes": "companion table" + (f" of {figure.name}" if parent else "")})
        listed.add(str(path))


def export_only_main(outputs_dir, plots_dir, preference_languages, metadata_path, include_identical=False,
                     exclude_freq=True, main_dir=None, main_plot_dir=None, glotcc=False):
    outputs_dir, plots_dir = _checked_dirs(outputs_dir, plots_dir)
    langs = parse_preference_languages(preference_languages)
    if not langs:
        raise ValueError("--only_main requires --use-preference-annotation LANG ...")
    metadata = load_metadata_table(metadata_path) if metadata_path else pd.DataFrame()
    if metadata.empty:
        raise FileNotFoundError(f"--only_main: metadata workbook not found or empty: {metadata_path}")
    for col in ("divergence_tl_realization_score", "other_language_strategy_divergence_tl_realization_score"):
        if col not in metadata.columns:
            raise ValueError(f"--only_main: metadata workbook {metadata_path} has no column {col!r}; rebuild it with "
                             "src/metadata/build_metadata.py (Other-Language Strategy Divergence)")
    print(f"[only_main] languages: {', '.join(langs)}; metadata: {metadata_path} ({len(metadata)} rows)")
    print(f"[only_main] figure font: {PAPER_FONT}"
          + ("" if PAPER_FONT_IS_TIMES else " (fallback: no Times-compatible font installed)"))
    df, warnings = load_all_data2(outputs_dir)
    for w in warnings:
        print(f"[warning] {w}")
    if df.empty:
        raise ValueError(f"no probability outputs found in {outputs_dir}")
    custom_main_dir = main_dir is not None
    main_dir = _main_dir(plots_dir, main_dir)
    dist_dir = (main_dir if custom_main_dir else Path(plots_dir)) / SPEARMANN_BOOTSTRAP_DIR
    main_plot_dir = Path(main_plot_dir) if main_plot_dir else default_main_plot_dir(
        plots_dir, main_dir if custom_main_dir else None)
    main_plot_dir.mkdir(parents=True, exist_ok=True)
    ja_workbook = (main_dir if custom_main_dir else Path(plots_dir)) / ja_control_lmm.OUTPUT_FILE
    print(f"[only_main] output directory: {main_dir.resolve()}; paper figures: {main_plot_dir.resolve()}")
    _DELEGATED_DIRS.append(main_dir.resolve())
    try:
        return _export_only_main(df, outputs_dir, plots_dir, langs, metadata, include_identical, exclude_freq,
                                 main_dir, dist_dir, main_plot_dir, ja_workbook, glotcc=glotcc)
    finally:
        _DELEGATED_DIRS.pop()


def _export_only_main(df, outputs_dir, plots_dir, langs, metadata, include_identical, exclude_freq, main_dir,
                      dist_dir, main_plot_dir, ja_workbook, glotcc=False):
    manifest, written = [], []
    w, _res = export_main_family_regressions(df, main_dir, langs, metadata, include_identical, manifest,
                                             exclude_freq=exclude_freq, glotcc=glotcc)
    written += w
    written += export_main_failure_rates(df, main_dir, langs, include_identical, manifest)
    written += export_main_failure_rate_bars(main_dir, manifest)
    written += export_main_translation_preference(outputs_dir, main_dir, langs, manifest)
    written += export_main_violins(df, main_dir, langs, include_identical, manifest)
    written += export_main_spearman(df, main_dir, langs, include_identical, manifest)
    written += export_main_spearman(df, main_dir, langs, include_identical, manifest, MAIN_SPEARMAN_MODELS_MANACA,
                                    MAIN_SPEARMAN_MANACA_STEM, "Spearman correlation (Manacá 1B instead of Lucie 7B)")
    written += export_main_consistency_all(df, main_dir, langs, include_identical, manifest)
    written += export_main_spearman_groups(df, main_dir, langs, include_identical, manifest, dist_dir=dist_dir)
    written += export_main_rq1_violins(df, main_dir, include_identical, manifest)
    w, rq1_res = export_main_rq1_lmm(df, main_dir, langs, include_identical, manifest)
    written += w
    w, reg_table, reg_diag = export_main_regression_cluster_robust(_res, main_dir, manifest)
    written += w
    w, rq1_table, rq1_checks = export_rq1_violin_summary(rq1_res, main_dir, manifest)
    written += w
    spearman_tables = {label: pd.read_csv(main_dir / f"{stem}_long.csv") for label, stem in (
        (f"Spearman ALL ({STAT_MODEL_COUNT} statistical models)", SPEARMAN_ALL_STEM),
        ("Spearman selected models", Path(MAIN_SPEARMAN_FIGURE).stem),
        ("Spearman selected models, Manacá 1B instead of Lucie 7B", MAIN_SPEARMAN_MANACA_STEM))
        if (main_dir / f"{stem}_long.csv").is_file()}
    path = main_dir / "main_statistical_inference_summary.txt"
    path.write_text(main_statistical_inference_text(spearman_tables, reg_table, reg_diag, rq1_table, rq1_checks,
                                                    rq1_res), encoding="utf-8")
    _record(written, path)
    _manifest_row(manifest, "consolidated statistical summary (Spearman descriptive, regression CGM, RQ1 LMMs)", path,
                  "", "see the referenced CSVs")
    written += export_main_japanese_control(df, main_dir, include_identical, manifest)
    written += export_main_model_size_win_rate(df, main_dir, langs, include_identical, manifest)
    w, _ja = export_main_ja_control_lmm(df, ja_workbook, include_identical, manifest)
    written += w
    w, produced = export_paper_figures(df, main_dir, main_plot_dir, langs, include_identical, manifest)
    written += w
    unrendered = [p for p in written if Path(p).name in PAPER_FIGURE_SCRIPTS and Path(p).parent.resolve()
                  == main_dir.resolve() and Path(p).name not in produced]
    if unrendered:
        raise AssertionError(f"delegated paper figures not produced by their scripts: {unrendered}")
    add_companion_manifest_rows(manifest, written)
    _write_table(written, main_dir / "main_fig_manifest.csv", pd.DataFrame(manifest))
    allowed = {(dist_dir / SPEARMANN_BOOTSTRAP_FILE).resolve(), Path(ja_workbook).resolve()}
    outside = [p for p in written if main_dir.resolve() not in Path(p).resolve().parents
               and Path(p).resolve() not in allowed and Path(main_plot_dir).resolve() not in Path(p).resolve().parents]
    if outside:
        raise AssertionError(f"--only_main wrote outside {main_dir}: {outside}")
    return written


DEFAULT_METADATA = "./data/20261011construction_data_expansion_metadata_no_freq.xlsx"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export publication-ready figures/tables from outputs2 results.")
    parser.add_argument("--outputs_dir", type=str, default="./outputs2")
    parser.add_argument("--plots_dir", type=str, default="./plots2")
    parser.add_argument(
        "--include_identical",
        action="store_true",
        help="Include identical pos_sentence/neg_sentence pairs (excluded by default, as in app2).",
    )
    parser.add_argument(
        "--use-preference-annotation", dest="use_preference_annotation", nargs="+", default=None, metavar="LANG",
        help="Target languages (e.g. de sv ja pt; 'se' = 'sv') whose target-language analyses keep only rows with "
             "'Target language preference annotation' in {>, >>}. Also selects the main-regression languages.",
    )
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA,
                        help="Metadata workbook joined into the main regression (never rebuilt here).")
    parser.add_argument("--run-regression", action="store_true",
                        help="Also run the two regression analyses (main ΔSLOR regression and gloss-feature ΔSLOR "
                             "regressions) during the full export. Off by default.")
    parser.add_argument("--regression-only", action="store_true",
                        help="Run ONLY the two regression analyses (implies --run-regression): remove stale regression "
                             "artifacts, write regression figures/tables, and regenerate nothing else. Requires "
                             "--use-preference-annotation.")
    parser.add_argument("--exclude_freq", action="store_true",
                        help="Fit the main regression without the corpus-frequency predictor (frequency_contrast); "
                             "missing frequency values then never remove observations. Gloss-feature regressions "
                             "are unchanged.")
    parser.add_argument("--main_dir", type=str, default=None,
                        help="With --only_main: write the paper figures/tables directly to this directory (created "
                             "if missing; the EC–EC vs EC–NEC workbook goes to <main_dir>/dist/) instead of "
                             "<plots_dir>/main_fig/")
    parser.add_argument("--only_main", action="store_true",
                        help="Write ONLY the paper figures/tables to <plots_dir>/main_fig/ (three family-specific "
                             "ΔSLOR regressions, side-specific SLOR failure rates, merged multilingual/monolingual "
                             "violins, selected-model Spearman heatmap, Japanese-control paired-baseline violins, "
                             "model size vs accuracy). Requires --use-preference-annotation; corpus "
                             "frequency is excluded unless --include_freq.")
    parser.add_argument("--include_freq", action="store_true",
                        help="With --only_main: keep the corpus-frequency predictor in the family regressions.")
    parser.add_argument("--glotcc", action="store_true",
                        help="Replace the language fixed effects of the pooled and family-specific ΔSLOR regressions "
                             "with the predictor log10(Common Crawl size by language) from data/lang2size.json. "
                             "Requires --only_main, --run-regression or --regression-only.")
    parser.add_argument("--main_plot_dir", type=str, default=None,
                        help="With --only_main: directory of the standalone paper figures (src/Fig2_4.py, Fig3.py, "
                             "Fig5.py, Fig6.py); default Main_plot/ next to --plots_dir (<main_dir>/Main_plot with "
                             "--main_dir)")
    args = parser.parse_args(argv)
    if args.only_main:
        if args.regression_only or args.run_regression:
            parser.error("--only_main cannot be combined with --regression-only / --run-regression")
        if not args.use_preference_annotation:
            parser.error("--only_main requires --use-preference-annotation LANG ...")
        if args.include_freq and args.exclude_freq:
            parser.error("--include_freq and --exclude_freq are mutually exclusive")
    elif args.include_freq:
        parser.error("--include_freq is only valid with --only_main")
    elif args.main_plot_dir:
        parser.error("--main_plot_dir is only valid with --only_main")
    if args.regression_only:
        args.run_regression = True
        if not args.use_preference_annotation:
            parser.error("--regression-only requires --use-preference-annotation LANG ...")
    if args.glotcc and not (args.only_main or args.run_regression):
        parser.error("--glotcc requires --only_main, --run-regression or --regression-only")
    return args


def remove_obsolete_pngs(written):
    removed = []
    for p in written:
        p = Path(p)
        if p.suffix != FIG_EXT:
            continue
        old = p.with_suffix(".png")
        if old.is_file():
            old.unlink()
            removed.append(old)
    if removed:
        print(f"Removed {len(removed)} obsolete PNG figure(s) superseded by their PDF versions")
    return removed


def main(argv=None):
    args = parse_args(argv)
    if args.only_main:
        written = export_only_main(args.outputs_dir, args.plots_dir, args.use_preference_annotation, args.metadata,
                                   include_identical=args.include_identical, exclude_freq=not args.include_freq,
                                   main_dir=args.main_dir, main_plot_dir=args.main_plot_dir, glotcc=args.glotcc)
        remove_obsolete_pngs(written)
        print(f"Generated {len(written)} main-figure artifacts in "
              f"{Path(args.main_dir or Path(args.plots_dir) / MAIN_FIG_DIR).resolve()}")
        return written
    if args.regression_only:
        written = export_regressions_only(args.outputs_dir, args.plots_dir, include_identical=args.include_identical,
                                          preference_languages=args.use_preference_annotation,
                                          metadata_path=args.metadata, exclude_freq=args.exclude_freq,
                                          glotcc=args.glotcc)
        remove_obsolete_pngs(written)
        n_pdf = sum(1 for p in written if p.suffix == FIG_EXT)
        n_csv = sum(1 for p in written if p.suffix == ".csv")
        print(f"Generated {len(written)} regression artifacts ({n_pdf} PDF figures, {n_csv} CSV) in {args.plots_dir}")
        return written
    written = export_all(args.outputs_dir, args.plots_dir, include_identical=args.include_identical,
                         preference_languages=args.use_preference_annotation, metadata_path=args.metadata,
                         run_regression=args.run_regression, exclude_freq=args.exclude_freq, glotcc=args.glotcc)
    remove_obsolete_pngs(written)
    n_pdf = sum(1 for p in written if p.suffix == FIG_EXT)
    n_csv = sum(1 for p in written if p.suffix == ".csv")
    n_xlsx = sum(1 for p in written if p.suffix == ".xlsx")
    print(f"Generated {len(written)} artifacts ({n_pdf} PDF figures, {n_csv} CSV, {n_xlsx} XLSX) in {args.plots_dir}")


if __name__ == "__main__":
    main()
