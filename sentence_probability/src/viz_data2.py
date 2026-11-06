import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from data_loading import (
    EXPERIMENT_JAPANESE_CONTROL,
    EXPERIMENT_STANDARD,
    TARGET_LANGUAGE_CANONICAL_COLUMN,
    canonical_language_code,
    comparison_experiment,
)
from model_metadata2 import METADATA_FIELDS, lookup_model, model_metadata_record, unresolved_models


MEANLP_COL = "token_logprob_diff"
MEANLP_LABEL = "ΔmeanLP"
MEANLP_LONG_LABEL = "ΔmeanLP (meanLP difference)"
MEANLP_AXIS_LABEL = "ΔmeanLP = meanLP(positive) − meanLP(negative)"

SLOR_COL = "SLOR_token_logprob_diff"
SLOR_LABEL = "ΔSLOR"
SLOR_LONG_LABEL = "ΔSLOR (SLOR difference)"
SLOR_AXIS_LABEL = "ΔSLOR = SLOR(positive) − SLOR(negative)"

METRIC_VIEW_OPTIONS = [
    (MEANLP_COL, MEANLP_LONG_LABEL),
    (SLOR_COL, SLOR_LONG_LABEL),
    ("diff_comparison", "Diff-token vs after-diff comparison"),
    ("pairwise_signed_choice_confidence", "Pairwise signed choice confidence"),
]

METRIC_Y_LABELS = {
    MEANLP_COL: MEANLP_AXIS_LABEL,
    SLOR_COL: SLOR_AXIS_LABEL,
    "pairwise_signed_choice_confidence": "Pairwise signed choice confidence",
    "only_diff_tokens_tokenwise_log_prob_diff": "Diff-token log probability difference (positive - negative)",
    "after_diff_tokens_tokenwise_log_prob_diff": "After-diff log probability difference (positive - negative)",
}

DIFF_METRIC_COLUMNS = {
    "only_diff": ("only_diff_tokens_tokenwise_log_prob_diff", "Diff-token"),
    "after_diff": ("after_diff_tokens_tokenwise_log_prob_diff", "After-diff"),
}

METRIC_SLUGS = {MEANLP_COL: "delta-meanLP", SLOR_COL: "delta-SLOR"}

DELTA_MEANLP = "delta_meanLP"
DELTA_SLOR = "delta_SLOR"
SCORE_METRICS = {
    MEANLP_COL: {"delta": DELTA_MEANLP, "label": MEANLP_LABEL, "slug": "delta-meanLP", "axis": MEANLP_AXIS_LABEL,
                 "table": "meanLP"},
    SLOR_COL: {"delta": DELTA_SLOR, "label": SLOR_LABEL, "slug": "delta-SLOR", "axis": SLOR_AXIS_LABEL,
               "table": "SLOR"},
}


def score_metric(metric):
    try:
        return SCORE_METRICS[metric]
    except KeyError:
        raise ValueError(f"Unknown score metric: {metric!r}") from None


def metric_slug(column):
    return METRIC_SLUGS.get(column, column)


def metric_display_name(column):
    return dict(METRIC_VIEW_OPTIONS).get(column) or METRIC_Y_LABELS.get(column, column)


VIOLIN_METRICS = [
    MEANLP_COL,
    SLOR_COL,
    "pairwise_signed_choice_confidence",
    "only_diff_tokens_tokenwise_log_prob_diff",
    "after_diff_tokens_tokenwise_log_prob_diff",
]


GROUP_COL = "Group"
ALL_GROUPS = "__all_groups__"
ALL_GROUPS_LABEL = "All groups"

EXPERIMENT_COL = "experiment"
EXPERIMENTS = [
    (EXPERIMENT_STANDARD, "Standard comparisons"),
    (EXPERIMENT_JAPANESE_CONTROL, "Japanese control"),
]
EXPERIMENT_LABELS = dict(EXPERIMENTS)
TARGET_LANGUAGE_SOURCE_COL = "target_language_source"


def experiment_title(experiment):
    if experiment == EXPERIMENT_JAPANESE_CONTROL:
        return " — Japanese control (Japanese sentences; x = original target language / strategy origin)"
    return ""


XAXIS_MODES = {
    "language": ("Target language", "Target language"),
    "model": ("model_name", "Model"),
}

GROUP_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

TRANSLATION_GENERATION_DETAIL_SUFFIX = "__translation_generation.xlsx"
TRANSLATION_GENERATION_SUMMARY_SUFFIX = "__translation_generation_summary.xlsx"

TRANSLATION_MATCH_LABELS = ["proper", "literal", "both", "neither"]

TRANSLATION_FILTER_DIMENSIONS = [
    ("model_name", "Model"),
    ("Target language", "Target language"),
    ("Subject", "Subject"),
]

FILTER_DIMENSIONS = [
    ("model_name", "Model"),
    ("lang", "Lang"),
    ("Target language", "Target language"),
    ("pos", "Positive column (pos)"),
    ("neg", "Negative column (neg)"),
    ("Subgroup", "Subgroup"),
    ("Subject", "Subject"),
    ("English preference annotation", "English preference annotation"),
    ("Target language preference annotation", "Target language preference annotation"),
    ("Japanese preference annotation", "Japanese preference annotation"),
    ("Temporal interpretation", "Temporal interpretation"),
]

PAGE_SIZE = 50

PREFERENCE_COL = "P_proper_given_proper_or_literal"


def _read_xlsx_files(files):
    frames, warnings = [], []
    for f in files:
        try:
            frames.append(pd.read_excel(f, keep_default_na=False, na_values=[]))
        except Exception as e:
            warnings.append(f"Failed to read {f}: {e}")
    return frames, warnings


def normalize_loaded_frame2(df):
    if df.empty:
        return df
    out = df.copy()
    derived = (
        pd.Series([comparison_experiment(p, n) for p, n in zip(out["pos"], out["neg"])], index=out.index)
        if {"pos", "neg"} <= set(out.columns)
        else pd.Series(EXPERIMENT_STANDARD, index=out.index)
    )
    if EXPERIMENT_COL in out.columns:
        saved = out[EXPERIMENT_COL]
        blank = saved.isna() | (saved.astype(str).str.strip() == "")
        out[EXPERIMENT_COL] = saved.where(~blank, derived)
    else:
        out[EXPERIMENT_COL] = derived

    if "Target language" in out.columns:
        if TARGET_LANGUAGE_SOURCE_COL not in out.columns:
            out[TARGET_LANGUAGE_SOURCE_COL] = out["Target language"]
        canonical = out["Target language"].map(canonical_language_code)
        if TARGET_LANGUAGE_CANONICAL_COLUMN in out.columns:
            saved = out[TARGET_LANGUAGE_CANONICAL_COLUMN]
            blank = saved.isna() | (saved.astype(str).str.strip() == "")
            canonical = saved.where(~blank, canonical)
        out["Target language"] = canonical
    if "lang" in out.columns:
        out["lang"] = out["lang"].map(canonical_language_code)
    return out


def _add_pair_id(df, warnings):
    if "Proper English strategy" in df.columns:
        df["_pair_id"] = df["Proper English strategy"]
    else:
        df["_pair_id"] = df.index.astype(str)
        warnings.append("'Proper English strategy' column missing; falling back to row index as pair id.")
    return df


def load_all_data2(outputs_dir):
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.exists():
        return pd.DataFrame(), [f"outputs2 directory does not exist: {outputs_dir}"]

    files = sorted(
        f for f in outputs_dir.glob("*/*.xlsx")
        if not f.name.endswith(TRANSLATION_GENERATION_DETAIL_SUFFIX)
        and not f.name.endswith(TRANSLATION_GENERATION_SUMMARY_SUFFIX)
    )
    files, superseded = _drop_superseded_japanese_control_files(files)
    if not files:
        return pd.DataFrame(), [f"No result xlsx files found under {outputs_dir} (*/*.xlsx)"]

    frames, warnings = _read_xlsx_files(files)
    warnings += [f"Ignored {f} (superseded by its japanese_control-tagged file)" for f in superseded]
    if not frames:
        return pd.DataFrame(), warnings

    df = normalize_loaded_frame2(pd.concat(frames, ignore_index=True))
    return _add_pair_id(df, warnings), warnings


JAPANESE_CONTROL_FILE_TAG = "__japanese_control__"


def _drop_superseded_japanese_control_files(files):
    tagged = {f.parent / f.name.replace(JAPANESE_CONTROL_FILE_TAG, "__", 1) for f in files if JAPANESE_CONTROL_FILE_TAG in f.name}
    kept = [f for f in files if f not in tagged]
    return kept, [f for f in files if f in tagged]


def load_all_translation_data2(outputs_dir):
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.exists():
        return pd.DataFrame(), [f"outputs2 directory does not exist: {outputs_dir}"]

    files = sorted(f for f in outputs_dir.glob("*/*.xlsx") if f.name.endswith(TRANSLATION_GENERATION_DETAIL_SUFFIX))
    if not files:
        return pd.DataFrame(), []

    frames, warnings = _read_xlsx_files(files)
    if not frames:
        return pd.DataFrame(), warnings

    df = normalize_loaded_frame2(pd.concat(frames, ignore_index=True))
    return _add_pair_id(df, warnings), warnings


def _sorted_choices(df, col):
    return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []


def group_choices(*dfs):
    values = set()
    for df in dfs:
        if df is not None and GROUP_COL in df.columns:
            values.update(str(v) for v in df[GROUP_COL].dropna().unique() if str(v) != "")
    return sorted(values)


def group_options(*dfs):
    return [{"label": ALL_GROUPS_LABEL, "value": ALL_GROUPS}] + [{"label": g, "value": g} for g in group_choices(*dfs)]


def filter_by_group(df, group):
    if group in (None, ALL_GROUPS):
        return df
    if GROUP_COL not in df.columns:
        return df.iloc[0:0]
    return df.loc[df[GROUP_COL].astype(str) == str(group)]


def groups_to_display(df, group):
    if group in (None, ALL_GROUPS):
        return group_choices(df)
    return [group]


def group_color(group, all_groups):
    all_groups = list(all_groups)
    idx = all_groups.index(group) if group in all_groups else len(all_groups)
    return GROUP_PALETTE[idx % len(GROUP_PALETTE)]


PREFERENCE_ANNOTATION_BY_POS = {
    "Proper English strategy": "English preference annotation",
    "Proper strategy in target language": "Target language preference annotation",
    "Proper strategy in Japanese": "Japanese preference annotation",
}

PREFERENCE_ALL = "all"
PREFERENCE_SUBSETS = [
    (PREFERENCE_ALL, "All", None),
    ("gt", "> only", frozenset({">"})),
    ("gtgt", ">> only", frozenset({">>"})),
    ("gt-or-gtgt", "> or >>", frozenset({">", ">>"})),
]
PREFERENCE_SUBSET_LABELS = {value: label for value, label, _ in PREFERENCE_SUBSETS}
_PREFERENCE_SUBSET_VALUES = {value: accepted for value, _, accepted in PREFERENCE_SUBSETS}


def preference_annotation_column(pos_column):
    try:
        return PREFERENCE_ANNOTATION_BY_POS[pos_column]
    except KeyError:
        raise ValueError(f"No preference annotation column defined for comparison pos={pos_column!r}") from None


def comparison_preference_annotation(df):
    out = pd.Series("", index=df.index, dtype=object)
    if "pos" not in df.columns:
        return out
    for pos_column, annotation_col in PREFERENCE_ANNOTATION_BY_POS.items():
        mask = df["pos"] == pos_column
        if mask.any() and annotation_col in df.columns:
            values = df.loc[mask, annotation_col]
            out.loc[mask] = values.where(values.notna(), "").astype(str).str.strip()
    return out


def filter_by_preference(df, subset):
    if subset in (None, PREFERENCE_ALL):
        return df
    if subset not in _PREFERENCE_SUBSET_VALUES:
        raise ValueError(f"Unknown preference subset: {subset!r}")
    accepted = _PREFERENCE_SUBSET_VALUES[subset]
    return df.loc[comparison_preference_annotation(df).isin(accepted)]


TARGET_COMPARISON_POS = "Proper strategy in target language"
ENGLISH_COMPARISON_POS = "Proper English strategy"
JAPANESE_COMPARISON_POS = "Proper strategy in Japanese"
TARGET_PREFERENCE_COL = PREFERENCE_ANNOTATION_BY_POS[TARGET_COMPARISON_POS]
STRONG_PREFERENCE_VALUES = frozenset({">", ">>"})


def parse_preference_languages(values):
    out = []
    for v in values or []:
        code = canonical_language_code(str(v).strip())
        if code and code not in out:
            out.append(code)
    return tuple(out)


def preference_filter_label(languages):
    langs = parse_preference_languages(languages)
    if not langs:
        return "none"
    return f"Target language preference annotation in {{>, >>}} for {', '.join(langs)}"


def _stripped(series):
    return series.where(series.notna(), "").astype(str).str.strip()


def target_preference_mask(df, languages):
    langs = set(parse_preference_languages(languages))
    keep = pd.Series(True, index=df.index)
    if not langs or df.empty or "Target language" not in df.columns:
        return keep
    is_target = df["pos"] == TARGET_COMPARISON_POS if "pos" in df.columns else pd.Series(True, index=df.index)
    affected = is_target & df["Target language"].map(canonical_language_code).isin(langs)
    if TARGET_PREFERENCE_COL in df.columns:
        strong = _stripped(df[TARGET_PREFERENCE_COL]).isin(STRONG_PREFERENCE_VALUES)
    else:
        strong = pd.Series(False, index=df.index)
    return keep & (~affected | strong)


def filter_target_preference(df, languages):
    if not parse_preference_languages(languages):
        return df
    return df.loc[target_preference_mask(df, languages)]


def side_preference_annotation(frame):
    out = pd.Series("", index=frame.index, dtype=object)
    if frame.empty:
        return out
    is_en = (frame[TARGET_WITH_EN_SOURCE_COL] == ENGLISH_COMPARISON_POS if TARGET_WITH_EN_SOURCE_COL in frame.columns
             else frame["pos"] == ENGLISH_COMPARISON_POS)
    for mask, col in ((is_en, PREFERENCE_ANNOTATION_BY_POS[ENGLISH_COMPARISON_POS]), (~is_en, TARGET_PREFERENCE_COL)):
        if mask.any() and col in frame.columns:
            out.loc[mask] = _stripped(frame.loc[mask, col])
    return out


def filter_side_preference(frame, accepted):
    accepted = {accepted} if isinstance(accepted, str) else set(accepted)
    return frame.loc[side_preference_annotation(frame).isin(accepted)]


FAMILY_SCOPE_ALL = "all_models"
FAMILY_SCOPES = [
    (FAMILY_SCOPE_ALL, "All models", None),
    ("aya", "Aya", frozenset({"Aya Expanse", "Tiny Aya"})),
    ("eurollm", "EuroLLM", frozenset({"EuroLLM"})),
    ("gemma", "Gemma", frozenset({"Gemma 4"})),
]
FAMILY_SCOPE_LABELS = {slug: label for slug, label, _ in FAMILY_SCOPES}
_FAMILY_SCOPE_MEMBERS = {slug: families for slug, _, families in FAMILY_SCOPES}


def family_scope_of(family):
    for slug, _label, families in FAMILY_SCOPES:
        if families is not None and family in families:
            return slug
    return None


def filter_by_family_scope(df, scope):
    if scope not in _FAMILY_SCOPE_MEMBERS:
        raise ValueError(f"Unknown family scope: {scope!r}")
    families = _FAMILY_SCOPE_MEMBERS[scope]
    if families is None or df.empty:
        return df
    return df.loc[_model_property(df, "family").isin(families)]


def models_in(df):
    return sorted(df["model_name"].dropna().astype(str).unique()) if "model_name" in df.columns else []


SCOPE_KIND_ALL = "all"
SCOPE_KIND_PICKED = "picked"
SCOPE_KIND_FAMILY = "family"
SCOPE_KIND_MODEL = "model"

PICKED_MODELS_SCOPE = "picked_models"
PICKED_MODELS_LABEL = "Picked models"
PICKED_MODELS = (
    "CohereLabs/aya-expanse-32b",
    "google/gemma-4-31B",
    "google/gemma-4-31B-it",
    "utter-project/EuroLLM-22B-2512",
    "utter-project/EuroLLM-22B-Instruct-2512",
)
_PICKED_MODEL_NAMES = frozenset(PICKED_MODELS) | frozenset(m.replace("/", "_", 1) for m in PICKED_MODELS)


def is_picked_model(model_name):
    return str(model_name) in _PICKED_MODEL_NAMES
MODEL_SCOPE_DIR = "models"


def plot_scopes(df):
    scopes = [(SCOPE_KIND_ALL if fam is None else SCOPE_KIND_FAMILY, slug, label, slug)
              for slug, label, fam in FAMILY_SCOPES]
    scopes.insert(1, (SCOPE_KIND_PICKED, PICKED_MODELS_SCOPE, PICKED_MODELS_LABEL, PICKED_MODELS_SCOPE))
    meta = model_metadata_frame(models_in(df)) if models_in(df) else None
    for m in models_in(df):
        label = meta.loc[m, "display_name"] or short_model_label(m)
        scopes.append((SCOPE_KIND_MODEL, m, label, f"{MODEL_SCOPE_DIR}/{m}"))
    return scopes


def filter_by_scope(df, kind, key):
    if kind in (SCOPE_KIND_MODEL, SCOPE_KIND_PICKED):
        if df.empty or "model_name" not in df.columns:
            return df.iloc[0:0]
        if kind == SCOPE_KIND_PICKED:
            return df.loc[df["model_name"].map(is_picked_model).astype(bool)]
        return df.loc[df["model_name"].astype(str) == str(key)]
    return filter_by_family_scope(df, key)


def scope_options(df):
    out = []
    for kind, key, label, _d in plot_scopes(df):
        prefix = {SCOPE_KIND_ALL: "", SCOPE_KIND_PICKED: "", SCOPE_KIND_FAMILY: "Family: ",
                  SCOPE_KIND_MODEL: "Model: "}[kind]
        out.append({"label": f"{prefix}{label}", "value": f"{kind}:{key}"})
    return out


def filter_by_scope_value(df, value):
    if not value:
        return df
    kind, _, key = str(value).partition(":")
    return filter_by_scope(df, kind, key)


MODEL_PALETTE = [
    "#F3C300", "#875692", "#F38400", "#A1CAF1", "#BE0032", "#C2B280", "#848482", "#008856",
    "#E68FAC", "#0067A5", "#F99379", "#604E97", "#F6A600", "#B3446C", "#DCD300", "#882D17",
    "#8DB600", "#654522", "#E25822", "#2B3D26", "#17BECF", "#1F3A93", "#D62728", "#7F3C8D",
    "#11A579", "#3969AC",
    "#00A6A6", "#9C6ADE", "#6B8E23", "#C71585",
]
UNREGISTERED_MODEL_PALETTE = ["#4D4D4D", "#9C9C9C", "#E73F74", "#80BA5A", "#008695", "#CF1C90"]

PAPER_FAMILY_OF_REGISTRY_FAMILY = {"Aya Expanse": "Aya", "Tiny Aya": "Aya"}
UNREGISTERED_MODEL_FAMILIES = {
    "OpenLLM-France/Claire-Mistral-7B-0.1": "Claire",
    "OpenLLM-France/Lucie-7B": "Lucie",
    "AIInstituteLNCC/manaca-1b-base": "Manacá",
    "maritaca-ai/sabia-7b": "Sabiá",
    "recogna-nlp/bode-13b-alpaca-pt-br-no-peft": "Bode",
    "hfl/chinese-llama-2-13b": "Chinese-LLaMA-2",
    "baichuan-inc/Baichuan2-13B-Base": "Baichuan 2",
    "tokyotech-llm/Swallow-70b-hf": "Swallow",
    "tokyotech-llm/Swallow-13b-hf": "Swallow",
    "llm-jp/llm-jp-3-13b": "LLM-jp-3",
}
_UNREGISTERED_FAMILY_KEYS = {**UNREGISTERED_MODEL_FAMILIES,
                             **{m.replace("/", "_", 1): f for m, f in UNREGISTERED_MODEL_FAMILIES.items()}}
FAMILY_COLORS = {
    "Aya": "#FFB000",
    "EuroLLM": "#FE6100",
    "Gemma 4": "#DC267F",
    "Apertus": "#648FFF",
    "Claire": "#7B3294",
    "Lucie": "#1B9E9E",
    "Sabiá": "#A6761D",
    "Manacá": "#7FBC41",
    "Bode": "#6B4226",
    "Chinese-LLaMA-2": "#D01C8B",
    "Baichuan 2": "#525252",
    "Swallow": "#56B4E9",
    "LLM-jp-3": "#E6AB02",
    "rinna GPT-NeoX": "#F781BF",
    "UER GPT-2": "#999933",
}
MAJOR_PAPER_FAMILIES = ("Aya", "EuroLLM", "Gemma 4", "Apertus")


def paper_family(model_name):
    meta = lookup_model(model_name)
    if meta is not None:
        return PAPER_FAMILY_OF_REGISTRY_FAMILY.get(meta.family, meta.family)
    return _UNREGISTERED_FAMILY_KEYS.get(str(model_name))


def family_color(family):
    import hashlib

    if family in FAMILY_COLORS:
        return FAMILY_COLORS[family]
    digest = int(hashlib.sha256(str(family).encode("utf-8")).hexdigest(), 16)
    return UNREGISTERED_MODEL_PALETTE[digest % len(UNREGISTERED_MODEL_PALETTE)]


def _hex(rgb):
    return "#" + "".join(f"{int(round(255 * min(max(c, 0.0), 1.0))):02X}" for c in rgb)


def _rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4))


def tint(color, amount):
    return _hex(tuple(c + (1 - c) * amount for c in _rgb(color)))


def shade(color, amount):
    return _hex(tuple(c * (1 - amount) for c in _rgb(color)))


def _paper_family_members(family):
    from model_metadata2 import _ENTRIES

    members = [e.model_id for e in _ENTRIES if PAPER_FAMILY_OF_REGISTRY_FAMILY.get(e.family, e.family) == family]
    return members + [m for m, f in UNREGISTERED_MODEL_FAMILIES.items() if f == family]


def _canonical_id(model_name):
    meta = lookup_model(model_name)
    if meta is not None:
        return meta.model_id
    name = str(model_name)
    for m in UNREGISTERED_MODEL_FAMILIES:
        if name in (m, m.replace("/", "_", 1)):
            return m
    return name


def model_color(model_name):
    import hashlib

    family = paper_family(model_name)
    if family is None:
        digest = int(hashlib.sha256(str(model_name).encode("utf-8")).hexdigest(), 16)
        return UNREGISTERED_MODEL_PALETTE[digest % len(UNREGISTERED_MODEL_PALETTE)]
    base = family_color(family)
    members = _paper_family_members(family)
    key = _canonical_id(model_name)
    if len(members) <= 1 or key not in members:
        return base
    t = -0.3 + 0.85 * members.index(key) / (len(members) - 1)
    return shade(base, -t) if t < 0 else tint(base, t)


BASE_VARIANT_HATCH = "///"
BASE_VARIANT_TINT = 0.45


def paper_model_style(model_name):
    from model_metadata2 import _ENTRIES

    family = paper_family(model_name)
    if family is None:
        return model_color(model_name), None
    color = family_color(family)
    meta = lookup_model(model_name)
    if meta is not None and not meta.instruction_tuned and any(
            e.family == meta.family and e.instruction_tuned for e in _ENTRIES):
        return tint(color, BASE_VARIANT_TINT), BASE_VARIANT_HATCH
    return color, None


LANGUAGE_COLORS = {
    "de": "#0072B2", "sv": "#56B4E9", "fr": "#009E73", "pt": "#E69F00",
    "ja": "#D55E00", "zh": "#CC79A7", "en": "#404040",
}

LANGUAGE_NAMES = {"en": "English", "de": "German", "sv": "Swedish", "fr": "French", "pt": "Portuguese",
                  "zh": "Chinese", "ja": "Japanese"}

STRICT_PALETTE = {"yellow": "#ffb000", "orange": "#fe6100", "magenta": "#dc267f", "purple": "#785ef0",
                  "blue": "#648fff"}
_P = STRICT_PALETTE
VIOLIN_MODEL_COLORS = {
    "utter-project/EuroLLM-22B-Instruct-2512": _P["orange"],
    "utter-project/EuroLLM-22B-2512": _P["orange"],
    "google/gemma-4-31B-it": _P["magenta"],
    "google/gemma-4-31B": _P["magenta"],
    "CohereLabs/aya-expanse-32b": _P["yellow"],
    "swiss-ai/Apertus-70B-2509": _P["blue"],
    "OpenLLM-France/Lucie-7B": _P["blue"],
    "maritaca-ai/sabia-7b": _P["orange"],
    "AIInstituteLNCC/manaca-1b-base": _P["yellow"],
    "hfl/chinese-llama-2-13b": _P["magenta"],
    "llm-jp/llm-jp-3-13b": _P["purple"],
    "OpenLLM-France/Claire-Mistral-7B-0.1": _P["blue"],
    "recogna-nlp/bode-13b-alpaca-pt-br-no-peft": _P["orange"],
    "tokyotech-llm/Swallow-13b-hf": _P["purple"],
}
del _P
_VIOLIN_COLOR_KEYS = {**VIOLIN_MODEL_COLORS,
                      **{m.replace("/", "_", 1): c for m, c in VIOLIN_MODEL_COLORS.items()}}
VIOLIN_UNASSIGNED_COLOR = "#8D8D8D"


def violin_model_color(model_name):
    return _VIOLIN_COLOR_KEYS.get(str(model_name), VIOLIN_UNASSIGNED_COLOR)


def violin_model_style(model_name):
    from model_metadata2 import _ENTRIES

    meta = lookup_model(model_name)
    hatch = None
    if meta is not None and not meta.instruction_tuned and any(
            e.family == meta.family and e.instruction_tuned for e in _ENTRIES):
        hatch = BASE_VARIANT_HATCH
    return violin_model_color(model_name), None


FAILURE_BAR_SECTIONS = [
    ("Tiny Aya", "Aya", ["CohereLabs/tiny-aya-base", "CohereLabs/tiny-aya-global"]),
    ("Aya Expanse", "Aya", ["CohereLabs/aya-expanse-8b", "CohereLabs/aya-expanse-32b"]),
    ("EuroLLM", "EuroLLM", ["utter-project/EuroLLM-1.7B", "utter-project/EuroLLM-1.7B-Instruct",
                            "utter-project/EuroLLM-9B", "utter-project/EuroLLM-9B-Instruct",
                            "utter-project/EuroLLM-22B-2512", "utter-project/EuroLLM-22B-Instruct-2512"]),
    ("Gemma 4", "Gemma 4", ["google/gemma-4-E2B", "google/gemma-4-E2B-it", "google/gemma-4-E4B",
                            "google/gemma-4-E4B-it", "google/gemma-4-12B", "google/gemma-4-12B-it",
                            "google/gemma-4-26B-A4B", "google/gemma-4-26B-A4B-it", "google/gemma-4-31B",
                            "google/gemma-4-31B-it"]),
    ("Apertus", "Apertus", ["swiss-ai/Apertus-8B-2509", "swiss-ai/Apertus-70B-2509"]),
    ("Apertus IT", "Apertus", ["swiss-ai/Apertus-8B-Instruct-2509", "swiss-ai/Apertus-70B-Instruct-2509"]),
    ("GPT-OSS", None, ["openai/gpt-oss-120b"]),
    ("Other non-English-centric models", None, [
        "OpenLLM-France/Claire-Mistral-7B-0.1", "OpenLLM-France/Lucie-7B", "AIInstituteLNCC/manaca-1b-base",
        "AIInstituteLNCC/manaca-1b-instruct", "maritaca-ai/sabia-7b",
        "recogna-nlp/bode-13b-alpaca-pt-br-no-peft", "hfl/chinese-llama-2-13b",
        "tokyotech-llm/Swallow-13b-hf", "llm-jp/llm-jp-3-13b", "llm-jp/llm-jp-3-172b",
        "Aleph-Alpha/Kolibri-1-BF16"]),
]
FAILURE_BAR_MODEL_COUNT = 36
FAILURE_BAR_MODELS = [m for _s, _f, models in FAILURE_BAR_SECTIONS for m in models]
STAT_MODEL_SECTIONS = [
    ("Tiny Aya", "Aya", ["CohereLabs/tiny-aya-base", "CohereLabs/tiny-aya-global", "CohereLabs/tiny-aya-earth",
                         "CohereLabs/tiny-aya-fire", "CohereLabs/tiny-aya-water"]),
    ("Aya Expanse", "Aya", ["CohereLabs/aya-expanse-8b", "CohereLabs/aya-expanse-32b"]),
    ("EuroLLM", "EuroLLM", ["utter-project/EuroLLM-1.7B", "utter-project/EuroLLM-1.7B-Instruct",
                            "utter-project/EuroLLM-9B", "utter-project/EuroLLM-9B-Instruct",
                            "utter-project/EuroLLM-22B-2512", "utter-project/EuroLLM-22B-Instruct-2512"]),
    ("Gemma 4", "Gemma 4", ["google/gemma-4-E2B", "google/gemma-4-E2B-it", "google/gemma-4-E4B",
                            "google/gemma-4-E4B-it", "google/gemma-4-12B", "google/gemma-4-12B-it",
                            "google/gemma-4-26B-A4B", "google/gemma-4-26B-A4B-it", "google/gemma-4-31B",
                            "google/gemma-4-31B-it"]),
    ("Apertus", "Apertus", ["swiss-ai/Apertus-8B-2509", "swiss-ai/Apertus-70B-2509"]),
    ("Apertus IT", "Apertus", ["swiss-ai/Apertus-8B-Instruct-2509", "swiss-ai/Apertus-70B-Instruct-2509"]),
    ("GPT-OSS", None, ["openai/gpt-oss-120b"]),
    ("Other non-English-centric models", None, [
        "OpenLLM-France/Claire-Mistral-7B-0.1", "OpenLLM-France/Lucie-7B", "AIInstituteLNCC/manaca-1b-base",
        "AIInstituteLNCC/manaca-1b-instruct", "maritaca-ai/sabia-7b", "recogna-nlp/bode-13b-alpaca-pt-br-no-peft",
        "hfl/chinese-llama-2-13b", "tokyotech-llm/Swallow-13b-hf", "llm-jp/llm-jp-3-13b", "llm-jp/llm-jp-3-172b",
        "Aleph-Alpha/Kolibri-1-BF16"]),
]
STAT_MODEL_COUNT = 39
STAT_MODELS = [m for _s, _f, models in STAT_MODEL_SECTIONS for m in models]
assert len(STAT_MODELS) == STAT_MODEL_COUNT == len(set(STAT_MODELS))
FAILURE_BAR_LABELS = {
    "CohereLabs/tiny-aya-base": "Base", "CohereLabs/tiny-aya-global": "Global",
    "CohereLabs/tiny-aya-earth": "Earth", "CohereLabs/tiny-aya-fire": "Fire", "CohereLabs/tiny-aya-water": "Water",
    "CohereLabs/aya-expanse-8b": "8B", "CohereLabs/aya-expanse-32b": "32B",
    "utter-project/EuroLLM-1.7B": "1.7B", "utter-project/EuroLLM-1.7B-Instruct": "1.7B IT",
    "utter-project/EuroLLM-9B": "9B", "utter-project/EuroLLM-9B-Instruct": "9B IT",
    "utter-project/EuroLLM-22B-2512": "22B", "utter-project/EuroLLM-22B-Instruct-2512": "22B IT",
    "google/gemma-4-E2B": "E2B", "google/gemma-4-E2B-it": "E2B IT", "google/gemma-4-E4B": "E4B",
    "google/gemma-4-E4B-it": "E4B IT", "google/gemma-4-12B": "12B", "google/gemma-4-12B-it": "12B IT",
    "google/gemma-4-26B-A4B": "26B-A4B", "google/gemma-4-26B-A4B-it": "26B-A4B IT",
    "google/gemma-4-31B": "31B", "google/gemma-4-31B-it": "31B IT",
    "swiss-ai/Apertus-8B-2509": "8B", "swiss-ai/Apertus-70B-2509": "70B",
    "swiss-ai/Apertus-8B-Instruct-2509": "8B", "swiss-ai/Apertus-70B-Instruct-2509": "70B",
    "openai/gpt-oss-120b": "120B",
    "OpenLLM-France/Claire-Mistral-7B-0.1": "Claire 7B", "OpenLLM-France/Lucie-7B": "Lucie 7B",
    "AIInstituteLNCC/manaca-1b-base": "Manacá 1B", "AIInstituteLNCC/manaca-1b-instruct": "Manacá Instruct 1B",
    "maritaca-ai/sabia-7b": "Sabiá 7B", "recogna-nlp/bode-13b-alpaca-pt-br-no-peft": "Bode 13B",
    "hfl/chinese-llama-2-13b": "Chinese-LLaMA-2 13B",
    "tokyotech-llm/Swallow-13b-hf": "Swallow 13B", "llm-jp/llm-jp-3-13b": "LLM-jp-3 13B",
    "llm-jp/llm-jp-3-172b": "LLM-jp-3 172B", "Aleph-Alpha/Kolibri-1-BF16": "Kolibri",
}


def failure_workbook_model_key(model_name):
    return str(model_name).replace("/", "_", 1)


SOURCE_ROW_ID_COL = "source_row_id"


def add_source_row_id(df):
    from metadata.expand import source_row_key

    out = df.copy()
    if out.empty or "Proper English strategy" not in out.columns:
        out[SOURCE_ROW_ID_COL] = pd.Series(dtype=object, index=out.index)
        return out
    target_col = TARGET_LANGUAGE_SOURCE_COL if TARGET_LANGUAGE_SOURCE_COL in out.columns else "Target language"
    out[SOURCE_ROW_ID_COL] = [
        source_row_key(t, s) for t, s in zip(out[target_col], out["Proper English strategy"])
    ]
    return out


MODEL_VIEW_ALL = "all"
MODEL_VIEW_IT = "it"
MODEL_VIEW_EC = "ec"
MODEL_VIEWS = [
    (MODEL_VIEW_ALL, "All models"),
    (MODEL_VIEW_IT, "Instruction-tuned vs non-instruction-tuned (English-centric models only)"),
    (MODEL_VIEW_EC, "English-centric vs non-English-centric"),
]
MODEL_VIEW_LABELS = dict(MODEL_VIEWS)

IT_LABELS = {False: "Non-instruction-tuned", True: "Instruction-tuned"}
EC_LABELS = {True: "English-centric", False: "Non-English-centric"}


def model_metadata_frame(model_names):
    names = sorted({str(n) for n in model_names})
    rows = [{"model_name": n, **model_metadata_record(n)} for n in names]
    return pd.DataFrame(rows, columns=["model_name"] + METADATA_FIELDS).set_index("model_name")


def add_model_metadata(df):
    out = df.copy()
    if "model_name" not in out.columns or out.empty:
        for f in METADATA_FIELDS:
            out[f] = None
        return out
    meta = model_metadata_frame(out["model_name"].unique())
    keys = out["model_name"].astype(str)
    for f in METADATA_FIELDS:
        out[f] = keys.map(meta[f])
    return out


def _model_property(df, field):
    if "model_name" not in df.columns or df.empty:
        return pd.Series(dtype=object, index=df.index)
    meta = model_metadata_frame(df["model_name"].unique())
    return df["model_name"].astype(str).map(meta[field])


def unresolved_model_names(*dfs):
    names = set()
    for df in dfs:
        if df is not None and "model_name" in df.columns:
            names.update(df["model_name"].dropna().astype(str).unique())
    return unresolved_models(names)


def filter_english_centric(df):
    return df.loc[_model_property(df, "english_centric") == True]


def model_view_panels(df, view):
    if view in (None, MODEL_VIEW_ALL):
        return [(None, df)]
    if view == MODEL_VIEW_IT:
        ec_df = filter_english_centric(df)
        it = _model_property(ec_df, "instruction_tuned")
        return [(IT_LABELS[v], ec_df.loc[it == v]) for v in (False, True)]
    if view == MODEL_VIEW_EC:
        ec = _model_property(df, "english_centric")
        return [(EC_LABELS[v], df.loc[ec == v]) for v in (True, False)]
    raise ValueError(f"Unknown model view: {view!r}")


def compute_filter_choices2(df):
    choices = {col: _sorted_choices(df, col) for col, _label in FILTER_DIMENSIONS}
    choices["metric_options"] = [
        {"label": label, "value": value} for value, label in METRIC_VIEW_OPTIONS if _metric_view_available(df, value)
    ]
    return choices


def filter_by_experiment(df, experiment):
    if experiment is None or df.empty:
        return df
    if EXPERIMENT_COL not in df.columns:
        df = normalize_loaded_frame2(df)
    return df.loc[df[EXPERIMENT_COL] == experiment]


def apply_filters2(df, selections, include_identical, group=ALL_GROUPS, preference_subset=PREFERENCE_ALL,
                   experiment=EXPERIMENT_STANDARD, preference_languages=None):
    filtered = filter_by_experiment(df, experiment)
    filtered = filter_by_group(filtered, group)
    filtered = filter_by_preference(filtered, preference_subset)
    filtered = filter_target_preference(filtered, preference_languages)
    for col, _label in FILTER_DIMENSIONS:
        values = (selections or {}).get(col)
        if values and col in filtered.columns:
            filtered = filtered.loc[filtered[col].isin(values)]

    if not include_identical and {"pos_sentence", "neg_sentence"} <= set(filtered.columns):
        filtered = filtered.loc[filtered["pos_sentence"] != filtered["neg_sentence"]]

    return filtered


def compute_translation_filter_choices2(df):
    return {col: _sorted_choices(df, col) for col, _label in TRANSLATION_FILTER_DIMENSIONS}


def apply_translation_filters2(df, selections, group=ALL_GROUPS, preference_languages=None):
    filtered = filter_by_group(df, group)
    filtered = filter_target_preference(filtered, preference_languages)
    for col, _label in TRANSLATION_FILTER_DIMENSIONS:
        values = (selections or {}).get(col)
        if values and col in filtered.columns:
            filtered = filtered.loc[filtered[col].isin(values)]
    return filtered


def metric_available(df, column):
    if column not in df.columns:
        return False
    return pd.to_numeric(df[column], errors="coerce").notna().any()


def _metric_view_available(df, value):
    if value == "diff_comparison":
        return any(metric_available(df, col) for col, _label in DIFF_METRIC_COLUMNS.values())
    return metric_available(df, value)


def matching_examples(df, metric, lower, upper):
    if metric not in df.columns:
        return df.iloc[0:0]
    ys = pd.to_numeric(df[metric], errors="coerce")
    mask = ys.notna()
    if lower is not None:
        mask &= ys >= lower
    if upper is not None:
        mask &= ys <= upper
    out = df.loc[mask].copy()
    out["_y_value"] = ys.loc[mask]
    return out


def paginate(df, page, page_size=PAGE_SIZE):
    n_pages = max(1, math.ceil(len(df) / page_size))
    page = min(max(1, int(page or 1)), n_pages)
    start = (page - 1) * page_size
    return df.iloc[start:start + page_size], page, n_pages


def category_order(df, x_col):
    return _sorted_choices(df, x_col)


def short_model_label(model_name):
    name = str(model_name)
    return name.split("_", 1)[1] if "_" in name else name


def category_labels(categories, x_col):
    if x_col != "model_name":
        return [str(c) for c in categories]
    short = [short_model_label(c) for c in categories]
    return short if len(set(short)) == len(short) else [str(c) for c in categories]


def metric_distributions(df, metric, x_col, groups):
    if df.empty or metric not in df.columns or x_col not in df.columns:
        return [], []
    ys = pd.to_numeric(df[metric], errors="coerce")
    work = pd.DataFrame({"x": df[x_col], "g": df[GROUP_COL].astype(str) if GROUP_COL in df.columns else "", "y": ys})
    work = work.loc[work["y"].notna()]
    categories = category_order(work, "x")
    grouped = {key: sub["y"].to_numpy(dtype=float) for key, sub in work.groupby(["x", "g"], sort=False)}
    out = []
    for cat in categories:
        for g in groups:
            values = grouped.get((cat, str(g)))
            if values is not None and values.size:
                out.append((cat, g, values))
    return categories, out


def summarize_distributions(df, metric, x_col, groups):
    _categories, dists = metric_distributions(df, metric, x_col, groups)
    rows = [
        {x_col: cat, GROUP_COL: g, "n": int(v.size), "mean": float(np.mean(v)), "median": float(np.median(v))}
        for cat, g, v in dists
    ]
    return pd.DataFrame(rows, columns=[x_col, GROUP_COL, "n", "mean", "median"])


def compute_translation_summary2(df):
    columns = ["model_name", "Target language", "classification", "count", "proportion"]
    if df.empty or "translation_match" not in df.columns:
        return pd.DataFrame(columns=columns)

    group_cols = [c for c in ("model_name", "Target language") if c in df.columns]
    groups = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]

    rows = []
    for keys, group in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        total = len(group)
        counts = group["translation_match"].value_counts()
        for label in TRANSLATION_MATCH_LABELS:
            count = int(counts.get(label, 0))
            row = dict(zip(group_cols, keys))
            row.update({"classification": label, "count": count, "proportion": (count / total) if total else float("nan")})
            rows.append(row)

    return pd.DataFrame(rows, columns=columns if not group_cols else group_cols + columns[2:])


def compute_translation_preference(df, x_col):
    count_cols = ["N_proper", "N_literal", "N_both", "N_neither", "N_total", "N_proper_or_literal"]
    columns = [x_col, GROUP_COL] + count_cols + [PREFERENCE_COL, "discriminative_coverage", "exact_reference_coverage"]
    if df.empty or "translation_match" not in df.columns or x_col not in df.columns:
        return pd.DataFrame(columns=columns)

    work = pd.DataFrame({
        x_col: df[x_col],
        GROUP_COL: df[GROUP_COL].astype(str) if GROUP_COL in df.columns else "",
        "m": df["translation_match"].astype(str),
    })
    rows = []
    for (x, g), sub in work.groupby([x_col, GROUP_COL], sort=True):
        counts = sub["m"].value_counts()
        n = {label: int(counts.get(label, 0)) for label in TRANSLATION_MATCH_LABELS}
        total = len(sub)
        disc = n["proper"] + n["literal"]
        rows.append({
            x_col: x,
            GROUP_COL: g,
            "N_proper": n["proper"],
            "N_literal": n["literal"],
            "N_both": n["both"],
            "N_neither": n["neither"],
            "N_total": total,
            "N_proper_or_literal": disc,
            PREFERENCE_COL: (n["proper"] / disc) if disc else float("nan"),
            "discriminative_coverage": (disc / total) if total else float("nan"),
            "exact_reference_coverage": ((disc + n["both"]) / total) if total else float("nan"),
        })
    return pd.DataFrame(rows, columns=columns)


TRANSFERRED_PROBABILITY_COL = "P_transferred_given_native_or_transferred"
TRANSLATION_STATUS_OK = "ok"
TRANSLATION_STATUS_NO_DENOMINATOR = "no native/transferred match (N_native + N_transferred = 0)"
TRANSLATION_STATUS_NO_ROWS = "no translation rows for this model/language"
TRANSLATION_STATUS_NO_MODEL = "no translation outputs for this model"


def compute_transferred_probability(df, models, languages, group=None):
    cols = ["model_name", "language", "group", "N_native", "N_transferred", "N_both", "N_neither", "N_total",
            "denominator", TRANSFERRED_PROBABILITY_COL, "status"]
    work = df
    if group is not None and len(work):
        work = work.loc[work[GROUP_COL].astype(str) == str(group)]
    has_model = set(df["model_name"]) if len(df) else set()
    counts = (work.groupby(["model_name", "Target language"])["translation_match"].value_counts().unstack(fill_value=0)
              if len(work) else pd.DataFrame())
    out = []
    for m in models:
        for lang in languages:
            key = (m, lang)
            n = {lab: int(counts.loc[key].get(lab, 0)) if key in counts.index else 0 for lab in TRANSLATION_MATCH_LABELS}
            total = int(sum(counts.loc[key])) if key in counts.index else 0
            denom = n["proper"] + n["literal"]
            status = (TRANSLATION_STATUS_NO_MODEL if m not in has_model else
                      TRANSLATION_STATUS_NO_ROWS if total == 0 else
                      TRANSLATION_STATUS_NO_DENOMINATOR if denom == 0 else TRANSLATION_STATUS_OK)
            out.append({"model_name": m, "language": lang, "group": group if group is not None else "all groups",
                        "N_native": n["proper"], "N_transferred": n["literal"], "N_both": n["both"],
                        "N_neither": n["neither"], "N_total": total, "denominator": denom,
                        TRANSFERRED_PROBABILITY_COL: (n["literal"] / denom) if denom else float("nan"),
                        "status": status})
    return pd.DataFrame(out, columns=cols)


def sanitize_filename(value):
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower()
    return slug or "unnamed"


NO_CONTEXT_PROMPT_LANGUAGE = "raw"
ALL_COMPARISONS = "All comparisons"


def analysis_rows(df, metric=MEANLP_COL, include_identical=False, preference_subset=PREFERENCE_ALL,
                  experiment=EXPERIMENT_STANDARD, preference_languages=None):
    value_col = score_metric(metric)["delta"]
    selected = apply_filters2(
        df, {}, include_identical=include_identical, preference_subset=preference_subset, experiment=experiment,
        preference_languages=preference_languages,
    )
    if "prompt_language" in selected.columns:
        selected = selected.loc[selected["prompt_language"].astype(str) == NO_CONTEXT_PROMPT_LANGUAGE]
    out = selected.copy()
    for col, spec in SCORE_METRICS.items():
        out[spec["delta"]] = pd.to_numeric(out[col], errors="coerce") if col in out.columns else np.nan
    return out.loc[out[value_col].notna()]


def meanlp_analysis_rows(df, include_identical=False, preference_subset=PREFERENCE_ALL, experiment=EXPERIMENT_STANDARD,
                         preference_languages=None):
    return analysis_rows(df, MEANLP_COL, include_identical, preference_subset, experiment, preference_languages)


FAILURE_RATE_COLUMNS = [
    "metric", "experiment", "preference_subset", "comparison", "model_name", "display_name", "instruction_tuned",
    "english_centric", "Target language", "N", "N_failure", "N_tie", "failure_rate",
]


def compute_failure_rates(df, preference_subset=PREFERENCE_ALL, include_identical=False, experiment=EXPERIMENT_STANDARD,
                          preference_languages=None, metric=MEANLP_COL):
    spec = score_metric(metric)
    rows_df = analysis_rows(
        df, metric, include_identical=include_identical, preference_subset=preference_subset, experiment=experiment,
        preference_languages=preference_languages,
    )
    if rows_df.empty:
        return pd.DataFrame(columns=FAILURE_RATE_COLUMNS)

    meta = model_metadata_frame(rows_df["model_name"].unique())
    pos = rows_df["pos"].astype(str) if "pos" in rows_df.columns else pd.Series(ALL_COMPARISONS, index=rows_df.index)
    parts = [(ALL_COMPARISONS, rows_df)] + [(p, rows_df.loc[pos == p]) for p in sorted(pos.unique())]

    out = []
    for comparison, sub in parts:
        for (model, lang), g in sub.groupby(["model_name", "Target language"], sort=True):
            y = g[spec["delta"]].to_numpy(dtype=float)
            n = int(y.size)
            n_fail = int((y < 0).sum())
            out.append({
                "metric": spec["label"],
                "experiment": experiment,
                "preference_subset": PREFERENCE_SUBSET_LABELS.get(preference_subset, preference_subset),
                "comparison": comparison,
                "model_name": model,
                "display_name": meta.loc[str(model), "display_name"],
                "instruction_tuned": meta.loc[str(model), "instruction_tuned"],
                "english_centric": meta.loc[str(model), "english_centric"],
                "Target language": lang,
                "N": n,
                "N_failure": n_fail,
                "N_tie": int((y == 0).sum()),
                "failure_rate": (n_fail / n) if n else float("nan"),
            })
    return pd.DataFrame(out, columns=FAILURE_RATE_COLUMNS)


FAILURE_RATE_PREFERENCE_SUBSET = "gt-or-gtgt"


def failure_rate_observations(df, preference_languages=None, include_identical=False):
    obs = add_source_row_id(analysis_rows(df, SLOR_COL, include_identical=include_identical,
                                          preference_subset=FAILURE_RATE_PREFERENCE_SUBSET,
                                          preference_languages=preference_languages))
    return obs.loc[obs["pos"].isin([TARGET_COMPARISON_POS, ENGLISH_COMPARISON_POS])] if len(obs) else obs


FAILURE_MATRIX_COLUMNS = ["en", "de", "sv", "fr", "pt", "ja", "zh"]


def compute_failure_rate_matrix(df, preference_languages=None, include_identical=False, metric=MEANLP_COL,
                                preference_subset=PREFERENCE_ALL):
    long = compute_failure_rates(
        df, preference_subset, include_identical=include_identical, experiment=EXPERIMENT_STANDARD,
        preference_languages=preference_languages, metric=metric,
    )
    long = long.loc[long["comparison"].isin([TARGET_COMPARISON_POS, ENGLISH_COMPARISON_POS])].copy()
    long["column"] = np.where(long["comparison"] == ENGLISH_COMPARISON_POS, "en", long["Target language"].astype(str))
    sums = long.groupby(["model_name", "column"])[["N", "N_failure", "N_tie"]].sum()
    models = sorted(long["model_name"].astype(str).unique())

    def wide(col):
        if sums.empty:
            return pd.DataFrame(0, index=pd.Index(models, name="model_name"), columns=FAILURE_MATRIX_COLUMNS)
        w = sums[col].unstack("column").reindex(index=models, columns=FAILURE_MATRIX_COLUMNS)
        w.index.name = "model_name"
        return w.fillna(0).astype(int)

    n_total, n_fail, n_tie = wide("N"), wide("N_failure"), wide("N_tie")
    rate = (n_fail / n_total.where(n_total > 0)).astype(float)
    return {"failure_rate": rate, "N_failure": n_fail, "N_total": n_total, "N_tie": n_tie}


CONTROL_CATEGORY_COL = "control_category"
CONTROL_JAPANESE = "Japanese"
CONTROL_ENGLISH = "English"
JAPANESE_CONTROL_PREFERENCE_SUBSET = "gt-or-gtgt"


def japanese_control_rows(df, metric=MEANLP_COL, include_identical=False, group=ALL_GROUPS,
                          preference_subset=JAPANESE_CONTROL_PREFERENCE_SUBSET):
    rows = analysis_rows(
        df, metric, include_identical=include_identical,
        preference_subset=preference_subset,
        experiment=EXPERIMENT_JAPANESE_CONTROL,
    )
    if not rows.empty:
        rows = rows.loc[rows["pos"] == JAPANESE_COMPARISON_POS]
    rows = filter_by_group(rows, group)
    out = add_source_row_id(rows)
    out[CONTROL_CATEGORY_COL] = CONTROL_JAPANESE
    return out


def japanese_control_frame(df, include_identical=False, metric=MEANLP_COL):
    ja_control = japanese_control_rows(df, metric=metric, include_identical=include_identical)

    english_strategy_set = set()
    if not ja_control.empty and "Proper English strategy" in ja_control.columns:
        english_strategy_set = set(ja_control["Proper English strategy"].dropna().astype(str))

    en = analysis_rows(df, metric, include_identical=include_identical, experiment=EXPERIMENT_STANDARD)
    if not en.empty:
        en = en.loc[
            (en["pos"] == TARGET_COMPARISON_POS)
            & (en["Target language"].map(canonical_language_code) == "ja")
        ].copy()
        if english_strategy_set and "Proper English strategy" in en.columns:
            en = en.loc[en["Proper English strategy"].astype(str).isin(english_strategy_set)].copy()
        elif not english_strategy_set:
            en = en.iloc[0:0].copy()
    en = add_source_row_id(en)

    if not en.empty:
        en["Target language"] = "en"
    en[CONTROL_CATEGORY_COL] = CONTROL_ENGLISH
    return pd.concat([ja_control, en], ignore_index=True)


ORIGINAL_TARGET_LANGUAGE_COL = "original_target_language"
PAIRED_CATEGORY_COL = "paired_category"
PAIRED_PREFERENCE_SUBSET_COL = "preference_subset"
SEMANTIC_CONTENT_KEY_COL = "semantic_content_key"
JAPANESE_PREFERENCE_COL = PREFERENCE_ANNOTATION_BY_POS[JAPANESE_COMPARISON_POS]
CONTROL_JA_BASELINE = "JA baseline (English strategy transfer)"
JA_BASELINE_LANGUAGE = "ja"
JA_BASELINE_POS = TARGET_COMPARISON_POS
JA_BASELINE_NEG = "Literal translation of Proper English into target language"
CONTROL_SHORT_LABELS = {CONTROL_JAPANESE: "JA control", CONTROL_JA_BASELINE: "JA baseline"}


class JapaneseBaselineIdentityError(ValueError):
    """More than one Japanese-control row per (original language, model,
    content) or Japanese-baseline row per (model, content): the join would no
    longer be one-to-one, so it is refused."""


def paired_category_label(language, control_category, sep=" "):
    return f"{language}{sep}{CONTROL_SHORT_LABELS[control_category]}"


def paired_category_order(languages):
    return [(lang, cat) for lang in language_axis_order(languages)
            for cat in (CONTROL_JAPANESE, CONTROL_JA_BASELINE)]


def _semantic_content_key(rows):
    if rows.empty or "Proper English strategy" not in rows.columns:
        return pd.Series(dtype=object, index=rows.index)
    return rows["Proper English strategy"].astype(str)


def japanese_baseline_rows(df, metric=SLOR_COL, include_identical=False, group=ALL_GROUPS):
    rows = analysis_rows(df, metric, include_identical=include_identical, experiment=EXPERIMENT_STANDARD)
    if not rows.empty:
        rows = rows.loc[(rows["pos"] == JA_BASELINE_POS) & (rows["neg"] == JA_BASELINE_NEG)
                        & (rows["Target language"].map(canonical_language_code) == JA_BASELINE_LANGUAGE)]
    return add_source_row_id(filter_by_group(rows, group))


def paired_baseline_inputs(df, metric=SLOR_COL, include_identical=False,
                           preference_subset=JAPANESE_CONTROL_PREFERENCE_SUBSET, group=ALL_GROUPS):
    return (japanese_control_rows(df, metric=metric, include_identical=include_identical, group=group,
                                  preference_subset=preference_subset),
            japanese_baseline_rows(df, metric=metric, include_identical=include_identical, group=group))


def _require_unique(rows, keys, side):
    if not len(rows):
        return
    dup = rows.duplicated(keys, keep=False)
    if dup.any():
        sample = rows.loc[dup, keys].drop_duplicates().head(5)
        raise JapaneseBaselineIdentityError(
            f"{int(dup.sum())} {side} rows share a {tuple(keys)} key; e.g. {sample.to_dict('records')}")


def _control_with_keys(japanese):
    ja = japanese.copy()
    ja[ORIGINAL_TARGET_LANGUAGE_COL] = (ja["Target language"].map(canonical_language_code)
                                        if "Target language" in ja.columns else pd.Series(dtype=object))
    ja[SEMANTIC_CONTENT_KEY_COL] = _semantic_content_key(ja)
    return ja


def pair_japanese_baseline(japanese, baseline, preference_subset=JAPANESE_CONTROL_PREFERENCE_SUBSET):
    ja = _control_with_keys(japanese)
    ja = ja.loc[ja["Proper English strategy"].notna()] if "Proper English strategy" in ja.columns else ja
    base = baseline.copy()
    if preference_subset in ("gt", "gtgt"):
        required = ">" if preference_subset == "gt" else ">>"
        if TARGET_PREFERENCE_COL not in base.columns:
            base = base.iloc[0:0].copy()
        else:
            base = base.loc[_stripped(base[TARGET_PREFERENCE_COL]) == required].copy()
    base[SEMANTIC_CONTENT_KEY_COL] = _semantic_content_key(base)
    base = base.loc[base["Proper English strategy"].notna()] if "Proper English strategy" in base.columns else base
    keys = [ORIGINAL_TARGET_LANGUAGE_COL, "model_name", SEMANTIC_CONTENT_KEY_COL]
    _require_unique(ja, keys, "Japanese-control")
    _require_unique(base, ["model_name", SEMANTIC_CONTENT_KEY_COL], "Japanese-baseline")

    ja_keys = ja[keys] if len(ja) else pd.DataFrame(columns=keys)
    base_keys = (base[["model_name", SEMANTIC_CONTENT_KEY_COL]] if len(base)
                 else pd.DataFrame(columns=["model_name", SEMANTIC_CONTENT_KEY_COL]))
    matched = ja_keys.merge(base_keys, on=["model_name", SEMANTIC_CONTENT_KEY_COL], how="inner", validate="many_to_one")
    matched_set = set(map(tuple, matched[keys].itertuples(index=False, name=None)))
    control = ja.loc[[k in matched_set for k in ja_keys.itertuples(index=False, name=None)]].copy()
    control[CONTROL_CATEGORY_COL] = CONTROL_JAPANESE
    paired = matched.merge(base, on=["model_name", SEMANTIC_CONTENT_KEY_COL], how="inner", validate="many_to_one")
    paired[CONTROL_CATEGORY_COL] = CONTROL_JA_BASELINE
    out = pd.concat([control, paired], ignore_index=True)
    out[PAIRED_CATEGORY_COL] = [paired_category_label(l, c)
                                for l, c in zip(out[ORIGINAL_TARGET_LANGUAGE_COL], out[CONTROL_CATEGORY_COL])]
    out[PAIRED_PREFERENCE_SUBSET_COL] = PREFERENCE_SUBSET_LABELS.get(preference_subset, preference_subset)
    return out


EXCLUSION_REASONS = [
    ("missing_proper_english_strategy", "the control row has no Proper English strategy"),
    ("no_ja_source_row", "no standard ja row of this content for any model"),
    ("no_ja_output_for_model", "the model has no standard ja row of this content"),
    ("ja_pair_identical", "the model's ja pair has identical pos/neg sentences (excluded by default)"),
    ("ja_score_missing_or_filtered", "the model's ja row has no valid no-context score or is outside the "
                                     "construction group"),
]
EXCLUSION_COLUMNS = ([ORIGINAL_TARGET_LANGUAGE_COL, "model_name", PAIRED_PREFERENCE_SUBSET_COL,
                      "n_control_candidates", "n_matched", "n_control_excluded"] + [r for r, _d in EXCLUSION_REASONS])


def _raw_japanese_baseline(df):
    rows = filter_by_experiment(df, EXPERIMENT_STANDARD)
    if rows.empty or not {"pos", "neg", "Target language"} <= set(rows.columns):
        return rows.iloc[0:0]
    return rows.loc[(rows["pos"] == JA_BASELINE_POS) & (rows["neg"] == JA_BASELINE_NEG)
                    & (rows["Target language"].map(canonical_language_code) == JA_BASELINE_LANGUAGE)]


def paired_baseline_exclusions(df, japanese, baseline, preference_subset=JAPANESE_CONTROL_PREFERENCE_SUBSET,
                               include_identical=False):
    ja = _control_with_keys(japanese)
    if ja.empty:
        return pd.DataFrame(columns=EXCLUSION_COLUMNS)
    base_keys = set(zip(baseline["model_name"], _semantic_content_key(baseline))) if len(baseline) else set()
    raw = _raw_japanese_baseline(df)
    raw_content = _semantic_content_key(raw)
    any_content = set(raw_content)
    by_model = set(zip(raw["model_name"], raw_content))
    identical = raw["pos_sentence"] == raw["neg_sentence"] if len(raw) else pd.Series(dtype=bool)
    identical_keys = set(zip(raw.loc[identical, "model_name"], raw_content[identical])) if len(raw) else set()

    def reason(model, content, missing):
        if missing:
            return "missing_proper_english_strategy"
        if (model, content) in base_keys:
            return None
        if content not in any_content:
            return "no_ja_source_row"
        if (model, content) not in by_model:
            return "no_ja_output_for_model"
        if not include_identical and (model, content) in identical_keys:
            return "ja_pair_identical"
        return "ja_score_missing_or_filtered"

    missing = ja["Proper English strategy"].isna() if "Proper English strategy" in ja.columns else True
    ja["_reason"] = [reason(m, c, x) for m, c, x in zip(ja["model_name"], ja[SEMANTIC_CONTENT_KEY_COL],
                                                         pd.Series(missing, index=ja.index))]
    rows = []
    for (lang, model), g in ja.groupby([ORIGINAL_TARGET_LANGUAGE_COL, "model_name"], sort=True):
        counts = g["_reason"].value_counts()
        rows.append({ORIGINAL_TARGET_LANGUAGE_COL: lang, "model_name": model,
                     PAIRED_PREFERENCE_SUBSET_COL: PREFERENCE_SUBSET_LABELS.get(preference_subset, preference_subset),
                     "n_control_candidates": len(g), "n_matched": int(g["_reason"].isna().sum()),
                     "n_control_excluded": int(g["_reason"].notna().sum()),
                     **{r: int(counts.get(r, 0)) for r, _d in EXCLUSION_REASONS}})
    out = pd.DataFrame(rows, columns=EXCLUSION_COLUMNS)
    order = {l: i for i, l in enumerate(language_axis_order(out[ORIGINAL_TARGET_LANGUAGE_COL].unique()))}
    return (out.assign(_o=out[ORIGINAL_TARGET_LANGUAGE_COL].map(order))
            .sort_values(["_o", "model_name"], kind="stable").drop(columns="_o").reset_index(drop=True))


def japanese_control_with_ja_baseline_frame(df, metric=SLOR_COL, include_identical=False,
                                            preference_subset=JAPANESE_CONTROL_PREFERENCE_SUBSET,
                                            group=ALL_GROUPS, scope=None):
    japanese, baseline = paired_baseline_inputs(df, metric, include_identical, preference_subset, group)
    return pair_japanese_baseline(filter_by_scope_value(japanese, scope), filter_by_scope_value(baseline, scope),
                                  preference_subset)


ENGLISH_CONTROL_CATEGORY = "en"
TARGET_WITH_EN_SOURCE_COL = "category_source"


def target_language_with_english_frame(df, metric=SLOR_COL, preference_languages=None, include_identical=False):
    rows = add_source_row_id(analysis_rows(df, metric, include_identical=include_identical,
                                           preference_languages=preference_languages))
    if rows.empty or "pos" not in rows.columns:
        return rows.iloc[0:0].assign(**{TARGET_WITH_EN_SOURCE_COL: [], "target_language_origin": []})
    target = rows.loc[rows["pos"] == TARGET_COMPARISON_POS].copy()
    target[TARGET_WITH_EN_SOURCE_COL] = TARGET_COMPARISON_POS
    target["target_language_origin"] = target["Target language"]
    english = rows.loc[rows["pos"] == ENGLISH_COMPARISON_POS].copy()
    english[TARGET_WITH_EN_SOURCE_COL] = ENGLISH_COMPARISON_POS
    english["target_language_origin"] = english["Target language"]
    english["Target language"] = ENGLISH_CONTROL_CATEGORY
    return pd.concat([target, english], ignore_index=True)


ENGLISH_COMPARISON_NEG = "Literal English translation of target language strategy"
ENGLISH_PREFERENCE_COL = PREFERENCE_ANNOTATION_BY_POS[ENGLISH_COMPARISON_POS]
RQ1_ORIGINS = ["fr", "pt", "zh", "ja"]
RQ1_PREFERENCE = ">>"
STRATEGY_ORIGIN_COL = "strategy_origin"


def english_side_origin_frame(df, metric=SLOR_COL, origins=RQ1_ORIGINS, annotation=RQ1_PREFERENCE,
                              include_identical=False):
    rows = add_source_row_id(analysis_rows(df, metric, include_identical=include_identical))
    if rows.empty or "pos" not in rows.columns:
        return rows.iloc[0:0].assign(**{STRATEGY_ORIGIN_COL: []})
    rows = rows.loc[(rows["pos"] == ENGLISH_COMPARISON_POS) & (rows["neg"] == ENGLISH_COMPARISON_NEG)].copy()
    rows = rows.loc[_stripped(rows[ENGLISH_PREFERENCE_COL]) == annotation] if ENGLISH_PREFERENCE_COL in rows \
        else rows.iloc[0:0]
    rows[STRATEGY_ORIGIN_COL] = rows["Target language"].map(canonical_language_code)
    rows = rows.loc[rows[STRATEGY_ORIGIN_COL].isin(set(origins))].copy()
    rows["Target language"] = rows[STRATEGY_ORIGIN_COL]
    dup = rows.duplicated(["model_name", STRATEGY_ORIGIN_COL, SOURCE_ROW_ID_COL, "pos_sentence", "neg_sentence"])
    if dup.any():
        raise ValueError(f"RQ1 English-side frame: {int(dup.sum())} duplicated (model, origin, sentence pair) rows")
    return rows


LANGUAGE_AXIS_ORDER = ["de", "sv", "fr", "pt", "zh", "ja"]


def language_axis_order(values):
    present = {str(v) for v in values}
    rest = sorted(present - set(LANGUAGE_AXIS_ORDER) - {ENGLISH_CONTROL_CATEGORY})
    tail = [ENGLISH_CONTROL_CATEGORY] if ENGLISH_CONTROL_CATEGORY in present else []
    return [c for c in LANGUAGE_AXIS_ORDER if c in present] + rest + tail


def english_control_mean(frame, metric=SLOR_COL):
    rows = frame.loc[frame[TARGET_WITH_EN_SOURCE_COL] == ENGLISH_COMPARISON_POS] if len(frame) else frame
    values = rows[score_metric(metric)["delta"]].to_numpy(dtype=float) if len(rows) else np.array([])
    return (float(values.mean()) if values.size else float("nan")), int(values.size)


DIFF_CONDITION_COL = "diff_condition"
DIFF_VALUE_COL = "diff_value"
DIFF_CONDITIONS = [
    ("only_diff_tokens_tokenwise_log_prob_diff", "diff-token"),
    ("after_diff_tokens_tokenwise_log_prob_diff", "after-diff-token"),
]


def diff_comparison_frame(df, preference_languages=None, include_identical=False, experiment=EXPERIMENT_STANDARD):
    selected = apply_filters2(df, {}, include_identical=include_identical, experiment=experiment,
                              preference_languages=preference_languages)
    if "prompt_language" in selected.columns:
        selected = selected.loc[selected["prompt_language"].astype(str) == NO_CONTEXT_PROMPT_LANGUAGE]
    parts = []
    for col, label in DIFF_CONDITIONS:
        if col not in selected.columns:
            continue
        values = pd.to_numeric(selected[col], errors="coerce")
        part = selected.loc[values.notna()].copy()
        part[DIFF_CONDITION_COL] = label
        part[DIFF_VALUE_COL] = values.loc[values.notna()].astype(float)
        parts.append(part)
    if not parts:
        return selected.iloc[0:0].assign(**{DIFF_CONDITION_COL: [], DIFF_VALUE_COL: []})
    return pd.concat(parts, ignore_index=True)


def _betacf(a, b, x, max_iter=300, eps=3e-16):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc_reg(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_bt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_bt) * _betacf(b, a, 1.0 - x) / b


def student_t_sf2(t, df):
    if not np.isfinite(t):
        return 0.0 if np.isinf(t) else float("nan")
    return _betainc_reg(df / 2.0, 0.5, df / (df + t * t))


def student_t_ppf(q, df):
    target = 2.0 * (1.0 - q)
    lo, hi = 0.0, 1.0
    while student_t_sf2(hi, df) > target:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if student_t_sf2(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


OLS_COLUMNS = ["model", "term", "beta", "std_error", "t_value", "ci_lower", "ci_upper", "p_value", "n", "n_models"]


def fit_ols(y, X, terms, alpha=0.05):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    dof = n - k
    rank = np.linalg.matrix_rank(X) if n else 0
    if n == 0 or rank < k or dof <= 0:
        return [{"term": t, "beta": np.nan, "std_error": np.nan, "t_value": np.nan, "ci_lower": np.nan,
                 "ci_upper": np.nan, "p_value": np.nan, "n": n, "r_squared": np.nan, "adj_r_squared": np.nan}
                for t in terms]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)#ordinary least squaresを使用している
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / dof if ss_tot > 0 else np.nan
    sigma2 = ss_res / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    crit = student_t_ppf(1.0 - alpha / 2.0, dof)
    rows = []
    for t, b, s in zip(terms, beta, se):
        tval = b / s if s > 0 else np.nan
        rows.append({
            "term": t, "beta": float(b), "std_error": float(s), "t_value": float(tval),
            "ci_lower": float(b - crit * s), "ci_upper": float(b + crit * s),
            "p_value": float(student_t_sf2(tval, dof)) if np.isfinite(tval) else np.nan, "n": n,
            "r_squared": r2, "adj_r_squared": adj_r2,
        })
    return rows


LOG_SIZE_TERM = "log10(model_size_b)"
IT_TERM = "instruction_tuned"

OLS_MODELS = [
    ("size", [LOG_SIZE_TERM]),
    ("instruction tuning", [IT_TERM]),
    ("size + instruction tuning", [LOG_SIZE_TERM, IT_TERM]),
]


REGRESSION_OUTCOME_COL = "outcome"
REGRESSION_OUTCOME_LABEL_COL = "outcome_label"


def regression_frame(df, preference_subset=PREFERENCE_ALL, include_identical=False, experiment=EXPERIMENT_STANDARD,
                     preference_languages=None, comparison=None, languages=None, report=None, metric=MEANLP_COL):
    value_col = score_metric(metric)["delta"]
    rows_df = analysis_rows(df, metric, include_identical, preference_subset, experiment, preference_languages)
    if comparison is not None:
        rows_df = rows_df.loc[rows_df["pos"] == comparison] if "pos" in rows_df.columns else rows_df.iloc[0:0]
    if languages is not None:
        wanted = set(parse_preference_languages(languages))
        rows_df = rows_df.loc[rows_df["Target language"].map(canonical_language_code).isin(wanted)]
    all_models = set(models_in(rows_df))
    ec_df = filter_english_centric(rows_df)
    rows_df = add_model_metadata(ec_df)
    size = pd.to_numeric(rows_df["model_size_b"], errors="coerce")
    known = size.gt(0) & rows_df["instruction_tuned"].isin([True, False])
    if report is not None:
        ec_models = set(models_in(ec_df))
        report["models_excluded_not_english_centric_or_unregistered"] = sorted(all_models - ec_models)
        report["models_excluded_unresolved_registry"] = unresolved_models(all_models)
        report["models_excluded_missing_size_or_it"] = sorted(set(models_in(rows_df.loc[~known])) - set(models_in(rows_df.loc[known])))
    rows_df = add_source_row_id(rows_df.loc[known])
    annotation = comparison_preference_annotation(rows_df)
    return pd.DataFrame({
        "model_name": rows_df["model_name"].astype(str),
        "y": rows_df[value_col].astype(float),
        REGRESSION_OUTCOME_COL: metric,
        REGRESSION_OUTCOME_LABEL_COL: value_col,
        metric: pd.to_numeric(rows_df[metric], errors="coerce").astype(float) if metric in rows_df.columns else np.nan,
        LOG_SIZE_TERM: np.log10(size.loc[known].astype(float)),
        IT_TERM: rows_df["instruction_tuned"].astype(bool).astype(float),
        "language": rows_df["Target language"].astype(str) if "Target language" in rows_df.columns else "",
        "comparison": rows_df["pos"].astype(str) if "pos" in rows_df.columns else "",
        SOURCE_ROW_ID_COL: rows_df[SOURCE_ROW_ID_COL],
        "pos_sentence": rows_df["pos_sentence"] if "pos_sentence" in rows_df.columns else None,
        "neg_sentence": rows_df["neg_sentence"] if "neg_sentence" in rows_df.columns else None,
        STRONG_PREF_TERM: annotation.map(STRONG_PREFERENCE_CODES).astype(float),
    })


def fit_meanlp_ols(df, preference_subset=PREFERENCE_ALL, include_identical=False, experiment=EXPERIMENT_STANDARD,
                   preference_languages=None, languages=None):
    data = regression_frame(df, preference_subset, include_identical, experiment, preference_languages,
                            languages=languages if languages else None, metric=MEANLP_COL)
    out = []
    for label, predictors in OLS_MODELS:
        X = np.column_stack([np.ones(len(data))] + [data[p].to_numpy(dtype=float) for p in predictors])
        for row in fit_ols(data["y"].to_numpy(dtype=float), X, ["intercept"] + predictors):
            out.append({"model": label, **row, "n_models": int(data["model_name"].nunique())})
    result = pd.DataFrame(out, columns=OLS_COLUMNS)
    result.insert(0, "preference_subset", PREFERENCE_SUBSET_LABELS.get(preference_subset, preference_subset))
    result.insert(0, "experiment", experiment)
    return result


MAIN_REGRESSION_OUTCOME = SLOR_COL
MAIN_REGRESSION_OUTCOME_LABEL = DELTA_SLOR

STRONG_PREF_TERM = "strong_preference"
STRONG_PREFERENCE_CODES = {">": 0.0, ">>": 1.0}
LANGUAGE_FE_PREFIX = "language="

STRATEGY_DISTANCE_TERM = "divergence"
OLSD_TERM = "other_language_strategy_divergence"
METADATA_FEATURES = [
    (STRATEGY_DISTANCE_TERM, "divergence_tl_realization_score", "Strategy Distance (target-origin realization)"),
    (OLSD_TERM, "other_language_strategy_divergence_tl_realization_score",
     "Other-Language Strategy Divergence (target-origin realization)"),
    ("word_order_divergence", "wo_score", "Word-order divergence (normalized inversions, EN→target)"),
    ("non11_alignment", "tl_non11_prop", "Non-1:1 alignment share (target side)"),
    ("frequency_contrast", "tl_freq_contrast", "Corpus-frequency contrast log1p(f_proper) − log1p(f_transferred) (target side)"),
]
METADATA_FEATURE_COLUMNS = {term: col for term, col, _ in METADATA_FEATURES}
GLOSS_FEATURE_PREFIX = "gloss_feature_"

#ここでpretraining langauge proportionの実装
COMMON_CRAWL_TERM = "log10_common_crawl_size"
COMMON_CRAWL_LABEL = "log10(Common Crawl size by language)"
COMMON_CRAWL_SIZE_PATH = Path(__file__).resolve().parent.parent / "data" / "lang2size.json"
COMMON_CRAWL_LANGUAGE_KEYS = {"zh": "cmn", "de": "deu", "fr": "fra", "pt": "por", "sv": "swe", "ja": "jpn"}
COMMON_CRAWL_CAVEAT = (
    f"{COMMON_CRAWL_TERM} is a language-level predictor that varies across only the languages in the sample; it "
    "replaces the language fixed effects, so language-specific unobserved factors are not controlled, and model x "
    "sentence-pair clustered inference does not account for language-level dependence (its p-value is not robust)")

TERM_LABELS = {
    LOG_SIZE_TERM: "log10(model size, B)",
    COMMON_CRAWL_TERM: COMMON_CRAWL_LABEL,
    IT_TERM: "Instruction-tuned (vs non-IT)",
    STRONG_PREF_TERM: "Strong preference (>> vs >)",
    **{term: label for term, _col, label in METADATA_FEATURES},
}

MAIN_FIG_TERM_LABELS = {
    LOG_SIZE_TERM: TERM_LABELS[LOG_SIZE_TERM],
    COMMON_CRAWL_TERM: COMMON_CRAWL_LABEL,
    IT_TERM: "Instruction-tuned",
    STRONG_PREF_TERM: "Human preference",
    STRATEGY_DISTANCE_TERM: "Strategy Distance",
    OLSD_TERM: "Other-Language Strategy Divergence",
    "word_order_divergence": "Word-order divergence",
    "non11_alignment": "Non-1:1 Alignment Rate",
    "frequency_contrast": "Corpus-frequency contrast",
}

METADATA_JOIN_COL = "metadata_join_status"
_META_POS_COL = "Proper strategy in target language"
_META_NEG_COL = "Literal translation of Proper English into target language"


def load_metadata_table(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_excel(path, sheet_name="metadata", keep_default_na=False, na_values=[""])


def metadata_feature_frame(metadata):
    if metadata is None or metadata.empty:
        return pd.DataFrame()
    m = metadata.loc[(metadata["sentence_role"] == "tl_proper") & (metadata["experiment"] == "standard")].copy()
    m["_base_id"] = m["source_row_id"].astype(str).str.split("#").str[0]
    ambiguous = set(m.loc[m["_base_id"].duplicated(keep=False), "_base_id"])
    m = m.drop_duplicates("_base_id").set_index("_base_id")
    cols = [c for _t, c, _l in METADATA_FEATURES if c in m.columns]
    out = pd.DataFrame(index=m.index)
    for c in cols:
        out[c] = pd.to_numeric(m[c], errors="coerce")
    gloss = gloss_pair_feature_frame(metadata)
    if not gloss.empty:
        out = out.join(gloss, how="left")
    out["_meta_pos"] = m[_META_POS_COL] if _META_POS_COL in m.columns else None
    out["_meta_neg"] = m[_META_NEG_COL] if _META_NEG_COL in m.columns else None
    out["_ambiguous"] = out.index.isin(ambiguous)
    return out


def gloss_feature_columns(frame):
    out = []
    for c in frame.columns:
        if str(c).startswith(GLOSS_FEATURE_PREFIX) and pd.to_numeric(frame[c], errors="coerce").notna().any():
            out.append(c)
    return out


GLOSS_COLUMN = "Interlinear gloss"
GLOSS_SPLIT_RE = re.compile(r"[\s.\-()]+")
GLOSS_FEATURE_RE = re.compile(r"^[A-Z0-9]+$")
_GLOSS_PAIR_ROLES = ("tl_proper", "tl_transferred")


def gloss_features(gloss):
    if gloss is None or (isinstance(gloss, float) and math.isnan(gloss)):
        return None
    text = str(gloss).strip()
    if not text:
        return None
    return {seg for seg in GLOSS_SPLIT_RE.split(text) if GLOSS_FEATURE_RE.fullmatch(seg)}


def _gloss_pair_rows(metadata):
    if metadata is None or metadata.empty or GLOSS_COLUMN not in metadata.columns:
        return pd.DataFrame()
    m = metadata.loc[metadata["sentence_role"].isin(_GLOSS_PAIR_ROLES) & (metadata["experiment"] == "standard")].copy()
    m["_base_id"] = m["source_row_id"].astype(str).str.split("#").str[0]
    m["_features"] = m[GLOSS_COLUMN].map(gloss_features)
    return m


def gloss_pair_feature_frame(metadata):
    m = _gloss_pair_rows(metadata)
    if m.empty:
        return pd.DataFrame()
    m = m.drop_duplicates(["_base_id", "sentence_role"])
    proper = m.loc[m["sentence_role"] == "tl_proper"].set_index("_base_id")["_features"]
    transferred = m.loc[m["sentence_role"] == "tl_transferred"].set_index("_base_id")["_features"]
    ids = sorted(set(proper.index) | set(transferred.index))
    vocab = sorted(set().union(*[f for f in m["_features"] if f is not None]))
    values = np.full((len(ids), len(vocab)), np.nan)
    col = {f: j for j, f in enumerate(vocab)}
    for i, sid in enumerate(ids):
        p, t = proper.get(sid), transferred.get(sid)
        if p is None or t is None:
            continue
        row = np.zeros(len(vocab))
        for f in p:
            row[col[f]] += 1.0
        for f in t:
            row[col[f]] -= 1.0
        values[i] = row
    return pd.DataFrame(values, index=pd.Index(ids, name="_base_id"),
                        columns=[f"{GLOSS_FEATURE_PREFIX}{f}" for f in vocab])


def gloss_feature_inventory(metadata):
    m = _gloss_pair_rows(metadata)
    columns = ["language", "gloss_feature", "feature", "n_sentences_with_feature", "n_glossed_sentences"]
    if m.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for lang, sub in m.groupby("language", sort=True):
        sets = [f for f in sub["_features"] if f is not None]
        for feat in sorted(set().union(*sets)) if sets else []:
            rows.append({"language": canonical_language_code(lang), "gloss_feature": f"{GLOSS_FEATURE_PREFIX}{feat}",
                         "feature": feat, "n_sentences_with_feature": sum(feat in f for f in sets),
                         "n_glossed_sentences": len(sets)})
    return pd.DataFrame(rows, columns=columns)


def join_metadata_features(frame, features):
    out = frame.copy()
    if features is None or features.empty:
        out[METADATA_JOIN_COL] = "no_metadata_table"
        return out
    feat = features.reindex(out[SOURCE_ROW_ID_COL].astype(str))
    found = feat.index.isin(features.index)
    match = (feat["_meta_pos"].to_numpy() == out["pos_sentence"].to_numpy()) & (
        feat["_meta_neg"].to_numpy() == out["neg_sentence"].to_numpy())
    ambiguous = feat["_ambiguous"].eq(True).to_numpy()
    status = np.where(~found, "no_metadata_row", np.where(ambiguous, "ambiguous_duplicate",
                                                           np.where(match, "ok", "sentence_mismatch")))
    ok = status == "ok"
    for c in features.columns:
        if c.startswith("_"):
            continue
        values = feat[c].to_numpy(dtype=float)
        out[c] = np.where(ok, values, np.nan)
    out[METADATA_JOIN_COL] = status
    return out


FREQUENCY_TERM = "frequency_contrast"


def main_regression_feature_terms(exclude_freq=False):
    return [t for t, _c, _l in METADATA_FEATURES if not (exclude_freq and t == FREQUENCY_TERM)]


def common_crawl_log10_sizes(languages, path=COMMON_CRAWL_SIZE_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"--glotcc: language-size file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        sizes = json.load(fh)
    if not isinstance(sizes, dict):
        raise ValueError(f"--glotcc: {path} must hold a JSON object (language key -> size)")
    out = {}
    for lang in languages:
        key = COMMON_CRAWL_LANGUAGE_KEYS.get(lang)
        if key is None:
            raise ValueError(f"--glotcc: no Common Crawl key defined for regression language {lang!r} "
                             f"(defined: {COMMON_CRAWL_LANGUAGE_KEYS})")
        if key not in sizes:
            raise KeyError(f"--glotcc: key {key!r} (language {lang!r}) not found in {path}")
        value = sizes[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"--glotcc: size of {key!r} (language {lang!r}) in {path} must be finite and > 0, "
                             f"got {value!r}")
        out[lang] = math.log10(value)
    return out


def main_regression_frame(df, preference_languages, metadata=None, include_identical=False, report=None,
                          exclude_freq=False, glotcc=False):
    langs = parse_preference_languages(preference_languages)
    frame = regression_frame(
        df, include_identical=include_identical, preference_languages=langs, comparison=TARGET_COMPARISON_POS,
        languages=langs, report=report, metric=MAIN_REGRESSION_OUTCOME,
    )
    frame = frame.loc[frame[STRONG_PREF_TERM].notna()]
    features = metadata_feature_frame(metadata) if metadata is not None else pd.DataFrame()
    if exclude_freq and not features.empty:
        features = features.drop(columns=[METADATA_FEATURE_COLUMNS[FREQUENCY_TERM]], errors="ignore")
    frame = join_metadata_features(frame, features)
    for term in main_regression_feature_terms(exclude_freq):
        col = METADATA_FEATURE_COLUMNS[term]
        frame[term] = frame[col] if col in frame.columns else np.nan
    if glotcc:
        codes = frame["language"].map(canonical_language_code)
        sizes = common_crawl_log10_sizes(sorted(set(codes)))
        frame[COMMON_CRAWL_TERM] = codes.map(sizes).astype(float)
    return frame.reset_index(drop=True)


GLOSS_PAIR_KEYS = ["language", SOURCE_ROW_ID_COL, "pos_sentence", "neg_sentence"]


def gloss_regression_frame(df, preference_languages, metadata=None, include_identical=False):
    langs = parse_preference_languages(preference_languages)
    rows = analysis_rows(df, GLOSS_OUTCOME, include_identical=include_identical, preference_languages=langs)
    if not rows.empty:
        rows = rows.loc[(rows["pos"] == TARGET_COMPARISON_POS)
                        & rows["Target language"].map(canonical_language_code).isin(set(langs))]
    rows = add_source_row_id(rows)
    if rows.empty:
        return pd.DataFrame(columns=GLOSS_PAIR_KEYS + ["y", REGRESSION_OUTCOME_COL, REGRESSION_OUTCOME_LABEL_COL,
                                                       "n_models", METADATA_JOIN_COL])
    work = pd.DataFrame({
        "language": rows["Target language"].map(canonical_language_code).astype(str),
        SOURCE_ROW_ID_COL: rows[SOURCE_ROW_ID_COL].astype(str),
        "pos_sentence": rows["pos_sentence"], "neg_sentence": rows["neg_sentence"],
        "model_name": rows["model_name"].astype(str), "y": rows[DELTA_SLOR].astype(float),
    })
    per_model = work.groupby(GLOSS_PAIR_KEYS + ["model_name"], sort=True)["y"].mean().reset_index()
    pairs = per_model.groupby(GLOSS_PAIR_KEYS, sort=True).agg(y=("y", "mean"), n_models=("model_name", "nunique"))
    pairs = pairs.reset_index()
    pairs[REGRESSION_OUTCOME_COL] = GLOSS_OUTCOME
    pairs[REGRESSION_OUTCOME_LABEL_COL] = GLOSS_OUTCOME_LABEL
    features = metadata_feature_frame(metadata) if metadata is not None else pd.DataFrame()
    pairs = join_metadata_features(pairs, features)
    keep = (GLOSS_PAIR_KEYS + ["y", REGRESSION_OUTCOME_COL, REGRESSION_OUTCOME_LABEL_COL, "n_models", METADATA_JOIN_COL]
            + gloss_feature_columns(pairs))
    return pairs[[c for c in keep if c in pairs.columns]]


def gloss_pair_counts(pair_frame):
    cols = gloss_feature_columns(pair_frame)
    out = []
    for lang, sub in pair_frame.groupby("language", sort=True):
        valid = sub[cols].notna().all(axis=1) if cols else pd.Series(False, index=sub.index)
        out.append({"language": lang, "n_pairs": int(len(sub)), "n_pairs_valid_gloss": int(valid.sum())})
    return pd.DataFrame(out, columns=["language", "n_pairs", "n_pairs_valid_gloss"])


def available_metadata_terms(frame):
    return [t for t, _c, _l in METADATA_FEATURES if t in frame.columns and frame[t].notna().any()]


CORE_TERMS = [LOG_SIZE_TERM, IT_TERM, STRONG_PREF_TERM]
FULL_SPEC_LABEL = "full (core + all available metadata)"


def main_regression_specs(feature_terms, full_terms=None):
    full_terms = list(feature_terms) if full_terms is None else list(full_terms)
    specs = [(label, preds) for label, preds in OLS_MODELS]
    specs.append(("core (size + IT + strong preference)", CORE_TERMS))
    specs += [(f"core + {t}", CORE_TERMS + [t]) for t in feature_terms]
    if full_terms:
        specs.append((FULL_SPEC_LABEL, CORE_TERMS + full_terms))
    return specs


def full_specification_terms(frame, terms):
    terms, omitted = list(terms), []
    while terms:
        complete = frame.dropna(subset=["y"] + CORE_TERMS + terms)
        constant = [t for t in terms if complete[t].nunique() < 2]
        if not constant:
            break
        omitted += [f"{t} (constant in full-specification sample)" for t in constant]
        terms = [t for t in terms if t not in constant]
    return terms, omitted


REGRESSION_COEF_COLUMNS = [
    REGRESSION_OUTCOME_COL, REGRESSION_OUTCOME_LABEL_COL,
    "scope", "specification", "term", "beta", "standardized_beta", "std_error", "t_value", "p_value",
    "ci_lower", "ci_upper", "n", "r_squared", "adj_r_squared", "n_models",
]
REGRESSION_SUMMARY_COLUMNS = [
    REGRESSION_OUTCOME_COL, REGRESSION_OUTCOME_LABEL_COL,
    "scope", "specification", "predictors", "language_fixed_effects", "languages", "n", "n_dropped_missing_predictor",
    "n_models", "models", "n_source_contrasts", "r_squared", "adj_r_squared", "features_omitted_in_scope",
]


def fit_specification(frame, predictors, language_fe=False):
    needed = ["y"] + list(predictors)
    complete = frame.dropna(subset=needed)
    columns = [np.ones(len(complete))]
    terms = ["intercept"]
    for p in predictors:
        columns.append(complete[p].to_numpy(dtype=float))
        terms.append(p)
    fe_levels = sorted(complete["language"].unique()) if language_fe and len(complete) else []
    for level in fe_levels[1:]:
        columns.append((complete["language"] == level).to_numpy(dtype=float))
        terms.append(f"{LANGUAGE_FE_PREFIX}{level}")
    X = np.column_stack(columns) if len(complete) else np.empty((0, len(terms)))
    y = complete["y"].to_numpy(dtype=float)
    rows = fit_ols(y, X, terms)
    sd_y = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
    for row, j in zip(rows, range(len(terms))):
        sd_x = float(np.std(X[:, j], ddof=1)) if len(y) > 1 else np.nan
        row["standardized_beta"] = (row["beta"] * sd_x / sd_y) if j > 0 and sd_y and sd_y > 0 else np.nan
        row["n_models"] = int(complete["model_name"].nunique())
    summary = {
        "predictors": " + ".join(predictors),
        "language_fixed_effects": bool(fe_levels[1:]),
        "languages": ",".join(sorted(complete["language"].unique())),
        "n": int(len(complete)),
        "n_dropped_missing_predictor": int(len(frame) - len(complete)),
        "n_models": int(complete["model_name"].nunique()),
        "models": ",".join(sorted(complete["model_name"].unique())),
        "n_source_contrasts": int(complete[SOURCE_ROW_ID_COL].nunique()) if SOURCE_ROW_ID_COL in complete else 0,
        "r_squared": rows[0]["r_squared"] if rows else np.nan,
        "adj_r_squared": rows[0]["adj_r_squared"] if rows else np.nan,
    }
    return rows, summary


def fit_regression_specs(frame, specs, scope, language_fe, omitted="", outcome=MAIN_REGRESSION_OUTCOME):
    out = {REGRESSION_OUTCOME_COL: outcome, REGRESSION_OUTCOME_LABEL_COL: score_metric(outcome)["delta"]}
    coefs, summaries = [], []
    for label, predictors in specs:
        rows, summary = fit_specification(frame, predictors, language_fe=language_fe)
        coefs += [{**out, "scope": scope, "specification": label, **r} for r in rows]
        summaries.append({**out, "scope": scope, "specification": label, **summary,
                          "features_omitted_in_scope": omitted})
    return (pd.DataFrame(coefs, columns=REGRESSION_COEF_COLUMNS),
            pd.DataFrame(summaries, columns=REGRESSION_SUMMARY_COLUMNS))


def usable_feature_terms(frame, terms):
    usable, omitted = [], []
    for t in terms:
        values = frame[t].dropna() if t in frame.columns else pd.Series(dtype=float)
        if values.empty:
            omitted.append(f"{t} (no values)")
        elif values.nunique() < 2:
            omitted.append(f"{t} (constant)")
        else:
            usable.append(t)
    return usable, omitted


def run_main_regression(df, preference_languages, metadata=None, include_identical=False, exclude_freq=False,
                        glotcc=False):
    report = {}
    frame = main_regression_frame(df, preference_languages, metadata, include_identical, report=report,
                                  exclude_freq=exclude_freq, glotcc=glotcc)
    #dslor, model size, instruction tuning, human preference etcなどを揃える
    if len(frame) and not (frame[REGRESSION_OUTCOME_COL] == MAIN_REGRESSION_OUTCOME).all():
        raise AssertionError(f"main regression outcome must be {MAIN_REGRESSION_OUTCOME}")

    candidate_terms = main_regression_feature_terms(exclude_freq)
    excluded = [t for t, _c, _l in METADATA_FEATURES if t not in candidate_terms]
    available = [t for t in available_metadata_terms(frame) if t in candidate_terms]
    missing = [t for t in candidate_terms if t not in available]

    full_terms, constant_omitted = full_specification_terms(frame, available)
    predictors = CORE_TERMS + ([COMMON_CRAWL_TERM] if glotcc else []) + full_terms
    specification = "full multivariable model"
    coefs, summary = fit_regression_specs(
        frame,
        [(specification, predictors)],
        "pooled",
        language_fe=not glotcc,
        omitted="; ".join(constant_omitted),
        outcome=MAIN_REGRESSION_OUTCOME,
    )

    report["metadata_join_status"] = frame[METADATA_JOIN_COL].value_counts().to_dict() if len(frame) else {}
    report["metadata_join_not_ok_by_model"] = (
        frame.loc[frame[METADATA_JOIN_COL] != "ok"].groupby("model_name").size().to_dict() if len(frame) else {}
    )
    report["feature_non_missing"] = {t: int(frame[t].notna().sum()) for t in candidate_terms}
    report["main_predictors"] = predictors
    report["constant_unidentifiable_in_joint_sample"] = constant_omitted
    report["features_excluded_by_option"] = excluded
    if glotcc:
        report["language_size_predictor_note"] = COMMON_CRAWL_CAVEAT

    return {
        "frame": frame,
        "coefficients": coefs,
        "summary": summary,
        "features_used": full_terms,
        "features_missing": missing,
        "features_excluded": excluded,
        "outcome": MAIN_REGRESSION_OUTCOME,
        "outcome_label": MAIN_REGRESSION_OUTCOME_LABEL,
        "report": report,
        "language_fe": not glotcc,
    }


REGRESSION_FAMILIES = [
    ("aya", "Aya", ("Aya Expanse", "Tiny Aya")),
    ("eurollm", "EuroLLM", ("EuroLLM",)),
    ("gemma", "Gemma 4", ("Gemma 4",)),
    ("apertus", "Apertus", ("Apertus",)),
]

FAMILY_REGRESSION_SUMMARY_COLUMNS = [
    REGRESSION_OUTCOME_COL, REGRESSION_OUTCOME_LABEL_COL, "family", "family_label", "registry_families",
    "registered_models", "registered_models_without_rows", "models", "n_models", "n_models_it", "n_models_non_it",
    "model_sizes_b", "n_rows_before_complete_case", "n", "n_dropped_missing_predictor", "n_unique_sentence_pairs",
    "languages", "predictors_intended", "predictors_included", "predictors_omitted", "missing_by_predictor",
    "language_fixed_effects", "design_rank", "design_columns", "rank_deficient", "r_squared", "adj_r_squared",
    "status",
]


def registered_family_models(registry_families):
    from model_metadata2 import _ENTRIES

    return [m.model_id for m in _ENTRIES if m.family in set(registry_families)]


def identifiable_terms(frame, terms):
    kept, omitted = list(terms), []
    while kept:
        complete = frame.dropna(subset=["y"] + kept)
        if complete.empty:
            break
        constant = [t for t in kept if complete[t].nunique() < 2]
        if not constant:
            break
        omitted += [f"{t} (constant in family complete-case sample: value {complete[t].iloc[0]:g})"
                    for t in constant]
        kept = [t for t in kept if t not in constant]
    return kept, omitted

#論文のsec5.2のregressionのコード
def run_family_regressions(df, preference_languages, metadata=None, include_identical=False, exclude_freq=True,
                           families=None, glotcc=False):
    families = list(REGRESSION_FAMILIES if families is None else families)
    language_fe = not glotcc
    report = {}
    base = main_regression_frame(df, preference_languages, metadata, include_identical, report=report,
                                 exclude_freq=exclude_freq, glotcc=glotcc)
    if len(base) and not (base[REGRESSION_OUTCOME_COL] == MAIN_REGRESSION_OUTCOME).all():
        raise AssertionError(f"main regression outcome must be {MAIN_REGRESSION_OUTCOME}")
    if len(base) and (base["comparison"] != TARGET_COMPARISON_POS).any():
        raise AssertionError("family regressions must contain target-language comparisons only")
    base = base.assign(_registry_family=_model_property(base, "family"))
    candidate = main_regression_feature_terms(exclude_freq)
    excluded = [t for t, _c, _l in METADATA_FEATURES if t not in candidate]
    intended = CORE_TERMS + ([COMMON_CRAWL_TERM] if glotcc else []) + candidate

    frames, coef_tables, summaries, membership = [], [], [], []
    for slug, label, reg_families in families:
        registered = registered_family_models(reg_families)
        out_names = {m: m.replace("/", "_", 1) for m in registered}
        frame = base.loc[base["_registry_family"].isin(reg_families)].drop(columns="_registry_family")
        present = set(models_in(frame))
        meta = model_metadata_frame(sorted(present)) if present else None
        for m in registered:
            name = out_names[m]
            membership.append({
                "family": slug, "family_label": label, "model_id": m, "model_name": name,
                "registry_family": lookup_model(m).family, "instruction_tuned": lookup_model(m).instruction_tuned,
                "model_size_b": lookup_model(m).model_size_b,
                "n_regression_rows": int((frame["model_name"] == name).sum()) if len(frame) else 0,
                "in_regression": name in present,
            })
        missing_by = {t: int(frame[t].isna().sum()) if t in frame.columns else int(len(frame)) for t in intended}
        no_values = [t for t in candidate if t not in frame.columns or frame[t].notna().sum() == 0]
        terms = [t for t in intended if t not in no_values]
        kept, omitted = identifiable_terms(frame, terms)
        omitted = [f"{t} (no values in family sample)" for t in no_values] + omitted
        omitted += [f"{t} (excluded by option: corpus frequency)" for t in excluded]
        status = "ok"
        if frame.empty:
            status = "no_observations"
        rows, summary = fit_specification(frame, kept, language_fe=language_fe)
        complete = frame.dropna(subset=["y"] + kept)
        n_fe = max(complete["language"].nunique() - 1, 0) if len(complete) and language_fe else 0
        k = 1 + len(kept) + n_fe
        X_rank = None
        if len(complete):
            cols = [np.ones(len(complete))] + [complete[t].to_numpy(dtype=float) for t in kept]
            if language_fe:
                cols += [(complete["language"] == lv).to_numpy(dtype=float)
                         for lv in sorted(complete["language"].unique())[1:]]
            X_rank = int(np.linalg.matrix_rank(np.column_stack(cols)))
        rank_def = bool(X_rank is not None and X_rank < k)
        if status == "ok" and rank_def:
            status = "rank_deficient (coefficients not identified; reported as NA)"
        elif status == "ok" and len(complete) <= k:
            status = "insufficient_observations"
        outcome = {REGRESSION_OUTCOME_COL: MAIN_REGRESSION_OUTCOME, REGRESSION_OUTCOME_LABEL_COL: DELTA_SLOR}
        coef_tables.append(pd.DataFrame([{**outcome, "scope": f"family={slug}", "family": slug, "family_label": label,
                                          "specification": "full multivariable model (family-specific)", **r}
                                         for r in rows]))
        fam_meta = meta.loc[sorted(set(complete["model_name"]))] if meta is not None and len(complete) else None
        summaries.append({
            **outcome, "family": slug, "family_label": label, "registry_families": ",".join(reg_families),
            "registered_models": ",".join(out_names[m] for m in registered),
            "registered_models_without_rows": ",".join(out_names[m] for m in registered if out_names[m] not in present),
            "models": summary["models"], "n_models": summary["n_models"],
            "n_models_it": int(fam_meta["instruction_tuned"].eq(True).sum()) if fam_meta is not None else 0,
            "n_models_non_it": int(fam_meta["instruction_tuned"].eq(False).sum()) if fam_meta is not None else 0,
            "model_sizes_b": ",".join(f"{v:g}" for v in sorted(set(fam_meta["model_size_b"]))) if fam_meta is not None else "",
            "n_rows_before_complete_case": int(len(frame)), "n": summary["n"],
            "n_dropped_missing_predictor": summary["n_dropped_missing_predictor"],
            "n_unique_sentence_pairs": int(complete[["language", SOURCE_ROW_ID_COL, "pos_sentence", "neg_sentence"]]
                                           .drop_duplicates().shape[0]) if len(complete) else 0,
            "languages": summary["languages"], "predictors_intended": " + ".join(intended),
            "predictors_included": " + ".join(kept), "predictors_omitted": "; ".join(omitted),
            "missing_by_predictor": ", ".join(f"{t}={n}" for t, n in missing_by.items()),
            "language_fixed_effects": summary["language_fixed_effects"], "design_rank": X_rank,
            "design_columns": k, "rank_deficient": rank_def,
            "r_squared": summary["r_squared"], "adj_r_squared": summary["adj_r_squared"], "status": status,
        })
        frames.append(frame.assign(family=slug))
    coefs = pd.concat(coef_tables, ignore_index=True) if coef_tables else pd.DataFrame()
    if glotcc:
        report["language_size_predictor_note"] = COMMON_CRAWL_CAVEAT
    return {
        "frame": pd.concat(frames, ignore_index=True) if frames else base.iloc[0:0],
        "coefficients": coefs,
        "summary": pd.DataFrame(summaries, columns=FAMILY_REGRESSION_SUMMARY_COLUMNS),
        "membership": pd.DataFrame(membership),
        "features_excluded": excluded,
        "report": report,
        "outcome": MAIN_REGRESSION_OUTCOME,
        "language_fe": language_fe,
    }


GLOSS_REGRESSION_COLUMNS = [
    REGRESSION_OUTCOME_COL, REGRESSION_OUTCOME_LABEL_COL, "language", "gloss_feature", "specification", "design_terms", "status", "intercept", "beta", "standardized_beta",
    "standard_error", "t_value", "p_value", "ci_lower", "ci_upper", "n", "n_minus1", "n_zero", "n_plus1",
    "N_contrast", "models_per_pair_min", "models_per_pair_max", "r_squared", "rank",
]
GLOSS_OUTCOME_LABEL = "pair_mean_delta_SLOR"
GLOSS_OUTCOME = SLOR_COL
GLOSS_MIN_CONTRAST = 20
GLOSS_STATUS_OK = "ok"
GLOSS_STATUS_NO_VARIATION = "no_variation"
GLOSS_STATUS_INSUFFICIENT = "insufficient_contrast"


def gloss_design(frame, feature):
    complete = frame.dropna(subset=["y", feature])
    X = np.column_stack([np.ones(len(complete)), complete[feature].to_numpy(dtype=float)]) if len(complete) \
        else np.empty((0, 2))
    return complete, complete["y"].to_numpy(dtype=float), X, ["intercept", feature]


def run_gloss_feature_regressions(frame, languages=None, inventory=None, outcome_label=GLOSS_OUTCOME_LABEL,
                                  min_contrast=GLOSS_MIN_CONTRAST):
    features = gloss_feature_columns(frame)
    if languages is not None:
        wanted = set(parse_preference_languages(languages))
        frame = frame.loc[frame["language"].map(canonical_language_code).isin(wanted)]
    if not features:
        return pd.DataFrame(columns=GLOSS_REGRESSION_COLUMNS), (
            "skipped: no gloss-derived feature columns "
            f"('{GLOSS_FEATURE_PREFIX}*'): the metadata has no usable '{GLOSS_COLUMN}' annotations for the "
            "target-language pairs"
        )
    by_lang = None
    if inventory is not None and len(inventory):
        by_lang = inventory.groupby("language")["gloss_feature"].apply(list).to_dict()
    rows = []
    for lang in sorted(frame["language"].unique()):
        sub = frame.loc[frame["language"] == lang]
        if by_lang is not None:
            lang_features = [f for f in by_lang.get(lang, []) if f in sub.columns]
        else:
            lang_features = [f for f in features if (sub[f].fillna(0) != 0).any()]
        for feat in lang_features:
            complete, y, X, terms = gloss_design(sub, feat)
            if not len(complete):
                continue
            g = X[:, 1]
            base = {
                REGRESSION_OUTCOME_COL: GLOSS_OUTCOME, REGRESSION_OUTCOME_LABEL_COL: outcome_label,
                "language": lang, "gloss_feature": feat,
                "specification": f"{outcome_label} ~ 1 + {feat} (univariate)",
                "design_terms": " + ".join(terms), "n": int(len(complete)),
                "models_per_pair_min": int(complete["n_models"].min()) if "n_models" in complete else np.nan,
                "models_per_pair_max": int(complete["n_models"].max()) if "n_models" in complete else np.nan,
                "n_plus1": int((g == 1).sum()), "n_zero": int((g == 0).sum()), "n_minus1": int((g == -1).sum()),
                "N_contrast": int((g != 0).sum()),
            }
            if np.unique(g).size < 2:
                rows.append({**base, "status": GLOSS_STATUS_NO_VARIATION})
                continue
            if base["N_contrast"] < min_contrast:
                rows.append({**base, "status": GLOSS_STATUS_INSUFFICIENT})
                continue
            coef = fit_ols(y, X, terms)
            icpt, r = coef[0], coef[1]
            sd_y = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
            sd_x = float(np.std(g, ddof=1)) if len(y) > 1 else np.nan
            rows.append({
                **base, "status": GLOSS_STATUS_OK, "intercept": icpt["beta"], "beta": r["beta"],
                "standardized_beta": (r["beta"] * sd_x / sd_y) if sd_y and sd_y > 0 else np.nan,
                "standard_error": r["std_error"], "t_value": r["t_value"], "p_value": r["p_value"],
                "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"], "r_squared": r["r_squared"],
            })
    table = pd.DataFrame(rows, columns=GLOSS_REGRESSION_COLUMNS[:-1])
    if table.empty:
        return pd.DataFrame(columns=GLOSS_REGRESSION_COLUMNS), "skipped: gloss features have no values for the selected rows"
    table["rank"] = table.groupby("language")["standardized_beta"].transform(
        lambda v: v.abs().rank(ascending=False, method="min"))
    table = table.sort_values(["language", "rank", "gloss_feature"], na_position="last").reset_index(drop=True)
    n_ok = int((table["status"] == GLOSS_STATUS_OK).sum())
    per_lang = ", ".join(
        f"{l}={int((t['status'] == GLOSS_STATUS_OK).sum())} fitted/"
        f"{int((t['status'] == GLOSS_STATUS_INSUFFICIENT).sum())} insufficient_contrast/"
        f"{int((t['status'] == GLOSS_STATUS_NO_VARIATION).sum())} no_variation"
        for l, t in table.groupby("language"))
    return table[GLOSS_REGRESSION_COLUMNS], (
        f"ok: {n_ok} univariate OLS fits (intercept + one gloss feature; N_contrast >= {min_contrast}; {per_lang})")


WIN_RATE_SIZE_FIELD = "total_parameters_b"
WIN_RATE_COLUMNS = ["model_name", "display_name", "family", "instruction_tuned", "model_size",
                    "model_size_definition", "N_total", "N_win", "N_tie", "N_loss", "win_rate", "win_rate_percent"]


def model_size_win_rate_frame(df, preference_languages=None, include_identical=False):
    rows = analysis_rows(df, SLOR_COL, include_identical=include_identical, preference_languages=preference_languages)
    if not rows.empty:
        rows = rows.loc[rows["pos"] == TARGET_COMPARISON_POS]
    if rows.empty:
        return pd.DataFrame(columns=WIN_RATE_COLUMNS), []
    meta = model_metadata_frame(rows["model_name"].unique())
    out, unresolved = [], []
    for model, g in rows.groupby("model_name", sort=True):
        m = meta.loc[str(model)]
        size = m[WIN_RATE_SIZE_FIELD]
        if size is None or pd.isna(size) or m["family"] is None or m["instruction_tuned"] not in (True, False):
            unresolved.append(str(model))
            continue
        y = g[DELTA_SLOR].to_numpy(dtype=float)
        n, n_win, n_tie = int(y.size), int((y > 0).sum()), int((y == 0).sum())
        rate = n_win / n if n else float("nan")
        out.append({"model_name": str(model), "display_name": m["display_name"], "family": m["family"],
                    "instruction_tuned": bool(m["instruction_tuned"]), "model_size": float(size),
                    "model_size_definition": f"{WIN_RATE_SIZE_FIELD} (billions; total parameters, incl. MoE)",
                    "N_total": n, "N_win": n_win, "N_tie": n_tie, "N_loss": int((y < 0).sum()),
                    "win_rate": rate, "win_rate_percent": 100.0 * rate})
    return pd.DataFrame(out, columns=WIN_RATE_COLUMNS), unresolved


def family_colors():
    from model_metadata2 import _ENTRIES

    out, seen = {}, {}
    for f in dict.fromkeys(e.family for e in _ENTRIES):
        paper = PAPER_FAMILY_OF_REGISTRY_FAMILY.get(f, f)
        k = seen.get(paper, 0)
        seen[paper] = k + 1
        out[f] = family_color(paper) if k == 0 else tint(family_color(paper), 0.4 * k)
    return out


MAIN_MULTILINGUAL_MODELS = [
    ("utter-project_EuroLLM-22B-2512", "EuroLLM 22B"),
    ("google_gemma-4-31B", "Gemma 4 31B"),
    ("CohereLabs_aya-expanse-32b", "Aya Expanse 32B"),
    ("swiss-ai_Apertus-70B-2509", "Apertus 70B"),
]
MAIN_MONOLINGUAL_SET_A = [
    ("OpenLLM-France_Lucie-7B", "Lucie 7B"),
    ("maritaca-ai_sabia-7b", "Sabiá 7B"),
    ("hfl_chinese-llama-2-13b", "Chinese-LLaMA-2 13B"),
    ("llm-jp_llm-jp-3-13b", "LLM-jp-3 13B"),
]
MAIN_MANACA_MODEL = "AIInstituteLNCC_manaca-1b-base"
MAIN_SABIA_MODEL = "maritaca-ai_sabia-7b"
MAIN_MONOLINGUAL_SET_A_MANACA = [(MAIN_MANACA_MODEL, "Manacá 1B") if m == MAIN_SABIA_MODEL else (m, l)
                                 for m, l in MAIN_MONOLINGUAL_SET_A]
MAIN_MONOLINGUAL_SET_B = [
    ("OpenLLM-France_Claire-Mistral-7B-0.1", "Claire-Mistral 7B"),
    ("recogna-nlp_bode-13b-alpaca-pt-br-no-peft", "Bode 13B"),
    ("hfl_chinese-llama-2-13b", "Chinese-LLaMA-2 13B"),
    ("tokyotech-llm_Swallow-13b-hf", "Swallow 13B"),
]
MAIN_VIOLIN_SETS = [
    ("multilingual", MAIN_MULTILINGUAL_MODELS),
    ("monolingual-setA", MAIN_MONOLINGUAL_SET_A),
    ("monolingual-setB", MAIN_MONOLINGUAL_SET_B),
]
MAIN_SPEARMAN_MODELS = [
    ("utter-project_EuroLLM-22B-2512", "EuroLLM 22B"),
    ("google_gemma-4-31B", "Gemma 4 31B"),
    ("CohereLabs_aya-expanse-32b", "Aya Expanse 32B"),
    ("swiss-ai_Apertus-70B-2509", "Apertus 70B"),
    ("OpenLLM-France_Lucie-7B", "Lucie 7B"),
    ("llm-jp_llm-jp-3-13b", "LLM-jp-3 13B"),
]
MAIN_SPEARMAN_MODELS_MANACA = [(MAIN_MANACA_MODEL, "Manacá 1B") if m == "OpenLLM-France_Lucie-7B" else (m, l)
                               for m, l in MAIN_SPEARMAN_MODELS]
MAIN_ALL_MODELS = [(failure_workbook_model_key(m), m) for m in STAT_MODELS]
MAIN_VIOLIN_PREFERENCE_SUBSETS = [("gtgt", ">>"), ("gt", ">")]


def resolve_main_models(df, models):
    loaded = set(models_in(df))
    present = [m for m, _l in models if m in loaded]
    missing = [m for m, _l in models if m not in loaded]
    return present, missing
