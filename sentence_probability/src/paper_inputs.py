from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loading import (
    EXPERIMENT_STANDARD,
    TARGET_LANGUAGE_CANONICAL_COLUMN,
    canonical_language_code,
    comparison_experiment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs2"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "plots2"
DEFAULT_MAIN_PLOT_DIR = PROJECT_ROOT / "Main_plot"

TRANSLATION_DETAIL_SUFFIX = "__translation_generation.xlsx"
TRANSLATION_SUMMARY_SUFFIX = "__translation_generation_summary.xlsx"
JAPANESE_CONTROL_FILE_TAG = "__japanese_control__"


def _probability_files(outputs_dir):
    files = sorted(f for f in Path(outputs_dir).glob("*/*.xlsx")
                   if not f.name.endswith(TRANSLATION_DETAIL_SUFFIX)
                   and not f.name.endswith(TRANSLATION_SUMMARY_SUFFIX))
    tagged = {f.parent / f.name.replace(JAPANESE_CONTROL_FILE_TAG, "__", 1)
              for f in files if JAPANESE_CONTROL_FILE_TAG in f.name}
    return [f for f in files if f not in tagged], [f for f in files if f in tagged]


def _normalize(df):
    out = df.copy()
    if {"pos", "neg"} <= set(out.columns):
        derived = pd.Series([comparison_experiment(p, n) for p, n in zip(out["pos"], out["neg"])], index=out.index)
    else:
        derived = pd.Series(EXPERIMENT_STANDARD, index=out.index)
    if "experiment" in out.columns:
        saved = out["experiment"]
        blank = saved.isna() | (saved.astype(str).str.strip() == "")
        out["experiment"] = saved.where(~blank, derived)
    else:
        out["experiment"] = derived
    if "Target language" in out.columns:
        if "target_language_source" not in out.columns:
            out["target_language_source"] = out["Target language"]
        canonical = out["Target language"].map(canonical_language_code)
        if TARGET_LANGUAGE_CANONICAL_COLUMN in out.columns:
            saved = out[TARGET_LANGUAGE_CANONICAL_COLUMN]
            blank = saved.isna() | (saved.astype(str).str.strip() == "")
            canonical = saved.where(~blank, canonical)
        out["Target language"] = canonical
    if "lang" in out.columns:
        out["lang"] = out["lang"].map(canonical_language_code)
    return out


def load_probability_outputs(outputs_dir=DEFAULT_OUTPUTS_DIR):
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"outputs directory not found: {outputs_dir}")
    files, superseded = _probability_files(outputs_dir)
    frames, warnings = [], [f"Ignored {f} (superseded by its japanese_control-tagged file)" for f in superseded]
    for f in files:
        try:
            frames.append(pd.read_excel(f, keep_default_na=False, na_values=[]))
        except Exception as e:
            warnings.append(f"Failed to read {f}: {e}")
    if not frames:
        return pd.DataFrame(), warnings + [f"No result xlsx files found under {outputs_dir} (*/*.xlsx)"]
    df = _normalize(pd.concat(frames, ignore_index=True))
    if "Proper English strategy" in df.columns:
        df["_pair_id"] = df["Proper English strategy"]
    else:
        df["_pair_id"] = df.index.astype(str)
    return df, warnings
