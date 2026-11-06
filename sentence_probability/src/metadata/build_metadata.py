from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from data_loading import canonical_language_code
from metadata import PIPELINE_VERSION
from metadata import alignment, corpus_frequency, divergence, gloss
from metadata.cache import FeatureCache, canonical_json
from metadata.expand import build_sentence_table
from metadata.normalize import NORMALIZATION_VERSION
from metadata.schema import load_source
from metadata.validation import validate

log = logging.getLogger("metadata")

DEFAULT_INPUT = Path("./data/20261001construction_data_expansion_preprocessed.xlsx")


def default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_preprocessed"):
        stem = stem[: -len("_preprocessed")]
    return input_path.with_name(f"{stem}_metadata.xlsx")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--sheet", default=None, help="worksheet name (default: auto-detect by schema)")
    p.add_argument("--output", type=Path, default=None,
                   help="default: <input stem without _preprocessed>_metadata.xlsx next to the input")
    p.add_argument("--cache", type=Path, default=None,
                   help="SQLite cache (default: metadata_cache.sqlite next to the input)")

    g = p.add_argument_group("FineWeb2 corpus frequency (opt-in)")
    g.add_argument("--fineweb2", action="store_true",
                   help="stream FineWeb2 (5 GiB of text per language) for uncached sentences")
    g.add_argument("--fineweb2-languages", nargs="+", default=None,
                   help="restrict scanning to these dataset language codes (e.g. de fr pt sv zh ja en; se = sv)")
    g.add_argument("--fineweb2-boundary", choices=sorted(corpus_frequency.BOUNDARY_RULES), default="sentence")

    g = p.add_argument_group("Stanza + Awesome-Align (opt-in)")
    g.add_argument("--align", action="store_true", help="compute uncached alignments (downloads models)")
    g.add_argument("--align-checkpoint", choices=sorted(alignment.CHECKPOINTS), default=alignment.DEFAULT_CHECKPOINT,
                   help="awesome-without-co (default; official Drive archive, pass its unzipped dir via "
                        "--align-model), awesome-with-co (pinned HF mirror), mbert (no fine-tuning)")
    g.add_argument("--align-model", default=None, help="local checkpoint directory")
    g.add_argument("--align-model-sha256", default=None,
                   help="expected SHA-256 of the local checkpoint's weights file (optional pin)")
    g.add_argument("--align-device", default="auto", help="auto, cpu, cuda, cuda:0 ...")
    g.add_argument("--stanza-dir", default=None, help="Stanza resources directory")

    g = p.add_argument_group("Gloss annotation (opt-in)")
    g.add_argument("--gloss", nargs="+", default=None, metavar="LANG",
                   help="sentence languages to gloss (dataset codes, e.g. en de ja se; or 'all')")
    g.add_argument("--gloss-scheme", type=Path, default=None,
                   help="default: gloss_annotation_scheme.txt next to the input")
    g.add_argument("--gloss-model", default=gloss.GlossSettings.model)
    g.add_argument("--gloss-temperature", type=float, default=None)
    g.add_argument("--gloss-batch-size", type=int, default=gloss.GlossSettings.batch_size)
    g.add_argument("--gloss-max-attempts", type=int, default=gloss.GlossSettings.max_attempts)

    p.add_argument("--incomplete_preference_annotation", action="store_true",
                   help="divergence eligibility ignores 'English preference annotation' (only the target-language "
                        "annotation must be > or >>); the proper English realization is still always included")

    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def write_workbook_atomic(path: Path, sheets: Dict[str, pd.DataFrame]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp.xlsx", dir=str(path.parent))
    os.close(fd)
    try:
        with pd.ExcelWriter(tmp, engine="openpyxl") as xw:
            for name, df in sheets.items():
                df.to_excel(xw, sheet_name=name, index=False)
                ws = xw.sheets[name]
                ws.freeze_panes = "A2"
                if len(df.columns):
                    ws.auto_filter.ref = ws.dimensions
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _status_counts(table: pd.DataFrame, col: str) -> str:
    vc = table[col].value_counts(dropna=False)
    return ", ".join(f"{k}={v}" for k, v in vc.items())


def run(args: argparse.Namespace) -> Dict[str, object]:
    started = _dt.datetime.now(_dt.timezone.utc)
    input_path = args.input
    output_path = args.output or default_output_path(input_path)
    cache_path = args.cache or input_path.with_name("metadata_cache.sqlite")
    scheme_path = args.gloss_scheme or input_path.with_name("gloss_annotation_scheme.txt")

    src = load_source(input_path, args.sheet)
    table, expand_warnings = build_sentence_table(src.df)
    log.info("source: %d rows -> %d sentence rows", len(src.df), len(table))

    cache = FeatureCache(cache_path)
    try:
        incomplete_pref = bool(getattr(args, "incomplete_preference_annotation", False))
        div_config = divergence.divergence_config(incomplete_pref)
        table, div_audit = divergence.annotate(table, cache, incomplete_preference_annotation=incomplete_pref)

        fineweb2_settings = corpus_frequency.FineWeb2Settings(boundary_mode=args.fineweb2_boundary)
        fineweb2_languages = ([canonical_language_code(c) for c in args.fineweb2_languages]
                              if args.fineweb2_languages else None)
        table, freq_report = corpus_frequency.run_frequency_stage(
            table, cache, fineweb2_settings, run=args.fineweb2, languages=fineweb2_languages)

        align_settings = alignment.AlignmentSettings(
            checkpoint=args.align_checkpoint, expected_weights_sha256=args.align_model_sha256)
        table, align_report = alignment.annotate(
            table, cache, run=args.align, align_settings=align_settings,
            device=args.align_device, stanza_dir=args.stanza_dir, model_path=args.align_model)

        gloss_langs = args.gloss
        if gloss_langs and "all" in gloss_langs:
            gloss_langs = sorted(table["language"].unique())
        elif gloss_langs:
            gloss_langs = list(dict.fromkeys(canonical_language_code(c) for c in gloss_langs))
        gloss_settings = gloss.GlossSettings(
            model=args.gloss_model, batch_size=args.gloss_batch_size,
            max_attempts=args.gloss_max_attempts, temperature=args.gloss_temperature)
        table, gloss_report = gloss.annotate(table, cache, scheme_path, gloss_settings, gloss_langs)

        checks = validate(table, src.df)
        cache_stats = dict(cache.stats)
    finally:
        cache.close()

    unique_sentences = table[["language", "sentence"]].drop_duplicates()
    stages = {
        "divergence": "Strategy Distance + Other-Language Strategy Divergence computed/cached (always on)"
                      + (", incomplete preference annotation mode (English preference annotation ignored)"
                         if incomplete_pref else ""),
        "fineweb2": ("EXECUTED" + (f" for {fineweb2_languages}" if fineweb2_languages else "")
                     if args.fineweb2 else "not executed (cache read only)")
                    + f", budget={fineweb2_settings.budget_bytes} bytes (5 GiB)/language",
        "alignment": "EXECUTED" if args.align else "not executed (cache read only)",
        "gloss": f"EXECUTED for {gloss_langs}" if gloss_langs else "not executed (cache read only)",
    }
    run_info = {
        "pipeline_version": PIPELINE_VERSION,
        "run_started_utc": started.isoformat(timespec="seconds"),
        "input_file": str(input_path),
        "input_sheet": src.sheet,
        "input_sha256": src.file_sha256,
        "cache_file": str(cache_path),
        "normalization_version": NORMALIZATION_VERSION,
        "n_source_rows": len(src.df),
        "n_metadata_rows": len(table),
        "n_unique_language_sentence": len(unique_sentences),
        "n_semantic_content_groups": table["semantic_content_id"].nunique(),
        "source_warnings": " | ".join(src.warnings + expand_warnings),
        "divergence_config": canonical_json(div_config),
        "divergence_incomplete_preference_annotation": incomplete_pref,
        "strategy_distance_columns": "divergence_score, divergence_en_realization_score, divergence_tl_realization_score",
        "other_language_strategy_divergence_definition": divergence.OLSD_DEFINITION,
        "other_language_strategy_divergence_result_schema": divergence.RESULT_SCHEMA,
        "divergence_status_counts": _status_counts(table, "divergence_status"),
        "other_language_strategy_divergence_status_counts": _status_counts(
            table, "other_language_strategy_divergence_status"),
        "fineweb2_settings": canonical_json(fineweb2_settings.__dict__),
        "fineweb2_sources": canonical_json({**{l: [corpus_frequency.FINEWEB2_DATASET, c]
                                                for l, c in corpus_frequency.LANGS.items()},
                                             "en": list(corpus_frequency.ENGLISH_SOURCE)}),
        "fineweb2_report": canonical_json(freq_report),
        "alignment_config": canonical_json(alignment.alignment_config(alignment.StanzaSettings(), align_settings)),
        "alignment_report": canonical_json(align_report),
        "alignment_runtime_versions_installed": canonical_json(alignment.runtime_versions()),
        "gloss_report": canonical_json(gloss_report),
        "cache_stats": canonical_json(cache_stats),
    }
    for k, v in stages.items():
        run_info[f"stage_{k}"] = v

    write_workbook_atomic(output_path, {
        "metadata": table,
        "divergence_groups": div_audit,
        "validation": pd.DataFrame(checks),
        "run_info": pd.DataFrame({"key": list(run_info), "value": [str(v) for v in run_info.values()]}),
    })

    print("\n=== metadata build report ===")
    print(f"input:   {input_path} (sheet {src.sheet!r}, sha256 {src.file_sha256[:12]})")
    print(f"output:  {output_path}")
    print(f"cache:   {cache_path}")
    print(f"rows:    {len(src.df)} source rows -> {len(table)} sentence rows; "
          f"{len(unique_sentences)} unique (language, sentence); "
          f"{table['semantic_content_id'].nunique()} semantic-content groups")
    print("rows by language: " + ", ".join(f"{k}={v}" for k, v in table["language"].value_counts().items()))
    print("rows by experiment: " + ", ".join(f"{k}={v}" for k, v in table["experiment"].value_counts().items()))
    for ns, s in sorted(cache_stats.items()):
        extra = (f" upgraded_to_pairwise={s['upgraded_to_pairwise']}" if "upgraded_to_pairwise" in s else "")
        print(f"cache {ns:14s} hits={s['hits']} misses={s['misses']} writes={s['writes']}{extra}")
    for k, v in stages.items():
        print(f"stage {k:10s} {v}")
    for col in ("divergence_status", "other_language_strategy_divergence_status", "freq_status", "freq_contrast_status", "align_status", "gloss_status"):
        print(f"{col}: {_status_counts(table, col)}")
    print(f"validation: {len(checks)} checks passed")
    if src.warnings or expand_warnings:
        print("warnings: " + " | ".join(src.warnings + expand_warnings))
    return {"table": table, "run_info": run_info, "output": output_path}


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(args)


if __name__ == "__main__":
    main()
