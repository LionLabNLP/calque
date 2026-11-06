#!/usr/bin/env python3
"""Compute sentence-level log-probability / SLOR metrics for classified constructional error
candidates.

Reads the classified output produced by `scripts/classify_errors.py` (CSV or Parquet), selects
`pure`/`constructional+others` rows with `stanza_word_edit_distance` <= 5 (a Stanza word-level
Levenshtein distance precomputed once by `scripts/classify_errors.py` and stored in the classified
output -- never recomputed here), scores the reconstructed correction (positive) and learner
sentence (negative) with a local Hugging Face causal-LM, and saves the resulting per-candidate
metrics. Does not require the review SQLite database.

Example:
    python scripts/calculate_sent_prob.py \\
        --input data/processed/full/classified/all_classified.csv \\
        --output data/processed/full/logprob/sentence_probabilities.parquet \\
        --model meta-llama/Llama-3.1-8B \\
        --model-name llama_3_1_8b \\
        --condition "No context"

Resuming a partially completed run:
    python scripts/calculate_sent_prob.py \\
        --input data/processed/full/classified/all_classified.csv \\
        --output data/processed/full/logprob/sentence_probabilities.parquet \\
        --model meta-llama/Llama-3.1-8B \\
        --condition "No context" \\
        --resume

Resume/completion semantics:

Every result is keyed by (candidate_id, --model, --condition) -- the *probability-scoring*
model, never the reconstruction model (`api_model`) that produced the classified candidate,
which this script does not read or depend on at all.

* --resume with the same --model and --condition as a previous run skips only candidates that
  already have a *successful* result for that exact key; a candidate whose previous attempt
  failed is always retried.
* A different --model (e.g. switching from one local Hugging Face model to another, such as
  Aya) is a separate, complete calculation over every eligible candidate from scratch. It never
  treats another model's results as already-completed work for the new model.
* A different --condition is likewise a separate calculation, independent of any other
  condition's results.
* Results for every (candidate_id, model, condition) ever computed into a given --output are
  preserved side by side in that same output file -- an appendable dataset keyed by
  (candidate_id, model, condition) -- so switching --model or --condition and re-running never
  overwrites, relabels, or drops another model's/condition's previously saved results.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from learner_corrections.io import append_jsonl, atomic_write_dataframe, read_jsonl, setup_logging
from learner_corrections.probability import compute_sentence_pair_metrics, to_json_field
from learner_corrections.reproducibility import RunSummary
from learner_corrections.review_app.display_labels import classification_display_label

logger = logging.getLogger("calculate_sent_prob")

ELIGIBLE_CLASSIFICATIONS = frozenset({"pure", "constructional+others"})
MAX_EDIT_DISTANCE = 5

DEFAULT_CONDITION = "No context"

REQUIRED_OUTPUT_COLUMNS = [
    "candidate_id",
    "positive_sentence",
    "negative_sentence",
    "reconstructed_sentence",
    "learner_sentence",
    "token_logprob_diff",
    "SLOR_token_logprob_diff",
    "positive_sentence_logprob",
    "negative_sentence_logprob",
    "positive_avg_token_logprob",
    "negative_avg_token_logprob",
    "SLOR_positive_avg_token_logprob",
    "SLOR_negative_avg_token_logprob",
    "positive_token_count",
    "negative_token_count",
    "classification",
    "error_type",
    "edit_distance",
    "stanza_word_edit_distance",
    "learning_language",
    "learning_code",
    "construction_group_id",
    "construction_group_type",
    "construction_group_signature",
    "construction_group_key",
    "construction_edit_count",
    "model_name",
    "model",
    "condition",
    "prompt",
]

OPTIONAL_DETAIL_COLUMNS = [
    "positive_tokens",
    "negative_tokens",
    "positive_token_surprisals",
    "negative_token_surprisals",
    "pos_diff_tokens",
    "neg_diff_tokens",
    "pos_after_diff_tokens",
    "neg_after_diff_tokens",
]

STATUS_COLUMNS = ["probability_status", "probability_error"]

OUTPUT_COLUMNS = REQUIRED_OUTPUT_COLUMNS + OPTIONAL_DETAIL_COLUMNS + STATUS_COLUMNS

_NUMERIC_METRIC_FIELDS = (
    "token_logprob_diff",
    "SLOR_token_logprob_diff",
    "positive_sentence_logprob",
    "negative_sentence_logprob",
    "positive_avg_token_logprob",
    "negative_avg_token_logprob",
    "SLOR_positive_avg_token_logprob",
    "SLOR_negative_avg_token_logprob",
    "positive_token_count",
    "negative_token_count",
)
_LIST_DETAIL_FIELDS = (
    "positive_tokens",
    "negative_tokens",
    "positive_token_surprisals",
    "negative_token_surprisals",
    "pos_diff_tokens",
    "neg_diff_tokens",
    "pos_after_diff_tokens",
    "neg_after_diff_tokens",
)


def _is_missing_sentence(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return not str(value).strip()


@dataclass
class RowSelection:
    eligible_rows: list[dict] = field(default_factory=list)
    skipped_counts: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped_counts[reason] = self.skipped_counts.get(reason, 0) + 1


def _parse_stanza_word_edit_distance(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


def select_eligible_rows(records: list[dict]) -> RowSelection:
    selection = RowSelection()
    for record in records:
        row = dict(record)

        classification = row.get("classification")
        if classification not in ELIGIBLE_CLASSIFICATIONS:
            selection.skip("classification_not_eligible")
            continue

        parse_status = row.get("parse_status")
        if parse_status not in (None, "", "ok"):
            selection.skip("invalid_parse")
            continue

        learner_sentence = row.get("learner_sentence")
        reconstructed_sentence = row.get("reconstructed_sentence")
        if _is_missing_sentence(learner_sentence) or _is_missing_sentence(reconstructed_sentence):
            selection.skip("missing_sentence")
            continue

        learner_sentence = str(learner_sentence)
        reconstructed_sentence = str(reconstructed_sentence)
        if learner_sentence == reconstructed_sentence:
            selection.skip("identical_sentence")
            continue

        edit_distance = _parse_stanza_word_edit_distance(row.get("stanza_word_edit_distance"))
        if edit_distance is None:
            selection.skip("missing_stanza_word_edit_distance")
            continue
        if edit_distance > MAX_EDIT_DISTANCE:
            selection.skip("edit_distance_too_large")
            continue

        row["learner_sentence"] = learner_sentence
        row["reconstructed_sentence"] = reconstructed_sentence
        row["edit_distance"] = edit_distance
        row["stanza_word_edit_distance"] = edit_distance
        selection.eligible_rows.append(row)

    return selection


def _nan_metric_fields() -> dict[str, Any]:
    fields: dict[str, Any] = {name: float("nan") for name in _NUMERIC_METRIC_FIELDS}
    fields.update({name: to_json_field([]) for name in _LIST_DETAIL_FIELDS})
    return fields


def score_row(
    row: dict,
    *,
    model,
    tokenizer,
    vocab_size: int,
    unigram_stats_by_lang: dict[str, dict],
    flores_langs: frozenset,
    prompt_text: str | None,
    model_arg: str,
    model_name: str,
    condition: str,
) -> dict:
    positive_sentence = row["reconstructed_sentence"]
    negative_sentence = row["learner_sentence"]
    learning_code = row.get("learning_code")

    base = {
        "candidate_id": row["candidate_id"],
        "positive_sentence": positive_sentence,
        "negative_sentence": negative_sentence,
        "reconstructed_sentence": positive_sentence,
        "learner_sentence": negative_sentence,
        "classification": row.get("classification"),
        "error_type": classification_display_label(row.get("classification")),
        "edit_distance": row["edit_distance"],
        "stanza_word_edit_distance": row["stanza_word_edit_distance"],
        "learning_language": row.get("learning_language"),
        "learning_code": learning_code,
        "construction_group_id": row.get("construction_group_id") or "",
        "construction_group_type": row.get("construction_group_type") or "",
        "construction_group_signature": row.get("construction_group_signature") or "",
        "construction_group_key": row.get("construction_group_key") or "",
        "construction_edit_count": row.get("construction_edit_count"),
        "model_name": model_name,
        "model": model_arg,
        "condition": condition,
        "prompt": prompt_text or "",
    }

    if learning_code not in flores_langs:
        return {
            **base,
            **_nan_metric_fields(),
            "probability_status": "error",
            "probability_error": f"unsupported language for SLOR: {learning_code!r}",
        }

    try:
        unigram_stats = unigram_stats_by_lang[learning_code]
        metrics = compute_sentence_pair_metrics(
            model,
            tokenizer,
            positive_sentence,
            negative_sentence,
            prompt_text,
            unigram_stats,
            vocab_size,
        )
    except Exception as exc:
        logger.warning("Scoring failed for candidate_id=%s: %s", row.get("candidate_id"), exc)
        return {
            **base,
            **_nan_metric_fields(),
            "probability_status": "error",
            "probability_error": f"{exc.__class__.__name__}: {exc}",
        }

    return {
        **base,
        "token_logprob_diff": metrics["token_logprob_diff"],
        "SLOR_token_logprob_diff": metrics["SLOR_token_logprob_diff"],
        "positive_sentence_logprob": metrics["positive_sentence_logprob"],
        "negative_sentence_logprob": metrics["negative_sentence_logprob"],
        "positive_avg_token_logprob": metrics["positive_avg_token_logprob"],
        "negative_avg_token_logprob": metrics["negative_avg_token_logprob"],
        "SLOR_positive_avg_token_logprob": metrics["SLOR_positive_avg_token_logprob"],
        "SLOR_negative_avg_token_logprob": metrics["SLOR_negative_avg_token_logprob"],
        "positive_token_count": metrics["positive_token_count"],
        "negative_token_count": metrics["negative_token_count"],
        "positive_tokens": to_json_field(metrics["positive_tokens"]),
        "negative_tokens": to_json_field(metrics["negative_tokens"]),
        "positive_token_surprisals": to_json_field(metrics["positive_token_surprisals"]),
        "negative_token_surprisals": to_json_field(metrics["negative_token_surprisals"]),
        "pos_diff_tokens": to_json_field(metrics["pos_diff_tokens"]),
        "neg_diff_tokens": to_json_field(metrics["neg_diff_tokens"]),
        "pos_after_diff_tokens": to_json_field(metrics["pos_after_diff_tokens"]),
        "neg_after_diff_tokens": to_json_field(metrics["neg_after_diff_tokens"]),
        "probability_status": "ok",
        "probability_error": "",
    }


def resume_key(candidate_id: str, model_arg: str, condition: str) -> str:
    return f"{candidate_id}\x1f{model_arg}\x1f{condition}"


def load_done_keys(checkpoint_path: Path) -> set[str]:
    done: set[str] = set()
    if checkpoint_path.exists():
        for row in read_jsonl(checkpoint_path):
            if row.get("probability_status") == "ok":
                done.add(resume_key(row["candidate_id"], row["model"], row["condition"]))
    return done


def _read_classified(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False, na_values=[""])
    return pd.read_parquet(path)


def _require_stanza_word_edit_distance_column(df: pd.DataFrame, path: Path) -> None:
    if "stanza_word_edit_distance" not in df.columns:
        raise SystemExit(
            f"{path} is missing the required 'stanza_word_edit_distance' column. This classified "
            "file was produced by an older version of scripts/classify_errors.py that computed "
            "edit distance at scoring time instead of during classification. Rerun "
            "scripts/classify_errors.py to regenerate the classified data before running "
            "scripts/calculate_sent_prob.py."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="all_classified.csv/.parquet from scripts/classify_errors.py")
    parser.add_argument("--output", required=True, type=Path, help="Output .parquet (or .csv) path")
    parser.add_argument("--model", required=True, help="Hugging Face model path/identifier used for scoring")
    parser.add_argument("--model-name", default=None, help="Stable display/output label; derived from --model if omitted")
    parser.add_argument("--condition", default=DEFAULT_CONDITION)
    parser.add_argument("--prompt-text", default=None, help="Prompt text for a non-'No context' --condition")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory (passed through to model loading)")
    parser.add_argument("--csv", action="store_true", help="Also write a .csv sibling of --output")
    parser.add_argument("--checkpoint-every-rows", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    interim_dir = output.parent / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output.parent / "logs" / "calculate_sent_prob.log")

    from learner_corrections.model_utils import safe_model_key

    model_name = args.model_name or safe_model_key(args.model)

    if args.condition == DEFAULT_CONDITION:
        prompt_text = None
    else:
        prompt_text = args.prompt_text

    summary = RunSummary(
        command="calculate_sent_prob",
        argv=sys.argv,
        config={
            "model": args.model,
            "model_name": model_name,
            "condition": args.condition,
            "prompt_text": prompt_text,
        },
        seed=None,
    )
    summary.record_inputs({"classified_input": args.input})

    logger.info("Reading classified candidates from %s", args.input)
    df = _read_classified(args.input)
    logger.info("Loaded %d classified row(s)", len(df))
    _require_stanza_word_edit_distance_column(df, args.input)

    selection = select_eligible_rows(df.to_dict(orient="records"))
    logger.info(
        "Row selection: %d eligible, skipped=%s", len(selection.eligible_rows), selection.skipped_counts
    )
    summary.record("row_selection", {"eligible": len(selection.eligible_rows), "skipped": selection.skipped_counts})

    checkpoint_path = interim_dir / f"{output.stem}_checkpoint.jsonl"
    if not args.resume:
        if checkpoint_path.exists() and not args.overwrite:
            raise SystemExit(
                f"A previous checkpoint exists at {checkpoint_path}. Pass --resume to continue "
                "it, or --overwrite to discard it and start fresh."
            )
        if checkpoint_path.exists() and args.overwrite:
            checkpoint_path.unlink()

    done_keys = load_done_keys(checkpoint_path)
    already_done_count = sum(
        1 for row in selection.eligible_rows if resume_key(row["candidate_id"], args.model, args.condition) in done_keys
    )
    pending = [
        row
        for row in selection.eligible_rows
        if resume_key(row["candidate_id"], args.model, args.condition) not in done_keys
    ]
    logger.info(
        "%d row(s) already successfully scored for model=%r condition=%r, %d pending",
        already_done_count,
        args.model,
        args.condition,
        len(pending),
    )

    languages_needed = sorted({row.get("learning_code") for row in pending if row.get("learning_code")})

    if pending:
        from learner_corrections import slor
        from learner_corrections.model_utils import load_model_and_tokenizer

        logger.info("Loading model and tokenizer: %s", args.model)
        tokenizer, model = load_model_and_tokenizer(args.model, cache_dir=args.cache_dir)
        vocab_size = model.config.vocab_size

        unigram_stats_by_lang: dict[str, dict] = {}
        for lang_code in languages_needed:
            if lang_code not in slor.FLORES_LANGS:
                continue
            logger.info("Loading FLORES unigram counts for language: %s", lang_code)
            unigram_stats_by_lang[lang_code] = slor.get_unigram_token_counts(tokenizer, lang_code)

        buffer: list[dict] = []
        n_ok = 0
        n_error = 0
        for i, row in enumerate(pending, start=1):
            result = score_row(
                row,
                model=model,
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                unigram_stats_by_lang=unigram_stats_by_lang,
                flores_langs=slor.FLORES_LANGS,
                prompt_text=prompt_text,
                model_arg=args.model,
                model_name=model_name,
                condition=args.condition,
            )
            buffer.append(result)
            if result["probability_status"] == "ok":
                n_ok += 1
            else:
                n_error += 1

            if len(buffer) >= args.checkpoint_every_rows:
                append_jsonl(checkpoint_path, buffer)
                buffer = []
                logger.info("Scored %d/%d pending row(s)", i, len(pending))

        if buffer:
            append_jsonl(checkpoint_path, buffer)

        summary.record("scoring", {"n_ok": n_ok, "n_error": n_error})
        logger.info("Scoring complete: %d ok, %d error", n_ok, n_error)
    else:
        summary.record("scoring", {"n_ok": 0, "n_error": 0})

    all_rows = list(read_jsonl(checkpoint_path)) if checkpoint_path.exists() else []
    eligible_ids = {row["candidate_id"] for row in selection.eligible_rows}
    by_key: dict[str, dict] = {}
    for row in all_rows:
        if row["candidate_id"] not in eligible_ids:
            continue
        key = resume_key(row["candidate_id"], row["model"], row["condition"])
        existing = by_key.get(key)
        if existing is None or (existing.get("probability_status") != "ok" and row.get("probability_status") == "ok"):
            by_key[key] = row
    kept_rows = sorted(by_key.values(), key=lambda r: (r["candidate_id"], r["model"], r["condition"]))

    result_df = pd.DataFrame(kept_rows)
    if not result_df.empty:
        result_df = result_df[OUTPUT_COLUMNS]
    else:
        result_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    if output.suffix == ".csv":
        formats = ("csv",)
    elif args.csv:
        formats = ("parquet", "csv")
    else:
        formats = ("parquet",)
    atomic_write_dataframe(result_df, output.parent, output.stem, overwrite=args.overwrite or args.resume, formats=formats)

    summary.record_counts(
        {
            "input_rows": len(df),
            "eligible_rows": len(selection.eligible_rows),
            "output_rows": len(result_df),
        }
    )
    summary.write(output.parent / f"{output.stem}_summary.json")

    logger.info("Done. Wrote %d row(s) to %s", len(result_df), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
