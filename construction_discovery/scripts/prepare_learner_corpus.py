#!/usr/bin/env python3
"""Component A CLI: preprocess + filter the learner corpus, reconstruct clean corrections with
either the OpenAI Responses API or an OpenAI-compatible vLLM chat-completions endpoint, and
deduplicate the result -- in one command.

Example (full mode, OpenAI backend):
    python scripts/prepare_learner_corpus.py \\
        --input data/raw/learner_corpus.dat \\
        --output-dir data/processed/full \\
        --model gpt-5.6 --batch-size 25 --seed 42

Example (toy mode, OpenAI backend):
    python scripts/prepare_learner_corpus.py \\
        --input data/raw/learner_corpus.dat \\
        --output-dir data/processed/toy \\
        --model gpt-5.6 --batch-size 10 --toy --toy-size 50 --seed 42

Example (vLLM backend -- selected automatically for model IDs starting with "vllm-" or listed in
the comma-separated VLLM_MODEL_IDS environment variable; the endpoint is read from
VLLM_CHAT_COMPLETIONS_URL or vllm.chat_completions_url and the key from VLLM_API_KEY; the model ID
is sent to the endpoint unchanged, so pass the ID your server exposes for Llama 3.3 70B Instruct FP8):
    VLLM_CHAT_COMPLETIONS_URL=http://localhost:8000/v1/chat/completions \\
    VLLM_MODEL_IDS=nvidia/Llama-3.3-70B-Instruct-FP8 \\
    python scripts/prepare_learner_corpus.py \\
        --input data/lang-8-20111007-L1-v2.dat \\
        --output-dir data/processed/full \\
        --model nvidia/Llama-3.3-70B-Instruct-FP8 \\
        --batch-size 50 --seed 42

Example (extend an existing run with a different model, keeping prior candidates' api_model):
    VLLM_MODEL_IDS=nvidia/Llama-3.3-70B-Instruct-FP8 \\
    python scripts/prepare_learner_corpus.py \\
        --input data/lang-8-20111007-L1-v2.dat \\
        --output-dir data/processed/full \\
        --model nvidia/Llama-3.3-70B-Instruct-FP8 \\
        --batch-size 50 --seed 42 --resume --allow-model-change
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from learner_corrections.config import ALLOWED_PAIRS, JAPANESE_LANGUAGES, load_config
from learner_corrections.correction_reconstruction import (
    VLLM_BACKEND,
    ReconstructionRunResult,
    load_candidate_rows,
    load_failed_correction_ids,
    load_successful_correction_ids,
    reconstruct_corrections,
    resolve_backend,
)
from learner_corrections.deduplication import deduplicate_candidates
from learner_corrections.io import (
    atomic_write_dataframe,
    atomic_write_jsonl,
    guard_overwrite,
    setup_logging,
)
from learner_corrections.learner_preprocessing import preprocess_learner_corpus
from learner_corrections.reproducibility import (
    STABLE_ID_SCHEME_VERSION,
    ReconstructionRunState,
    RunStateIncompatibleError,
    RunSummary,
    load_run_state,
    validate_resume_compatibility,
    write_run_state,
)
from learner_corrections.sampling import (
    SAMPLING_STRATEGY_VERSION,
    compute_sentence_completion,
    downsample_japanese,
    group_by_pair_key,
    sample_toy,
)

logger = logging.getLogger("prepare_learner_corpus")

CANDIDATE_COLUMNS = [
    "candidate_id",
    "source_row_id",
    "sentence_id",
    "correction_id",
    "candidate_type",
    "suggestion_index",
    "learning_language",
    "native_language",
    "learning_code",
    "native_code",
    "learner_sentence",
    "original_tagged_correction",
    "reconstructed_sentence",
    "suggestion_evidence",
    "source_line_number",
    "source_sentence_index",
    "source_correction_index",
    "api_model",
    "api_status",
    "api_raw_response_path",
    "is_toy",
]


def _log_pair_table(pair_stats: dict[str, dict]) -> None:
    header = f"{'Pair':<22}{'Available':>10}{'Target':>8}{'Selected':>10}{'Complete rows':>15}{'Pending corrections':>21}"
    lines = [header]
    for pair_key, stats in sorted(pair_stats.items()):
        available = stats.get("available_sentence_rows")
        available_str = "n/a" if available is None else str(available)
        target = stats.get("target_sentence_rows")
        target_str = "n/a" if target is None else str(target)
        lines.append(
            f"{pair_key:<22}{available_str:>10}{target_str:>8}"
            f"{stats['selected_sentence_rows']:>10}{stats['complete_sentence_rows_before_run']:>15}"
            f"{stats['pending_corrections_before_run']:>21}"
        )
    logger.info("Per-language-pair plan:\n%s", "\n".join(lines))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Path to the raw learner-corpus JSONL (.dat) file")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", action="append", default=[], type=Path, help="Extra YAML config file(s), applied after config/default.yaml (and config/toy.yaml if --toy)")

    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--retry-backoff-seconds", type=float, default=None)
    parser.add_argument("--request-timeout-seconds", type=float, default=None)
    parser.add_argument("--checkpoint-every-batches", type=int, default=None)

    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--toy-size", type=int, default=None)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-api", action="store_true", help="Use only cached reconstruction results; do not call the OpenAI API")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing final (non-checkpoint) outputs")
    parser.add_argument(
        "--allow-model-change",
        action="store_true",
        help="Allow --resume to continue an existing checkpoint even if --model differs from the "
        "model recorded in the previous run (candidate rows keep their original api_model).",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    interim_dir = output_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / "logs" / "prepare_learner_corpus.log")

    overrides = {
        "openai.model": args.model,
        "openai.batch_size": args.batch_size,
        "random_seed": args.seed,
        "openai.reasoning_effort": args.reasoning_effort,
        "openai.max_retries": args.max_retries,
        "openai.retry_backoff_seconds": args.retry_backoff_seconds,
        "openai.request_timeout_seconds": args.request_timeout_seconds,
        "openai.checkpoint_every_batches": args.checkpoint_every_batches,
        "learner_corpus.toy_size_per_pair": args.toy_size,
    }
    config = load_config(extra_config_paths=args.config, toy=args.toy, overrides=overrides)

    seed = config["random_seed"]
    model = config["openai"]["model"]
    backend = resolve_backend(model)
    batch_size = config["learner_corpus"]["batch_size"]
    reasoning_effort = config["openai"]["reasoning_effort"]
    max_retries = config["openai"]["max_retries"]
    retry_backoff_seconds = config["openai"]["retry_backoff_seconds"]
    checkpoint_every_batches = config["openai"]["checkpoint_every_batches"]
    toy_size_per_pair = config["learner_corpus"]["toy_size_per_pair"]
    vllm_url = config["vllm"].get("chat_completions_url")
    vllm_api_key_env = config["vllm"]["api_key_env"]
    vllm_max_tokens = config["vllm"]["max_tokens"]

    if args.request_timeout_seconds is not None:
        request_timeout_seconds = args.request_timeout_seconds
    elif backend == VLLM_BACKEND:
        request_timeout_seconds = config["vllm"]["request_timeout_seconds"]
    else:
        request_timeout_seconds = config["openai"]["request_timeout_seconds"]

    logger.info("Reconstruction model: %s (backend=%s)", model, backend)

    summary = RunSummary(command="prepare_learner_corpus", argv=sys.argv, config=config, seed=seed)
    summary.record_inputs({"learner_corpus": args.input})

    final_overwrite = args.overwrite or args.resume or args.skip_api

    logger.info("Parsing and filtering %s", args.input)
    parsed = preprocess_learner_corpus(args.input)

    logger.info("Language pairs seen (raw): %s", dict(parsed.pair_counts_seen))
    logger.info("Language pairs retained (allowed, >=1 surviving correction): %s", dict(parsed.pair_counts_allowed_rows))

    malformed_path = output_dir / "malformed_input_rows.jsonl"
    guard_overwrite(malformed_path, final_overwrite)
    atomic_write_jsonl(malformed_path, parsed.malformed_rows)

    removed_df = pd.DataFrame(parsed.removed_corrections)
    atomic_write_dataframe(removed_df, output_dir, "removed_initial_filters", overwrite=final_overwrite, formats=("csv",))

    sentence_rows, corrections, ja_log = downsample_japanese(
        parsed.sentence_rows, parsed.retained_corrections, seed=seed
    )
    summary.record("japanese_downsampling", ja_log)

    if args.toy:
        sentence_rows, corrections, toy_log = sample_toy(
            sentence_rows, corrections, toy_size_per_pair=toy_size_per_pair, seed=seed
        )
        summary.record("toy_sampling", toy_log)

    logger.info("Sentence rows selected for reconstruction: %d", len(sentence_rows))
    logger.info("Corrections selected for reconstruction: %d", len(corrections))

    checkpoint_path = interim_dir / "reconstruction_checkpoint.jsonl"
    failures_path = output_dir / "api_failures.jsonl"
    raw_responses_dir = interim_dir / "api_raw_responses"
    run_state_path = interim_dir / "reconstruction_run_state.json"

    if not args.resume and not args.skip_api:
        stale = checkpoint_path.exists() or failures_path.exists()
        if stale and not args.overwrite:
            raise SystemExit(
                f"A previous reconstruction checkpoint exists at {checkpoint_path} (or "
                f"{failures_path}). Pass --resume to continue it, or --overwrite to discard it "
                "and start a fresh reconstruction run."
            )
        if stale and args.overwrite:
            import shutil

            for p in (checkpoint_path, failures_path, run_state_path):
                if p.exists():
                    p.unlink()
            if raw_responses_dir.exists():
                shutil.rmtree(raw_responses_dir)

    current_run_state = ReconstructionRunState(
        input_path=str(args.input),
        input_sha256=summary.data["inputs"]["learner_corpus"]["sha256"],
        seed=seed,
        model=model,
        sampling_strategy_version=SAMPLING_STRATEGY_VERSION,
        stable_id_scheme_version=STABLE_ID_SCHEME_VERSION,
        allowed_pairs=sorted([list(pair) for pair in ALLOWED_PAIRS]),
        japanese_languages=sorted(JAPANESE_LANGUAGES),
    )
    if args.resume:
        previous_run_state = load_run_state(run_state_path)
        if previous_run_state is None:
            logger.warning(
                "No previous run-state file found at %s; skipping resume-compatibility validation.",
                run_state_path,
            )
        else:
            try:
                validate_resume_compatibility(
                    previous_run_state, current_run_state, allow_model_change=args.allow_model_change
                )
            except RunStateIncompatibleError as exc:
                raise SystemExit(str(exc))
    write_run_state(run_state_path, current_run_state)

    pre_run_successful_ids = load_successful_correction_ids(checkpoint_path)
    corrections_by_pair = group_by_pair_key(corrections)
    pair_stats: dict[str, dict] = {}
    for pair_key, pair_corrections in sorted(corrections_by_pair.items()):
        correction_ids = [c["correction_id"] for c in pair_corrections]
        successful_before = sum(1 for cid in correction_ids if cid in pre_run_successful_ids)
        pending_before = len(correction_ids) - successful_before
        complete_before, _incomplete_before = compute_sentence_completion(
            pair_corrections, pre_run_successful_ids
        )
        sampling_info = toy_log["pairs"].get(pair_key, {}) if args.toy else {}
        pair_stats[pair_key] = {
            "available_sentence_rows": sampling_info.get("available_sentence_rows"),
            "target_sentence_rows": sampling_info.get("target_sentence_rows"),
            "selected_sentence_rows": sampling_info.get(
                "selected_sentence_rows", len({c["sentence_id"] for c in pair_corrections})
            ),
            "complete_sentence_rows_before_run": len(complete_before),
            "selected_corrections": len(correction_ids),
            "successful_corrections_before_run": successful_before,
            "pending_corrections_before_run": pending_before,
        }
        if pending_before == 0:
            logger.info("Skipping %s: all selected corrections are already complete", pair_key)

    _log_pair_table(pair_stats)

    run_result: ReconstructionRunResult = reconstruct_corrections(
        corrections,
        client=None,
        model=model,
        batch_size=batch_size,
        reasoning_effort=reasoning_effort,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        request_timeout_seconds=request_timeout_seconds,
        checkpoint_every_batches=checkpoint_every_batches,
        checkpoint_path=checkpoint_path,
        failures_path=failures_path,
        raw_responses_dir=raw_responses_dir,
        resume=args.resume,
        fail_fast=args.fail_fast,
        skip_api=args.skip_api,
        is_toy=args.toy,
        vllm_url=vllm_url,
        vllm_api_key_env=vllm_api_key_env,
        vllm_max_tokens=vllm_max_tokens,
    )
    summary.record(
        "reconstruction",
        {
            "n_already_successful": run_result.n_already_successful,
            "n_pending": run_result.n_pending,
            "n_newly_successful": run_result.n_newly_successful,
            "n_failed": run_result.n_failed,
            "n_batches_called": run_result.n_batches_called,
        },
    )

    post_run_successful_ids = load_successful_correction_ids(checkpoint_path)
    post_run_failed_ids = load_failed_correction_ids(failures_path)
    for pair_key, pair_corrections in corrections_by_pair.items():
        correction_ids = [c["correction_id"] for c in pair_corrections]
        successful_after = sum(1 for cid in correction_ids if cid in post_run_successful_ids)
        failed_after = sum(1 for cid in correction_ids if cid in post_run_failed_ids)
        stats = pair_stats[pair_key]
        stats["newly_successful_corrections"] = successful_after - stats["successful_corrections_before_run"]
        stats["failed_corrections"] = failed_after

    if args.toy:
        summary.record(
            "incremental_sampling",
            {
                "requested_target_per_pair": toy_size_per_pair,
                "sampling_strategy": SAMPLING_STRATEGY_VERSION,
                "pairs": pair_stats,
            },
        )

    complete_sentences, incomplete_sentences = compute_sentence_completion(
        corrections, post_run_successful_ids
    )
    logger.info(
        "Sentence-row completion: %d complete, %d incomplete", len(complete_sentences), len(incomplete_sentences)
    )

    all_candidate_rows = load_candidate_rows(checkpoint_path)
    selected_correction_ids = {c["correction_id"] for c in corrections}
    all_candidate_rows = [r for r in all_candidate_rows if r["correction_id"] in selected_correction_ids]

    kept_rows, dedup_map_rows = deduplicate_candidates(all_candidate_rows)
    logger.info(
        "Deduplication: %d candidates -> %d kept (%d merged)",
        len(all_candidate_rows),
        len(kept_rows),
        len(dedup_map_rows),
    )

    candidates_df = pd.DataFrame(kept_rows)
    if not candidates_df.empty:
        candidates_df = candidates_df[CANDIDATE_COLUMNS]
    else:
        candidates_df = pd.DataFrame(columns=CANDIDATE_COLUMNS)
    atomic_write_dataframe(candidates_df, output_dir, "prepared_candidates", overwrite=final_overwrite)

    dedup_map_df = pd.DataFrame(dedup_map_rows)
    atomic_write_dataframe(dedup_map_df, output_dir, "deduplication_map", overwrite=final_overwrite, formats=("csv",))

    summary.record_counts(
        {
            "malformed_input_rows": len(parsed.malformed_rows),
            "removed_initial_filters": len(parsed.removed_corrections),
            "sentence_rows_selected": len(sentence_rows),
            "corrections_selected": len(corrections),
            "corrections_already_successful": run_result.n_already_successful,
            "corrections_pending_this_run": run_result.n_pending,
            "corrections_newly_successful": run_result.n_newly_successful,
            "candidates_before_dedup": len(all_candidate_rows),
            "prepared_candidates": len(kept_rows),
            "deduplication_map_rows": len(dedup_map_rows),
            "api_failures": run_result.n_failed,
            "complete_sentence_rows": len(complete_sentences),
            "incomplete_sentence_rows": len(incomplete_sentences),
        }
    )
    summary.write(output_dir / "preparation_summary.json")

    logger.info("Done. Outputs written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
