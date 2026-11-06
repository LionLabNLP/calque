
from __future__ import annotations

import hashlib
import logging
import random
from collections import Counter, defaultdict
from typing import Any

from .config import JAPANESE_LANGUAGES

logger = logging.getLogger(__name__)

SAMPLING_STRATEGY_VERSION = "stable_pair_permutation_prefix_v1"


def _pair_of(row: dict) -> tuple[str, str]:
    return (row["learning_language"], row["native_language"])


def _group_sentence_ids_by_pair(sentence_rows: list[dict]) -> dict[tuple[str, str], list[str]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in sentence_rows:
        grouped[_pair_of(row)].append(row["sentence_id"])
    return grouped


def deterministic_sample(ids: list[str], k: int, seed: int) -> list[str]:
    if k >= len(ids):
        return sorted(ids)
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(ids), k))


def _pair_seed(seed: int, pair: tuple[str, str]) -> int:
    digest = hashlib.sha256(f"{seed}\x00{pair[0]}\x00{pair[1]}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def deterministic_order(ids: list[str], seed: int, pair: tuple[str, str]) -> list[str]:
    ordered = sorted(ids)
    rng = random.Random(_pair_seed(seed, pair))
    rng.shuffle(ordered)
    return ordered


def stable_prefix_sample(ids: list[str], k: int, seed: int, pair: tuple[str, str]) -> list[str]:
    ordered = deterministic_order(ids, seed, pair)
    return ordered[: min(k, len(ordered))]


def _filter_by_sentence_ids(
    sentence_rows: list[dict], corrections: list[dict], keep_ids: set[str]
) -> tuple[list[dict], list[dict]]:
    kept_sentences = [r for r in sentence_rows if r["sentence_id"] in keep_ids]
    kept_corrections = [c for c in corrections if c["sentence_id"] in keep_ids]
    return kept_sentences, kept_corrections


def downsample_japanese(
    sentence_rows: list[dict],
    corrections: list[dict],
    seed: int,
    japanese_languages: frozenset[str] = JAPANESE_LANGUAGES,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    grouped = _group_sentence_ids_by_pair(sentence_rows)

    counts_before = {pair: len(ids) for pair, ids in grouped.items()}
    non_japanese_counts = {
        pair: n for pair, n in counts_before.items() if pair[0] not in japanese_languages
    }
    cap = max(non_japanese_counts.values()) if non_japanese_counts else 0

    keep_ids: set[str] = set()
    counts_after: dict[tuple[str, str], int] = {}
    for pair, ids in grouped.items():
        if pair[0] in japanese_languages and len(ids) > cap:
            selected = deterministic_sample(ids, cap, seed)
        else:
            selected = ids
        keep_ids.update(selected)
        counts_after[pair] = len(selected)

    kept_sentences, kept_corrections = _filter_by_sentence_ids(sentence_rows, corrections, keep_ids)

    log_info = {
        "japanese_cap": cap,
        "counts_before": {f"{p[0]}->{p[1]}": n for p, n in counts_before.items()},
        "counts_after": {f"{p[0]}->{p[1]}": n for p, n in counts_after.items()},
    }
    for pair, n_before in counts_before.items():
        n_after = counts_after[pair]
        logger.info(
            "Japanese downsampling: %s->%s rows %d -> %d", pair[0], pair[1], n_before, n_after
        )

    return kept_sentences, kept_corrections, log_info


def sample_toy(
    sentence_rows: list[dict],
    corrections: list[dict],
    toy_size_per_pair: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    grouped = _group_sentence_ids_by_pair(sentence_rows)

    keep_ids: set[str] = set()
    counts_before = {pair: len(ids) for pair, ids in grouped.items()}
    pair_info: dict[str, dict[str, int]] = {}
    for pair, ids in grouped.items():
        available = len(ids)
        selected = stable_prefix_sample(ids, toy_size_per_pair, seed, pair)
        keep_ids.update(selected)

        pair_key = f"{pair[0]}->{pair[1]}"
        pair_info[pair_key] = {
            "available_sentence_rows": available,
            "target_sentence_rows": toy_size_per_pair,
            "selected_sentence_rows": len(selected),
        }
        if toy_size_per_pair > available:
            logger.info(
                "%s: requested %d rows, only %d available; using all %d",
                pair_key,
                toy_size_per_pair,
                available,
                available,
            )
        else:
            logger.info("Toy sampling: %s rows -> %d", pair_key, len(selected))

    kept_sentences, kept_corrections = _filter_by_sentence_ids(sentence_rows, corrections, keep_ids)

    log_info = {
        "toy_size_per_pair": toy_size_per_pair,
        "sampling_strategy": SAMPLING_STRATEGY_VERSION,
        "counts_before": {f"{p[0]}->{p[1]}": n for p, n in counts_before.items()},
        "counts_after": {k: v["selected_sentence_rows"] for k, v in pair_info.items()},
        "pairs": pair_info,
    }

    return kept_sentences, kept_corrections, log_info


def group_by_pair_key(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['learning_language']}->{row['native_language']}"].append(row)
    return grouped


def group_correction_ids_by_sentence(corrections: list[dict]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for correction in corrections:
        grouped[correction["sentence_id"]].append(correction["correction_id"])
    return grouped


def compute_sentence_completion(
    corrections: list[dict], successful_correction_ids: set[str]
) -> tuple[set[str], set[str]]:
    grouped = group_correction_ids_by_sentence(corrections)
    complete: set[str] = set()
    incomplete: set[str] = set()
    for sentence_id, correction_ids in grouped.items():
        if all(cid in successful_correction_ids for cid in correction_ids):
            complete.add(sentence_id)
        else:
            incomplete.add(sentence_id)
    return complete, incomplete
