
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from .io import append_jsonl, atomic_write_jsonl, read_jsonl

logger = logging.getLogger(__name__)

_TAG_MARKER_RE = re.compile(
    r"\[/?[A-Za-z][A-Za-z0-9_-]*\]"
)


def contains_tag_markers(text: str) -> bool:
    return bool(_TAG_MARKER_RE.search(text))

SYSTEM_PROMPT = """You recover native-speaker corrections from noisy tagged Lang-8 data.

For each item, you are given:
- a learner sentence
- a tagged correction written by a native speaker

Recover the complete correction that the native speaker most likely intended.

The markup is noisy and inconsistent. Do not assume that every tag type always
has one fixed meaning. Interpret the tagged correction as a whole, using the
learner sentence to understand what was changed.

Rules:
- Recover the primary native correction as faithfully as possible.
- Preserve the native corrector's wording.
- Make the smallest reconstruction supported by the tagged correction.
- Do not independently improve, rewrite, or translate the sentence.
- Remove all markup tags from the output.
- Remove editorial comments, explanations, arrows, and metalinguistic notes
  from the primary correction.
- Preserve unchanged text, spacing, punctuation, capitalization, and quotation
  marks unless the correction clearly changes them.
- Treat explicitly struck-through text as deleted.
- If the tagged correction clearly provides multiple complete alternative
  corrections, choose the most direct one as Correction and output each other
  complete alternative as a Suggestion.
- Do not output a Suggestion line when there is no genuine alternative.
- Do not explain your reasoning.
- Treat every item independently.
- Return exactly one block for each requested correction_id, in the same order.
- Write every Correction and Suggestion on a single line.

Use exactly this format:

ID: <correction_id>
Correction: <complete corrected sentence>
END

Only when a genuine alternative exists, insert one or more lines before END:

Suggestion: <complete alternative sentence>
"""



def build_batch_user_content(items: list[dict]) -> str:
    blocks: list[str] = []

    for item in items:
        blocks.append(
            "\n".join(
                [
                    f"ID: {item['correction_id']}",
                    f"Language: {item['learning_language']}",
                    f"Learner: {item['learner_sentence']}",
                    f"Tagged correction: {item['original_tagged_correction']}",
                    "END INPUT",
                ]
            )
        )

    return "\n\n".join(blocks)


_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3040, 0x30FF), (0xFF66, 0xFF9F))


def _has_cjk(text: str) -> bool:
    return any(any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES) for ch in text)


def plausible_language_match(text: str, learning_language: str) -> bool:
    if not text or not text.strip():
        return False
    if learning_language in ("Japanese", "Mandarin"):
        return _has_cjk(text)
    if _has_cjk(text):
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True
    latin_count = sum(1 for ch in letters if ord(ch) < 0x0250 or 0x1E00 <= ord(ch) <= 0x1EFF)
    return (latin_count / len(letters)) > 0.8




@dataclass
class BatchValidationResult:
    candidate_rows: list[dict] = field(default_factory=list)
    valid_ids: set[str] = field(default_factory=set)
    failed_ids: set[str] = field(default_factory=set)
    unknown_ids: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)


def _make_candidate_row(
    item_meta: dict,
    candidate_type: str,
    suggestion_index: int | None,
    reconstructed_sentence: str,
    suggestion_evidence: str,
    api_model: str,
    api_raw_response_path: str,
    is_toy: bool,
) -> dict:
    correction_id = item_meta["correction_id"]
    candidate_id = (
        f"{correction_id}:cand:primary"
        if candidate_type == "primary"
        else f"{correction_id}:cand:suggestion:{suggestion_index}"
    )
    return {
        "candidate_id": candidate_id,
        "source_row_id": item_meta["source_row_id"],
        "sentence_id": item_meta["sentence_id"],
        "correction_id": correction_id,
        "candidate_type": candidate_type,
        "suggestion_index": suggestion_index,
        "learning_language": item_meta["learning_language"],
        "native_language": item_meta["native_language"],
        "learning_code": item_meta["learning_code"],
        "native_code": item_meta["native_code"],
        "learner_sentence": item_meta["learner_sentence"],
        "original_tagged_correction": item_meta["original_tagged_correction"],
        "reconstructed_sentence": reconstructed_sentence,
        "suggestion_evidence": suggestion_evidence,
        "source_line_number": item_meta["source_line_number"],
        "source_sentence_index": item_meta["source_sentence_index"],
        "source_correction_index": item_meta["source_correction_index"],
        "api_model": api_model,
        "api_status": "ok",
        "api_raw_response_path": api_raw_response_path,
        "is_toy": is_toy,
    }


def validate_and_expand_batch(
    requested_items: list[dict],
    parsed_response: dict,
    api_model: str,
    api_raw_response_path: str,
    is_toy: bool,
) -> BatchValidationResult:
    result = BatchValidationResult()
    requested_by_id = {item["correction_id"]: item for item in requested_items}
    requested_ids = set(requested_by_id)

    raw_items = parsed_response.get("items")
    if not isinstance(raw_items, list):
        result.errors.append("response missing 'items' array")
        result.failed_ids = set(requested_ids)
        return result

    seen_ids: dict[str, int] = {}
    for entry in raw_items:
        if not isinstance(entry, dict) or "correction_id" not in entry:
            result.errors.append(f"malformed item entry: {entry!r}")
            continue
        cid = entry["correction_id"]
        seen_ids[cid] = seen_ids.get(cid, 0) + 1

    for cid, count in seen_ids.items():
        if cid not in requested_ids:
            result.unknown_ids.add(cid)
            result.errors.append(f"unknown correction_id returned: {cid}")
        elif count > 1:
            result.errors.append(f"duplicate correction_id returned: {cid} (x{count})")

    processed_ids: set[str] = set()
    for entry in raw_items:
        if not isinstance(entry, dict):
            continue
        cid = entry.get("correction_id")
        if cid not in requested_by_id:
            continue
        if cid in processed_ids:
            continue
        if seen_ids.get(cid, 0) > 1:
            result.failed_ids.add(cid)
            processed_ids.add(cid)
            continue
        processed_ids.add(cid)

        item_meta = requested_by_id[cid]
        status = entry.get("status")
        corrected_sentence = entry.get("corrected_sentence")

        if status != "ok" or not isinstance(corrected_sentence, str) or not corrected_sentence.strip():
            result.failed_ids.add(cid)
            result.errors.append(f"{cid}: empty/invalid corrected_sentence or status={status!r}")
            continue

        if contains_tag_markers(corrected_sentence):
            result.failed_ids.add(cid)
            result.errors.append(f"{cid}: corrected_sentence still contains raw edit-tag markers")
            continue

        if not plausible_language_match(corrected_sentence, item_meta["learning_language"]):
            result.failed_ids.add(cid)
            result.errors.append(f"{cid}: corrected_sentence fails language-plausibility check")
            continue

        rows = [
            _make_candidate_row(
                item_meta,
                "primary",
                None,
                corrected_sentence,
                "",
                api_model,
                api_raw_response_path,
                is_toy,
            )
        ]

        raw_suggestions = entry.get("suggestions") or []
        accepted_sentences: set[str] = {corrected_sentence}
        suggestion_index = 0
        for sug in raw_suggestions:
            if not isinstance(sug, dict):
                continue
            sentence = sug.get("sentence")
            evidence = sug.get("evidence", "")
            if not isinstance(sentence, str) or not sentence.strip():
                continue
            if sentence == corrected_sentence:
                continue
            if sentence in accepted_sentences:
                continue
            if contains_tag_markers(sentence):
                continue
            if not plausible_language_match(sentence, item_meta["learning_language"]):
                continue
            accepted_sentences.add(sentence)
            rows.append(
                _make_candidate_row(
                    item_meta,
                    "suggestion",
                    suggestion_index,
                    sentence,
                    evidence if isinstance(evidence, str) else "",
                    api_model,
                    api_raw_response_path,
                    is_toy,
                )
            )
            suggestion_index += 1

        result.candidate_rows.extend(rows)
        result.valid_ids.add(cid)

    result.failed_ids |= requested_ids - result.valid_ids - result.unknown_ids
    return result




class ReconstructionClient(Protocol):
    model_name: str

    def request_batch(
        self, items: list[dict], model: str, reasoning_effort: str, timeout: float
    ) -> str:
        ...


class OpenAIReconstructionClient:

    def __init__(self, model_name: str = "") -> None:
        self.model_name = model_name
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set; cannot call the OpenAI API."
            )
        import openai

        self._client = openai.OpenAI(api_key=api_key)

    def request_batch(
        self, items: list[dict], model: str, reasoning_effort: str, timeout: float
    ) -> str:
        response = self._client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_batch_user_content(items)},
            ],
            reasoning={"effort": reasoning_effort},
            timeout=timeout,
        )
        return response.output_text



OPENAI_BACKEND = "openai"
VLLM_BACKEND = "vllm"

VLLM_URL_ENV = "VLLM_CHAT_COMPLETIONS_URL"
VLLM_MODEL_IDS_ENV = "VLLM_MODEL_IDS"
VLLM_CHAT_COMPLETIONS_URL = os.environ.get(VLLM_URL_ENV, "")
VLLM_API_KEY_ENV = "VLLM_API_KEY"
DEFAULT_VLLM_REQUEST_TIMEOUT_SECONDS = 600.0
DEFAULT_VLLM_MAX_TOKENS = 8192

EXPLICIT_MODEL_BACKENDS: dict[str, str] = {
    m.strip(): VLLM_BACKEND for m in os.environ.get(VLLM_MODEL_IDS_ENV, "").split(",") if m.strip()
}


def resolve_backend(model: str) -> str:
    if model in EXPLICIT_MODEL_BACKENDS:
        return EXPLICIT_MODEL_BACKENDS[model]
    if model.startswith("vllm-"):
        return VLLM_BACKEND
    return OPENAI_BACKEND


class VLLMChatCompletionsReconstructionClient:

    def __init__(
        self,
        model_name: str = "",
        *,
        url: str | None = VLLM_CHAT_COMPLETIONS_URL,
        api_key_env: str = VLLM_API_KEY_ENV,
        max_tokens: int = DEFAULT_VLLM_MAX_TOKENS,
    ) -> None:
        self.model_name = model_name
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} environment variable is not set; cannot call the vLLM "
                "chat-completions endpoint."
            )
        url = url or os.environ.get(VLLM_URL_ENV, "")
        if not url:
            raise RuntimeError(
                f"No vLLM chat-completions URL configured; set {VLLM_URL_ENV} or "
                "vllm.chat_completions_url in the config."
            )
        self._api_key = api_key
        self._url = url
        self._max_tokens = max_tokens

    def request_batch(
        self, items: list[dict], model: str, reasoning_effort: str, timeout: float
    ) -> str:
        import requests

        response = requests.post(
            self._url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_batch_user_content(items)},
                ],
                "temperature": 0,
                "max_tokens": self._max_tokens,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Malformed vLLM chat-completions response: {payload!r}") from exc


def build_reconstruction_client(
    model: str,
    *,
    vllm_url: str | None = VLLM_CHAT_COMPLETIONS_URL,
    vllm_api_key_env: str = VLLM_API_KEY_ENV,
    vllm_max_tokens: int = DEFAULT_VLLM_MAX_TOKENS,
) -> "ReconstructionClient":
    backend = resolve_backend(model)
    if backend == VLLM_BACKEND:
        return VLLMChatCompletionsReconstructionClient(
            model, url=vllm_url, api_key_env=vllm_api_key_env, max_tokens=vllm_max_tokens
        )
    return OpenAIReconstructionClient(model)




def _chunk(items: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


@dataclass
class ReconstructionRunResult:
    n_already_successful: int = 0
    n_pending: int = 0
    n_newly_successful: int = 0
    n_failed: int = 0
    n_batches_called: int = 0

def parse_plaintext_response(raw_text: str, requested_ids: Iterable[str] | None = None) -> dict:

    items: list[dict] = []
    current: dict | None = None
    seen_ids: set[str] = set()

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("ID:"):
            if current is not None:
                raise ValueError(
                    f"Started a new item before END for "
                    f"{current.get('correction_id')!r}"
                )

            correction_id = line.removeprefix("ID:").strip()
            if not correction_id:
                raise ValueError("Empty correction ID")
            if correction_id in seen_ids:
                raise ValueError(f"Duplicate correction ID in response: {correction_id!r}")
            seen_ids.add(correction_id)

            current = {
                "correction_id": correction_id,
                "corrected_sentence": "",
                "suggestions": [],
            }
            continue

        if current is None:
            raise ValueError(f"Content outside an item block: {line!r}")

        if line.startswith("Correction:"):
            if current["corrected_sentence"]:
                raise ValueError(
                    f"Duplicate Correction line for "
                    f"{current['correction_id']}"
                )

            current["corrected_sentence"] = (
                line.removeprefix("Correction:").strip()
            )
            continue

        if line.startswith("Suggestion:"):
            suggestion = line.removeprefix("Suggestion:").strip()
            if suggestion:
                current["suggestions"].append(
                    {
                        "sentence": suggestion,
                        "evidence": "",
                    }
                )
            continue

        if line == "END":
            if not current["corrected_sentence"]:
                raise ValueError(
                    f"Missing Correction for "
                    f"{current['correction_id']}"
                )

            current["status"] = "ok"
            current["notes"] = ""
            items.append(current)
            current = None
            continue

        raise ValueError(
            f"Unexpected output line for "
            f"{current['correction_id']}: {line!r}"
        )

    if current is not None:
        logger.warning(
            "Response ended before END for %s; treating it as an incomplete final item.",
            current["correction_id"],
        )

    missing_ids: list[str] = []

    if requested_ids is not None:
        requested = set(requested_ids)
        returned = {item["correction_id"] for item in items}
        unknown = returned - requested
        if unknown:
            raise ValueError(...)
        missing_ids = sorted(requested - returned)

    return {
        "items": items,
        "missing_ids": missing_ids,
    }

def load_successful_correction_ids(checkpoint_path: Path) -> set[str]:
    checkpoint_path = Path(checkpoint_path)
    done: set[str] = set()
    if checkpoint_path.exists():
        for row in read_jsonl(checkpoint_path):
            cid = row.get("correction_id")
            if cid:
                done.add(cid)
    return done


def load_failed_correction_ids(failures_path: Path) -> set[str]:
    failures_path = Path(failures_path)
    failed: set[str] = set()
    if failures_path.exists():
        for row in read_jsonl(failures_path):
            cid = row.get("correction_id")
            if cid:
                failed.add(cid)
    return failed


def reconstruct_corrections(
    retained_corrections: list[dict],
    *,
    client: ReconstructionClient | None,
    model: str,
    batch_size: int,
    reasoning_effort: str,
    max_retries: int,
    retry_backoff_seconds: float,
    request_timeout_seconds: float,
    checkpoint_every_batches: int,
    checkpoint_path: Path,
    failures_path: Path,
    raw_responses_dir: Path,
    resume: bool,
    fail_fast: bool,
    skip_api: bool,
    is_toy: bool,
    vllm_url: str | None,
    vllm_api_key_env: str,
    vllm_max_tokens: int,
) -> ReconstructionRunResult:
    checkpoint_path = Path(checkpoint_path)
    failures_path = Path(failures_path)
    raw_responses_dir = Path(raw_responses_dir)
    raw_responses_dir.mkdir(parents=True, exist_ok=True)

    run_result = ReconstructionRunResult()

    pre_run_successful = load_successful_correction_ids(checkpoint_path) if resume else set()
    pending = [c for c in retained_corrections if c["correction_id"] not in pre_run_successful]
    run_result.n_already_successful = len(retained_corrections) - len(pending)
    run_result.n_pending = len(pending)
    logger.info(
        "Reconstruction: %d already successful (resume=%s), %d pending",
        run_result.n_already_successful,
        resume,
        len(pending),
    )

    if skip_api:
        if pending:
            raise RuntimeError(
                f"--skip-api was passed but {len(pending)} correction(s) have no cached "
                f"reconstruction result in {checkpoint_path}. Run without --skip-api first, "
                "or use --resume to complete a previous run."
            )
        return run_result

    if client is None:
        client = build_reconstruction_client(
            model,
            vllm_url=vllm_url,
            vllm_api_key_env=vllm_api_key_env,
            vllm_max_tokens=vllm_max_tokens,
        )

    pending_buffer: list[dict] = list(pending)
    buffered_rows: list[dict] = []
    batches_since_flush = 0
    global_batch_counter = 0
    newly_successful_ids: set[str] = set()

    for round_idx in range(max_retries + 1):
        if not pending_buffer:
            break
        next_round: list[dict] = []

        for batch in _chunk(pending_buffer, batch_size):
            global_batch_counter += 1
            run_result.n_batches_called += 1
            try:
                raw_text = client.request_batch(
                    batch, model, reasoning_effort, request_timeout_seconds
                )
            except Exception as exc:
                logger.warning(
                    "Batch %d (round %d) call failed: %s", global_batch_counter, round_idx, exc
                )
                if fail_fast:
                    raise
                next_round.extend(batch)
                continue

            raw_path = raw_responses_dir / f"batch_{global_batch_counter:06d}.json"
            raw_path = (
                raw_responses_dir
                / f"batch_{global_batch_counter:06d}.txt"
            )
            
            try:
                parsed = parse_plaintext_response(
                    raw_text, requested_ids=[item["correction_id"] for item in batch]
                )
            except ValueError as exc:
                logger.warning(
                    "Batch %d returned invalid plaintext output: %s",
                    global_batch_counter,
                    exc,
                )
                if fail_fast:
                    raise
                next_round.extend(batch)
                continue

            validation = validate_and_expand_batch(
                batch, parsed, model, str(raw_path), is_toy
            )
            for err in validation.errors:
                logger.warning("Batch %d validation issue: %s", global_batch_counter, err)

            buffered_rows.extend(validation.candidate_rows)
            run_result.n_newly_successful += len(validation.valid_ids)
            newly_successful_ids |= validation.valid_ids

            failed_items = [item for item in batch if item["correction_id"] in validation.failed_ids]
            if failed_items:
                if fail_fast:
                    raise RuntimeError(
                        f"Batch {global_batch_counter} had {len(failed_items)} failed item(s): "
                        f"{validation.errors}"
                    )
                next_round.extend(failed_items)

            batches_since_flush += 1
            if batches_since_flush >= checkpoint_every_batches:
                if buffered_rows:
                    append_jsonl(checkpoint_path, buffered_rows)
                    buffered_rows = []
                batches_since_flush = 0

        pending_buffer = next_round
        if pending_buffer and round_idx < max_retries:
            sleep_s = retry_backoff_seconds * (2**round_idx)
            logger.info(
                "Round %d: %d item(s) failed, retrying in %.1fs", round_idx, len(pending_buffer), sleep_s
            )
            time.sleep(sleep_s)

    if buffered_rows:
        append_jsonl(checkpoint_path, buffered_rows)

    final_successful = pre_run_successful | newly_successful_ids
    retained_by_id = {c["correction_id"]: c for c in retained_corrections}
    still_failing_ids = (
        load_failed_correction_ids(failures_path) | {item["correction_id"] for item in pending_buffer}
    ) & retained_by_id.keys()
    still_failing_ids -= final_successful

    failure_rows = [
        {**retained_by_id[cid], "reason": "exhausted_retries"} for cid in sorted(still_failing_ids)
    ]
    atomic_write_jsonl(failures_path, failure_rows)
    run_result.n_failed = len(still_failing_ids)
    if still_failing_ids:
        logger.warning("%d item(s) still failed after exhausting retries", len(still_failing_ids))

    return run_result


def load_candidate_rows(checkpoint_path: Path) -> list[dict]:
    return list(read_jsonl(Path(checkpoint_path)))
