
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .alignment import AlignmentOp, align_tokens, build_alignment_maps
from .io import append_jsonl, read_jsonl
from .parenthetical import (
    CLASSIFICATION_LEARNER_SENTENCE_FIELD,
    CLASSIFICATION_RECONSTRUCTED_SENTENCE_FIELD,
    apply_classification_preprocessing,
)
from .stanza_parsing import ParsedSentence, StanzaPipelineManager, parse_feats, parse_sentence
from .text_metrics import token_sequence_edit_distance

logger = logging.getLogger(__name__)

FUNCTION_WORD_UPOS: frozenset[str] = frozenset({"ADP", "AUX", "CCONJ", "SCONJ", "DET", "PART", "PRON"})

PUNCT_UPOS = "PUNCT"


@dataclass
class ClassificationResult:
    classification: str
    contains_constructional: bool
    contains_lexical: bool
    contains_morphological: bool
    function_word_changes: list[dict] = field(default_factory=list)
    dependency_arc_changes: list[dict] = field(default_factory=list)
    lemma_changes: list[dict] = field(default_factory=list)
    morphological_feature_changes: list[dict] = field(default_factory=list)
    orthographic_changes: list[dict] = field(default_factory=list)
    token_alignment: list[dict] = field(default_factory=list)
    constructional_edits: list[dict] = field(default_factory=list)
    learner_conllu: str = ""
    correction_conllu: str = ""
    parse_status: str = "ok"
    classification_notes: str = ""
    stanza_word_edit_distance: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "contains_constructional": self.contains_constructional,
            "contains_lexical": self.contains_lexical,
            "contains_morphological": self.contains_morphological,
            "function_word_changes": self.function_word_changes,
            "dependency_arc_changes": self.dependency_arc_changes,
            "lemma_changes": self.lemma_changes,
            "morphological_feature_changes": self.morphological_feature_changes,
            "orthographic_changes": self.orthographic_changes,
            "token_alignment": self.token_alignment,
            "constructional_edits": self.constructional_edits,
            "learner_conllu": self.learner_conllu,
            "correction_conllu": self.correction_conllu,
            "parse_status": self.parse_status,
            "classification_notes": self.classification_notes,
            "stanza_word_edit_distance": self.stanza_word_edit_distance,
        }


def _serialize_alignment(ops: list[AlignmentOp], correction_tokens: list[dict], learner_tokens: list[dict]) -> list[dict]:
    serialized = []
    for op in ops:
        c_tok = correction_tokens[op.source_index] if op.source_index is not None else None
        l_tok = learner_tokens[op.target_index] if op.target_index is not None else None
        serialized.append(
            {
                "op": op.op,
                "correction_index": op.source_index,
                "correction_text": c_tok["text"] if c_tok else None,
                "correction_upos": c_tok["upos"] if c_tok else None,
                "correction_deprel": c_tok["deprel"] if c_tok else None,
                "learner_index": op.target_index,
                "learner_text": l_tok["text"] if l_tok else None,
                "learner_upos": l_tok["upos"] if l_tok else None,
                "learner_deprel": l_tok["deprel"] if l_tok else None,
            }
        )
    return serialized


def _is_word_order_only(correction_tokens: list[dict], learner_tokens: list[dict]) -> bool:
    correction_texts = [t["text"] for t in correction_tokens]
    learner_texts = [t["text"] for t in learner_tokens]
    return correction_texts != learner_texts and sorted(correction_texts) == sorted(learner_texts)


def classify_pair(
    correction_tokens: list[dict],
    learner_tokens: list[dict],
    function_word_upos: frozenset[str] = FUNCTION_WORD_UPOS,
) -> ClassificationResult:
    ops = align_tokens(correction_tokens, learner_tokens)
    source_to_target, _target_to_source = build_alignment_maps(ops)

    lemma_changes: list[dict] = []
    morphological_feature_changes: list[dict] = []
    orthographic_changes: list[dict] = []
    function_word_changes: list[dict] = []
    dependency_arc_changes: list[dict] = []
    constructional_edits: list[dict] = []
    constructional_pairs_seen: set[tuple[int | None, int | None]] = set()

    for op in ops:
        if op.op == "match":
            continue

        if op.op == "substitution":
            c_idx, l_idx = op.source_index, op.target_index
            c_tok, l_tok = correction_tokens[c_idx], learner_tokens[l_idx]
            if c_tok["upos"] == PUNCT_UPOS or l_tok["upos"] == PUNCT_UPOS:
                continue

            is_function_word = c_tok["upos"] in function_word_upos or l_tok["upos"] in function_word_upos
            if is_function_word:
                function_word_changes.append(
                    {
                        "type": "substitution",
                        "correction_index": c_idx,
                        "correction_text": c_tok["text"],
                        "correction_lemma": c_tok["lemma"],
                        "correction_upos": c_tok["upos"],
                        "learner_index": l_idx,
                        "learner_text": l_tok["text"],
                        "learner_lemma": l_tok["lemma"],
                        "learner_upos": l_tok["upos"],
                    }
                )
                constructional_edits.append(
                    {
                        "operation": "substitution",
                        "correction_index": c_idx,
                        "correction_text": c_tok["text"],
                        "correction_upos": c_tok["upos"],
                        "correction_deprel": c_tok["deprel"],
                        "learner_index": l_idx,
                        "learner_text": l_tok["text"],
                        "learner_upos": l_tok["upos"],
                        "learner_deprel": l_tok["deprel"],
                    }
                )
                constructional_pairs_seen.add((c_idx, l_idx))
                continue

            if c_tok["lemma"] != l_tok["lemma"]:
                lemma_changes.append(
                    {
                        "correction_index": c_idx,
                        "correction_text": c_tok["text"],
                        "correction_lemma": c_tok["lemma"],
                        "learner_index": l_idx,
                        "learner_text": l_tok["text"],
                        "learner_lemma": l_tok["lemma"],
                        "upos": c_tok["upos"],
                    }
                )
                continue

            feats_c = parse_feats(c_tok["feats"])
            feats_l = parse_feats(l_tok["feats"])
            changed_features = {
                k: [feats_c[k], feats_l[k]]
                for k in (set(feats_c) & set(feats_l))
                if feats_c[k] != feats_l[k]
            }
            if changed_features:
                morphological_feature_changes.append(
                    {
                        "correction_index": c_idx,
                        "learner_index": l_idx,
                        "correction_text": c_tok["text"],
                        "learner_text": l_tok["text"],
                        "lemma": c_tok["lemma"],
                        "changed_features": changed_features,
                    }
                )
            if c_tok["text"] != l_tok["text"]:
                orthographic_changes.append(
                    {
                        "correction_index": c_idx,
                        "learner_index": l_idx,
                        "correction_text": c_tok["text"],
                        "learner_text": l_tok["text"],
                        "lemma": c_tok["lemma"],
                    }
                )

        elif op.op in ("insertion", "deletion"):
            if op.op == "insertion":
                tok = learner_tokens[op.target_index]
            else:
                tok = correction_tokens[op.source_index]
            if tok["upos"] == PUNCT_UPOS:
                continue

            entry = {
                "type": op.op,
                "correction_index": op.source_index,
                "learner_index": op.target_index,
                "text": tok["text"],
                "lemma": tok["lemma"],
                "upos": tok["upos"],
            }
            if tok["upos"] in function_word_upos:
                function_word_changes.append(entry)
            dependency_arc_changes.append(entry)

            if op.op == "insertion":
                constructional_edits.append(
                    {
                        "operation": "insertion",
                        "correction_index": None,
                        "correction_text": None,
                        "correction_upos": None,
                        "correction_deprel": None,
                        "learner_index": op.target_index,
                        "learner_text": tok["text"],
                        "learner_upos": tok["upos"],
                        "learner_deprel": tok["deprel"],
                    }
                )
            else:
                constructional_edits.append(
                    {
                        "operation": "deletion",
                        "correction_index": op.source_index,
                        "correction_text": tok["text"],
                        "correction_upos": tok["upos"],
                        "correction_deprel": tok["deprel"],
                        "learner_index": None,
                        "learner_text": None,
                        "learner_upos": None,
                        "learner_deprel": None,
                    }
                )
            constructional_pairs_seen.add((op.source_index, op.target_index))

    for c_idx, l_idx in source_to_target.items():
        c_tok, l_tok = correction_tokens[c_idx], learner_tokens[l_idx]
        if c_tok["upos"] == PUNCT_UPOS or l_tok["upos"] == PUNCT_UPOS:
            continue

        c_head_id, l_head_id = c_tok["head"], l_tok["head"]
        c_is_root, l_is_root = c_head_id == 0, l_head_id == 0
        c_head_repr: Any = "ROOT" if c_is_root else c_head_id - 1
        l_head_repr: Any = "ROOT" if l_is_root else l_head_id - 1

        head_changed = False
        if c_is_root != l_is_root:
            head_changed = True
        elif not c_is_root:
            expected_l_head_idx0 = source_to_target.get(c_head_id - 1)
            if expected_l_head_idx0 is None or expected_l_head_idx0 != l_head_id - 1:
                head_changed = True

        deprel_changed = c_tok["deprel"] != l_tok["deprel"]

        if head_changed or deprel_changed:
            dependency_arc_changes.append(
                {
                    "type": "arc_change",
                    "correction_index": c_idx,
                    "correction_text": c_tok["text"],
                    "correction_head": c_head_repr,
                    "correction_deprel": c_tok["deprel"],
                    "learner_index": l_idx,
                    "learner_text": l_tok["text"],
                    "learner_head": l_head_repr,
                    "learner_deprel": l_tok["deprel"],
                    "head_changed": head_changed,
                    "deprel_changed": deprel_changed,
                }
            )
            if (c_idx, l_idx) not in constructional_pairs_seen:
                constructional_edits.append(
                    {
                        "operation": "substitution",
                        "correction_index": c_idx,
                        "correction_text": c_tok["text"],
                        "correction_upos": c_tok["upos"],
                        "correction_deprel": c_tok["deprel"],
                        "learner_index": l_idx,
                        "learner_text": l_tok["text"],
                        "learner_upos": l_tok["upos"],
                        "learner_deprel": l_tok["deprel"],
                    }
                )
                constructional_pairs_seen.add((c_idx, l_idx))

    contains_lexical = bool(lemma_changes)
    contains_morphological = bool(morphological_feature_changes) or bool(orthographic_changes)
    contains_constructional = bool(function_word_changes) or bool(dependency_arc_changes)

    evidence_types = set()
    if contains_lexical:
        evidence_types.add("lexical")
    if contains_morphological:
        evidence_types.add("morphological")
    if contains_constructional:
        evidence_types.add("constructional")

    if not evidence_types:
        classification = "unchanged"
    elif len(evidence_types) == 1:
        (only_type,) = evidence_types
        classification = "pure" if only_type == "constructional" else only_type
    elif contains_constructional:
        classification = "constructional+others"
    else:
        classification = "mixed"

    if _is_word_order_only(correction_tokens, learner_tokens):
        classification = "word_order"
        contains_constructional = False

    if evidence_types:
        arc_change_count = len([d for d in dependency_arc_changes if d["type"] == "arc_change"])
        constructional_count = len(function_word_changes) + arc_change_count
        notes = (
            f"{len(lemma_changes)} lexical, "
            f"{len(morphological_feature_changes) + len(orthographic_changes)} morphological, "
            f"{constructional_count} constructional edit(s) found"
        )
    else:
        notes = "no meaningful edits found"

    return ClassificationResult(
        classification=classification,
        contains_constructional=contains_constructional,
        contains_lexical=contains_lexical,
        contains_morphological=contains_morphological,
        function_word_changes=function_word_changes,
        dependency_arc_changes=dependency_arc_changes,
        lemma_changes=lemma_changes,
        morphological_feature_changes=morphological_feature_changes,
        orthographic_changes=orthographic_changes,
        token_alignment=_serialize_alignment(ops, correction_tokens, learner_tokens),
        constructional_edits=constructional_edits,
        parse_status="ok",
        classification_notes=notes,
    )


def classify_candidate(
    correction_parsed: ParsedSentence,
    learner_parsed: ParsedSentence,
    function_word_upos: frozenset[str] = FUNCTION_WORD_UPOS,
) -> ClassificationResult:
    if not correction_parsed.ok or not learner_parsed.ok:
        reasons = []
        if not correction_parsed.ok:
            reasons.append(f"correction parse failed ({correction_parsed.status})")
        if not learner_parsed.ok:
            reasons.append(f"learner parse failed ({learner_parsed.status})")
        return ClassificationResult(
            classification="invalid",
            contains_constructional=False,
            contains_lexical=False,
            contains_morphological=False,
            learner_conllu=learner_parsed.conllu,
            correction_conllu=correction_parsed.conllu,
            parse_status="error",
            classification_notes="; ".join(reasons),
        )

    result = classify_pair(correction_parsed.tokens, learner_parsed.tokens, function_word_upos)
    result.learner_conllu = learner_parsed.conllu
    result.correction_conllu = correction_parsed.conllu
    result.stanza_word_edit_distance = token_sequence_edit_distance(
        [t["text"] for t in correction_parsed.tokens],
        [t["text"] for t in learner_parsed.tokens],
    )
    return result




@dataclass
class ClassifyRunResult:
    n_processed: int = 0
    n_skipped_already_done: int = 0
    n_invalid: int = 0


def _load_already_done_ids(checkpoint_path: Path) -> set[str]:
    done: set[str] = set()
    if checkpoint_path.exists():
        for row in read_jsonl(checkpoint_path):
            cid = row.get("candidate_id")
            if cid:
                done.add(cid)
    return done


def classify_candidates(
    candidates: list[dict],
    pipeline_manager: StanzaPipelineManager,
    *,
    function_word_upos: frozenset[str] = FUNCTION_WORD_UPOS,
    checkpoint_path: Path,
    resume: bool,
    fail_fast: bool,
    batch_size: int,
) -> ClassifyRunResult:
    checkpoint_path = Path(checkpoint_path)
    already_done = _load_already_done_ids(checkpoint_path) if resume else set()
    pending = [c for c in candidates if c["candidate_id"] not in already_done]
    run_result = ClassifyRunResult(n_skipped_already_done=len(candidates) - len(pending))
    logger.info(
        "Classification: %d already done (resume=%s), %d pending",
        run_result.n_skipped_already_done,
        resume,
        len(pending),
    )

    buffer: list[dict] = []
    for i, cand in enumerate(pending, start=1):
        cand = apply_classification_preprocessing(cand)
        try:
            learning_code = cand["learning_code"]
            learner_parsed = parse_sentence(
                pipeline_manager,
                learning_code,
                cand[CLASSIFICATION_LEARNER_SENTENCE_FIELD],
                f"{cand['candidate_id']}:learner",
            )
            correction_parsed = parse_sentence(
                pipeline_manager,
                learning_code,
                cand[CLASSIFICATION_RECONSTRUCTED_SENTENCE_FIELD],
                f"{cand['candidate_id']}:correction",
            )
            classification = classify_candidate(correction_parsed, learner_parsed, function_word_upos)
        except Exception as exc:
            logger.warning("Classification failed for %s: %s", cand.get("candidate_id"), exc)
            if fail_fast:
                raise
            classification = ClassificationResult(
                classification="invalid",
                contains_constructional=False,
                contains_lexical=False,
                contains_morphological=False,
                parse_status="error",
                classification_notes=f"unexpected exception during classification: {exc}",
            )

        row = {**cand, **classification.to_dict()}
        buffer.append(row)
        run_result.n_processed += 1
        if classification.classification == "invalid":
            run_result.n_invalid += 1

        if len(buffer) >= batch_size:
            append_jsonl(checkpoint_path, buffer)
            buffer = []
            logger.info("Classified %d/%d pending candidates", i, len(pending))

    if buffer:
        append_jsonl(checkpoint_path, buffer)

    return run_result


def load_classified_rows(checkpoint_path: Path) -> list[dict]:
    return list(read_jsonl(Path(checkpoint_path)))
