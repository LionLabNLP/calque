
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from .error_classification import FUNCTION_WORD_UPOS

IGNORED_GROUPING_UPOS: frozenset[str] = frozenset({"PUNCT", "SYM", "X"})

GROUPABLE_CLASSIFICATIONS: frozenset[str] = frozenset({"pure", "constructional+others"})

_OPERATION_ORDER: tuple[str, ...] = ("substitution", "insertion", "deletion")

_NO_GROUPABLE_EDIT_NOTE = (
    "grouping: constructional evidence present but no groupable constructional token edit "
    "could be extracted (all candidate edits were punctuation/symbol/other UPOS)"
)

EMPTY_GROUP_COLUMNS: dict[str, Any] = {
    "construction_group_id": "",
    "construction_group_rank": "",
    "construction_group_size": "",
    "construction_group_type": "",
    "construction_operation_pattern": "",
    "construction_group_signature": "",
    "construction_group_key": "",
    "construction_edit_count": "",
}


def classify_group_token_kind(
    upos: str | None, function_word_upos: frozenset[str] = FUNCTION_WORD_UPOS
) -> str | None:
    if not upos or upos in IGNORED_GROUPING_UPOS:
        return None
    return "function_word" if upos in function_word_upos else "content_word"


def _side_node(kind: str, surface: str | None, upos: str, deprel: str | None) -> dict[str, Any]:
    node: dict[str, Any] = {"kind": kind, "upos": upos, "deprel": deprel or ""}
    if kind == "function_word":
        node["surface"] = surface or ""
    return node


def _build_edit_side(
    edit: dict, side: str, function_word_upos: frozenset[str]
) -> tuple[dict | None, str | None]:
    index = edit.get(f"{side}_index")
    if index is None:
        return None, None
    upos = edit.get(f"{side}_upos")
    kind = classify_group_token_kind(upos, function_word_upos)
    if kind is None:
        return None, None
    node = _side_node(kind, edit.get(f"{side}_text"), upos, edit.get(f"{side}_deprel"))
    return node, kind


@dataclass(frozen=True)
class CanonicalEdit:

    operation: str
    correction_index: int | None
    learner_index: int | None
    correction: dict[str, Any] | None
    learner: dict[str, Any] | None
    category: str

    def signature_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, "correction": self.correction, "learner": self.learner}

    def display(self) -> str:
        def fmt(node: dict[str, Any] | None) -> str:
            if node is None:
                return "∅"
            if node["kind"] == "function_word":
                return f"{node['surface']}/{node['upos']}/{node['deprel']}"
            return f"{node['upos']}/{node['deprel']}"

        return f"{fmt(self.correction)} → {fmt(self.learner)}"


def build_constructional_edit_signature(
    constructional_edits: Iterable[dict],
    function_word_upos: frozenset[str] = FUNCTION_WORD_UPOS,
) -> list[CanonicalEdit]:
    canonical: list[CanonicalEdit] = []
    for edit in constructional_edits:
        operation = edit["operation"]
        correction_node, correction_kind = _build_edit_side(edit, "correction", function_word_upos)
        learner_node, learner_kind = _build_edit_side(edit, "learner", function_word_upos)

        if operation == "substitution":
            if correction_kind is None or learner_kind is None:
                continue
        else:
            if correction_kind is None and learner_kind is None:
                continue

        kinds = {k for k in (correction_kind, learner_kind) if k is not None}
        category = "mixed" if len(kinds) > 1 else next(iter(kinds))

        canonical.append(
            CanonicalEdit(
                operation=operation,
                correction_index=edit.get("correction_index"),
                learner_index=edit.get("learner_index"),
                correction=correction_node,
                learner=learner_node,
                category=category,
            )
        )

    def _sort_key(e: CanonicalEdit) -> tuple[int, str, str]:
        primary = e.correction_index if e.correction_index is not None else e.learner_index
        primary_key = primary if primary is not None else 10**9
        return (primary_key, e.operation, json.dumps(e.signature_dict(), sort_keys=True, ensure_ascii=False))

    canonical.sort(key=_sort_key)
    return canonical


def canonicalize_group_key(canonical_edits: list[CanonicalEdit]) -> str:
    return json.dumps(
        [e.signature_dict() for e in canonical_edits],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _group_id_from_key(signature_key: str) -> str:
    digest = hashlib.sha256(signature_key.encode("utf-8")).hexdigest()
    return f"cg_{digest[:12]}"


def _format_operation_pattern(operation_counts: dict[str, int]) -> str:
    parts: list[str] = []
    for op in _OPERATION_ORDER:
        n = operation_counts.get(op, 0)
        if n == 0:
            continue
        parts.append(op if n == 1 else f"{n}× {op}")
    return " + ".join(parts)


def build_pair_group_signature(
    canonical_edits: list[CanonicalEdit],
) -> dict[str, Any] | None:
    if not canonical_edits:
        return None

    signature_key = canonicalize_group_key(canonical_edits)
    group_id = _group_id_from_key(signature_key)
    display_signature = "; ".join(e.display() for e in canonical_edits)

    categories = {e.category for e in canonical_edits}
    if categories == {"function_word"}:
        group_type = "function_word"
    elif categories == {"content_word"}:
        group_type = "content_word"
    else:
        group_type = "function_and_content"

    operation_counts = {"substitution": 0, "insertion": 0, "deletion": 0}
    for e in canonical_edits:
        operation_counts[e.operation] += 1

    return {
        "construction_group_id": group_id,
        "construction_group_type": group_type,
        "construction_operation_pattern": _format_operation_pattern(operation_counts),
        "construction_group_signature": display_signature,
        "construction_group_key": signature_key,
        "construction_edit_count": len(canonical_edits),
        "operation_counts": operation_counts,
    }


def _append_note(existing: str, note: str) -> str:
    existing = (existing or "").strip()
    return f"{existing}; {note}" if existing else note


def assign_construction_groups(
    rows: list[dict],
    function_word_upos: frozenset[str] = FUNCTION_WORD_UPOS,
) -> tuple[list[dict], list[dict]]:
    groups: dict[str, dict[str, Any]] = {}

    for i, row in enumerate(rows):
        for col, default in EMPTY_GROUP_COLUMNS.items():
            row.setdefault(col, default)
            row[col] = default

        if row.get("classification") not in GROUPABLE_CLASSIFICATIONS:
            continue

        raw_edits = row.get("constructional_edits") or []
        canonical_edits = build_constructional_edit_signature(raw_edits, function_word_upos)
        signature = build_pair_group_signature(canonical_edits)
        if signature is None:
            row["classification_notes"] = _append_note(
                row.get("classification_notes", ""), _NO_GROUPABLE_EDIT_NOTE
            )
            continue

        group_id = signature["construction_group_id"]
        group = groups.setdefault(
            group_id,
            {
                "construction_group_id": group_id,
                "construction_group_type": signature["construction_group_type"],
                "construction_operation_pattern": signature["construction_operation_pattern"],
                "construction_group_signature": signature["construction_group_signature"],
                "construction_group_key": signature["construction_group_key"],
                "construction_edit_count": signature["construction_edit_count"],
                "member_indices": [],
                "languages": set(),
            },
        )
        group["member_indices"].append(i)
        language = row.get("learning_language")
        if language:
            group["languages"].add(language)

    ordered_group_ids = sorted(
        groups,
        key=lambda gid: (
            -len(groups[gid]["member_indices"]),
            groups[gid]["construction_group_signature"],
            gid,
        ),
    )

    group_summary_rows: list[dict] = []
    for rank, gid in enumerate(ordered_group_ids, start=1):
        group = groups[gid]
        size = len(group["member_indices"])
        for idx in group["member_indices"]:
            row = rows[idx]
            row["construction_group_id"] = gid
            row["construction_group_rank"] = rank
            row["construction_group_size"] = size
            row["construction_group_type"] = group["construction_group_type"]
            row["construction_operation_pattern"] = group["construction_operation_pattern"]
            row["construction_group_signature"] = group["construction_group_signature"]
            row["construction_group_key"] = group["construction_group_key"]
            row["construction_edit_count"] = group["construction_edit_count"]

        languages = sorted(group["languages"])
        group_summary_rows.append(
            {
                "construction_group_id": gid,
                "construction_group_rank": rank,
                "construction_group_type": group["construction_group_type"],
                "construction_operation_pattern": group["construction_operation_pattern"],
                "construction_group_signature": group["construction_group_signature"],
                "construction_group_size": size,
                "construction_edit_count": group["construction_edit_count"],
                "learning_languages": ", ".join(languages),
                "language_count": len(languages),
            }
        )

    return rows, group_summary_rows



_SIZE_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("size = 1", 1, 1),
    ("size = 2", 2, 2),
    ("size = 3–5", 3, 5),
    ("size = 6–10", 6, 10),
    ("size > 10", 11, None),
)


def _size_bucket(size: int) -> str:
    for label, lo, hi in _SIZE_BUCKETS:
        if size >= lo and (hi is None or size <= hi):
            return label
    return "size = 1"


def summarize_construction_groups(
    rows: list[dict], group_summary_rows: list[dict], *, top_n: int = 20
) -> dict[str, Any]:
    total_classified = len(rows)
    constructional_rows = [r for r in rows if r.get("classification") in GROUPABLE_CLASSIFICATIONS]
    total_constructional = len(constructional_rows)
    grouped_rows = [r for r in constructional_rows if r.get("construction_group_id")]
    total_grouped = len(grouped_rows)
    total_ungrouped = total_constructional - total_grouped

    total_groups = len(group_summary_rows)
    singleton_groups = sum(1 for g in group_summary_rows if g["construction_group_size"] == 1)
    multi_groups = total_groups - singleton_groups

    pairs_by_type: dict[str, int] = {}
    groups_by_type: dict[str, int] = {}
    for r in grouped_rows:
        t = r.get("construction_group_type") or ""
        pairs_by_type[t] = pairs_by_type.get(t, 0) + 1
    for g in group_summary_rows:
        t = g["construction_group_type"]
        groups_by_type[t] = groups_by_type.get(t, 0) + 1

    pairs_by_operation_pattern: dict[str, int] = {}
    groups_by_operation_pattern: dict[str, int] = {}
    for r in grouped_rows:
        p = r.get("construction_operation_pattern") or ""
        pairs_by_operation_pattern[p] = pairs_by_operation_pattern.get(p, 0) + 1
    for g in group_summary_rows:
        p = g["construction_operation_pattern"]
        groups_by_operation_pattern[p] = groups_by_operation_pattern.get(p, 0) + 1

    size_distribution: dict[str, int] = {label: 0 for label, _, _ in _SIZE_BUCKETS}
    for g in group_summary_rows:
        size_distribution[_size_bucket(g["construction_group_size"])] += 1

    top_groups = [
        {
            "rank": g["construction_group_rank"],
            "group_id": g["construction_group_id"],
            "group_type": g["construction_group_type"],
            "operation_pattern": g["construction_operation_pattern"],
            "signature": g["construction_group_signature"],
            "pair_count": g["construction_group_size"],
        }
        for g in sorted(group_summary_rows, key=lambda g: g["construction_group_rank"])[:top_n]
    ]

    return {
        "total_classified_pairs": total_classified,
        "total_constructional_pairs": total_constructional,
        "total_grouped_pairs": total_grouped,
        "total_ungrouped_constructional_pairs": total_ungrouped,
        "total_construction_groups": total_groups,
        "singleton_groups": singleton_groups,
        "groups_with_at_least_2_pairs": multi_groups,
        "pairs_by_group_type": pairs_by_type,
        "groups_by_group_type": groups_by_type,
        "pairs_by_operation_pattern": pairs_by_operation_pattern,
        "groups_by_operation_pattern": groups_by_operation_pattern,
        "group_size_distribution": size_distribution,
        "top_groups": top_groups,
    }


def format_construction_group_report(stats: dict[str, Any]) -> str:
    lines = ["Constructional grouping summary"]
    lines.append(f"  Total classified pairs: {stats['total_classified_pairs']}")
    lines.append(f"  Total pairs containing constructional evidence: {stats['total_constructional_pairs']}")
    lines.append(f"  Total pairs successfully assigned to a construction group: {stats['total_grouped_pairs']}")
    lines.append(f"  Total ungrouped constructional pairs: {stats['total_ungrouped_constructional_pairs']}")
    lines.append(f"  Total construction groups: {stats['total_construction_groups']}")
    lines.append(f"  Singleton groups: {stats['singleton_groups']}")
    lines.append(f"  Groups with at least 2 pairs: {stats['groups_with_at_least_2_pairs']}")

    lines.append("  Pairs by group type:")
    for t, n in sorted(stats["pairs_by_group_type"].items()):
        lines.append(f"    {t}: {n}")
    lines.append("  Groups by group type:")
    for t, n in sorted(stats["groups_by_group_type"].items()):
        lines.append(f"    {t}: {n}")

    lines.append("  Pairs by operation pattern:")
    for p, n in sorted(stats["pairs_by_operation_pattern"].items()):
        lines.append(f"    {p}: {n}")
    lines.append("  Groups by operation pattern:")
    for p, n in sorted(stats["groups_by_operation_pattern"].items()):
        lines.append(f"    {p}: {n}")

    lines.append("  Group-size distribution:")
    for label, _, _ in _SIZE_BUCKETS:
        lines.append(f"    {label}: {stats['group_size_distribution'][label]}")

    lines.append("  Most frequent groups:")
    for g in stats["top_groups"]:
        lines.append(
            f"    #{g['rank']} {g['group_id']} [{g['group_type']}] {g['operation_pattern']} "
            f"-- {g['signature']} ({g['pair_count']} pairs)"
        )

    return "\n".join(lines)
