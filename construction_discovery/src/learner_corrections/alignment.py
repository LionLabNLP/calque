
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

SUBSTITUTION_COST_DIFFERENT_LEMMA = 1.0
SUBSTITUTION_COST_SAME_LEMMA = 0.4
GAP_COST = 1.0


def normalize_token_form(text: str) -> str:
    return unicodedata.normalize("NFC", text)


@dataclass(frozen=True)
class AlignmentOp:
    op: str
    source_index: int | None
    target_index: int | None


def _substitution_cost(source_tok: dict, target_tok: dict) -> float:
    if normalize_token_form(source_tok["text"]) == normalize_token_form(target_tok["text"]):
        return 0.0
    source_lemma = normalize_token_form(source_tok.get("lemma") or source_tok["text"])
    target_lemma = normalize_token_form(target_tok.get("lemma") or target_tok["text"])
    if source_lemma == target_lemma:
        return SUBSTITUTION_COST_SAME_LEMMA
    return SUBSTITUTION_COST_DIFFERENT_LEMMA


def align_tokens(source_tokens: list[dict], target_tokens: list[dict]) -> list[AlignmentOp]:
    n, m = len(source_tokens), len(target_tokens)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * GAP_COST
    for j in range(1, m + 1):
        dp[0][j] = j * GAP_COST

    sub_costs: list[list[float]] = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            sub_costs[i][j] = _substitution_cost(source_tokens[i], target_tokens[j])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + sub_costs[i - 1][j - 1]
            up = dp[i - 1][j] + GAP_COST
            left = dp[i][j - 1] + GAP_COST
            dp[i][j] = min(diag, up, left)

    ops: list[AlignmentOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diag = dp[i - 1][j - 1] + sub_costs[i - 1][j - 1]
        else:
            diag = None
        if i > 0 and j > 0 and dp[i][j] == diag:
            cost = sub_costs[i - 1][j - 1]
            ops.append(
                AlignmentOp(
                    op="match" if cost == 0.0 else "substitution",
                    source_index=i - 1,
                    target_index=j - 1,
                )
            )
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + GAP_COST:
            ops.append(AlignmentOp(op="deletion", source_index=i - 1, target_index=None))
            i -= 1
        else:
            ops.append(AlignmentOp(op="insertion", source_index=None, target_index=j - 1))
            j -= 1

    ops.reverse()
    return ops


def build_alignment_maps(ops: list[AlignmentOp]) -> tuple[dict[int, int], dict[int, int]]:
    source_to_target: dict[int, int] = {}
    target_to_source: dict[int, int] = {}
    for op in ops:
        if op.op in ("match", "substitution"):
            source_to_target[op.source_index] = op.target_index
            target_to_source[op.target_index] = op.source_index
    return source_to_target, target_to_source
