from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .cache import STATUS_OK, FeatureCache, canonical_json
from .normalize import DIVERGENCE_TOKENIZER_VERSION, NORMALIZATION_VERSION, divergence_tokens
from .schema import ENGLISH_PREFERENCE_COLUMN, TARGET_PREFERENCE_COLUMN

NAMESPACE = "divergence"
DIVERGENCE_CONFIG = {
    "feature": "cross_lingual_realization_divergence",
    "version": "div-v2",
    "normalization": NORMALIZATION_VERSION,
    "tokenizer": DIVERGENCE_TOKENIZER_VERSION,
    "distance": "token_levenshtein_unit_costs / max(len_a, len_b)",
    "eligibility": "English and Target language preference annotation in {>, >>}",
}

INCOMPLETE_PREFERENCE_ELIGIBILITY = (
    "Target language preference annotation in {>, >>} (English preference annotation ignored)"
)


def divergence_config(incomplete_preference_annotation: bool = False) -> Dict[str, object]:
    if not incomplete_preference_annotation:
        return dict(DIVERGENCE_CONFIG)
    return {**DIVERGENCE_CONFIG, "eligibility": INCOMPLETE_PREFERENCE_ELIGIBILITY,
            "incomplete_preference_annotation": True}


RESULT_SCHEMA = "pairwise-v1"
OLSD_DEFINITION = ("OLSD_l = 2/((n-1)(n-2)) * sum_{i<j; i,j != l} d(S_i, S_j) over the same eligible realizations "
                   "and distance d as Strategy Distance; NA (insufficient_realizations) for n < 3")

ELIGIBLE_PREFERENCES = frozenset({">", ">>"})
STATUS_INELIGIBLE = "ineligible_preference"


def _annotation(row, column) -> str:
    v = row.get(column) if hasattr(row, "get") else None
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def realization_eligible(row, incomplete_preference_annotation: bool = False) -> bool:
    if _annotation(row, TARGET_PREFERENCE_COLUMN) not in ELIGIBLE_PREFERENCES:
        return False
    return incomplete_preference_annotation or _annotation(row, ENGLISH_PREFERENCE_COLUMN) in ELIGIBLE_PREFERENCES

ENGLISH_LABEL = "en"


def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def normalized_distance(a: Sequence[str], b: Sequence[str]) -> float:
    m = max(len(a), len(b))
    if m == 0:
        raise ValueError("both token sequences are empty")
    return levenshtein(a, b) / m


def compute_group(realizations: List[Tuple[str, str]]) -> Dict[str, object]:
    toks = [(label, divergence_tokens(s)) for label, s in realizations]
    valid = [(label, t) for label, t in toks if len(t) > 0]
    invalid = [label for label, t in toks if len(t) == 0]
    scores: Dict[str, Optional[float]] = {label: None for label, _ in toks}
    n = len(valid)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = normalized_distance(valid[i][1], valid[j][1])
    if n >= 2:
        for i, (label, _) in enumerate(valid):
            scores[label] = float(dist[i].sum() / (n - 1))
    return {"scores": scores, "n_valid": n, "invalid_labels": invalid,
            "valid_labels": [label for label, _ in valid], "pairwise": dist.tolist(),
            "result_schema": RESULT_SCHEMA}


def has_pairwise(result) -> bool:
    return isinstance(result, dict) and result.get("result_schema") == RESULT_SCHEMA and "pairwise" in result


def olsd_scores(result):
    if not has_pairwise(result):
        raise ValueError("group result has no pairwise distance matrix")

    labels = list(result["valid_labels"])
    out = {label: None for label in result["scores"]}

    if len(labels) < 3:
        return out

    dist = np.asarray(result["pairwise"], dtype=float)

    for i, label in enumerate(labels):
        others = np.delete(np.delete(dist, i, axis=0), i, axis=1)
        pair_distances = others[np.triu_indices(len(labels) - 1, k=1)]
        out[label] = float(pair_distances.mean())

    return out


def group_hash(realizations: List[Tuple[str, str]]) -> str:
    payload = canonical_json(sorted([list(r) for r in realizations]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _collect_realizations(table: pd.DataFrame,
                          incomplete_preference_annotation: bool = False) -> Dict[str, Dict[str, object]]:
    groups: Dict[str, Dict[str, object]] = {}
    for sc_id, g in table.groupby("semantic_content_id", sort=False):
        english = g["Proper English strategy"].iloc[0]
        realizations = [(ENGLISH_LABEL, english)]
        row_label: Dict[str, str] = {}
        ineligible = []
        lit = g[g["sentence_role"] == "en_transferred"]
        per_lang_count: Dict[str, int] = {}
        for _, r in lit.iterrows():
            lang = r["target_language"]
            if not realization_eligible(r, incomplete_preference_annotation):
                ineligible.append((r["source_row_id"], lang, r["sentence"]))
                continue
            per_lang_count[lang] = per_lang_count.get(lang, 0) + 1
            label = lang if per_lang_count[lang] == 1 else f"{lang}#{per_lang_count[lang]}"
            row_label[r["source_row_id"]] = label
            realizations.append((label, r["sentence"]))
        groups[sc_id] = {"realizations": realizations, "row_label": row_label, "ineligible": ineligible,
                         "english": english}
    return groups


def annotate(table: pd.DataFrame, cache: FeatureCache,
             incomplete_preference_annotation: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cid = cache.register_config(NAMESPACE, divergence_config(incomplete_preference_annotation))
    groups = _collect_realizations(table, incomplete_preference_annotation)
    hashes = {sc: group_hash(g["realizations"]) for sc, g in groups.items()}
    cached = cache.get_many(NAMESPACE, cid, [(h,) for h in hashes.values()])

    results: Dict[str, Dict[str, object]] = {}
    to_store = []
    n_upgraded = 0
    for sc, g in groups.items():
        rec = cached.get((hashes[sc],))
        if rec is not None and rec.ok and has_pairwise(rec.value):
            results[sc] = rec.value
            continue
        res = compute_group(g["realizations"])
        if rec is not None and rec.ok:
            n_upgraded += 1
            old = rec.value.get("scores", {}) if isinstance(rec.value, dict) else {}
            if old != res["scores"]:
                raise RuntimeError(f"recomputed Strategy Distance differs from the cached value for group {sc}")
        results[sc] = res
        to_store.append(((hashes[sc],), STATUS_OK, res, None))
    if to_store:
        cache.put_many(NAMESPACE, cid, to_store)
    cache.stats.setdefault(NAMESPACE, {"hits": 0, "misses": 0, "writes": 0})
    cache.stats[NAMESPACE]["upgraded_to_pairwise"] = cache.stats[NAMESPACE].get("upgraded_to_pairwise", 0) + n_upgraded
    olsd = {sc: olsd_scores(res) for sc, res in results.items()}

    own, own_status, en_score, tl_score, n_real, n_cmp, ghash = [], [], [], [], [], [], []
    o_own, o_status, o_en, o_tl, o_pairs = [], [], [], [], []
    for _, r in table.iterrows():
        sc = r["semantic_content_id"]
        if r.get("experiment", "standard") != "standard":
            own.append(None)
            own_status.append("not_applicable")
            en_score.append(None)
            tl_score.append(None)
            n_real.append(None)
            n_cmp.append(None)
            ghash.append(None)
            o_own.append(None)
            o_status.append("not_applicable")
            o_en.append(None)
            o_tl.append(None)
            o_pairs.append(None)
            continue
        g, res, o_scores = groups[sc], results[sc], olsd[sc]
        scores = res["scores"]
        n_valid = res["n_valid"]
        tl_label = g["row_label"].get(r["source_row_id"])
        en_s = scores.get(ENGLISH_LABEL)
        tl_s = scores.get(tl_label) if tl_label else None
        if r["sentence_role"] == "en_proper":
            label = ENGLISH_LABEL
        elif r["sentence_role"] == "en_transferred":
            label = tl_label
        else:
            label = None
        if r["sentence_role"] == "en_transferred" and tl_label is None:
            s, st = None, STATUS_INELIGIBLE
        elif label is None:
            s, st = None, "not_applicable"
        elif label in res["invalid_labels"]:
            s, st = None, "invalid_realization"
        elif n_valid < 2:
            s, st = None, "insufficient_realizations"
        else:
            s, st = scores[label], "ok"
        if st == "ok" and n_valid < 3:
            o_s, o_st = None, "insufficient_realizations"
        elif st == "ok":
            o_s, o_st = o_scores[label], "ok"
        else:
            o_s, o_st = None, st
        o_own.append(o_s)
        o_status.append(o_st)
        o_en.append(o_scores.get(ENGLISH_LABEL))
        o_tl.append(o_scores.get(tl_label) if tl_label else None)
        o_pairs.append((n_valid - 1) * (n_valid - 2) // 2 if n_valid >= 3 else None)
        own.append(s)
        own_status.append(st)
        en_score.append(en_s)
        tl_score.append(tl_s)
        n_real.append(n_valid)
        n_cmp.append(n_valid - 1 if n_valid >= 2 else None)
        ghash.append(hashes[sc])

    out = table.copy()
    out["divergence_score"] = pd.array(own, dtype="Float64")
    out["divergence_status"] = own_status
    out["divergence_en_realization_score"] = pd.array(en_score, dtype="Float64")
    out["divergence_tl_realization_score"] = pd.array(tl_score, dtype="Float64")
    out["divergence_n_realizations"] = pd.array(n_real, dtype="Int64")
    out["divergence_n_comparisons"] = pd.array(n_cmp, dtype="Int64")
    out["divergence_group_hash"] = ghash
    out["divergence_config_id"] = cid
    out["other_language_strategy_divergence_score"] = pd.array(o_own, dtype="Float64")
    out["other_language_strategy_divergence_status"] = o_status
    out["other_language_strategy_divergence_en_realization_score"] = pd.array(o_en, dtype="Float64")
    out["other_language_strategy_divergence_tl_realization_score"] = pd.array(o_tl, dtype="Float64")
    out["other_language_strategy_divergence_n_pairs"] = pd.array(o_pairs, dtype="Int64")

    audit_rows = []
    for sc, g in groups.items():
        res = results[sc]
        for label, sentence in g["realizations"]:
            audit_rows.append({
                "semantic_content_id": sc,
                "Proper English strategy": g["english"],
                "realization_label": label,
                "realization_sentence": sentence,
                "tokens": " ".join(divergence_tokens(sentence)),
                "n_valid_realizations": res["n_valid"],
                "divergence_score": res["scores"].get(label),
                "other_language_strategy_divergence_score": olsd[sc].get(label),
                "eligible": True,
                "group_hash": hashes[sc],
            })
        for source_row_id, lang, sentence in g["ineligible"]:
            audit_rows.append({
                "semantic_content_id": sc,
                "Proper English strategy": g["english"],
                "realization_label": f"{lang} (ineligible: {source_row_id})",
                "realization_sentence": sentence,
                "tokens": " ".join(divergence_tokens(sentence)),
                "n_valid_realizations": res["n_valid"],
                "divergence_score": None,
                "other_language_strategy_divergence_score": None,
                "eligible": False,
                "group_hash": hashes[sc],
            })
    return out, pd.DataFrame(audit_rows)
