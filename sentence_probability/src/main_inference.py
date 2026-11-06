from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cross_model_consistency import MIN_SHARED_PAIRS_SPEARMAN
from viz_data2 import COMMON_CRAWL_TERM, LANGUAGE_FE_PREFIX, SOURCE_ROW_ID_COL, student_t_ppf, student_t_sf2

MIN_MODEL_CLUSTERS = 4
FEW_MODEL_CLUSTERS = 10


def spearman_long(rho, n, labels, observed, min_shared=MIN_SHARED_PAIRS_SPEARMAN):
    names = list(rho.index)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            k = int(n.loc[a, b])
            if a not in observed or b not in observed:
                status = "missing model (no rows)"
            elif k == 0:
                status = "no shared sentence pairs"
            elif k < min_shared:
                status = f"insufficient overlap (< {min_shared} shared pairs)"
            elif not np.isfinite(rho.loc[a, b]):
                status = "undefined (constant ranks)"
            else:
                status = "ok"
            rows.append({"model_a": a, "model_b": b, "label_a": labels.get(a, a), "label_b": labels.get(b, b),
                         "rho": rho.loc[a, b], "n_shared_pairs": k, "status": status})
    out = pd.DataFrame(rows)
    if len(out):
        out.loc[out["status"] != "ok", ["rho"]] = np.nan
    return out


def family_design(frame, predictors, language_fe=True):
    complete = frame.dropna(subset=["y"] + list(predictors))
    cols, terms = [np.ones(len(complete))], ["intercept"]
    for p in predictors:
        cols.append(complete[p].to_numpy(dtype=float))
        terms.append(p)
    levels = sorted(complete["language"].unique()) if language_fe and len(complete) else []
    for lv in levels[1:]:
        cols.append((complete["language"] == lv).to_numpy(dtype=float))
        terms.append(f"{LANGUAGE_FE_PREFIX}{lv}")
    X = np.column_stack(cols) if len(complete) else np.empty((0, len(terms)))
    return complete, X, complete["y"].to_numpy(dtype=float), terms


def _codes(values):
    return pd.factorize(pd.Series(values).astype(str))[0]


def _cluster_meat(X, e, codes):
    G = int(codes.max()) + 1
    U = np.zeros((G, X.shape[1]))
    np.add.at(U, codes, X * e[:, None])
    return U.T @ U, G


def cluster_covariances(X, e, model_codes, pair_codes):
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    out = {}
    finite = (n - 1) / (n - k)
    covs = {}
    for name, codes in (("model", model_codes), ("pair", pair_codes)):
        meat, G = _cluster_meat(X, e, codes)
        covs[name] = (G / (G - 1) * finite * bread @ meat @ bread if G > 1 else np.full((k, k), np.nan), G)
    inter = _codes([f"{a}|{b}" for a, b in zip(model_codes, pair_codes)])
    meat, G_int = _cluster_meat(X, e, inter)
    covs["intersection"] = (G_int / (G_int - 1) * finite * bread @ meat @ bread, G_int)
    out["CR1 one-way by model"] = (*covs["model"], "")
    out["CR1 one-way by sentence pair"] = (*covs["pair"], "")
    two = covs["model"][0] + covs["pair"][0] - covs["intersection"][0]#Vmodel+Vpair-V(model and pair)をここでやってる
    note = ""
    w, v = np.linalg.eigh((two + two.T) / 2)
    if (w < 0).any():
        two = v @ np.diag(np.clip(w, 0, None)) @ v.T
        note = f"{int((w < 0).sum())} negative eigenvalue(s) set to 0 (Cameron-Gelbach-Miller PSD fix)"
    out["two-way CGM (model x sentence pair)"] = (two, min(covs["model"][1], covs["pair"][1]), note)
    return out


def vif(X, terms):
    out = {}
    for j, t in enumerate(terms):
        if t == "intercept":
            continue
        others = [c for c in range(X.shape[1]) if c != j]
        b, *_ = np.linalg.lstsq(X[:, others], X[:, j], rcond=None)
        res = X[:, j] - X[:, others] @ b
        tss = ((X[:, j] - X[:, j].mean()) ** 2).sum()
        r2 = 1 - (res @ res) / tss if tss > 0 else np.nan
        out[t] = 1 / (1 - r2) if np.isfinite(r2) and r2 < 1 else np.inf
    return out


def predictor_level(complete, term):
    if term == "intercept" or term.startswith(LANGUAGE_FE_PREFIX) or term == COMMON_CRAWL_TERM:
        return "intercept" if term == "intercept" else "language"
    col = complete[term]
    if (complete.assign(_v=col).groupby("model_name")["_v"].nunique(dropna=True) <= 1).all():
        return "model"
    if (complete.assign(_v=col).groupby(["language", SOURCE_ROW_ID_COL])["_v"].nunique(dropna=True) <= 1).all():
        return "sentence pair"
    return "observation"


def cluster_robust_family(frame, predictors, saved_coefs, family, language_fe=True):
    complete, X, y, terms = family_design(frame, predictors, language_fe)
    n, k = X.shape
    rank = int(np.linalg.matrix_rank(X)) if n else 0
    diag = {"family": family, "n": n, "k": k, "design_rank": rank, "rank_deficient": rank < k}
    if n == 0 or rank < k or n <= k:
        diag["status"] = "not estimable (rank deficient or too few rows)"
        return [], diag
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    saved = saved_coefs.set_index("term")["beta"].reindex(terms).to_numpy(dtype=float)
    if not np.allclose(beta, saved, rtol=1e-7, atol=1e-9):
        raise AssertionError(f"cluster-robust design for {family} does not reproduce the saved coefficients")
    e = y - X @ beta
    pair_ids = complete["language"].astype(str) + ":" + complete[SOURCE_ROW_ID_COL].astype(str)
    model_codes, pair_codes = _codes(complete["model_name"]), _codes(pair_ids)
    G_model, G_pair = int(model_codes.max()) + 1, int(pair_codes.max()) + 1
    covs = cluster_covariances(X, e, model_codes, pair_codes)
    vifs = vif(X, terms)
    cond = float(np.linalg.cond((X - X.mean(axis=0) * (np.arange(k) > 0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1)))
    saved_idx = saved_coefs.set_index("term")
    rows = []
    for j, t in enumerate(terms):
        level = predictor_level(complete, t)
        r = {"family": family, "term": t, "predictor_level": level, "beta": beta[j],
             "standardized_beta": saved_idx.loc[t, "standardized_beta"] if t in saved_idx.index else np.nan,
             "classical_std_error": saved_idx.loc[t, "std_error"], "classical_p_value": saved_idx.loc[t, "p_value"],
             "n": n, "n_model_clusters": G_model, "n_sentence_pair_clusters": G_pair, "vif": vifs.get(t, np.nan)}
        if level == "model":
            vals = complete.groupby("model_name")[t].first()
            counts = vals.value_counts()
            r["n_distinct_model_values"] = int(len(counts))
            r["model_level_support"] = ", ".join(f"{v:g}: {int(c)} model(s)" for v, c in counts.sort_index().items())
        elif t == COMMON_CRAWL_TERM:
            r["n_languages"] = int(complete["language"].nunique())
        for name in ("two-way CGM (model x sentence pair)",):
            V, G, _note = covs[name]
            short = "cr_twoway"
            se = math.sqrt(V[j, j]) if np.isfinite(V[j, j]) and V[j, j] > 0 else np.nan
            df = G - 1
            crit = student_t_ppf(0.975, df) if df >= 1 else np.nan
            tv = beta[j] / se if se and np.isfinite(se) else np.nan
            r[f"{short}_std_error"] = se
            r[f"{short}_df"] = df
            r[f"{short}_ci_lower"] = beta[j] - crit * se if np.isfinite(crit) else np.nan
            r[f"{short}_ci_upper"] = beta[j] + crit * se if np.isfinite(crit) else np.nan
            r[f"{short}_p_value"] = student_t_sf2(tv, df) if np.isfinite(tv) and df >= 1 else np.nan
        r["inference_flag"] = inference_flag(r, G_model)
        rows.append(r)
    diag.update({"status": "ok", "n_model_clusters": G_model, "n_sentence_pair_clusters": G_pair,
                 "max_vif": float(np.nanmax([v for v in vifs.values() if np.isfinite(v)] or [np.nan])),
                 "terms_vif_gt_10": ", ".join(t for t, v in vifs.items() if v > 10),
                 "condition_number_standardized": cond,
                 "two_way_psd_fix": covs["two-way CGM (model x sentence pair)"][2],
                 "covariance_estimators": "CR1 = G/(G-1)*(N-1)/(N-K) (X'X)^-1 (sum_g X_g'e_g e_g'X_g) (X'X)^-1; "
                                          "two-way = CR1(model) + CR1(pair) - CR1(model x pair) (CGM 2011); "
                                          "t with G-1 df (two-way: min(G_model, G_pair) - 1)",
                 "few_model_clusters": G_model < FEW_MODEL_CLUSTERS})
    return rows, diag


def inference_flag(r, G_model):
    flags = []
    if r["term"] == "intercept":
        return "intercept (not interpreted)"
    if G_model < MIN_MODEL_CLUSTERS:
        flags.append(f"only {G_model} model clusters: model-clustered inference not possible")
    elif G_model < FEW_MODEL_CLUSTERS:
        flags.append(f"few model clusters ({G_model}): two-way t uses df = {G_model - 1} (conservative)")
    if r["predictor_level"] == "model":
        nd = r.get("n_distinct_model_values", 0)
        flags.append(f"model-level predictor: identified by {nd} distinct value(s) across {G_model} models "
                     "(effective N = models, not observations)")
        support = r.get("model_level_support", "")
        if nd == 2 and " 1 model(s)" in support:
            flags.append("one side of the contrast is a single model (confounded with that model)")
        if nd < 2:
            flags.append("no variation")
    if r["term"] == COMMON_CRAWL_TERM:
        flags.append(f"language-level predictor: varies across only {r.get('n_languages')} languages; model x "
                     "sentence-pair clustering does not account for language-level dependence (p-value not robust)")
    if np.isfinite(r.get("vif", np.nan)) and r["vif"] > 10:
        flags.append(f"high collinearity (VIF {r['vif']:.1f})")
    return "; ".join(flags) if flags else "ok"
