# このコードは論文のsec 5.1のjapanese controlに関するlinear mixed modelにおいて使われているコードです。


from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import rq1_lmm
from viz_data2 import (
    CONTROL_CATEGORY_COL,
    CONTROL_JA_BASELINE,
    CONTROL_JAPANESE,
    EXPERIMENT_JAPANESE_CONTROL,
    EXPERIMENT_STANDARD,
    JA_BASELINE_NEG,
    JAPANESE_COMPARISON_POS,
    JAPANESE_PREFERENCE_COL,
    NO_CONTEXT_PROMPT_LANGUAGE,
    ORIGINAL_TARGET_LANGUAGE_COL,
    SEMANTIC_CONTENT_KEY_COL,
    SLOR_COL,
    DELTA_SLOR,
    TARGET_COMPARISON_POS,
    TARGET_PREFERENCE_COL,
    canonical_language_code,
    language_axis_order,
    pair_japanese_baseline,
    paired_baseline_inputs,
    student_t_ppf,
    student_t_sf2,
)

PREFERENCE_SUBSET = "gtgt"
PREFERENCE = ">>"
OUTPUT_FILE = "last_ja_control_statistics.xlsx"
SHEETS = ("Summary", "Language_Contrasts", "Pooled_Contrast", "Model_Coverage", "Pairing_Audit",
          "Model_Diagnostics", "Methods")
ALPHA = 0.05
CONTROL_NEG = "Literal translation of Proper target language into Japanese"
TL_CONDITION = "target-language strategy transferred into Japanese (JA control)"
EN_CONDITION = "English strategy transferred into Japanese (JA baseline)"
COMPARISON = "target-language transfer − English transfer (both evaluated in Japanese)"

"""
primaryが論文sec5.1において使われているja control用のLMM
y=dslor(ja control)-dslor(ja baseline)
固定切片は0でモデルから除外している
"""

SPECS = [
    ("primary", "y ~ 0 + language + (1 | model) + (1 | model_language) + (1 | item)",
     "primary: language mean differences; random intercepts for model, model x language and semantic pair",
     rq1_lmm.SATTERTHWAITE, "main", "observed mean paired difference"),
]
POOLED_CONTRAST = "pooled (equal weight)"

_R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages(library(lme4))
spec <- read.csv(args[1], stringsAsFactors = FALSE)
out <- args[2]
""" + rq1_lmm.R_SATTERTHWAITE + r"""
for (k in seq_len(nrow(spec))) {
  s <- spec[k, ]
  d <- read.csv(s$data, stringsAsFactors = FALSE)
  d$language <- factor(d$language, levels = strsplit(s$levels, ",")[[1]])
  d$model <- factor(d$model)
  d$item <- factor(d$item)
  d$model_language <- factor(paste(d$model, d$language, sep = ":"))
  warns <- character(0)
  lmer_k <- fit_lmer(k)
  fit <- tryCatch(withCallingHandlers(
      lmer_k(as.formula(s$formula), data = d, REML = TRUE,
             control = lmerControl(optimizer = "bobyqa", calc.derivs = TRUE)),
      warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }),
    error = function(e) e)
  p <- file.path(out, s$fit_id)
  if (inherits(fit, "error")) {
    writeLines(c("error", conditionMessage(fit)), paste0(p, "_status.txt"))
    next
  }
  if (has_contrasts[k]) {
    tab <- tryCatch(satterthwaite_table(fit, s$contrasts), error = function(e) e)
    if (inherits(tab, "error")) {
      writeLines(c("error", conditionMessage(tab)), paste0(p, "_status.txt"))
      next
    }
    write.csv(tab, paste0(p, "_satterthwaite.csv"), row.names = FALSE)
  }
  cf <- summary(fit)$coefficients
  write.csv(data.frame(term = rownames(cf), estimate = cf[, "Estimate"], std_error = cf[, "Std. Error"],
                       t_value = cf[, "t value"]),
            paste0(p, "_fixef.csv"), row.names = FALSE)
  write.csv(as.matrix(vcov(fit)), paste0(p, "_vcov.csv"))
  write.csv(as.data.frame(VarCorr(fit)), paste0(p, "_varcorr.csv"), row.names = FALSE)
  conv <- fit@optinfo$conv$lme4$messages
  writeLines(c("ok",
               paste0("singular=", isSingular(fit)),
               paste0("nobs=", nobs(fit)),
               paste0("ngrps=", paste(names(ngrps(fit)), ngrps(fit), sep = ":", collapse = ";")),
               paste0("loglik=", as.numeric(logLik(fit))),
               paste0("npar=", attr(logLik(fit), "df")),
               paste0("optimizer_code=", fit@optinfo$conv$opt),
               paste0("lme4_messages=", paste(gsub("[\r\n]", " ", conv), collapse = " | ")),
               paste0("warnings=", paste(gsub("[\r\n]", " ", warns), collapse = " | "))),
             paste0(p, "_status.txt"))
}
"""


def english_centric_models():
    cls = rq1_lmm.model_classification()
    return cls, cls.loc[cls["english_centric"], "model_name"].tolist()


def _stripped(series):
    return series.where(series.notna(), "").astype(str).str.strip()


def _item_id(content):
    return "it_" + hashlib.sha256(str(content).encode("utf-8")).hexdigest()[:12]


def paired_frame(df, models, include_identical=False):
    japanese, baseline = paired_baseline_inputs(df, SLOR_COL, include_identical, PREFERENCE_SUBSET)
    keep = set(models)
    japanese = japanese.loc[japanese["model_name"].isin(keep)] if len(japanese) else japanese
    baseline = baseline.loc[baseline["model_name"].isin(keep)] if len(baseline) else baseline
    return pair_japanese_baseline(japanese, baseline, PREFERENCE_SUBSET)


def difference_frame(long):
    keys = ["model_name", ORIGINAL_TARGET_LANGUAGE_COL, SEMANTIC_CONTENT_KEY_COL]
    if long.empty:
        cols = keys + ["dslor_target_transfer", "dslor_english_transfer", "difference", "language", "content", "item",
                       "same_native_japanese_sentence", "control_origin_target_annotation"]
        return pd.DataFrame(columns=cols), {"paired observations": 0}
    control = long.loc[long[CONTROL_CATEGORY_COL] == CONTROL_JAPANESE]
    base = long.loc[long[CONTROL_CATEGORY_COL] == CONTROL_JA_BASELINE]
    checks = {
        "control rows: Japanese preference annotation exactly >>": bool((_stripped(control[JAPANESE_PREFERENCE_COL])
                                                                         == PREFERENCE).all()),
        "baseline rows: Target language preference annotation exactly >>": bool(
            (_stripped(base[TARGET_PREFERENCE_COL]) == PREFERENCE).all()),
        "control rows: Proper strategy in Japanese vs Literal translation of Proper target language into Japanese":
            bool(((control["pos"] == JAPANESE_COMPARISON_POS) & (control["neg"] == CONTROL_NEG)).all()),
        "baseline rows: ja target-language comparison (Proper strategy in target language vs Literal translation "
        "of Proper English into target language)": bool(
            ((base["pos"] == TARGET_COMPARISON_POS) & (base["neg"] == JA_BASELINE_NEG)
             & (base["Target language"].map(canonical_language_code) == "ja")).all()),
        "no duplicate (model, language, semantic pair, condition) observations": bool(
            not long.duplicated(keys + [CONTROL_CATEGORY_COL]).any()),
    }
    if not all(checks.values()):
        raise ValueError(f"Japanese-control pairing failed validation: {checks}")
    c = control[keys + [DELTA_SLOR, "pos_sentence", "neg_sentence", TARGET_PREFERENCE_COL]].rename(
        columns={DELTA_SLOR: "dslor_target_transfer", "pos_sentence": "native_ja_control",
                 "neg_sentence": "transfer_target", TARGET_PREFERENCE_COL: "control_origin_target_annotation"})
    b = base[keys + [DELTA_SLOR, "pos_sentence", "neg_sentence"]].rename(
        columns={DELTA_SLOR: "dslor_english_transfer", "pos_sentence": "native_ja_baseline",
                 "neg_sentence": "transfer_english"})
    d = c.merge(b, on=keys, how="inner", validate="one_to_one")
    if len(d) != len(c) or len(d) != len(b):
        raise ValueError("Japanese-control pairing is not one-to-one")
    d["difference"] = d["dslor_target_transfer"] - d["dslor_english_transfer"]
    d["language"] = d[ORIGINAL_TARGET_LANGUAGE_COL]
    d["content"] = d[SEMANTIC_CONTENT_KEY_COL].map(_item_id)
    d["item"] = d["language"] + ":" + d["content"]
    d["same_native_japanese_sentence"] = d["native_ja_control"] == d["native_ja_baseline"]
    d["control_origin_target_annotation"] = _stripped(d["control_origin_target_annotation"])
    return d, checks


def pairing_audit(df, models, include_identical=False):
    keep = set(models)
    raw = df.loc[df["model_name"].isin(keep)]
    if raw.empty or not {"experiment", "pos", JAPANESE_PREFERENCE_COL} <= set(raw.columns):
        return pd.DataFrame(), 0
    if "prompt_language" in raw.columns:
        raw = raw.loc[raw["prompt_language"].astype(str) == NO_CONTEXT_PROMPT_LANGUAGE]
    ctrl = raw.loc[(raw["experiment"] == EXPERIMENT_JAPANESE_CONTROL) & (raw["pos"] == JAPANESE_COMPARISON_POS)].copy()
    base = raw.loc[(raw["experiment"] == EXPERIMENT_STANDARD) & (raw["pos"] == TARGET_COMPARISON_POS)
                   & (raw["neg"] == JA_BASELINE_NEG)
                   & (raw["Target language"].map(canonical_language_code) == "ja")].copy()
    base["content"] = base["Proper English strategy"].astype(str)
    base_dups = int(base.duplicated(["model_name", "content"]).sum())
    base = base.drop_duplicates(["model_name", "content"])
    base_info = {(m, c): (p == n, pd.notna(pd.to_numeric(v, errors="coerce")), str(a).strip())
                 for m, c, p, n, v, a in zip(base["model_name"], base["content"], base["pos_sentence"],
                                             base["neg_sentence"], base[SLOR_COL],
                                             base[TARGET_PREFERENCE_COL].where(base[TARGET_PREFERENCE_COL].notna(),
                                                                               ""))}
    ctrl["language"] = ctrl["Target language"].map(canonical_language_code)
    ctrl_value = pd.to_numeric(ctrl[SLOR_COL], errors="coerce")
    ctrl_pref = _stripped(ctrl[JAPANESE_PREFERENCE_COL])
    reasons = []
    for (_i, r), v, pref in zip(ctrl.iterrows(), ctrl_value, ctrl_pref):
        content = r["Proper English strategy"]
        if not include_identical and r["pos_sentence"] == r["neg_sentence"]:
            reasons.append("control_pair_identical")
        elif pd.isna(v):
            reasons.append("control_score_invalid")
        elif pref != PREFERENCE:
            reasons.append("japanese_preference_not_>>")
        elif pd.isna(content):
            reasons.append("missing_proper_english_strategy")
        elif (r["model_name"], str(content)) not in base_info:
            reasons.append("missing_ja_baseline")
        else:
            identical, valid, bpref = base_info[(r["model_name"], str(content))]
            if identical and not include_identical:
                reasons.append("baseline_pair_identical")
            elif not valid:
                reasons.append("baseline_score_invalid")
            elif bpref != PREFERENCE:
                reasons.append("baseline_target_preference_not_>>")
            else:
                reasons.append("paired")
    ctrl["reason"] = reasons
    order = ["paired", "control_pair_identical", "control_score_invalid", "japanese_preference_not_>>",
             "missing_proper_english_strategy", "missing_ja_baseline", "baseline_pair_identical",
             "baseline_score_invalid", "baseline_target_preference_not_>>"]
    rows = []
    for (lang, model), g in ctrl.groupby(["language", "model_name"], sort=False):
        counts = g["reason"].value_counts()
        rows.append({"original_language": lang, "model_name": model, "n_control_candidates": int(len(g)),
                     **{f"n_{k}": int(counts.get(k, 0)) for k in order},
                     "n_duplicate_control_identities": int(g.duplicated(["language", "model_name",
                                                                         "Proper English strategy"]).sum())})
    out = pd.DataFrame(rows)
    if len(out):
        lang_order = {l: i for i, l in enumerate(language_axis_order(out["original_language"].unique()))}
        out = out.assign(_o=out["original_language"].map(lang_order)).sort_values(
            ["_o", "model_name"], kind="stable").drop(columns="_o").reset_index(drop=True)
        total = {"original_language": "ALL", "model_name": "ALL",
                 **{c: int(out[c].sum()) for c in out.columns if c.startswith("n_")}}
        out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)
    return out, base_dups


def fit_lmms(specs, timeout=3600):
    rscript = shutil.which("Rscript")
    if not rscript:
        return {s["fit_id"]: {"status": "error: Rscript not found on PATH"} for s in specs}
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rows = []
        for s in specs:
            path = tmp / f"{s['fit_id']}_data.csv"
            s["data"][["y", "language", "model", "item", "content"]].to_csv(path, index=False)
            contrasts = (rq1_lmm.write_contrast_file(tmp / f"{s['fit_id']}_contrasts.csv", s["satterthwaite"])
                         if s.get("satterthwaite") else "")
            rows.append({"fit_id": s["fit_id"], "data": str(path), "formula": s["formula"],
                         "levels": ",".join(s["levels"]), "contrasts": contrasts})
        pd.DataFrame(rows).to_csv(tmp / "spec.csv", index=False)
        (tmp / "fit.R").write_text(_R_SCRIPT, encoding="utf-8")
        proc = subprocess.run([rscript, str(tmp / "fit.R"), str(tmp / "spec.csv"), str(tmp)], capture_output=True,
                              text=True, timeout=timeout)
        rq1_lmm.check_lmertest(tmp, proc)
        for s in specs:
            fid = s["fit_id"]
            status_file = tmp / f"{fid}_status.txt"
            if not status_file.is_file():
                results[fid] = {"status": f"error: R failed ({proc.stderr.strip()[-400:]})"}
                continue
            lines = status_file.read_text(encoding="utf-8").splitlines()
            if lines[0] != "ok":
                results[fid] = {"status": "error: " + " ".join(lines[1:])}
                continue
            results[fid] = {"status": "ok", **dict(line.split("=", 1) for line in lines[1:]),
                            "fixef": pd.read_csv(tmp / f"{fid}_fixef.csv"),
                            "vcov": pd.read_csv(tmp / f"{fid}_vcov.csv", index_col=0),
                            "varcorr": pd.read_csv(tmp / f"{fid}_varcorr.csv")}
            if s.get("satterthwaite"):
                results[fid]["satterthwaite"] = pd.read_csv(tmp / f"{fid}_satterthwaite.csv")
    return results


def _converged(fit):
    msgs = [x for x in str(fit.get("lme4_messages", "")).split(" | ") if x and "boundary (singular)" not in x]
    return fit.get("optimizer_code") == "0" and not msgs and "converge" not in str(fit.get("warnings", "")).lower()


def _wald(estimate, se, df, satterthwaite=False):
    if not (np.isfinite(estimate) and np.isfinite(se) and se > 0):
        return np.nan, np.nan, np.nan, np.nan, "not estimable"
    stat = estimate / se
    if df is None:
        p, q = math.erfc(abs(stat) / math.sqrt(2.0)), rq1_lmm.Z975
        return stat, p, estimate - q * se, estimate + q * se, "Wald z"
    p, q = student_t_sf2(stat, df), student_t_ppf(0.975, df)
    label = f"Satterthwaite t (df = {df:.2f})" if satterthwaite else f"Wald t (df = {df})"
    return stat, p, estimate - q * se, estimate + q * se, label


def _varcorr_text(fit):
    vc = fit["varcorr"]
    parts = []
    for _i, r in vc.iterrows():
        if pd.isna(r.get("var2")):
            name = r["grp"] if r["var1"] in ("(Intercept)", None) or pd.isna(r["var1"]) else f"{r['grp']}[{r['var1']}]"
            parts.append(f"{name}: var={float(r['vcov']):.4g}")
    return "; ".join(parts)


def _descriptives(d, languages):
    out = {}
    for lang in languages:
        g = d.loc[d["language"] == lang]
        out[lang] = {"n_models": int(g["model_name"].nunique()), "n_semantic_pairs": int(g["item"].nunique()),
                     "n_paired_observations": int(len(g)),
                     "mean_dslor_target_transfer": float(g["dslor_target_transfer"].mean()),
                     "mean_dslor_english_transfer": float(g["dslor_english_transfer"].mean()),
                     "mean_paired_difference": float(g["difference"].mean())}
    return out


def _contrasts(fit, languages, df_, spec_id, desc, formula, desc_stats, d, estimand):
    sat = df_ == rq1_lmm.SATTERTHWAITE
    beta = fit["fixef"].set_index("term")["estimate"]
    vcov = fit["vcov"]
    names = [f"language{l}" for l in languages]
    rows = []
    for lang, name in zip(languages, names):
        est, se = float(beta[name]), float(math.sqrt(vcov.loc[name, name]))
        rows.append({"specification": spec_id, "target_language": lang,
                     "comparison": f"{lang}: {COMPARISON}", **desc_stats[lang],
                     "estimated_difference": est, "std_error": se,
                     "inference": "coefficient estimate and SE only (not tested)"})

    #ここで、論文で使ってるpooled contrastにおけるwald tを計算している
    w = np.full(len(names), 1.0 / len(names))
    b = beta.reindex(names).to_numpy(dtype=float)
    v = vcov.loc[names, names].to_numpy(dtype=float)
    est, se = float(w @ b), float(math.sqrt(w @ v @ w))
    df_p = rq1_lmm.satterthwaite_df(fit, POOLED_CONTRAST) if sat else df_
    stat, p, lo, hi, inf = _wald(est, se, df_p, sat)
    pooled = {"specification": spec_id, "contrast": "equally weighted mean over target languages of "
                                                    f"({COMPARISON})",
              "target_languages": ",".join(languages), "weights": ", ".join(f"{l}: 1/{len(languages)}"
                                                                           for l in languages),
              "n_models": int(d["model_name"].nunique()), "n_semantic_pairs": int(d["item"].nunique()),
              "n_paired_observations": int(len(d)),
              "mean_dslor_target_transfer_equal_weight": float(np.mean(
                  [desc_stats[l]["mean_dslor_target_transfer"] for l in languages])),
              "mean_dslor_english_transfer_equal_weight": float(np.mean(
                  [desc_stats[l]["mean_dslor_english_transfer"] for l in languages])),
              "estimated_difference": est, "std_error": se, "wald_statistic": stat, "inference": inf,
              "df": df_p if df_p is not None else np.inf, "p_value": p, "ci_lower_95": lo, "ci_upper_95": hi,
              "significant_0.05": bool(p < ALPHA) if np.isfinite(p) else False,
              "multiplicity": "single planned contrast (not Holm-adjusted)"}
    for r in rows + [pooled]:
        r.update({"estimand": estimand, "model_formula": formula, "description": desc,
                  "converged": _converged(fit), "singular": fit.get("singular") == "TRUE"})
    return rows, pooled


def per_model_mean_differences(d, languages):
    return d.groupby(["model_name", "language"])["difference"].mean().unstack().reindex(columns=languages)


def _empty_result(cls, d, languages, checks, audit, base_dups, n_ec):
    coverage = pd.DataFrame([{"model_name": r["model_name"], "model_id": r["model_id"], "label": r["label"],
                              "family": r["family"], "english_centric": True, "included": False,
                              "status": "unavailable: no valid paired Japanese-control observations",
                              "n_paired_observations": 0} for _i, r in cls.loc[cls["english_centric"]].iterrows()])
    status = pd.DataFrame([{"fit_id": "none", "status": "not fitted: no valid paired Japanese-control observations"}])
    return {"pairs": d, "languages": languages, "checks": checks, "audit": audit, "baseline_duplicates": base_dups,
            "language_contrasts": pd.DataFrame(), "pooled": pd.DataFrame(),
            "diagnostics": status, "coverage": coverage, "fits": {}, "n_ec_models": n_ec}


def run_ja_control_lmm(df, include_identical=False):
    cls, ec_models = english_centric_models()
    long = paired_frame(df, ec_models, include_identical)
    d, checks = difference_frame(long)
    languages = [l for l in language_axis_order(d["language"].unique())]
    audit, base_dups = pairing_audit(df, ec_models, include_identical)
    n_paired_audit = int(audit.loc[audit["model_name"] == "ALL", "n_paired"].iloc[0]) if len(audit) else 0
    checks["pairing audit reproduces the paired observations"] = n_paired_audit == len(d)
    if d.empty:
        return _empty_result(cls, d, languages, checks, audit, base_dups, len(ec_models))
    checks["every paired difference = control ΔSLOR − baseline ΔSLOR"] = bool(np.allclose(
        d["difference"], d["dslor_target_transfer"] - d["dslor_english_transfer"]))
    checks["native Japanese sentence identical in both conditions (share)"] = float(
        d["same_native_japanese_sentence"].mean())
    subsets = {"main": d}
    specs = []
    for fid, formula, desc, rule, subset, _est in SPECS:
        data = subsets[subset].rename(columns={"difference": "y", "model_name": "model"})
        levels = [l for l in languages if (data["language"] == l).any()]
        planned = None
        if rule == rq1_lmm.SATTERTHWAITE:
            names = [f"language{l}" for l in levels]
            planned = {POOLED_CONTRAST: {n: 1.0 / len(names) for n in names}}
        specs.append({"fit_id": fid, "data": data, "formula": formula, "levels": levels, "satterthwaite": planned})
    fits = fit_lmms(specs)
    lang_rows, pooled_rows, diagnostics = [], [], []
    for (fid, formula, desc, rule, subset, estimand), s in zip(SPECS, specs):
        fit = fits[fid]
        data = subsets[subset]
        n_models = int(data["model_name"].nunique())
        df_ = rule if rule == rq1_lmm.SATTERTHWAITE else None if rule is None else max(n_models - 1, 1)
        diag = {"fit_id": fid, "formula": formula, "description": desc, "data": subset,
                "n_obs": int(len(data)), "n_models": n_models, "n_items": int(data["item"].nunique()),
                "n_semantic_contents": int(data["content"].nunique()), "estimand": estimand, "status": fit["status"]}
        if fit["status"] == "ok":
            diag.update({"converged": _converged(fit), "singular": fit.get("singular") == "TRUE",
                         "optimizer_code": fit.get("optimizer_code"), "lme4_messages": fit.get("lme4_messages"),
                         "warnings": fit.get("warnings"), "loglik_REML": float(fit.get("loglik")),
                         "n_parameters": int(fit.get("npar")), "groups": fit.get("ngrps"),
                         "variance_components": _varcorr_text(fit),
                         "inference": "Wald z (large-sample)" if df_ is None
                         else "Satterthwaite t (lmerTest::contest1D, contrast-specific df)"
                         if df_ == rq1_lmm.SATTERTHWAITE
                         else f"Wald t, df = n_models - 1 = {df_} (conservative model-level df)"})
            desc_stats = _descriptives(data, s["levels"])
            rows, pooled = _contrasts(fit, s["levels"], df_, fid, desc, formula, desc_stats, data, estimand)
            lang_rows += rows
            pooled_rows.append(pooled)
        diagnostics.append(diag)
    per_model = per_model_mean_differences(d, languages)
    coverage = []
    by_model = d.groupby(["model_name", "language"]).size().unstack(fill_value=0).reindex(columns=languages,
                                                                                        fill_value=0)
    for _i, r in cls.loc[cls["english_centric"]].iterrows():
        m = r["model_name"]
        n = by_model.loc[m] if m in by_model.index else pd.Series(0, index=languages)
        total = int(n.sum())
        coverage.append({"model_name": m, "model_id": r["model_id"], "label": r["label"], "family": r["family"],
                         "english_centric": True, "included": total > 0,
                         "status": "included" if total > 0 else "unavailable: no valid paired Japanese-control "
                                                                "observations",
                         "n_paired_observations": total,
                         "n_semantic_pairs": int(d.loc[d["model_name"] == m, "item"].nunique()),
                         **{f"n_{l}": int(n[l]) for l in languages},
                         **{f"mean_difference_{l}": float(per_model.loc[m, l]) if m in per_model.index
                            and np.isfinite(per_model.loc[m, l]) else np.nan for l in languages}})
    return {"pairs": d, "languages": languages, "checks": checks, "audit": audit, "baseline_duplicates": base_dups,
            "language_contrasts": pd.DataFrame(lang_rows), "pooled": pd.DataFrame(pooled_rows),
            "diagnostics": pd.DataFrame(diagnostics), "coverage": pd.DataFrame(coverage), "fits": fits, "n_ec_models": len(ec_models)}


def _summary_table(res):
    if res["language_contrasts"].empty:
        return pd.DataFrame([{"contrast": "none", "interpretation": "no valid paired Japanese-control observations "
                                                                   "for any English-centric model; nothing estimated"}])
    lc = res["language_contrasts"]
    pc = res["pooled"]
    prim = lc.loc[lc["specification"] == "primary"]
    rows = []
    for _i, r in prim.iterrows():
        rows.append({"contrast": f"{r['target_language']} vs English transfer (in Japanese)",
                     "target_language": r["target_language"], "n_models": r["n_models"],
                     "n_semantic_pairs": r["n_semantic_pairs"], "n_paired_observations": r["n_paired_observations"],
                     "mean_dslor_target_transfer": r["mean_dslor_target_transfer"],
                     "mean_dslor_english_transfer": r["mean_dslor_english_transfer"],
                     "estimated_difference": r["estimated_difference"]})
    p = pc.loc[pc["specification"] == "primary"]
    if len(p):
        r = p.iloc[0]
        rows.append({"contrast": "pooled: equally weighted mean over languages", "target_language":
                     r["target_languages"], "n_models": r["n_models"], "n_semantic_pairs": r["n_semantic_pairs"],
                     "n_paired_observations": r["n_paired_observations"],
                     "mean_dslor_target_transfer": r["mean_dslor_target_transfer_equal_weight"],
                     "mean_dslor_english_transfer": r["mean_dslor_english_transfer_equal_weight"],
                     "estimated_difference": r["estimated_difference"], "ci_lower_95": r["ci_lower_95"],
                     "ci_upper_95": r["ci_upper_95"], "t": r["wald_statistic"], "df": r["df"],
                     "p_value": r["p_value"], "p_adjusted": r["p_value"], "adjustment": "none (planned contrast)",
                     "significant_0.05": r["significant_0.05"]})
    out = pd.DataFrame(rows, columns=["contrast", "target_language", "n_models", "n_semantic_pairs",
                                      "n_paired_observations", "mean_dslor_target_transfer",
                                      "mean_dslor_english_transfer", "estimated_difference", "ci_lower_95",
                                      "ci_upper_95", "t", "df", "p_value", "p_adjusted", "adjustment",
                                      "significant_0.05"])

    def interpret(r):
        if not str(r["contrast"]).startswith("pooled"):
            return "language-specific coefficient: descriptive estimate (not tested)"
        if not np.isfinite(r["p_adjusted"]):
            return "not estimable"
        direction = ("native Japanese is preferred more strongly over the target-language transfer than over the "
                     "English transfer" if r["estimated_difference"] > 0 else
                     "native Japanese is preferred more strongly over the English transfer than over the "
                     "target-language transfer")
        if r["significant_0.05"]:
            return f"significant at α = 0.05: {direction}"
        return "not significant at α = 0.05 (not evidence of equivalence)"

    if len(out):
        out["interpretation"] = out.apply(interpret, axis=1)
    return out


def _methods_table(res, include_identical):
    rows = [
        ("Question", "For English-centric models: does mean ΔSLOR of the target-language strategy transferred into "
                     "Japanese differ from mean ΔSLOR of the English strategy transferred into Japanese (both "
                     "evaluated in Japanese)?"),
        ("Dependent variable", "saved SLOR_token_logprob_diff (ΔSLOR = SLOR(positive) − SLOR(negative)); never "
                               "recomputed"),
        ("Target-language condition (JA control)", "experiment japanese_control; Proper strategy in Japanese vs "
                                                   "Literal translation of Proper target language into Japanese"),
        ("English condition (JA baseline)", "standard ja row; Proper strategy in target language (Japanese) vs "
                                            "Literal translation of Proper English into target language"),
        ("Preference filters", "JA control: Japanese preference annotation exactly >>; JA baseline: Target "
                               "language preference annotation (of the ja row) exactly >> -- the Figure 4 "
                               "dataset (pair_japanese_baseline, pref-gtgt). The control row's own origin-language "
                               "Target language preference annotation is not required"),
        ("Pairing keys", "one-to-one inner join on (model_name, exact Proper English strategy = semantic content), "
                         "separately per original target language; duplicates refused"),
        ("Rows", f"no-context rows only; identical pos/neg pairs {'included' if include_identical else 'excluded'}; "
                 "valid numeric ΔSLOR on both sides"),
        ("Model universe", f"all {res['n_ec_models']} English-centric models of the statistical universe "
                           "(viz_data2.STAT_MODEL_SECTIONS, registry english_centric), incl. Tiny Aya Earth / Fire "
                           "/ Water; not restricted to the four plotted models"),
        ("Paired difference / sign convention", "d = ΔSLOR(JA control) − ΔSLOR(JA baseline). Positive: greater "
                                                "preference for native Japanese over the target-language transfer "
                                                "than over the English transfer; negative: the reverse; zero: "
                                                "equal. Both conditions share the same native Japanese sentence "
                                                f"(share {res['checks'].get('native Japanese sentence identical in both conditions (share)', float('nan')):.3f}), "
                                                "so d = SLOR(English transfer) − SLOR(target-language transfer)"),
        ("Primary model", SPECS[0][1] + "  (R lme4 lmer, REML, bobyqa); item = semantic sentence pair as "
                                        "evaluated in that language (language x semantic content, nested in "
                                        "language; every item is observed for every model, so the design is "
                                        "balanced and the coefficient language<l> equals the observed mean paired "
                                        "difference of language l); model_language = model x language"),
        ("Shared JA baseline", "a semantic content used under several original languages shares ONE JA-baseline "
                               "observation per model across those languages; the nested-item LMM does not model "
                               "this dependence"),
        ("Contrasts", "language-specific: the cell-mean coefficient language<l> (= mean paired difference; "
                      "estimate and SE, descriptive, not tested); pooled: equally weighted mean of the language "
                      "coefficients (weight 1/L each), so languages with more items do not dominate"),
        ("Inference", "Pooled contrast only (planned test of H0: C_JA = 0): t with contrast-specific Satterthwaite "
                      "denominator df from lmerTest::contest1D (possibly fractional); 95% CI = estimate ± "
                      "t_{0.975, df} SE. Kenward-Roger is not used"),
        ("Multiple comparisons", "none: a single planned test (pooled contrast)"),
        ("Interpretation", "a non-significant contrast is not evidence of equivalence; results are observational"),
        ("Pairing validation", "; ".join(f"{k}: {v}" for k, v in res["checks"].items())),
        ("Paired observations", f"{len(res['pairs'])} pairs; languages {', '.join(res['languages']) or 'none'}"),
        ("Source", "src/ja_control_lmm.py (called by export_plots2.py --only_main)"),
    ]
    return pd.DataFrame(rows, columns=["item", "definition"])


def write_workbook(res, path, include_identical=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    diag = res["diagnostics"].copy()
    notes = []
    notes.append({"fit_id": "limitation", "description": "denominator df",
                  "status": "primary: contrast-specific Satterthwaite df (lmerTest)"})
    if res["baseline_duplicates"]:
        notes.append({"fit_id": "limitation", "description": "duplicate baseline identities",
                      "status": f"{res['baseline_duplicates']} duplicated (model, content) ja rows in the raw data"})
    diag = pd.concat([diag, pd.DataFrame(notes)], ignore_index=True)
    pooled = res["pooled"].assign(level="LMM")
    lang = res["language_contrasts"].assign(level="LMM")
    with pd.ExcelWriter(path) as writer:
        _summary_table(res).to_excel(writer, sheet_name="Summary", index=False)
        lang.to_excel(writer, sheet_name="Language_Contrasts", index=False)
        pooled.to_excel(writer, sheet_name="Pooled_Contrast", index=False)
        res["coverage"].to_excel(writer, sheet_name="Model_Coverage", index=False)
        res["audit"].to_excel(writer, sheet_name="Pairing_Audit", index=False)
        diag.to_excel(writer, sheet_name="Model_Diagnostics", index=False)
        _methods_table(res, include_identical).to_excel(writer, sheet_name="Methods", index=False)
    return path
