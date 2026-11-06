# このコードはcalque論文のsec5.1のLinear mixed modelにおいて使われてるコードです。


from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from model_metadata2 import lookup_model
from viz_data2 import (
    DELTA_SLOR,
    ENGLISH_COMPARISON_POS,
    FAILURE_BAR_LABELS,
    STAT_MODEL_SECTIONS,
    SOURCE_ROW_ID_COL,
    TARGET_COMPARISON_POS,
    canonical_language_code,
    failure_rate_observations,
    failure_workbook_model_key,
    student_t_ppf,
    student_t_sf2,
)

NON_ENGLISH = ["de", "sv", "fr", "pt", "ja", "zh"]
EN_REFERENCE = "en"
NEC_REFERENCE = "de"
NEC_SECTION = "Other non-English-centric models"
Z975 = 1.959963984540054
SATTERTHWAITE = "satterthwaite"

FOCUS_LANGUAGES = {
    "OpenLLM-France/Claire-Mistral-7B-0.1": (("fr",), "Mistral-7B-v0.1 adapted (causal LM, LoRA) on French dialogue "
                                             "data only", "HF model card OpenLLM-France/Claire-Mistral-7B-0.1",
                                             "adaptation on French dialogue (LoRA), not full pretraining"),
    "OpenLLM-France/Lucie-7B": (("fr",), "balanced pretraining En 33%, Fr 32% (bilingual; English excluded from "
                                "Analysis 2)", "project metadata (MAIN_MONOLINGUAL_TRAINING)", ""),
    "AIInstituteLNCC/manaca-1b-base": (("pt",), "monolingual Portuguese, 100% Portuguese pretraining",
                                       "CLAUDE.md addendum 2026-10-12 §5", ""),
    "maritaca-ai/sabia-7b": (("pt",), "continual pretraining on Portuguese (Pt 100%)",
                             "project metadata (MAIN_MONOLINGUAL_TRAINING)", ""),
    "recogna-nlp/bode-13b-alpaca-pt-br-no-peft": (("pt",), "Llama 2 fine-tuned on Portuguese Alpaca",
                                                  "HF model card recogna-nlp/bode-13b-alpaca-pt-br-no-peft",
                                                  "Portuguese instruction fine-tuning only, no Portuguese "
                                                  "pretraining"),
    "hfl/chinese-llama-2-13b": (("zh",), "continual pretraining on Chinese (Zh 100%)",
                                "project metadata (MAIN_MONOLINGUAL_TRAINING)", ""),
    "tokyotech-llm/Swallow-13b-hf": (("ja",), "Llama 2 continual pretraining mainly with Japanese data",
                                     "HF model card tokyotech-llm/Swallow-13b-hf (languages: Japanese, English)",
                                     ""),
    "llm-jp/llm-jp-3-13b": (("ja",), "balanced pretraining En 46%, Ja 49% (bilingual; English excluded from "
                            "Analysis 2)", "CLAUDE.md addendum 2026-10-12 §3", ""),
    "AIInstituteLNCC/manaca-1b-instruct": (("pt",), "instruction fine-tune of the monolingual Portuguese Manacá 1B "
                                           "base", "HF model card AIInstituteLNCC/manaca-1b-instruct", ""),
    "llm-jp/llm-jp-3-172b": (("ja",), "LLM-jp-3 family, Japanese focus", "CLAUDE.md 2026-10-30",
                             "focus from CLAUDE.md; 172B-specific training shares not recorded in this project"),
    "Aleph-Alpha/Kolibri-1-BF16": (("de", "en"), "bilingual pre-training corpus ~62.5% En, ~23.9% De, ~13.6% code "
                                   "(English excluded from Analysis 2; German stays its focus language)",
                                   "HF model card Aleph-Alpha/Kolibri-1-BF16", ""),
}
NOT_PRETRAINING_FOCUS = ("recogna-nlp/bode-13b-alpaca-pt-br-no-peft", "OpenLLM-France/Claire-Mistral-7B-0.1")


N_EC_MODELS = sum(len(ids) for s, _f, ids in STAT_MODEL_SECTIONS if s != NEC_SECTION)
N_NEC_MODELS = sum(len(ids) for s, _f, ids in STAT_MODEL_SECTIONS if s == NEC_SECTION)


def model_label(model_id):
    for section, _f, ids in STAT_MODEL_SECTIONS:
        if model_id in ids:
            short = FAILURE_BAR_LABELS.get(model_id, model_id)
            return short if section.startswith("Other") else f"{section} {short}"
    return model_id


def model_classification():
    rows = []
    for section, _family, ids in STAT_MODEL_SECTIONS:
        for model_id in ids:
            meta = lookup_model(model_id)
            nec_section = section == NEC_SECTION
            if meta is not None:
                ec, source = bool(meta.english_centric), "model_metadata2 registry (english_centric)"
                conflict = ec == nec_section
            else:
                ec, source, conflict = (not nec_section), f"line-plot section '{section}' (not in registry)", False
            focus, training, focus_source, ambiguity = FOCUS_LANGUAGES.get(model_id, ((), "", "", ""))
            if not ec and not focus:
                ambiguity = "no documented focus language (excluded from Analysis 2)"
            rows.append({
                "model_id": model_id, "model_name": failure_workbook_model_key(model_id),
                "label": model_label(model_id), "line_plot_section": section,
                "family": meta.family if meta is not None else section,
                "english_centric": ec, "classification_source": source,
                "classification_conflict": conflict,
                "analysis": "1: English-centric" if ec else "2: non-English-centric",
                "focus_languages": ",".join(focus), "training_language_metadata": training,
                "focus_language_source": focus_source, "ambiguity": ambiguity,
                "in_pretraining_focus_subset": (not ec) and model_id not in NOT_PRETRAINING_FOCUS,
            })
    return pd.DataFrame(rows)

#この関数で保存済みのdiff SLORからLMM用のデータフレームを作成
def lmm_frame(df, langs, include_identical=False):
    keys = {failure_workbook_model_key(m) for _s, _f, ids in STAT_MODEL_SECTIONS for m in ids}
    obs = failure_rate_observations(df, langs, include_identical)
    obs = obs.loc[obs["model_name"].astype(str).isin(keys)]
    is_en = (obs["pos"] == ENGLISH_COMPARISON_POS).to_numpy()
    origin = obs["Target language"].map(canonical_language_code).astype(str).to_numpy()
    out = pd.DataFrame({
        "model": obs["model_name"].astype(str).to_numpy(),
        "language": np.where(is_en, EN_REFERENCE, origin),
        "origin": origin,
        "comparison": np.where(is_en, "en", "tl"),
        "item": np.where(is_en, "en:", "tl:") + obs[SOURCE_ROW_ID_COL].astype(str).to_numpy(),
        "pos_sentence": obs["pos_sentence"].astype(str).to_numpy(),
        "neg_sentence": obs["neg_sentence"].astype(str).to_numpy(),
        "dslor": obs[DELTA_SLOR].astype(float).to_numpy(),
    })
    assert set(obs["pos"]) <= {TARGET_COMPARISON_POS, ENGLISH_COMPARISON_POS}
    return out.reset_index(drop=True)


def identity_report(frame):
    dup = int(frame.duplicated(["model", "item"]).sum())
    variants = frame.groupby("item")[["pos_sentence", "neg_sentence"]].nunique().max(axis=1)
    en = frame.loc[frame["comparison"] == "en"]
    g = en.groupby(["model", "pos_sentence", "neg_sentence"])["dslor"].agg(["size", "nunique"])
    shared = g.loc[g["size"] > 1]
    return {"duplicate_model_item_rows": dup, "items": int(frame["item"].nunique()),
            "items_with_sentence_variants": int((variants > 1).sum()),
            "en_rows": int(len(en)), "en_unique_model_sentence_pairs": int(len(g)),
            "en_pairs_under_several_origins": int(len(shared)),
            "en_rows_in_such_pairs": int(shared["size"].sum()),
            "en_shared_pairs_with_differing_scores": int((shared["nunique"] > 1).sum())}


#LMMはRで実行した方が良いためこうする。
#lmer(as.formula(s$formula), data = d, REML = TRUEここでREMLを指定

LMERTEST_MARKER = "LMERTEST_MISSING"
LMERTEST_MISSING_MESSAGE = (
    "The primary LMMs need Satterthwaite df from the R package lmerTest, which is not installed for this Rscript; "
    "they never fall back to Wald z or df = n_models - 1. Install it with: "
    "Rscript -e 'install.packages(\"lmerTest\", repos = \"https://cloud.r-project.org\")'")
R_SATTERTHWAITE = r"""
has_contrasts <- !is.na(spec$contrasts) & nzchar(as.character(spec$contrasts))
if (any(has_contrasts) && !requireNamespace("lmerTest", quietly = TRUE)) {
  file.create(file.path(out, "LMERTEST_MISSING"))
  quit(status = 3)
}
fit_lmer <- function(k) if (has_contrasts[k]) lmerTest::lmer else lme4::lmer
satterthwaite_table <- function(fit, path) {
  if (!inherits(fit, "lmerModLmerTest")) stop("lmerTest could not compute the Satterthwaite terms of this fit")
  ct <- read.csv(path, stringsAsFactors = FALSE)
  rows <- lapply(unique(ct$contrast), function(label) {
    w <- ct[ct$contrast == label, ]
    L <- setNames(rep(0, length(fixef(fit))), names(fixef(fit)))
    if (!all(w$term %in% names(L))) stop(paste("contrast", label, "uses a term that was not estimated"))
    L[w$term] <- w$weight
    r <- lmerTest::contest1D(fit, L, ddf = "Satterthwaite", confint = TRUE, level = 0.95)
    data.frame(contrast = label, estimate = r[["Estimate"]], std_error = r[["Std. Error"]], df = r[["df"]],
               t_value = r[["t value"]], ci_lower = r[["lower"]], ci_upper = r[["upper"]],
               p_value = r[["Pr(>|t|)"]])
  })
  do.call(rbind, rows)
}
"""

_R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages(library(lme4))
spec <- read.csv(args[1], stringsAsFactors = FALSE)
out <- args[2]
""" + R_SATTERTHWAITE + r"""
for (k in seq_len(nrow(spec))) {
  s <- spec[k, ]
  d <- read.csv(s$data, stringsAsFactors = FALSE)
  d$language <- relevel(factor(d$language), ref = s$reference)
  d$model <- factor(d$model)
  d$item <- factor(d$item)
  d$model_language <- factor(paste(d$model, d$language, sep = ":"))
  warns <- character(0)
  t0 <- Sys.time()
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
  vc <- as.data.frame(VarCorr(fit))
  write.csv(vc, paste0(p, "_varcorr.csv"), row.names = FALSE)
  conv <- fit@optinfo$conv$lme4$messages
  writeLines(c("ok",
               paste0("singular=", isSingular(fit)),
               paste0("nobs=", nobs(fit)),
               paste0("ngrps=", paste(names(ngrps(fit)), ngrps(fit), sep = ":", collapse = ";")),
               paste0("loglik=", as.numeric(logLik(fit))),
               paste0("seconds=", round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)),
               paste0("optimizer_code=", fit@optinfo$conv$opt),
               paste0("lme4_messages=", paste(gsub("[\r\n]", " ", conv), collapse = " | ")),
               paste0("warnings=", paste(gsub("[\r\n]", " ", warns), collapse = " | "))),
             paste0(p, "_status.txt"))
}
"""


def rscript_path():
    return shutil.which("Rscript")


def write_contrast_file(path, contrasts):
    pd.DataFrame([{"contrast": label, "term": t, "weight": w} for label, weights in contrasts.items()
                  for t, w in weights.items()]).to_csv(path, index=False)
    return str(path)


def check_lmertest(tmp, proc):
    if (Path(tmp) / LMERTEST_MARKER).exists():
        raise RuntimeError(f"{LMERTEST_MISSING_MESSAGE} (R: {proc.stderr.strip()[-200:]})")


def satterthwaite_df(fit, label):
    tab = fit["satterthwaite"].set_index("contrast")
    df = float(tab.loc[label, "df"]) if label in tab.index else np.nan
    if not np.isfinite(df):
        raise ValueError(f"no finite Satterthwaite df for contrast {label!r}")
    return df


def fit_lmms(specs, timeout=7200):
    rscript = rscript_path()
    if not rscript:
        return {s["fit_id"]: {"status": "error: Rscript not found on PATH"} for s in specs}
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rows = []
        for s in specs:
            path = tmp / f"{s['fit_id']}_data.csv"
            s["data"][["dslor", "model", "language", "item"] + [c for c in ("is_focus", "non_en") if c in s["data"]]
                      ].to_csv(path, index=False)
            contrasts = (write_contrast_file(tmp / f"{s['fit_id']}_contrasts.csv", s["satterthwaite"])
                         if s.get("satterthwaite") else "")
            rows.append({"fit_id": s["fit_id"], "data": str(path), "formula": s["formula"],
                         "reference": s["reference"], "contrasts": contrasts})
        pd.DataFrame(rows).to_csv(tmp / "spec.csv", index=False)
        (tmp / "fit.R").write_text(_R_SCRIPT, encoding="utf-8")
        proc = subprocess.run([rscript, str(tmp / "fit.R"), str(tmp / "spec.csv"), str(tmp)], capture_output=True,
                              text=True, timeout=timeout)
        check_lmertest(tmp, proc)
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
            info = dict(line.split("=", 1) for line in lines[1:])
            res = {"status": "ok", **info,
                   "fixef": pd.read_csv(tmp / f"{fid}_fixef.csv"),
                   "vcov": pd.read_csv(tmp / f"{fid}_vcov.csv", index_col=0),
                   "varcorr": pd.read_csv(tmp / f"{fid}_varcorr.csv")}
            if s.get("satterthwaite"):
                res["satterthwaite"] = pd.read_csv(tmp / f"{fid}_satterthwaite.csv")
            results[fid] = res
    return results


def wald(estimate, se, df=None):
    if not (np.isfinite(estimate) and np.isfinite(se) and se > 0):
        return np.nan, np.nan, np.nan, np.nan
    stat = estimate / se
    if df is None:
        p, q = math.erfc(abs(stat) / math.sqrt(2.0)), Z975
    else:
        p, q = student_t_sf2(stat, df), student_t_ppf(0.975, df)
    return stat, p, estimate - q * se, estimate + q * se


def linear_contrast(fit, weights):
    beta = fit["fixef"].set_index("term")["estimate"]
    vcov = fit["vcov"]
    names = list(weights)
    w = np.array([weights[n] for n in names], dtype=float)
    b = beta.reindex(names).to_numpy(dtype=float)
    v = vcov.loc[names, names].to_numpy(dtype=float)
    return float(w @ b), float(np.sqrt(w @ v @ w))


def variance_components(fit):
    vc = fit["varcorr"]
    out = {}
    for _i, r in vc.iterrows():
        name = r["grp"] if pd.isna(r.get("var2")) or r.get("var2") in ("", None) else f"{r['grp']}:{r['var1']}x{r['var2']}"
        if isinstance(r.get("var1"), str) and r["var1"] != "(Intercept)" and pd.isna(r.get("var2")):
            name = f"{r['grp']}[{r['var1']}]"
        key = "cov" if not pd.isna(r.get("var2")) else "var"
        out[f"{key}:{name}"] = float(r["vcov"])
    return out


"""
ec_model_languageが論文sec5.1で使用したEnglish centric model用のLMM
"""
EC_SPECS = [
    ("ec_model_language", "dslor ~ language + (1 | model) + (1 | model_language) + (1 | item)",
     "paper-primary: + model x language random intercepts (model-specific language deviations)", SATTERTHWAITE),
]
PLANNED_CONTRAST = "en - mean(non-English)"


def ec_contrasts(fit, df=None):
    terms = set(fit["fixef"]["term"])
    langs = [l for l in NON_ENGLISH if f"language{l}" in terms]
    rows = []
    est, se = linear_contrast(fit, {f"language{l}": -1.0 / len(langs) for l in langs})
    #fitはすでに学習されたLMMの結果。estが英語と非英語6言語の平均diff slorの推定差
    #seがstandard errorでその推定差の標準誤差
    df_ = satterthwaite_df(fit, PLANNED_CONTRAST) if df == SATTERTHWAITE else df
    stat, p, lo, hi = wald(est, se, df_)
    rows.append({"contrast": PLANNED_CONTRAST, "type": "planned", "estimate": est, "std_error": se,
                 "statistic": stat, "df": df_, "p_value": p, "ci_lower": lo, "ci_upper": hi})
    return rows


def ec_satterthwaite_contrasts(frame):
    langs = [l for l in NON_ENGLISH if l in set(frame["language"])]
    return {PLANNED_CONTRAST: {f"language{l}": -1.0 / len(langs) for l in langs}}


NEC_TESTED_TERM = "is_focus"


def nec_satterthwaite_contrasts(frame):
    return {NEC_TESTED_TERM: {NEC_TESTED_TERM: 1.0}}


def inference_label(rule_df, df):
    if rule_df is None:
        return "Wald z"
    return f"Satterthwaite t (df = {df:.2f})" if rule_df == SATTERTHWAITE else f"Wald t (df = {df})"


def per_model_ec_contrast(frame):
    rows = []
    for model, g in frame.groupby("model", sort=False):
        stats = g.groupby("language")["dslor"].agg(["mean", "var", "size"])
        non = [l for l in NON_ENGLISH if l in stats.index]
        if EN_REFERENCE not in stats.index or not non:
            continue
        est = stats.loc[EN_REFERENCE, "mean"] - stats.loc[non, "mean"].mean()
        se = math.sqrt(stats.loc[EN_REFERENCE, "var"] / stats.loc[EN_REFERENCE, "size"]
                       + (stats.loc[non, "var"] / stats.loc[non, "size"]).sum() / len(non) ** 2)
        rows.append({"model": model, "estimate": est, "std_error": se,
                     "n_obs": int(len(g)), "n_en": int(stats.loc[EN_REFERENCE, "size"]),
                     "mean_en": stats.loc[EN_REFERENCE, "mean"], "mean_non_english": stats.loc[non, "mean"].mean(),
                     **{f"en_minus_{l}": stats.loc[EN_REFERENCE, "mean"] - stats.loc[l, "mean"] for l in non}})
    return pd.DataFrame(rows)


"""
論文中で使用したものはnec_model_languageです。
"""


NEC_SPECS = [
    ("nec_model_language", "dslor ~ is_focus + language + (1 | model) + (1 | model_language) + (1 | item)",
     "paper-primary: + model x language random intercepts (is_focus varies only between these cells)", SATTERTHWAITE),
]


def focus_identifiability(frame):
    langs = sorted(set(frame["language"]) - {NEC_REFERENCE})
    cells = frame.drop_duplicates(["model", "language"])[["model", "language", "is_focus"]]
    X = np.column_stack([np.ones(len(cells))] + [(cells["language"] == l).to_numpy(float) for l in langs]
                        + [cells["is_focus"].to_numpy(float)])
    rank = int(np.linalg.matrix_rank(X))
    support = cells.groupby("language")["is_focus"].agg(["sum", "size"])
    mixed = support.loc[(support["sum"] > 0) & (support["sum"] < support["size"])].index.tolist()
    varies_within_model = cells.groupby("model")["is_focus"].nunique().gt(1)
    ok = rank == X.shape[1] and len(mixed) > 0 and bool(varies_within_model.all())
    return {"identifiable": ok, "design_rank": rank, "design_columns": X.shape[1],
            "focus_cells": int(cells["is_focus"].sum()), "model_language_cells": int(len(cells)),
            "languages_with_focus_and_nonfocus_models": ",".join(mixed),
            "models_without_within_model_variation": ",".join(varies_within_model.index[~varies_within_model])}


def per_model_focus_effects(frame, focus):
    means = frame.groupby(["model", "language"])["dslor"].agg(["mean", "var", "size"])
    models = list(dict.fromkeys(frame["model"]))
    langs = [l for l in NON_ENGLISH if l in set(frame["language"])]

    def contrast(m, f_langs):
        mu = means.loc[m]
        foc = [l for l in langs if l in f_langs and l in mu.index]
        non = [l for l in langs if l not in f_langs and l in mu.index]
        est = mu.loc[foc, "mean"].mean() - mu.loc[non, "mean"].mean()
        se = math.sqrt((mu.loc[foc, "var"] / mu.loc[foc, "size"]).sum() / len(foc) ** 2
                       + (mu.loc[non, "var"] / mu.loc[non, "size"]).sum() / len(non) ** 2)
        return est, se, foc, non

    rows = []
    for m in models:
        est, se, foc, non = contrast(m, focus[m])
        others = [contrast(o, focus[m])[0] for o in models if o != m]
        adj = est - float(np.mean(others))
        rows.append({"model": m, "focus_languages": ",".join(foc), "non_focus_languages": ",".join(non),
                     "raw_focus_minus_nonfocus": est, "raw_std_error": se, "other_models_same_contrast": float(np.mean(others)),
                     "language_adjusted_difference": adj,
                     "mean_focus": means.loc[m].loc[foc, "mean"].mean(),
                     "mean_non_focus": means.loc[m].loc[non, "mean"].mean(),
                     "n_focus": int(means.loc[m].loc[foc, "size"].sum()),
                     "n_non_focus": int(means.loc[m].loc[non, "size"].sum())})
    return pd.DataFrame(rows)


def _df_for(rule, n_models):
    if rule == SATTERTHWAITE:
        return SATTERTHWAITE
    return None if rule is None else max(n_models - 1, 1)


def _fixef_rows(fit, analysis, spec_id, description, df, n_obs, n_models, n_items, tested=()):
    rows = []
    for _i, r in fit["fixef"].iterrows():
        row = {"analysis": analysis, "specification": spec_id, "description": description,
               "term": r["term"], "estimate": r["estimate"], "std_error": r["std_error"]}
        if r["term"] in tested:
            df_ = satterthwaite_df(fit, r["term"]) if df == SATTERTHWAITE else df
            stat, p, lo, hi = wald(r["estimate"], r["std_error"], df_)
            row.update({"statistic": stat, "df": df_, "inference": inference_label(df, df_),
                        "p_value": p, "ci_lower": lo, "ci_upper": hi})
        else:
            row["inference"] = "estimate and SE only (not tested)"
        rows.append({**row, "n_obs": n_obs, "n_models": n_models, "n_items": n_items})
    return rows


def run_rq1_lmm(df, langs, include_identical=False):
    classification = model_classification()
    frame = lmm_frame(df, langs, include_identical)
    #diff SLORからLMM用のデータフレーむをここで作成
    ident = identity_report(frame)
    if ident["duplicate_model_item_rows"]:
        raise ValueError(f"RQ1 LMM: {ident['duplicate_model_item_rows']} duplicated (model, sentence pair) rows")
    present = set(frame["model"])
    cls = classification.assign(has_rows=classification["model_name"].isin(present))
    ec_models = cls.loc[cls["english_centric"] & cls["has_rows"], "model_name"].tolist()
    #ここで、English centric modelだけを抽出する
    nec = cls.loc[~cls["english_centric"] & cls["has_rows"] & (cls["focus_languages"] != "")]
    focus = {r["model_name"]: tuple(r["focus_languages"].split(",")) for _i, r in nec.iterrows()}
    family = dict(zip(cls["model_name"], cls["family"]))

    ec = frame.loc[frame["model"].isin(ec_models)].copy()
    ne = frame.loc[frame["model"].isin(focus) & (frame["language"] != EN_REFERENCE)].copy()
    ne["is_focus"] = [int(l in focus[m]) for m, l in zip(ne["model"], ne["language"])]
    ident_ne = focus_identifiability(ne)

    specs = []
    for fid, formula, _d, rule in EC_SPECS:
        specs.append({"fit_id": fid, "data": ec, "formula": formula, "reference": EN_REFERENCE,
                      "satterthwaite": ec_satterthwaite_contrasts(ec) if rule == SATTERTHWAITE else None})
        #ここで、englishをβ0として設定している。wald t検定において重要。
    if ident_ne["identifiable"]:
        for fid, formula, _d, rule in NEC_SPECS:
            specs.append({"fit_id": fid, "data": ne, "formula": formula, "reference": NEC_REFERENCE,
                          "satterthwaite": nec_satterthwaite_contrasts(ne) if rule == SATTERTHWAITE else None})
    fits = fit_lmms(specs)
    #ここで定義したLMMの式を渡している
    sizes = {s["fit_id"]: (int(len(s["data"])), int(s["data"]["model"].nunique()), int(s["data"]["item"].nunique()))
             for s in specs}

    ec_coef, ec_con = [], []
    for fid, _f, desc, rule in EC_SPECS:
        fit = fits[fid]
        n_obs, n_mod, n_items = sizes[fid]
        if fit["status"] != "ok":
            ec_con.append({"scope": "pooled", "specification": fid, "contrast": "NOT ESTIMATED",
                           "type": fit["status"]})
            continue
        df_ = _df_for(rule, n_mod)
        ec_coef += _fixef_rows(fit, "1: English-centric", fid, desc, df_, n_obs, n_mod, n_items)
        for r in ec_contrasts(fit, df_):
            ec_con.append({"scope": "pooled LMM", "specification": fid, "description": desc, **r,
                           "inference": inference_label(df_, r["df"]),
                           "n_obs": n_obs, "n_models": n_mod, "n_items": n_items})
    per_model = per_model_ec_contrast(ec)
    per_model.insert(1, "label", per_model["model"].map(dict(zip(cls["model_name"], cls["label"]))))
    per_model.insert(2, "family", per_model["model"].map(family))
    for _i, r in per_model.iterrows():
        ec_con.append({"scope": "per model (descriptive)", "specification": r["model"], "description": r["label"],
                       "contrast": PLANNED_CONTRAST, "type": "per model", "estimate": r["estimate"],
                       "std_error": r["std_error"], "inference": "difference of means (descriptive; not tested)",
                       "n_obs": r["n_obs"], "n_models": 1, "n_items": r["n_obs"]})

    ne_coef, ne_model = [], pd.DataFrame()
    if ident_ne["identifiable"]:
        for fid, _f, desc, rule in NEC_SPECS:
            fit = fits[fid]
            n_obs, n_mod, n_items = sizes[fid]
            if fit["status"] != "ok":
                ne_coef.append({"analysis": "2: non-English-centric", "specification": fid, "term": "NOT ESTIMATED",
                                "description": fit["status"]})
                continue
            ne_coef += _fixef_rows(fit, "2: non-English-centric", fid, desc, _df_for(rule, n_mod), n_obs, n_mod,
                                   n_items, tested=(NEC_TESTED_TERM,))
        ne_model = per_model_focus_effects(ne, focus)
        ne_model.insert(1, "label", ne_model["model"].map(dict(zip(cls["model_name"], cls["label"]))))

    diagnostics = []
    for s in specs:
        fit = fits[s["fit_id"]]
        n_obs, n_mod, n_items = sizes[s["fit_id"]]
        row = {"fit_id": s["fit_id"], "formula": s["formula"], "reference_language": s["reference"],
               "status": fit["status"], "n_obs": n_obs, "n_models": n_mod, "n_items": n_items}
        if fit["status"] == "ok":
            msgs = fit.get("lme4_messages", "")
            warns = fit.get("warnings", "")
            other_msgs = [x for x in msgs.split(" | ") if x and "boundary (singular)" not in x]
            row.update({"singular": fit.get("singular"), "optimizer_code": fit.get("optimizer_code"),
                        "converged": (fit.get("optimizer_code") == "0" and not other_msgs
                                      and "converge" not in warns.lower()),
                        "lme4_messages": msgs, "warnings": warns, "loglik_REML": fit.get("loglik"),
                        "seconds": fit.get("seconds"), **variance_components(fit)})
        diagnostics.append(row)
    return {"classification": cls, "frame": frame, "identity": ident, "ec": ec, "ne": ne,
            "ec_coefficients": pd.DataFrame(ec_coef), "ec_contrasts": pd.DataFrame(ec_con),
            "ec_per_model": per_model, "ne_coefficients": pd.DataFrame(ne_coef), "ne_model_effects": ne_model,
            "ne_identifiability": ident_ne, "diagnostics": pd.DataFrame(diagnostics), "fits": fits,
            "focus": focus}
