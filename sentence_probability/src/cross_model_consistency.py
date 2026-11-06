from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from viz_data2 import (
    DELTA_SLOR,
    NO_CONTEXT_PROMPT_LANGUAGE,
    SLOR_COL,
    SOURCE_ROW_ID_COL,
    TARGET_COMPARISON_POS,
    add_source_row_id,
    analysis_rows,
    apply_filters2,
    canonical_language_code,
    language_axis_order,
)

PAIR_ID_COL = "sentence_pair_id"
TARGET_LANGUAGE_COL = "target_language"
SIGN_COL = "sign"
SIGN_POSITIVE, SIGN_NEGATIVE, SIGN_TIE = "positive", "negative", "tie"
POOLED = "all"
MIN_SHARED_PAIRS_SPEARMAN = 3


class SentencePairIdentityError(ValueError):

    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


def _sign(values):
    v = np.asarray(values, dtype=float)
    return np.where(np.isnan(v), None, np.where(v > 0, SIGN_POSITIVE, np.where(v < 0, SIGN_NEGATIVE, SIGN_TIE)))


DIAGNOSTIC_COLUMNS = ["reason", "model_name", TARGET_LANGUAGE_COL, SOURCE_ROW_ID_COL, PAIR_ID_COL, "pos_sentence",
                      "neg_sentence", SLOR_COL]


def sentence_pair_id(source_row_id, pos_sentence, neg_sentence):
    digest = hashlib.sha256(f"{pos_sentence}\t{neg_sentence}".encode("utf-8")).hexdigest()[:10]
    return f"{source_row_id}:{digest}"


def duplicate_diagnostic(frame):
    if frame.empty:
        return pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)
    dup = frame.loc[frame.duplicated(["model_name", PAIR_ID_COL], keep=False)].assign(reason="duplicate_model_pair")
    return dup.reindex(columns=DIAGNOSTIC_COLUMNS).sort_values([PAIR_ID_COL, "model_name"]).reset_index(drop=True)


def identity_variants(frame):
    if frame.empty:
        return pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)
    n_variants = frame.drop_duplicates([SOURCE_ROW_ID_COL, PAIR_ID_COL]).groupby(SOURCE_ROW_ID_COL).size()
    rows = frame.loc[frame[SOURCE_ROW_ID_COL].isin(set(n_variants.index[n_variants > 1]))]
    rows = rows.assign(reason="sentence_variant_of_source_row")
    return rows.reindex(columns=DIAGNOSTIC_COLUMNS).sort_values([SOURCE_ROW_ID_COL, PAIR_ID_COL, "model_name"]) \
        .reset_index(drop=True)


def sentence_pair_frame(df, preference_languages=None, include_identical=False):
    rows = analysis_rows(df, SLOR_COL, include_identical=include_identical, preference_languages=preference_languages)
    if not rows.empty:
        rows = rows.loc[rows["pos"] == TARGET_COMPARISON_POS]
    rows = add_source_row_id(rows)
    out = pd.DataFrame({
        "model_name": rows["model_name"].astype(str),
        TARGET_LANGUAGE_COL: rows["Target language"].map(canonical_language_code).astype(str),
        PAIR_ID_COL: [sentence_pair_id(i, p, n) for i, p, n in zip(rows[SOURCE_ROW_ID_COL], rows["pos_sentence"],
                                                                   rows["neg_sentence"])],
        SOURCE_ROW_ID_COL: rows[SOURCE_ROW_ID_COL].astype(str),
        "Proper English strategy": rows["Proper English strategy"] if "Proper English strategy" in rows else None,
        "pos_sentence": rows["pos_sentence"],
        "neg_sentence": rows["neg_sentence"],
        SLOR_COL: rows[DELTA_SLOR].astype(float),
        SIGN_COL: _sign(rows[DELTA_SLOR]),
    }).reset_index(drop=True)
    diagnostic = duplicate_diagnostic(out)
    if len(diagnostic):
        reasons = diagnostic["reason"].value_counts().to_dict()
        raise SentencePairIdentityError(
            f"invalid cross-model sentence-pair identity ({reasons}); see the diagnostic rows", diagnostic)
    return out


def coverage_table(df, frame, preference_languages=None, include_identical=False):
    selected = apply_filters2(df, {}, include_identical=include_identical, preference_languages=preference_languages)
    if "prompt_language" in selected.columns:
        selected = selected.loc[selected["prompt_language"].astype(str) == NO_CONTEXT_PROMPT_LANGUAGE]
    selected = selected.loc[selected["pos"] == TARGET_COMPARISON_POS] if "pos" in selected.columns else selected.iloc[0:0]
    invalid = selected.loc[pd.to_numeric(selected[SLOR_COL], errors="coerce").isna()] if SLOR_COL in selected \
        else selected
    invalid = pd.DataFrame({"model_name": invalid["model_name"].astype(str),
                            TARGET_LANGUAGE_COL: invalid["Target language"].map(canonical_language_code).astype(str)})
    rows = []
    keys = sorted(set(zip(frame["model_name"], frame[TARGET_LANGUAGE_COL])) | set(zip(invalid["model_name"],
                                                                                       invalid[TARGET_LANGUAGE_COL])))
    for model in sorted({m for m, _l in keys}):
        langs = language_axis_order({l for m, l in keys if m == model})
        for lang in langs + [POOLED]:
            f = frame.loc[frame["model_name"] == model]
            inv = invalid.loc[invalid["model_name"] == model]
            if lang != POOLED:
                f = f.loc[f[TARGET_LANGUAGE_COL] == lang]
                inv = inv.loc[inv[TARGET_LANGUAGE_COL] == lang]
            rows.append({"model_name": model, TARGET_LANGUAGE_COL: lang, "n_sentence_pairs": int(len(f)),
                         "n_positive": int((f[SIGN_COL] == SIGN_POSITIVE).sum()),
                         "n_negative": int((f[SIGN_COL] == SIGN_NEGATIVE).sum()),
                         "n_tied": int((f[SIGN_COL] == SIGN_TIE).sum()),
                         "n_missing_or_invalid_slor": int(len(inv))})
    return pd.DataFrame(rows, columns=["model_name", TARGET_LANGUAGE_COL, "n_sentence_pairs", "n_positive",
                                       "n_negative", "n_tied", "n_missing_or_invalid_slor"])


def pair_matrix(frame, models=None):
    m = frame.pivot(index=PAIR_ID_COL, columns="model_name", values=SLOR_COL)
    if models is not None:
        m = m.reindex(columns=list(models))
    m.columns.name = "model_name"
    return m


def _shared_n(valid):
    v = valid.to_numpy(dtype=int)
    n = pd.DataFrame(v.T @ v if v.size else np.zeros((valid.shape[1],) * 2, dtype=int),
                     index=valid.columns, columns=valid.columns)
    n.index.name = "model_name"
    return n.astype(int)


def spearman_matrix(frame, models=None, min_shared=MIN_SHARED_PAIRS_SPEARMAN):
    m = pair_matrix(frame, models)
    rho = m.corr(method="spearman", min_periods=min_shared)
    rho.index.name = "model_name"
    return rho, _shared_n(m.notna())


def sign_agreement_and_kappa(frame, models=None):
    m = pair_matrix(frame, models)
    s = np.sign(m)
    s = s.where(s != 0)
    cols = list(s.columns)
    agree = pd.DataFrame(np.nan, index=cols, columns=cols)
    kappa = pd.DataFrame(np.nan, index=cols, columns=cols)
    values = s.to_numpy(dtype=float)
    valid = ~np.isnan(values)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            both = valid[:, i] & valid[:, j]
            n = int(both.sum())
            if not n:
                continue
            x, y = values[both, i], values[both, j]
            p_o = float((x == y).mean())
            pa, pb = float((x > 0).mean()), float((y > 0).mean())
            p_e = pa * pb + (1 - pa) * (1 - pb)
            agree.iloc[i, j] = p_o
            kappa.iloc[i, j] = (p_o - p_e) / (1 - p_e) if p_e < 1 else np.nan
    for t in (agree, kappa):
        t.index.name = "model_name"
        t.columns.name = "model_name"
    return agree, kappa, _shared_n(s.notna())


ICC_FORMULA = "dslor ~ 1 + (1 | sentence_pair) + (1 | model)"
ICC_RANDOM_EFFECTS = ("sentence_pair", "model")
MIN_LEVELS = 2

_R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages(library(lme4))
d <- read.csv(args[1], colClasses = c("character", "character", "character", "numeric"))
formula <- as.formula(args[4])
rows <- list()
summaries <- character(0)
for (g in unique(d$fit_id)) {
  sub <- d[d$fit_id == g, ]
  sub$sentence_pair <- factor(sub$sentence_pair)
  sub$model <- factor(sub$model)
  warns <- character(0)
  fit <- withCallingHandlers(
    tryCatch(lmer(formula, data = sub, REML = TRUE), error = function(e) e),
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })
  if (inherits(fit, "error")) {
    rows[[g]] <- data.frame(fit_id = g, status = "error", message = conditionMessage(fit),
                            var_sentence_pair = NA, var_model = NA, var_residual = NA, intercept = NA,
                            fixed_effects = NA, random_effects = NA, singular = NA, n_obs = nrow(sub),
                            n_sentence_pairs = nlevels(sub$sentence_pair), n_models = nlevels(sub$model),
                            reml_criterion = NA)
    next
  }
  vc <- as.data.frame(VarCorr(fit))
  conv <- c(warns, fit@optinfo$conv$lme4$messages)
  rows[[g]] <- data.frame(
    fit_id = g, status = "ok", message = paste(unique(conv), collapse = " | "),
    var_sentence_pair = vc$vcov[vc$grp == "sentence_pair"], var_model = vc$vcov[vc$grp == "model"],
    var_residual = vc$vcov[vc$grp == "Residual"], intercept = unname(fixef(fit)[1]),
    fixed_effects = paste(names(fixef(fit)), collapse = ";"),
    random_effects = paste(setdiff(vc$grp, "Residual"), collapse = ";"),
    singular = isSingular(fit), n_obs = nobs(fit), n_sentence_pairs = nlevels(sub$sentence_pair),
    n_models = nlevels(sub$model), reml_criterion = unname(REMLcrit(fit)))
  summaries <- c(summaries, paste0("===== fit: ", g, " ====="), capture.output(print(summary(fit))), "")
}
write.csv(do.call(rbind, rows), args[2], row.names = FALSE)
writeLines(summaries, args[3])
"""

VC_COLUMNS = ["fit_id", "status", "message", "formula", "estimation", "n_obs", "n_sentence_pairs", "n_models",
              "variance_sentence_pair", "variance_model", "variance_residual", "variance_total", "ICC_pair",
              "proportion_model", "proportion_residual", "intercept", "fixed_effects", "random_effects",
              "singular", "reml_criterion"]


def estimability(sub):
    n, n_pairs, n_models = len(sub), sub[PAIR_ID_COL].nunique(), sub["model_name"].nunique()
    if n_pairs < MIN_LEVELS or n_models < MIN_LEVELS:
        return f"not_estimable: {n_pairs} sentence pair(s), {n_models} model(s) (need >= {MIN_LEVELS} each)"
    if n <= n_pairs or n <= n_models:
        return f"not_estimable: {n} observations for {n_pairs} pairs / {n_models} models (no replication)"
    return None


def rscript_path():
    return shutil.which("Rscript")


def fit_variance_components(fits, rscript=None, timeout=3600):
    table, todo = [], {}
    for fit_id, sub in fits.items():
        reason = estimability(sub)
        if reason:
            table.append({"fit_id": fit_id, "status": reason, "n_obs": len(sub),
                          "n_sentence_pairs": sub[PAIR_ID_COL].nunique(), "n_models": sub["model_name"].nunique()})
        else:
            todo[fit_id] = sub
    summary = ""
    if todo:
        rscript = rscript or rscript_path()
        if not rscript:
            table += [{"fit_id": f, "status": "error", "message": "Rscript not found on PATH"} for f in todo]
        else:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                data = pd.concat([pd.DataFrame({"fit_id": f, "sentence_pair": s[PAIR_ID_COL].astype(str),
                                                "model": s["model_name"].astype(str),
                                                "dslor": s[SLOR_COL].astype(float)})
                                  for f, s in todo.items()], ignore_index=True)
                data.to_csv(tmp / "data.csv", index=False)
                (tmp / "fit.R").write_text(_R_SCRIPT, encoding="utf-8")
                try:
                    proc = subprocess.run([rscript, "--vanilla", str(tmp / "fit.R"), str(tmp / "data.csv"),
                                           str(tmp / "out.csv"), str(tmp / "summary.txt"), ICC_FORMULA],
                                          capture_output=True, text=True, timeout=timeout)
                except (OSError, subprocess.TimeoutExpired) as e:
                    proc = subprocess.CompletedProcess([rscript], 1, "", f"could not run Rscript: {e}")
                if proc.returncode != 0 or not (tmp / "out.csv").exists():
                    msg = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["Rscript failed"]
                    table += [{"fit_id": f, "status": "error", "message": msg[0]} for f in todo]
                else:
                    r = pd.read_csv(tmp / "out.csv", keep_default_na=True, dtype={"fit_id": str})
                    r = r.rename(columns={"var_sentence_pair": "variance_sentence_pair",
                                          "var_model": "variance_model", "var_residual": "variance_residual"})
                    table += r.to_dict("records")
                    summary = (tmp / "summary.txt").read_text(encoding="utf-8")
    out = pd.DataFrame(table).reindex(columns=VC_COLUMNS)
    out["formula"] = ICC_FORMULA
    out["estimation"] = "lme4::lmer, REML (via Rscript)"
    comps = out[["variance_sentence_pair", "variance_model", "variance_residual"]].astype(float)
    total = comps.sum(axis=1, min_count=3)
    out["variance_total"] = total
    out["ICC_pair"] = comps["variance_sentence_pair"] / total
    out["proportion_model"] = comps["variance_model"] / total
    out["proportion_residual"] = comps["variance_residual"] / total
    order = {f: i for i, f in enumerate(fits)}
    out = out.sort_values("fit_id", key=lambda s: s.map(order)).reset_index(drop=True)
    return out, summary


def languages_of(frame):
    return language_axis_order(frame[TARGET_LANGUAGE_COL].unique()) if len(frame) else []


def run_consistency_analysis(df, preference_languages=None, include_identical=False, models=None, rscript=None):
    frame = sentence_pair_frame(df, preference_languages, include_identical)
    models = list(models) if models is not None else sorted(frame["model_name"].unique())
    subsets = {POOLED: frame}
    subsets.update({lang: frame.loc[frame[TARGET_LANGUAGE_COL] == lang] for lang in languages_of(frame)})
    matrices = {}
    for key, sub in subsets.items():
        rho, rho_n = spearman_matrix(sub, models)
        agree, kappa, sign_n = sign_agreement_and_kappa(sub, models)
        matrices[key] = {"spearman": rho, "spearman_n": rho_n, "sign_agreement": agree,
                         "sign_agreement_n": sign_n, "cohens_kappa": kappa, "cohens_kappa_n": sign_n.copy()}
    icc, summary = fit_variance_components(subsets, rscript=rscript)
    return {"data": frame, "identity_variants": identity_variants(frame),
            "coverage": coverage_table(df, frame, preference_languages, include_identical),
            "matrices": matrices, "icc": icc, "summary": summary, "models": models}


SUMMARY_METRICS = [
    ("spearman", "mean_spearman_rho", "n_model_pairs_spearman"),
    ("sign_agreement", "mean_sign_agreement", "n_model_pairs_sign_agreement"),
    ("cohens_kappa", "mean_cohens_kappa", "n_model_pairs_cohens_kappa"),
]
LANGUAGE_SUMMARY_COLUMNS = (
    [TARGET_LANGUAGE_COL] + [c for _k, mean, n in SUMMARY_METRICS for c in (mean, n)]
    + ["ICC_pair", "icc_status", "icc_n_obs", "icc_n_sentence_pairs", "icc_n_models"]
)


def off_diagonal_values(matrix):
    values = matrix.to_numpy(dtype=float)
    upper = values[np.triu_indices(values.shape[0], k=1)] if values.size else np.array([])
    return upper[np.isfinite(upper)]


def language_summary(matrices, icc):
    icc = icc.set_index("fit_id") if "fit_id" in icc.columns else icc
    rows = []
    for lang in language_axis_order([k for k in matrices if k != POOLED]):
        row = {TARGET_LANGUAGE_COL: lang}
        for key, mean_col, n_col in SUMMARY_METRICS:
            v = off_diagonal_values(matrices[lang][key])
            row[mean_col] = float(v.mean()) if v.size else np.nan
            row[n_col] = int(v.size)
        r = icc.loc[lang] if lang in icc.index else None
        row.update({
            "ICC_pair": float(r["ICC_pair"]) if r is not None and pd.notna(r["ICC_pair"]) else np.nan,
            "icc_status": r["status"] if r is not None else "not_fitted",
            "icc_n_obs": r["n_obs"] if r is not None else np.nan,
            "icc_n_sentence_pairs": r["n_sentence_pairs"] if r is not None else np.nan,
            "icc_n_models": r["n_models"] if r is not None else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=LANGUAGE_SUMMARY_COLUMNS)


DIST_MATRIX_STEMS = {"spearman": "spearman_model_correlation", "sign_agreement": "sign_agreement",
                     "cohens_kappa": "cohens_kappa"}


def load_dist_results(dist_dir):
    dist_dir = Path(dist_dir)
    matrices = {}
    for path in sorted(dist_dir.glob(f"{DIST_MATRIX_STEMS['spearman']}__*.csv")):
        lang = path.stem.split("__", 1)[1]
        if lang.endswith("_n"):
            continue
        matrices[lang] = {key: pd.read_csv(dist_dir / f"{stem}__{lang}.csv", index_col=0)
                          for key, stem in DIST_MATRIX_STEMS.items()}
    icc = pd.concat([pd.read_csv(dist_dir / "icc_variance_components.csv"),
                     pd.read_csv(dist_dir / "icc_variance_components_by_language.csv")], ignore_index=True)
    return matrices, icc
