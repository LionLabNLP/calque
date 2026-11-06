from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from metadata.expand import source_row_key
from paper_inputs import DEFAULT_MAIN_PLOT_DIR, DEFAULT_OUTPUTS_DIR, load_probability_outputs

FIGURE = "Spearman_model_correlation_manaca.pdf"


DEFAULT_LANGUAGES = ("de", "fr", "pt", "sv", "zh", "ja")
MODELS = [
    ("utter-project_EuroLLM-22B-2512", "EuroLLM 22B"),
    ("google_gemma-4-31B", "Gemma 4 31B"),
    ("CohereLabs_aya-expanse-32b", "Aya Expanse 32B"),
    ("swiss-ai_Apertus-70B-2509", "Apertus 70B"),
    ("AIInstituteLNCC_manaca-1b-base", "Manacá 1B"),
    ("llm-jp_llm-jp-3-13b", "LLM-jp-3 13B"),
]
MIN_SHARED_PAIRS = 3

CMAP, NA_COLOR = "RdBu_r", "#e4e3df"
CELL_INCHES = 1.8
CELL_TEXT_SIZE = 48
CELL_TEXT_FRACTION = 0.45
DARK_TEXT_THRESHOLD = 0.6
TICK_SIZE, CBAR_LABEL_SIZE, CBAR_TICK_SIZE = 55, 35, 30
TEXT_COLOR = "#0b0b0b"

FONT_CANDIDATES = ["TeX Gyre Termes", "Times New Roman", "Times", "Nimbus Roman", "Nimbus Roman No9 L",
                   "STIX Two Text", "STIXGeneral"]


def _paper_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in FONT_CANDIDATES if n in installed), "DejaVu Serif")


STYLE = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": [_paper_font()] + FONT_CANDIDATES + ["DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.unicode_minus": True,
    "axes.edgecolor": "black", "axes.labelcolor": TEXT_COLOR,
    "xtick.color": "black", "ytick.color": "black", "xtick.labelcolor": TEXT_COLOR, "ytick.labelcolor": TEXT_COLOR,
}


@contextmanager
def figure_style():
    with plt.rc_context():
        matplotlib.rcdefaults()
        plt.rcParams.update(STYLE)
        yield


SLOR_COL = "SLOR_token_logprob_diff"
TARGET_POS = "Proper strategy in target language"
TARGET_PREF = "Target language preference annotation"
PAIR_ID = "sentence_pair_id"


def _canonical(code):
    return "sv" if str(code).strip() == "se" else str(code).strip()


def _pair_id(source_row_id, pos_sentence, neg_sentence):
    digest = hashlib.sha256(f"{pos_sentence}\t{neg_sentence}".encode("utf-8")).hexdigest()[:10]
    return f"{source_row_id}:{digest}"


def sentence_pair_frame(df, models, langs, include_identical=False):
    langs = {_canonical(l) for l in langs}
    rows = df.loc[(df["experiment"] == "standard") & df["model_name"].isin(set(models))]
    if TARGET_PREF in rows.columns:
        strong = rows[TARGET_PREF].where(rows[TARGET_PREF].notna(), "").astype(str).str.strip().isin({">", ">>"})
    else:
        strong = pd.Series(False, index=rows.index)
    affected = (rows["pos"] == TARGET_POS) & rows["Target language"].map(_canonical).isin(langs)
    rows = rows.loc[~affected | strong]
    if not include_identical:
        rows = rows.loc[rows["pos_sentence"] != rows["neg_sentence"]]
    if "prompt_language" in rows.columns:
        rows = rows.loc[rows["prompt_language"].astype(str) == "raw"]
    rows = rows.loc[rows["pos"] == TARGET_POS]
    value = pd.to_numeric(rows[SLOR_COL], errors="coerce")
    rows, value = rows.loc[value.notna()], value.loc[value.notna()]
    origin = rows["target_language_source"] if "target_language_source" in rows else rows["Target language"]
    source_ids = [source_row_key(t, s) for t, s in zip(origin, rows["Proper English strategy"])]
    pair_ids = [_pair_id(sid, p, n) for sid, p, n in zip(source_ids, rows["pos_sentence"], rows["neg_sentence"])]
    out = pd.DataFrame({"model_name": rows["model_name"].astype(str).to_numpy(), PAIR_ID: pair_ids,
                        SLOR_COL: value.astype(float).to_numpy()})
    dup = out.duplicated(["model_name", PAIR_ID], keep=False)
    if dup.any():
        raise ValueError(f"Fig6: {int(dup.sum())} rows share a (model, sentence pair) identity; "
                         f"e.g. {out.loc[dup].head(3).to_dict('records')}")
    return out


def spearman_matrix(frame, models):
    m = frame.pivot(index=PAIR_ID, columns="model_name", values=SLOR_COL).reindex(columns=list(models))
    rho = m.corr(method="spearman", min_periods=MIN_SHARED_PAIRS)
    valid = m.notna().to_numpy(dtype=int)
    n = pd.DataFrame(valid.T @ valid, index=m.columns, columns=m.columns)
    return rho, n


def _two_line(label):
    head, _sep, tail = str(label).rpartition(" ")
    return f"{head}\n{tail}" if head else str(label)


def plot_heatmap(rho, labels, out_path):
    k = len(rho)
    side = CELL_INCHES * k
    values = rho.to_numpy(dtype=float)
    with figure_style():
        fig = plt.figure(figsize=(side + 6.2, side + 3.0))
        left, bottom = 3.6, 2.6
        W, H = fig.get_size_inches()
        ax = fig.add_axes([left / W, bottom / H, side / W, side / H])
        cax = fig.add_axes([(left + side + 0.35) / W, bottom / H, 0.45 / W, side / H])
        cmap = matplotlib.colormaps[CMAP].copy()
        cmap.set_bad(NA_COLOR)
        ax.set_facecolor(cmap.get_bad())
        mesh = ax.pcolormesh(np.arange(k + 1) - 0.5, np.arange(k + 1) - 0.5, np.ma.masked_invalid(values), cmap=cmap,
                             vmin=-1.0, vmax=1.0, edgecolors="none", linewidth=0, antialiased=False)
        ax.set_xlim(-0.5, k - 0.5)
        ax.set_ylim(k - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels([labels[m] for m in rho.index], fontsize=TICK_SIZE, rotation=40, ha="right",
                           rotation_mode="anchor")
        ax.set_yticklabels([_two_line(labels[m]) for m in rho.index], fontsize=TICK_SIZE, linespacing=0.95)
        texts = []
        for i in range(k):
            for j in range(k):
                v = values[i, j]
                dark = np.isfinite(v) and abs(v) > DARK_TEXT_THRESHOLD
                texts.append(ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "n/a", ha="center", va="center",
                                     fontsize=CELL_TEXT_SIZE, color="white" if dark else TEXT_COLOR))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        cell_px = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
        widest = max(t.get_window_extent(renderer).width for t in texts)
        for t in texts:
            t.set_fontsize(CELL_TEXT_SIZE * CELL_TEXT_FRACTION * cell_px / widest)
        cb = fig.colorbar(mesh, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
        cb.solids.set_rasterized(False)
        cb.set_label("Spearman's ρ", fontsize=CBAR_LABEL_SIZE, labelpad=12)
        cb.ax.tick_params(labelsize=CBAR_TICK_SIZE, labelcolor=TEXT_COLOR, length=8, width=1.5)
        ax.tick_params(labelcolor=TEXT_COLOR, length=0, pad=10)
        ax.spines[:].set_visible(False)
        fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
    return True


def generate(df=None, outputs_dir=DEFAULT_OUTPUTS_DIR, out_dir=DEFAULT_MAIN_PLOT_DIR, langs=DEFAULT_LANGUAGES,
             include_identical=False):
    if df is None:
        df, warnings = load_probability_outputs(outputs_dir)
        for w in warnings:
            print(f"[Fig6] warning: {w}")
    loaded = set(df["model_name"].astype(str)) if len(df) else set()
    present = [m for m, _l in MODELS if m in loaded]
    missing = [m for m, _l in MODELS if m not in loaded]
    if missing:
        print(f"[Fig6] warning: no outputs for {missing} (left out, not replaced)")
    if not present:
        print(f"[Fig6] warning: {FIGURE} not written (none of the selected models has outputs)")
        return None
    frame = sentence_pair_frame(df, present, langs, include_identical)
    rho, n = spearman_matrix(frame, present)
    for a in present:
        low = [b for b in present if b != a and n.loc[a, b] < MIN_SHARED_PAIRS]
        if low:
            print(f"[Fig6] warning: {a} shares < {MIN_SHARED_PAIRS} sentence pairs with {low} (rho n/a)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FIGURE
    plot_heatmap(rho, dict(MODELS), path)
    off = n.to_numpy()[~np.eye(len(present), dtype=bool)]
    print(f"[Fig6] {path} ({len(present)} models; shared pairs per model pair {int(off.min())}..{int(off.max())})")
    return path, rho, n


def main(argv=None):
    parser = argparse.ArgumentParser(description="Figure 6: Spearman model correlation with Manacá 1B (standalone).")
    parser.add_argument("--outputs_dir", default=str(DEFAULT_OUTPUTS_DIR), help="saved outputs2 directory")
    parser.add_argument("--out_dir", default=str(DEFAULT_MAIN_PLOT_DIR), help="figure directory (default Main_plot/)")
    parser.add_argument("--use-preference-annotation", dest="langs", nargs="+", default=list(DEFAULT_LANGUAGES),
                        metavar="LANG", help="languages of the target preference filter (default: de fr pt sv zh ja)")
    parser.add_argument("--include_identical", action="store_true", help="keep identical pos/neg pairs")
    args = parser.parse_args(argv)
    generate(outputs_dir=args.outputs_dir, out_dir=args.out_dir, langs=args.langs,
             include_identical=args.include_identical)


if __name__ == "__main__":
    main()
