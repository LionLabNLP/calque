from __future__ import annotations

import argparse
from contextlib import contextmanager
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = PROJECT_ROOT / "plots2" / "main_fig" / "SLOR_failure_rates.xlsx"
DEFAULT_OUT_DIR = PROJECT_ROOT / "Main_plot"
FIGURE = "SLOR_failure_rates_line.pdf"


NEC_SECTION = "Non-English-centric models"
SECTIONS = [
    ("Tiny Aya", ["CohereLabs/tiny-aya-base", "CohereLabs/tiny-aya-global"]),
    ("Aya Expanse", ["CohereLabs/aya-expanse-8b", "CohereLabs/aya-expanse-32b"]),
    ("EuroLLM", ["utter-project/EuroLLM-1.7B", "utter-project/EuroLLM-9B", "utter-project/EuroLLM-22B-2512"]),
    ("EuroLLM IT", ["utter-project/EuroLLM-1.7B-Instruct", "utter-project/EuroLLM-9B-Instruct",
                    "utter-project/EuroLLM-22B-Instruct-2512"]),
    ("Gemma 4", ["google/gemma-4-E2B", "google/gemma-4-E4B", "google/gemma-4-12B",
                 "google/gemma-4-26B-A4B", "google/gemma-4-31B"]),
    ("Gemma 4 IT", ["google/gemma-4-E2B-it", "google/gemma-4-E4B-it", "google/gemma-4-12B-it",
                    "google/gemma-4-26B-A4B-it", "google/gemma-4-31B-it"]),
    ("Apertus", ["swiss-ai/Apertus-8B-2509", "swiss-ai/Apertus-70B-2509"]),
    ("Apertus IT", ["swiss-ai/Apertus-8B-Instruct-2509", "swiss-ai/Apertus-70B-Instruct-2509"]),
    ("GPT-OSS", ["openai/gpt-oss-120b"]),
    (NEC_SECTION, ["OpenLLM-France/Claire-Mistral-7B-0.1", "OpenLLM-France/Lucie-7B",
                   "AIInstituteLNCC/manaca-1b-base", "AIInstituteLNCC/manaca-1b-instruct", "maritaca-ai/sabia-7b",
                   "recogna-nlp/bode-13b-alpaca-pt-br-no-peft", "hfl/chinese-llama-2-13b",
                   "tokyotech-llm/Swallow-13b-hf", "llm-jp/llm-jp-3-13b", "llm-jp/llm-jp-3-172b",
                   "Aleph-Alpha/Kolibri-1-BF16"]),
]
MODEL_COUNT = 36
FIGURE_ROWS = [("EuroLLM", "EuroLLM IT", "Gemma 4", "Gemma 4 IT", "Apertus", "Apertus IT"),
               ("Tiny Aya", "Aya Expanse", "GPT-OSS", NEC_SECTION)]
LABELS = {
    "CohereLabs/tiny-aya-base": "Base", "CohereLabs/tiny-aya-global": "Global",
    "CohereLabs/aya-expanse-8b": "8B", "CohereLabs/aya-expanse-32b": "32B",
    "utter-project/EuroLLM-1.7B": "1.7B", "utter-project/EuroLLM-1.7B-Instruct": "1.7B",
    "utter-project/EuroLLM-9B": "9B", "utter-project/EuroLLM-9B-Instruct": "9B",
    "utter-project/EuroLLM-22B-2512": "22B", "utter-project/EuroLLM-22B-Instruct-2512": "22B",
    "google/gemma-4-E2B": "E2B", "google/gemma-4-E2B-it": "E2B", "google/gemma-4-E4B": "E4B",
    "google/gemma-4-E4B-it": "E4B", "google/gemma-4-12B": "12B", "google/gemma-4-12B-it": "12B",
    "google/gemma-4-26B-A4B": "26B-A4B", "google/gemma-4-26B-A4B-it": "26B-A4B",
    "google/gemma-4-31B": "31B", "google/gemma-4-31B-it": "31B",
    "swiss-ai/Apertus-8B-2509": "8B", "swiss-ai/Apertus-70B-2509": "70B",
    "swiss-ai/Apertus-8B-Instruct-2509": "8B", "swiss-ai/Apertus-70B-Instruct-2509": "70B",
    "openai/gpt-oss-120b": "120B",
    "OpenLLM-France/Claire-Mistral-7B-0.1": "Claire 7B", "OpenLLM-France/Lucie-7B": "Lucie 7B",
    "AIInstituteLNCC/manaca-1b-base": "Manacá 1B", "AIInstituteLNCC/manaca-1b-instruct": "Manacá Instruct 1B",
    "maritaca-ai/sabia-7b": "Sabiá 7B",
    "recogna-nlp/bode-13b-alpaca-pt-br-no-peft": "Bode 13B", "hfl/chinese-llama-2-13b": "Chinese-LLaMA-2 13B",
    "tokyotech-llm/Swallow-13b-hf": "Swallow 13B", "llm-jp/llm-jp-3-13b": "LLM-jp-3 13B",
    "llm-jp/llm-jp-3-172b": "LLM-jp-3 172B", "Aleph-Alpha/Kolibri-1-BF16": "Kolibri",
}
LANGUAGE_ORDER = ["de", "sv", "fr", "pt", "zh", "ja", "en"]
LANGUAGE_COLORS = {"de": "#fe6100", "sv": "#ffb000", "fr": "#4CAF50", "pt": "#648fff",
                   "zh": "#785ef0", "ja": "#dc267f", "en": "#808080"}
LANGUAGE_NAMES = {"en": "English", "de": "German", "sv": "Swedish", "fr": "French", "pt": "Portuguese",
                  "zh": "Chinese", "ja": "Japanese"}

STATUS_OK = "ok"
STATUS_MISSING = "no data (model not in workbook)"
STATUS_EMPTY = "no observations (N_total = 0)"

WIDTH, ROW_HEIGHT = 6.3, 1.75
TICK_SIZE, LABEL_SIZE, HEADER_SIZE, LEGEND_SIZE = 7, 8, 7.5, 7
SECTION_GAP = 0.7
TEXT_COLOR, MUTED_COLOR, GRID_COLOR = "#0b0b0b", "#52514e", "#e4e3df"
SPINE_WIDTH = 0.8
SECTION_RULE_COLOR, SECTION_RULE_WIDTH = "#6E6E6E", 0.45
SECTION_SEPARATOR_COLOR, SECTION_SEPARATOR_WIDTH = "#D9D9D9", 0.45
MARKER_SIZE, LINE_WIDTH = 3.0, 0.7
Y_MAX = 100

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
    "axes.spines.top": False, "axes.spines.right": False,
}


@contextmanager
def figure_style():
    with plt.rc_context():
        matplotlib.rcdefaults()
        plt.rcParams.update(STYLE)
        yield


def workbook_key(model_id):
    return str(model_id).replace("/", "_", 1)


def load_workbook(path):
    sheets = pd.read_excel(path, sheet_name=["failure_rate", "N_failure", "N_total"])
    return tuple(sheets[s].set_index("model_name") for s in ("failure_rate", "N_failure", "N_total"))


def failure_table(rates, n_failure, n_total):
    languages = [l for l in LANGUAGE_ORDER if l in rates.columns]
    keyed = {workbook_key(m): m for m in rates.index}
    rows, order = [], 0
    for section, ids in SECTIONS:
        for model_id in ids:
            key = workbook_key(model_id)
            for lang in languages:
                rate = nf = nt = np.nan
                if key in keyed:
                    rate, nf, nt = (float(t.loc[keyed[key], lang]) for t in (rates, n_failure, n_total))
                status = (STATUS_MISSING if key not in keyed
                          else STATUS_EMPTY if not (np.isfinite(nt) and nt > 0) else STATUS_OK)
                if status == STATUS_OK and (not np.isfinite(rate) or abs(rate - nf / nt) > 1e-9):
                    raise AssertionError(f"workbook inconsistent for {key} {lang}: {rate} vs {nf}/{nt}")
                if status == STATUS_EMPTY and np.isfinite(rate):
                    raise AssertionError(f"workbook has a rate without observations: {key} {lang}")
                rows.append({"order": order, "section": section, "model_id": model_id, "model_name": key,
                             "label": LABELS.get(model_id, key), "language": lang, "status": status,
                             "failure_rate_percent": 100 * rate if status == STATUS_OK else np.nan,
                             "N_failure": nf, "N_total": nt})
            order += 1
    if order != MODEL_COUNT:
        raise AssertionError(f"Figure 3 model list has {order} models, expected {MODEL_COUNT}")
    return pd.DataFrame(rows)


def _tick_label(section, label):
    if section == NEC_SECTION:
        return re.sub(r"(?<![\w.])(\d+(?:\.\d+)?B)$", lambda m: r"$_{" + m.group(1) + r"}$", label)
    return label


def plot_failure_rates(table, out_path):
    languages = list(dict.fromkeys(table["language"]))
    models = table.drop_duplicates("order").set_index("order")
    present = list(dict.fromkeys(models["section"]))
    rows_spec = [[s for s in r if s in present] for r in FIGURE_ROWS]
    rows_spec = [r for r in rows_spec if r] + [[s for s in present if not any(s in r for r in FIGURE_ROWS)]]
    rows_spec = [r for r in rows_spec if r]

    def row_width(secs):
        return sum((models["section"] == s).sum() for s in secs) + SECTION_GAP * (len(secs) - 1)

    width_units = max(row_width(r) for r in rows_spec)
    any_empty = False
    head_y, rule_y = 1.07 * Y_MAX, 1.04 * Y_MAX
    with figure_style():
        fig, axes = plt.subplots(len(rows_spec), 1, figsize=(WIDTH, ROW_HEIGHT * len(rows_spec) + 1.0), squeeze=False)
        for ax, secs in zip(axes[:, 0], rows_spec):
            x, ticks, tick_labels, tick_missing = 0.0, [], [], []
            for si, sec in enumerate(secs):
                start = x
                series = {lang: [] for lang in languages}
                for order, m in models.loc[models["section"] == sec].iterrows():
                    sub = table.loc[table["order"] == order].set_index("language")
                    missing = (sub["status"] == STATUS_MISSING).all()
                    if missing:
                        ax.text(x, Y_MAX / 2, "no data", rotation=90, ha="center", va="center",
                                fontsize=TICK_SIZE - 0.5, color=MUTED_COLOR, style="italic")
                    for li, lang in enumerate(languages):
                        r = sub.loc[lang]
                        ok = r["status"] == STATUS_OK and np.isfinite(r["failure_rate_percent"])
                        series[lang].append((x, r["failure_rate_percent"] if ok else np.nan))
                        if not ok and not missing:
                            any_empty = True
                            ax.plot([x + (li - (len(languages) - 1) / 2) * 0.1], [0.025 * Y_MAX], marker="x",
                                    markersize=2.6, color=LANGUAGE_COLORS[lang], markeredgewidth=0.7, zorder=3,
                                    clip_on=False)
                    ticks.append(x)
                    tick_labels.append(_tick_label(sec, str(m["label"])))
                    tick_missing.append(missing)
                    x += 1
                for lang, pts in series.items():
                    xs = np.array([p[0] for p in pts], dtype=float)
                    ys = np.array([p[1] for p in pts], dtype=float)
                    valid = np.isfinite(ys)
                    for i in range(len(xs) - 1):
                        if valid[i] and valid[i + 1]:
                            ax.plot(xs[i:i + 2], ys[i:i + 2], color=LANGUAGE_COLORS[lang], linewidth=LINE_WIDTH,
                                    alpha=0.85, zorder=2, solid_capstyle="round")
                    ax.plot(xs[valid], ys[valid], linestyle="none", marker="o", markersize=MARKER_SIZE,
                            color=LANGUAGE_COLORS[lang], markeredgecolor="white", markeredgewidth=0.3, zorder=3)
                end = x - 1
                ax.plot([start - 0.4, end + 0.4], [rule_y, rule_y], color=SECTION_RULE_COLOR,
                        linewidth=SECTION_RULE_WIDTH, clip_on=False, solid_capstyle="butt")
                ax.text((start + end) / 2, head_y, sec, ha="center", va="bottom", fontsize=HEADER_SIZE,
                        color=TEXT_COLOR, clip_on=False)
                if si < len(secs) - 1:
                    ax.axvline(x - 0.5 + SECTION_GAP / 2, color=SECTION_SEPARATOR_COLOR,
                               linewidth=SECTION_SEPARATOR_WIDTH, zorder=0.5)
                x += SECTION_GAP
            x_left, x_data_end = -0.6, ticks[-1] + 0.6
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels, rotation=30, ha="right", rotation_mode="anchor", fontsize=TICK_SIZE,
                               color=TEXT_COLOR)
            for lbl, miss in zip(ax.get_xticklabels(), tick_missing):
                if miss:
                    lbl.set_color(MUTED_COLOR)
                    lbl.set_fontstyle("italic")
            ax.set_xlim(x_left, width_units - 0.4)
            ax.set_ylim(0, Y_MAX)
            yticks = np.linspace(0, Y_MAX, 5)
            ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(Y_MAX, decimals=0))
            ax.tick_params(axis="y", labelsize=TICK_SIZE, length=3, width=0.8)
            ax.tick_params(axis="x", length=2, width=0.8, pad=1.5)
            ax.set_ylabel("Failure rate", fontsize=LABEL_SIZE)
            ax.hlines(yticks[1:], x_left, x_data_end, color=GRID_COLOR, linewidth=0.5, zorder=0.4)
            ax.spines["bottom"].set_bounds(x_left, x_data_end)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("black")
                ax.spines[side].set_linewidth(SPINE_WIDTH)
        handles = [Line2D([0], [0], color=LANGUAGE_COLORS[l], linewidth=LINE_WIDTH + 0.3, marker="o",
                          markersize=MARKER_SIZE + 0.5, label=LANGUAGE_NAMES[l]) for l in languages]
        if any_empty:
            handles.append(Line2D([0], [0], marker="x", linestyle="none", color=MUTED_COLOR, markersize=3.5,
                                  label="N = 0"))
        free, ri = max((width_units - row_width(r), i) for i, r in enumerate(rows_spec))
        if free >= 3:
            x_anchor = (width_units - free - 0.4 + 0.25 + 0.6) / width_units
            axes[ri, 0].legend(handles=handles, frameon=False, fontsize=LEGEND_SIZE, ncol=1, loc="center left",
                               bbox_to_anchor=(x_anchor, 0.5), handlelength=1.4, handletextpad=0.5,
                               labelspacing=0.3, borderaxespad=0.0)
        else:
            axes[0, 0].legend(handles=handles, frameon=True, framealpha=0.9, edgecolor="none", fontsize=LEGEND_SIZE,
                              ncol=1, loc="upper right", handlelength=1.4, labelspacing=0.3)
        fig.tight_layout(h_pad=0.8)
        fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    return True


def generate(workbook=DEFAULT_WORKBOOK, out_dir=DEFAULT_OUT_DIR):
    workbook = Path(workbook)
    if not workbook.is_file():
        raise FileNotFoundError(f"Fig3: failure-rate workbook not found: {workbook} (run export_plots2.py "
                                "--only_main first)")
    table = failure_table(*load_workbook(workbook))
    missing = sorted(table.loc[table["status"] == STATUS_MISSING, "model_name"].unique())
    if missing:
        print(f"[Fig3] warning: no workbook data, shown as 'no data': {missing}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FIGURE
    plot_failure_rates(table, path)
    print(f"[Fig3] {path} (from {workbook})")
    return path, table


def main(argv=None):
    parser = argparse.ArgumentParser(description="Figure 3: ΔSLOR failure rates (standalone).")
    parser.add_argument("--workbook", default=str(DEFAULT_WORKBOOK), help="saved SLOR_failure_rates.xlsx")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="figure directory (default Main_plot/)")
    args = parser.parse_args(argv)
    generate(args.workbook, args.out_dir)


if __name__ == "__main__":
    main()
