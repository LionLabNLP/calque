import argparse
import shutil
from pathlib import Path

import pandas as pd

GPT_PAIRWISE_COLUMNS = [
    "gpt_ab_raw_output",
    "gpt_ab_original_chosen_label",
    "gpt_ab_chosen_sentence",
    "gpt_ab_chosen_logprob",
    "gpt_ba_raw_output",
    "gpt_ba_original_chosen_label",
    "gpt_ba_chosen_sentence",
    "gpt_ba_chosen_logprob",
    "gpt_pairwise_winner",
    "gpt_signed_choice_confidence",
]

EXPERIMENT_KEY_FIELDS = [
    "lang",
    "pos_construction_lang",
    "neg_construction_lang",
    "pos_strategy",
    "neg_strategy",
]

MIGRATED_SUFFIX = "_pairwise.xlsx"


def is_legacy_gpt_workbook(df):
    if not any(c in df.columns for c in GPT_PAIRWISE_COLUMNS):
        return False
    if "prompt_length" not in df.columns or df.empty:
        return False
    normalized = df["prompt_length"].astype(str).str.strip().str.lower()
    return not (normalized == "pairwise").all()


def discover_legacy_gpt_files(outputs_dir):
    files = sorted(Path(outputs_dir).glob("*/*.xlsx"))
    legacy = []
    for f in files:
        if f.name.endswith(MIGRATED_SUFFIX):
            continue
        try:
            df = pd.read_excel(f)
        except Exception as e:
            print(f"WARNING: failed to read {f}: {e}")
            continue
        if is_legacy_gpt_workbook(df):
            legacy.append((f, df))
    return legacy


def experiment_key_for_file(df, model_dir):
    first = df.iloc[0]
    key_values = tuple(first.get(field) for field in EXPERIMENT_KEY_FIELDS)

    for field, value in zip(EXPERIMENT_KEY_FIELDS, key_values):
        if not (df[field] == value).all():
            raise ValueError(
                f"Workbook has rows with inconsistent {field!r} values (expected all rows to share one "
                f"experiment key); this script assumes one legacy workbook = one experiment."
            )

    return (str(model_dir),) + key_values


def destination_path(model_dir, exp_key):
    lang, pos_lang, neg_lang, pos_strategy, neg_strategy = exp_key
    out_name = f"{lang}_{pos_lang}_vs_{neg_lang}_{pos_strategy}_vs_{neg_strategy}_pairwise.xlsx"
    return Path(model_dir) / out_name


def run_migration(outputs_dir, dry_run=False, overwrite=False, archive_dir=None):
    legacy_files = discover_legacy_gpt_files(outputs_dir)
    if not legacy_files:
        print(f"No legacy GPT workbooks found under {outputs_dir}.")
        return {"written": 0, "skipped": 0}

    print(f"Discovered {len(legacy_files)} legacy GPT workbook(s).")

    groups = {}
    for f, df in legacy_files:
        key = experiment_key_for_file(df, f.parent)
        groups.setdefault(key, []).append((f, df))

    n_written = 0
    n_skipped = 0
    for key, entries in sorted(groups.items(), key=lambda kv: kv[0]):
        model_dir_str, lang, pos_lang, neg_lang, pos_strategy, neg_strategy = key
        model_dir = Path(model_dir_str)
        exp_key = (lang, pos_lang, neg_lang, pos_strategy, neg_strategy)

        entries_sorted = sorted(entries, key=lambda e: str(e[0]))
        source_file, source_df = entries_sorted[0]
        duplicate_files = [str(f) for f, _ in entries_sorted[1:]]

        print(
            f"\nExperiment: model_dir={model_dir} lang={lang} "
            f"{pos_lang} vs {neg_lang} {pos_strategy!r} vs {neg_strategy!r}"
        )
        print(f"  legacy workbooks found ({len(entries_sorted)}): repeated executions of the same GPT pairwise")
        print("    prompt (the per-condition prompt text is not used by gpt_pairwise.py), not independent data.")
        print(f"  selected source (first found): {source_file}")
        if duplicate_files:
            print(f"  ignored duplicate run(s) ({len(duplicate_files)}): {duplicate_files}")
        print(f"  source row count: {len(source_df)}")

        migrated_df = source_df.copy()
        migrated_df["prompt_language"] = "None"
        migrated_df["prompt_length"] = "Pairwise"
        migrated_df["prompt"] = ""
        migrated_df["positive_input"] = migrated_df["positive_sentence"]
        migrated_df["negative_input"] = migrated_df["negative_sentence"]

        print(f"  migrated row count: {len(migrated_df)}")

        out_path = destination_path(model_dir, exp_key)
        if out_path.exists() and not overwrite:
            print(f"  SKIPPED: destination already exists (use --overwrite to replace): {out_path}")
            n_skipped += 1
            continue

        print(f"  destination file: {out_path}")

        if dry_run:
            print("  (dry run: not writing)")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        migrated_df.to_excel(out_path, index=False)
        n_written += 1
        print(f"  wrote {len(migrated_df)} row(s) to {out_path}")

        if archive_dir:
            archive_target_dir = Path(archive_dir) / model_dir.name
            archive_target_dir.mkdir(parents=True, exist_ok=True)
            for src_file, _ in entries_sorted:
                dest = archive_target_dir / src_file.name
                if not dest.exists():
                    shutil.move(str(src_file), str(dest))
                    print(f"  archived legacy source: {src_file} -> {dest}")

    n_representative = len(groups)
    n_ignored_repeats = len(legacy_files) - n_representative
    print(
        f"\nUsed {n_representative} representative legacy GPT workbook(s); "
        f"ignored {n_ignored_repeats} repeated condition workbook(s)."
    )
    print(f"Done. {n_written} workbook(s) written, {n_skipped} group(s) skipped.")
    return {"written": n_written, "skipped": n_skipped}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs_dir", type=str, default="./outputs")
    parser.add_argument(
        "--dry_run", action="store_true", help="Report what would be migrated without writing any files."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite an existing destination _pairwise.xlsx if present."
    )
    parser.add_argument(
        "--archive_dir",
        type=str,
        default=None,
        help="If set, move all of an experiment's legacy source files (the selected one and the ignored "
        "repeated-run duplicates) here (outside outputs_dir) after writing the new workbook. Off by default -- "
        "legacy sources are left in place and excluded from later loading via "
        "viz_data.discover_result_files once a migrated sibling exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_migration(args.outputs_dir, dry_run=args.dry_run, overwrite=args.overwrite, archive_dir=args.archive_dir)


if __name__ == "__main__":
    main()
