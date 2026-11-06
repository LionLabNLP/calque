import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

import gpt_pairwise
import hf_pairwise
import slor
from data_loading import (
    EXPERIMENT_JAPANESE_CONTROL,
    EXPERIMENT_STANDARD,
    JAPANESE_CONTROL_COMPARISON,
    PAIR_ID_COLUMN2,
    TARGET_LANGUAGE_CANONICAL_COLUMN,
    TASK_LANGUAGE_BY_CODE,
    canonical_language_code,
    canonical_target_languages,
    comparison_experiment,
    japanese_control_availability,
    load_construction_data2,
    load_prompt_conditions,
    select_examples2,
    select_translation_rows2,
    sentence_column_construction_language,
    sentence_column_language,
    validate_sentence_columns2,
)
from distributed_utils import cleanup_distributed, distributed_barrier, is_main_process
from model_utils import load_model_and_tokenizer
from probability import compute_diff_subset_metrics, compute_sentence_logprob
from sentence_probability import (
    GPT_UNAVAILABLE_LOCAL_LM_FIELDS,
    build_prompted_input,
    generate_translation,
    to_json_field,
)
from copy import copy

INPUT_METADATA_COLUMNS = [
    "Target language",
    "Subject",
    "Proper English strategy",
    "English preference annotation",
    "Target language preference annotation",
    "Japanese preference annotation",
    "Group",
    "Subgroup",
    "Comment",
    "Temporal interpretation",
]

TARGET_LANGUAGES = ["de", "fr", "pt", "sv", "zh", "ja"]

COMPARISONS = [
    (
        "Proper English strategy",
        "Literal English translation of target language strategy",
    ),
    (
        "Proper strategy in target language",
        "Literal translation of Proper English into target language",
    ),
    (
        "Proper strategy in Japanese",
        "Literal translation of Proper target language into Japanese",
    ),
]

LANGUAGE_NAME_BY_CODE2 = {
    "de": "German",
    "fr": "French",
    "pt": "Portuguese",
    "sv": "Swedish",
    "zh": "Chinese",
    "ja": "Japanese",
}

FINAL_EXP_PREFERENCE_COLUMNS = {
    COMPARISONS[0]: "English preference annotation",
    COMPARISONS[1]: "Target language preference annotation",
    COMPARISONS[2]: "Japanese preference annotation",
}
FINAL_EXP_ALLOWED_PREFERENCES = frozenset({">", ">>"})


def filter_final_exp_rows2(selected_df, pos_col, neg_col):
    preference_col = FINAL_EXP_PREFERENCE_COLUMNS.get((pos_col, neg_col))
    if preference_col is None:
        raise ValueError(f"--final_exp has no preference mapping for: {(pos_col, neg_col)!r}")
    if preference_col not in selected_df.columns:
        raise ValueError(f"Missing --final_exp preference column: {preference_col!r}")
    mask = selected_df[preference_col].astype("string").str.strip().isin(FINAL_EXP_ALLOWED_PREFERENCES)
    return selected_df.loc[mask].reset_index(drop=True)


NO_CONTEXT_CONDITION = {
    "prompt_text": None,
    "prompt_language": "raw",
    "prompt_length": "raw",
}

def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")


def build_pair_base_metadata2(args, prompt_language, prompt_length, row, pos_sentence, neg_sentence):
    target_language = canonical_language_code(row["Target language"])
    experiment = comparison_experiment(args.pos, args.neg)
    base = {col: row.get(col) for col in INPUT_METADATA_COLUMNS}
    base.update({
        TARGET_LANGUAGE_CANONICAL_COLUMN: target_language,
        "experiment": experiment,
        "scoring_language": sentence_column_language(args.pos, target_language),
        "lang": args.lang,
        "pos": args.pos,
        "neg": args.neg,
        "pos_sentence": pos_sentence,
        "neg_sentence": neg_sentence,
        "pos_construction_lang": sentence_column_construction_language(args.pos, target_language),
        "neg_construction_lang": sentence_column_construction_language(args.neg, target_language),
        "model_name": args.model_name,
        "prompt_language": prompt_language,
        "prompt_length": prompt_length,
    })
    if experiment == EXPERIMENT_JAPANESE_CONTROL:
        for col in JAPANESE_CONTROL_COMPARISON:
            base[col] = row.get(col)
    return base


def compute_pair_row_fields2(model, tokenizer, pos_sentence, neg_sentence, prompt_text, unigram_stats, vocab_size):
    pos_result = compute_sentence_logprob(model, tokenizer, sentence=pos_sentence, prompt_text=prompt_text)
    neg_result = compute_sentence_logprob(model, tokenizer, sentence=neg_sentence, prompt_text=prompt_text)

    slor_pos = slor.compute_slor(
        pos_result["logprob"],
        pos_result["token_ids"],
        unigram_stats["token_counts"],
        unigram_stats["total_tokens"],
        vocab_size,
    )
    slor_neg = slor.compute_slor(
        neg_result["logprob"],
        neg_result["token_ids"],
        unigram_stats["token_counts"],
        unigram_stats["total_tokens"],
        vocab_size,
    )

    diff_metrics = compute_diff_subset_metrics(pos_result, neg_result)

    return {
        "prompt": prompt_text or "",
        "positive_input": build_prompted_input(prompt_text, pos_sentence),
        "negative_input": build_prompted_input(prompt_text, neg_sentence),
        "positive_sentence_logprob": pos_result["logprob"],
        "negative_sentence_logprob": neg_result["logprob"],
        "positive_avg_token_logprob": pos_result["avg_logprob"],
        "negative_avg_token_logprob": neg_result["avg_logprob"],
        "token_logprob_diff": pos_result["avg_logprob"] - neg_result["avg_logprob"],
        "SLOR_positive_avg_token_logprob": slor_pos,
        "SLOR_negative_avg_token_logprob": slor_neg,
        "SLOR_token_logprob_diff": slor_pos - slor_neg,
        "positive_only_diff_tokens_avg_token_logprob": diff_metrics["positive_only_diff_tokens_avg_token_logprob"],
        "negative_only_diff_tokens_avg_token_logprob": diff_metrics["negative_only_diff_tokens_avg_token_logprob"],
        "only_diff_tokens_tokenwise_log_prob_diff": diff_metrics["only_diff_tokens_tokenwise_log_prob_diff"],
        "positive_after_diff_tokens_avg_token_logprob": diff_metrics["positive_after_diff_tokens_avg_token_logprob"],
        "negative_after_diff_tokens_avg_token_logprob": diff_metrics["negative_after_diff_tokens_avg_token_logprob"],
        "after_diff_tokens_tokenwise_log_prob_diff": diff_metrics["after_diff_tokens_tokenwise_log_prob_diff"],
        "pos_diff_tokens": to_json_field(diff_metrics["pos_diff_tokens"]),
        "neg_diff_tokens": to_json_field(diff_metrics["neg_diff_tokens"]),
        "pos_after_diff_tokens": to_json_field(diff_metrics["pos_after_diff_tokens"]),
        "neg_after_diff_tokens": to_json_field(diff_metrics["neg_after_diff_tokens"]),
        "positive_tokens": to_json_field(pos_result["token_strs"]),
        "negative_tokens": to_json_field(neg_result["token_strs"]),
        "positive_token_surprisals": to_json_field(pos_result["token_surprisals"]),
        "negative_token_surprisals": to_json_field(neg_result["token_surprisals"]),
    }


def _pairwise_row_fields2(comparison, pos_sentence, neg_sentence, prompt_text=None):
    fields = {
        "prompt": prompt_text or "",
        "positive_input": build_prompted_input(prompt_text, pos_sentence),
        "negative_input": build_prompted_input(prompt_text, neg_sentence),
    }
    fields.update(comparison)

    for key in GPT_UNAVAILABLE_LOCAL_LM_FIELDS:
        fields[key] = float("nan")

    return fields


def compute_gpt_pair_row_fields2(client, gpt_model, pos_sentence, neg_sentence):
    comparison = gpt_pairwise.run_pairwise_comparison(client, gpt_model, None, pos_sentence, neg_sentence)
    return _pairwise_row_fields2(comparison, pos_sentence, neg_sentence, prompt_text=None)


def compute_hf_pairwise_pair_row_fields2(model, tokenizer, pos_sentence, neg_sentence):
    comparison = hf_pairwise.run_pairwise_comparison(model, tokenizer, pos_sentence, neg_sentence)
    return _pairwise_row_fields2(comparison, pos_sentence, neg_sentence, prompt_text=None)


def load_existing_results2(path):
    path = Path(path)
    if not path.exists():
        return {}

    try:
        df = pd.read_excel(path, keep_default_na=False, na_values=[])
    except Exception:
        return {}

    if PAIR_ID_COLUMN2 not in df.columns:
        return {}

    index = {}
    for _, existing_row in df.iterrows():
        pair_id = existing_row.get(PAIR_ID_COLUMN2)
        if pd.isna(pair_id) or pair_id == "":
            continue
        index[pair_id] = existing_row.to_dict()
    return index


def _sentences_unchanged(existing_row, pos_sentence, neg_sentence):
    return existing_row.get("pos_sentence") == pos_sentence and existing_row.get("neg_sentence") == neg_sentence


def compute_or_reuse_row(existing_index, pair_id, pos_sentence, neg_sentence, base_metadata, compute_fields_fn):
    existing_row = existing_index.get(pair_id)

    if existing_row is not None and _sentences_unchanged(existing_row, pos_sentence, neg_sentence):
        fields = dict(existing_row)
        fields.update(base_metadata)
        return fields, True

    fields = {**base_metadata, **compute_fields_fn()}
    return fields, False


def _save_results2(rows, out_path, label, skipped_count, recomputed_count):
    result_df = pd.DataFrame(rows)
    distributed_barrier()
    if is_main_process():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_excel(out_path, index=False)
        print(
            f"Saved {len(result_df)} {label} row(s) to {out_path} "
            f"(reused={skipped_count}, computed={recomputed_count})",
            flush=True,
        )
    return result_df


JAPANESE_CONTROL_FILE_TAG = "japanese_control"


def _comparison_file_stem2(args, legacy=False):
    tag = ""
    if not legacy and comparison_experiment(args.pos, args.neg) == EXPERIMENT_JAPANESE_CONTROL:
        tag = f"{JAPANESE_CONTROL_FILE_TAG}__"
    return f"{args.lang}__{tag}pos-{_slug(args.pos)}__neg-{_slug(args.neg)}"


def _pairwise_output_path2(args, output_dir, legacy=False):
    return output_dir / f"{_comparison_file_stem2(args, legacy)}__pairwise.xlsx"


def _condition_output_path2(args, output_dir, condition, legacy=False):
    return output_dir / (
        f"{_comparison_file_stem2(args, legacy)}__"
        f"{condition['prompt_language']}_{condition['prompt_length']}.xlsx"
    )


def _load_existing_for_resume2(out_path, legacy_path):
    existing = load_existing_results2(out_path)
    if not existing and legacy_path != out_path:
        existing = load_existing_results2(legacy_path)
    return existing


def run_pairwise_pass2(args, selected_df, output_dir, compute_row_fields, backend_label):
    out_path = _pairwise_output_path2(args, output_dir)
    existing_index = _load_existing_for_resume2(out_path, _pairwise_output_path2(args, output_dir, legacy=True))

    rows = []
    skipped = recomputed = 0
    for _, row in selected_df.iterrows():
        pos_sentence = row[args.pos]
        neg_sentence = row[args.neg]
        pair_id = row[PAIR_ID_COLUMN2]

        base_metadata = build_pair_base_metadata2(
            args, prompt_language="pairwise", prompt_length="pairwise", row=row,
            pos_sentence=pos_sentence, neg_sentence=neg_sentence,
        )

        fields, was_skipped = compute_or_reuse_row(
            existing_index, pair_id, pos_sentence, neg_sentence, base_metadata,
            lambda: compute_row_fields(pos_sentence, neg_sentence),
        )
        rows.append(fields)
        skipped += int(was_skipped)
        recomputed += int(not was_skipped)

    return _save_results2(rows, out_path, f"{backend_label} pairwise", skipped, recomputed)


def run_condition_pass2(args, selected_df, output_dir, condition, model, tokenizer, unigram_stats, vocab_size):
    out_path = _condition_output_path2(args, output_dir, condition)
    existing_index = _load_existing_for_resume2(
        out_path, _condition_output_path2(args, output_dir, condition, legacy=True)
    )

    rows = []
    skipped = recomputed = 0
    for _, row in selected_df.iterrows():
        pos_sentence = row[args.pos]
        neg_sentence = row[args.neg]
        pair_id = row[PAIR_ID_COLUMN2]

        base_metadata = build_pair_base_metadata2(
            args,
            prompt_language=condition["prompt_language"],
            prompt_length=condition["prompt_length"],
            row=row,
            pos_sentence=pos_sentence,
            neg_sentence=neg_sentence,
        )

        fields, was_skipped = compute_or_reuse_row(
            existing_index, pair_id, pos_sentence, neg_sentence, base_metadata,
            lambda: compute_pair_row_fields2(
                model, tokenizer, pos_sentence, neg_sentence, condition["prompt_text"], unigram_stats, vocab_size
            ),
        )
        rows.append(fields)
        skipped += int(was_skipped)
        recomputed += int(not was_skipped)

    label = f"{condition['prompt_language']}/{condition['prompt_length']}"
    return _save_results2(rows, out_path, label, skipped, recomputed)


def run_no_context_pass2(args, selected_df, output_dir, model, tokenizer, unigram_stats, vocab_size):
    return run_condition_pass2(
        args, selected_df, output_dir, NO_CONTEXT_CONDITION, model, tokenizer, unigram_stats, vocab_size
    )


TRANSLATION_MATCH_LABELS = ["proper", "literal", "both", "neither"]

TRANSLATION_METADATA_COLUMNS = [
    "Target language",
    "Subject",
    "Proper English strategy",
    "Proper strategy in target language",
    "Literal translation of Proper English into target language",
    "Group",
    "Subgroup",
    "Comment",
    "Temporal interpretation",
    "English preference annotation",
    "Target language preference annotation",
]


def _safe_text2(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


_CONTINUOUS_SCRIPT_LANGS2 = {"zh", "ja"}
_WHITESPACE_RE2 = re.compile(r"\s+")
_TRANSLATION_SENTENCE_FINAL2 = (".", "。")


def normalize_translation_match_text2(text):
    text = _WHITESPACE_RE2.sub(" ", _safe_text2(text)).strip()
    if text.endswith(_TRANSLATION_SENTENCE_FINAL2):
        text = text[:-1].strip()
    return text.casefold()


def _is_word_char2(ch):
    return ch.isalnum() or unicodedata.category(ch).startswith("M")


def contains_reference2(normalized_text, normalized_reference, target_lang):
    if not normalized_reference:
        return False
    check_boundaries = target_lang not in _CONTINUOUS_SCRIPT_LANGS2
    end_len = len(normalized_reference)
    start = normalized_text.find(normalized_reference)
    while start != -1:
        if not check_boundaries:
            return True
        end = start + end_len
        before_ok = start == 0 or not _is_word_char2(normalized_text[start - 1])
        after_ok = end == len(normalized_text) or not _is_word_char2(normalized_text[end])
        if before_ok and after_ok:
            return True
        start = normalized_text.find(normalized_reference, start + 1)
    return False


def reference_containment2(proper_reference, literal_reference, target_lang):
    proper = normalize_translation_match_text2(proper_reference)
    literal = normalize_translation_match_text2(literal_reference)
    if proper and proper == literal:
        return "identical"
    if contains_reference2(literal, proper, target_lang):
        return "proper_in_literal"
    if contains_reference2(proper, literal, target_lang):
        return "literal_in_proper"
    return "none"


TRANSLATION_SOURCE_LANGUAGE2 = "en"
_IDEOGRAPHIC_FULL_STOP_LANGS2 = {"ja", "zh"}
_SENTENCE_TERMINATORS2 = (".", "!", "?", "。", "！", "？")


def with_sentence_final_punctuation2(sentence, source_lang):
    text = _safe_text2(sentence).rstrip()
    if not text or text.endswith(_SENTENCE_TERMINATORS2):
        return text
    stop = "。" if canonical_language_code(source_lang) in _IDEOGRAPHIC_FULL_STOP_LANGS2 else "."
    return text + stop


def build_translation_prompt2(target_lang, english_source_sentence, source_lang=TRANSLATION_SOURCE_LANGUAGE2):
    target_lang = canonical_language_code(target_lang)
    if target_lang not in LANGUAGE_NAME_BY_CODE2:
        raise ValueError(f"No LANGUAGE_NAME_BY_CODE2 mapping for target_lang: {target_lang!r}")

    target_lang_name = LANGUAGE_NAME_BY_CODE2[target_lang]
    source_sentence = with_sentence_final_punctuation2(english_source_sentence, source_lang)
    return (
        f"Translate the following English sentence into {target_lang_name}.\n"
        f"English: {source_sentence}\n"
        f"{target_lang_name}:"
    )


def classify_translation2(raw_generated_text, proper_reference, literal_reference, target_lang):
    generated = normalize_translation_match_text2(raw_generated_text)
    proper = normalize_translation_match_text2(proper_reference)
    literal = normalize_translation_match_text2(literal_reference)
    containment = reference_containment2(proper_reference, literal_reference, target_lang)

    matches_proper = contains_reference2(generated, proper, target_lang)
    matches_literal = contains_reference2(generated, literal, target_lang)

    if matches_proper and matches_literal:
        if containment == "proper_in_literal":
            classification = "literal"
        elif containment == "literal_in_proper":
            classification = "proper"
        else:
            classification = "both"
    elif matches_proper:
        classification = "proper"
    elif matches_literal:
        classification = "literal"
    else:
        classification = "neither"

    return {
        "generated_translation_normalized": generated,
        "translation_match": classification,
        "reference_containment": containment,
    }


def build_translation_row_metadata2(args, row):
    base = {col: row.get(col) for col in TRANSLATION_METADATA_COLUMNS}
    base[TARGET_LANGUAGE_CANONICAL_COLUMN] = canonical_language_code(row.get("Target language"))
    base["translation_source_language"] = TRANSLATION_SOURCE_LANGUAGE2
    base["model_name"] = args.model_name
    return base


def _translation_generation_output_path2(output_dir, target_lang):
    return output_dir / f"{target_lang}__translation_generation.xlsx"


def _translation_generation_summary_path2(output_dir, target_lang):
    return output_dir / f"{target_lang}__translation_generation_summary.xlsx"


def _translation_identity_keys2(rows_iterable):
    seen_counts = {}
    keys = []
    for value in rows_iterable:
        seen_counts[value] = seen_counts.get(value, 0) + 1
        keys.append((value, seen_counts[value]))
    return keys


def _load_existing_translation_results2(path):
    path = Path(path)
    if not path.exists():
        return {}

    try:
        df = pd.read_excel(path, keep_default_na=False, na_values=[])
    except Exception:
        return {}

    if "Proper English strategy" not in df.columns:
        return {}

    keys = _translation_identity_keys2(df["Proper English strategy"].tolist())
    index = {}
    for key, (_, existing_row) in zip(keys, df.iterrows()):
        index[key] = existing_row.to_dict()
    return index


def _generation_reusable2(existing_row, english_source, translation_prompt):
    return (
        "generated_translation_raw" in existing_row
        and existing_row.get("Proper English strategy") == english_source
        and existing_row.get("translation_prompt") == translation_prompt
    )


def build_translation_summary2(args, target_lang, result_df):
    total = len(result_df)
    counts = result_df["translation_match"].value_counts() if "translation_match" in result_df.columns else {}

    summary_rows = [
        {
            "model_name": args.model_name,
            "Target language": target_lang,
            "classification": label,
            "count": int(counts.get(label, 0)),
            "proportion": (counts.get(label, 0) / total) if total else float("nan"),
        }
        for label in TRANSLATION_MATCH_LABELS
    ]
    summary_rows.append({
        "model_name": args.model_name,
        "Target language": target_lang,
        "classification": "total",
        "count": total,
        "proportion": 1.0 if total else float("nan"),
    })
    return pd.DataFrame(summary_rows)


def run_translation_pass2(args, translation_rows, output_dir, target_lang, model, tokenizer):
    out_path = _translation_generation_output_path2(output_dir, target_lang)
    summary_path = _translation_generation_summary_path2(output_dir, target_lang)

    existing_index = _load_existing_translation_results2(out_path)
    identity_keys = _translation_identity_keys2(translation_rows["Proper English strategy"].tolist())

    rows = []
    skipped = recomputed = 0

    for key, (_, row) in zip(identity_keys, translation_rows.iterrows()):
        english_source = row["Proper English strategy"]
        proper_reference = row.get("Proper strategy in target language")
        literal_reference = row.get("Literal translation of Proper English into target language")

        base_metadata = build_translation_row_metadata2(args, row)
        existing_row = existing_index.get(key)
        translation_prompt = build_translation_prompt2(target_lang, english_source)

        if existing_row is not None and _generation_reusable2(existing_row, english_source, translation_prompt):
            raw_generated_text = _safe_text2(existing_row.get("generated_translation_raw"))
            was_skipped = True
        else:
            raw_generated_text = generate_translation(model, tokenizer, translation_prompt)
            was_skipped = False

        fields = {
            **base_metadata,
            "translation_prompt": translation_prompt,
            "generated_translation_raw": raw_generated_text,
            **classify_translation2(raw_generated_text, proper_reference, literal_reference, target_lang),
        }

        rows.append(fields)
        skipped += int(was_skipped)
        recomputed += int(not was_skipped)

    result_df = _save_results2(rows, out_path, f"{target_lang} translation-generation", skipped, recomputed)

    summary_df = build_translation_summary2(args, target_lang, result_df)
    if is_main_process():
        summary_df.to_excel(summary_path, index=False)
        print(f"Saved translation-generation summary to {summary_path}", flush=True)

    return result_df, summary_df


def resolve_vocab_size2(model, tokenizer):
    config_vocab_size = getattr(model.config, "vocab_size", None)
    try:
        tokenizer_vocab_size = len(tokenizer)
    except TypeError:
        tokenizer_vocab_size = None

    if config_vocab_size is not None:
        if tokenizer_vocab_size is not None and config_vocab_size != tokenizer_vocab_size:
            print(
                f"[4] vocab_size={config_vocab_size} (model.config.vocab_size); "
                f"note: len(tokenizer)={tokenizer_vocab_size} differs by "
                f"{config_vocab_size - tokenizer_vocab_size}",
                flush=True,
            )
        else:
            print(f"[4] vocab_size={config_vocab_size} (model.config.vocab_size)", flush=True)
        return config_vocab_size

    if tokenizer_vocab_size is None:
        raise ValueError("model.config.vocab_size is unavailable and len(tokenizer) is not supported")

    text_config = getattr(model.config, "text_config", None)
    nested_vocab_size = getattr(text_config, "vocab_size", None) if text_config is not None else None
    nested_note = f"; model.config.text_config.vocab_size={nested_vocab_size}" if nested_vocab_size is not None else ""
    print(
        f"[4] model.config.vocab_size unavailable; using len(tokenizer)={tokenizer_vocab_size}{nested_note}",
        flush=True,
    )
    return tokenizer_vocab_size


def run_experiment2(args):
    print("[1] Loading construction data...", flush=True)
    construction_df = load_construction_data2(args.construction_data)
    print(f"[1] Done. rows={len(construction_df)}", flush=True)

    output_dir = Path("./outputs2") / args.model_name
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    print("[2] Checking model type...", flush=True)
    use_openai = gpt_pairwise.is_openai_model(args.model)
    print(f"[2] use_openai={use_openai}", flush=True)

    tokenizer = model = client = None
    vocab_size = None

    if use_openai:
        print("[3] Creating OpenAI client...", flush=True)
        client = gpt_pairwise.get_openai_client()
        print("[3] Done.", flush=True)

    else:
        print(f"[3] Loading model and tokenizer ONCE: {args.model}", flush=True)
        tokenizer, model = load_model_and_tokenizer(args.model)
        print("[3] Model loaded.", flush=True)

        vocab_size = resolve_vocab_size2(model, tokenizer)

    run_pairwise = bool(getattr(args, "pairwise", False))

    unigram_cache = {}

    target_languages, selected_comparisons, run_translation = resolve_run_plan2(args)
    if JAPANESE_CONTROL_COMPARISON in selected_comparisons:
        report_japanese_control_availability2(construction_df)

    for target_lang in target_languages:

        print(
            f"\n{'=' * 80}\n"
            f"TARGET LANGUAGE: {target_lang}\n"
            f"{'=' * 80}",
            flush=True,
        )

        comparisons = list(selected_comparisons)

        if target_lang == "ja":
            comparisons = [c for c in comparisons if c != JAPANESE_CONTROL_COMPARISON]

        if not use_openai and run_translation:
            translation_rows = select_translation_rows2(construction_df, target_lang)

            print(
                f"Translation-generation rows for {target_lang}: {len(translation_rows)}",
                flush=True,
            )

            if translation_rows.empty:
                print(
                    f"WARNING: no translation-generation rows selected; "
                    f"skipping translation generation for {target_lang}",
                    flush=True,
                )
            else:
                run_translation_pass2(
                    args,
                    translation_rows,
                    output_dir,
                    target_lang,
                    model,
                    tokenizer,
                )

        for pos, neg in comparisons:

            print(
                f"\n--- Comparison ---\n"
                f"experiment = {comparison_experiment(pos, neg)}\n"
                f"original target language (row selection) = {target_lang}\n"
                f"scoring language = {sentence_column_language(pos, target_lang)}\n"
                f"pos = {pos}\n"
                f"neg = {neg}",
                flush=True,
            )

            run_args = copy(args)
            run_args.lang = target_lang
            run_args.pos = pos
            run_args.neg = neg

            validate_sentence_columns2(
                construction_df,
                pos,
                neg,
            )

            selected_df = select_examples2(
                construction_df,
                target_lang,
                pos,
                neg,
            )

            if getattr(args, "final_exp", False):
                before_count = len(selected_df)
                selected_df = filter_final_exp_rows2(selected_df, pos, neg)
                print(
                    f"--final_exp: {FINAL_EXP_PREFERENCE_COLUMNS[(pos, neg)]} "
                    f"in {{'>', '>>'}}: {before_count} -> {len(selected_df)} rows",
                    flush=True,
                )

            print(
                f"Selected rows: {len(selected_df)}",
                flush=True,
            )

            if selected_df.empty:
                print(
                    f"WARNING: no rows selected; skipping "
                    f"{target_lang} / {pos} / {neg}",
                    flush=True,
                )
                continue

            if use_openai:
                if not run_pairwise:
                    print(
                        "OpenAI models only support pairwise comparison; "
                        "pass --pairwise to run it. Skipping.",
                        flush=True,
                    )
                    continue

                run_pairwise_pass2(
                    run_args,
                    selected_df,
                    output_dir,
                    compute_row_fields=lambda pos_s, neg_s: (
                        compute_gpt_pair_row_fields2(
                            client,
                            args.model,
                            pos_s,
                            neg_s,
                        )
                    ),
                    backend_label="GPT",
                )

                continue

            if run_pairwise:
                run_pairwise_pass2(
                    run_args,
                    selected_df,
                    output_dir,
                    compute_row_fields=lambda pos_s, neg_s: (
                        compute_hf_pairwise_pair_row_fields2(
                            model,
                            tokenizer,
                            pos_s,
                            neg_s,
                        )
                    ),
                    backend_label="Hugging Face",
                )

            sentence_lang = sentence_column_language(
                pos,
                target_lang,
            )

            neg_sentence_lang = sentence_column_language(
                neg,
                target_lang,
            )

            if sentence_lang != neg_sentence_lang:
                raise ValueError(
                    f"pos/neg sentence-language mismatch: "
                    f"{sentence_lang} vs {neg_sentence_lang}"
                )

            if sentence_lang not in unigram_cache:
                print(
                    f"Loading/building FLORES unigram counts "
                    f"for sentence language={sentence_lang}...",
                    flush=True,
                )

                unigram_cache[sentence_lang] = (
                    slor.get_unigram_token_counts(
                        tokenizer,
                        sentence_lang,
                    )
                )

                stats = unigram_cache[sentence_lang]

                print(
                    f"FLORES done for {sentence_lang}. "
                    f"sentences={stats['n_sentences']} "
                    f"tokens={stats['total_tokens']}",
                    flush=True,
                )

            unigram_stats = unigram_cache[sentence_lang]

            run_no_context_pass2(
                run_args,
                selected_df,
                output_dir,
                model,
                tokenizer,
                unigram_stats,
                vocab_size,
            )

            if args.context:
                if sentence_lang in TASK_LANGUAGE_BY_CODE:

                    prompt_conditions = load_prompt_conditions(
                        args.prompt_data,
                        sentence_lang,
                    )

                    prompt_conditions = [
                        condition for condition in prompt_conditions
                        if condition["prompt_language"] != "raw"
                    ]

                    for condition in prompt_conditions:
                        run_condition_pass2(
                            run_args,
                            selected_df,
                            output_dir,
                            condition,
                            model,
                            tokenizer,
                            unigram_stats,
                            vocab_size,
                        )

                else:
                    print(
                        f"No context-prompt Task language mapping "
                        f"for sentence language={sentence_lang!r}; "
                        f"skipping context conditions.",
                        flush=True,
                    )


def resolve_run_plan2(args):
    langs = getattr(args, "lang", None)
    if isinstance(langs, str):
        langs = [langs]
    if langs:
        target_languages = []
        for code in langs:
            canonical = canonical_language_code(code)
            if canonical not in TARGET_LANGUAGES:
                raise ValueError(f"--lang {code!r} is not one of {TARGET_LANGUAGES} (source code 'se' is accepted for 'sv')")
            if canonical not in target_languages:
                target_languages.append(canonical)
    else:
        target_languages = list(TARGET_LANGUAGES)

    pos, neg = getattr(args, "pos", None), getattr(args, "neg", None)
    if (pos is None) != (neg is None):
        raise ValueError("--pos and --neg must be given together")
    comparisons = [(pos, neg)] if pos is not None else list(COMPARISONS)

    run_translation = getattr(args, "translation", False) or pos is None
    return target_languages, comparisons, run_translation


def report_japanese_control_availability2(construction_df):
    table = japanese_control_availability(construction_df)
    print("Japanese-control pairs (usable = both Japanese sentences present):", flush=True)
    print(table.to_string(index=False), flush=True)
    return table


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--construction_data", type=str, required=True)
    parser.add_argument("--prompt_data", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--lang",
        nargs="+",
        default=None,
        help="Original target language(s) whose rows are selected (default: all). "
             "Source codes are canonicalized (se -> sv). This never sets the scoring "
             "language, which follows from --pos/--neg (Japanese-control columns are scored as ja).",
    )
    parser.add_argument("--pos", type=str, default=None,
                        help="Excel column of the positive sentence (with --neg; default: every comparison).")
    parser.add_argument("--neg", type=str, default=None,
                        help="Excel column of the negative sentence (with --pos).")
    parser.add_argument(
        "--translation",
        action="store_true",
        help="Also run translation generation in a --pos/--neg-restricted run "
             "(it always runs when --pos/--neg are omitted).",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="Also run prompt-context sentence-probability conditions.",
    )
    parser.add_argument(
        "--final_exp",
        action="store_true",
        help="For each probability/pairwise comparison, select only rows with > or >> "
             "in its corresponding English, target-language, or Japanese preference annotation. "
             "Does not filter independent translation generation.",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Also run the pairwise A/B comparison pass (off by default).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_experiment2(parse_args())
    cleanup_distributed()
