import argparse
import json
import string
from pathlib import Path

import pandas as pd
import torch

import gpt_pairwise
import hf_pairwise
import slor
from data_loading import (
    TASK_LANGUAGE_BY_CODE,
    find_translation_source_sentence,
    load_construction_data,
    load_prompt_conditions,
    match_positive_negative,
    select_examples,
)
from distributed_utils import model_input_device
from model_utils import load_model_and_tokenizer
from probability import compute_diff_subset_metrics, compute_sentence_logprob

LANGUAGE_NAME_BY_CODE = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "se": "Swedish",
    "pt": "Portuguese",
    "ja": "Japanese",
    "zh": "Chinese",
}

GPT_UNAVAILABLE_LOCAL_LM_FIELDS = [
    "positive_sentence_logprob",
    "negative_sentence_logprob",
    "positive_avg_token_logprob",
    "negative_avg_token_logprob",
    "token_logprob_diff",
    "SLOR_positive_avg_token_logprob",
    "SLOR_negative_avg_token_logprob",
    "SLOR_token_logprob_diff",
    "positive_only_diff_tokens_avg_token_logprob",
    "negative_only_diff_tokens_avg_token_logprob",
    "only_diff_tokens_tokenwise_log_prob_diff",
    "positive_after_diff_tokens_avg_token_logprob",
    "negative_after_diff_tokens_avg_token_logprob",
    "after_diff_tokens_tokenwise_log_prob_diff",
    "pos_diff_tokens",
    "neg_diff_tokens",
    "pos_after_diff_tokens",
    "neg_after_diff_tokens",
    "positive_tokens",
    "negative_tokens",
    "positive_token_surprisals",
    "negative_token_surprisals",
]


def build_prompted_input(prompt_text, sentence):
    if prompt_text:
        return f"{prompt_text} {sentence}"
    return sentence


def to_json_field(value):
    return json.dumps(value, ensure_ascii=False)


def safe_gloss_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def build_pair_base_metadata(args, prompt_language, prompt_length, pair):
    return {
        "lang": args.lang,
        "pos_construction_lang": args.pos_construction_lang,
        "neg_construction_lang": args.neg_construction_lang,
        "construction_lang": f"{args.pos_construction_lang}_vs_{args.neg_construction_lang}",
        "pos_strategy": args.pos_strategy,
        "neg_strategy": args.neg_strategy,
        "model_name": args.model_name,
        "prompt_language": prompt_language,
        "prompt_length": prompt_length,
        "English baseline": pair["English baseline"],
        "Subject/Undergoer Person+Number": pair["Subject/Undergoer Person+Number"],
        "Time": pair["Time"],
    }


def compute_pair_row_fields(model, tokenizer, pair, prompt_text, unigram_stats, vocab_size):
    pos_sentence = pair["Sentence_pos"]
    neg_sentence = pair["Sentence_neg"]

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
        "positive_sentence": pos_sentence,
        "negative_sentence": neg_sentence,
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
        "positive_Interlinear_gloss": safe_gloss_value(pair.get("Interlinear gloss_pos")),
        "negative_Interlinear_gloss": safe_gloss_value(pair.get("Interlinear gloss_neg")),
        "positive_Object_language_line": safe_gloss_value(pair.get("Object language line_pos")),
        "negative_Object_language_line": safe_gloss_value(pair.get("Object language line_neg")),
    }


def _build_pairwise_row_fields(comparison, pos_sentence, neg_sentence, pair, prompt_text=None):
    fields = {
        "prompt": prompt_text or "",
        "positive_sentence": pos_sentence,
        "negative_sentence": neg_sentence,
        "positive_input": build_prompted_input(prompt_text, pos_sentence),
        "negative_input": build_prompted_input(prompt_text, neg_sentence),
        "positive_Interlinear_gloss": safe_gloss_value(pair.get("Interlinear gloss_pos")),
        "negative_Interlinear_gloss": safe_gloss_value(pair.get("Interlinear gloss_neg")),
        "positive_Object_language_line": safe_gloss_value(pair.get("Object language line_pos")),
        "negative_Object_language_line": safe_gloss_value(pair.get("Object language line_neg")),
    }
    fields.update(comparison)

    for key in GPT_UNAVAILABLE_LOCAL_LM_FIELDS:
        fields[key] = float("nan")

    return fields


def compute_gpt_pair_row_fields(client, gpt_model, pair, prompt_text):
    pos_sentence = pair["Sentence_pos"]
    neg_sentence = pair["Sentence_neg"]

    pair_metadata = {
        "English baseline": pair["English baseline"],
        "Subject/Undergoer Person+Number": pair["Subject/Undergoer Person+Number"],
        "Time": pair["Time"],
    }

    comparison = gpt_pairwise.run_pairwise_comparison(
        client, gpt_model, prompt_text, pos_sentence, neg_sentence, pair_metadata=pair_metadata
    )

    return _build_pairwise_row_fields(comparison, pos_sentence, neg_sentence, pair, prompt_text)


def compute_hf_pairwise_pair_row_fields(model, tokenizer, pair):
    pos_sentence = pair["Sentence_pos"]
    neg_sentence = pair["Sentence_neg"]

    comparison = hf_pairwise.run_pairwise_comparison(model, tokenizer, pos_sentence, neg_sentence)

    return _build_pairwise_row_fields(comparison, pos_sentence, neg_sentence, pair, prompt_text=None)


def _pairwise_output_path(args, output_dir):
    out_name = (
        f"{args.lang}_{args.pos_construction_lang}_vs_"
        f"{args.neg_construction_lang}_"
        f"{args.pos_strategy}_vs_{args.neg_strategy}_"
        f"pairwise.xlsx"
    )
    return output_dir / out_name


def run_pairwise_pass(args, matched_df, output_dir, compute_row_fields, backend_label):
    rows = []
    for _, pair in matched_df.iterrows():
        base = build_pair_base_metadata(args, prompt_language="None", prompt_length="Pairwise", pair=pair)
        fields = compute_row_fields(pair)
        rows.append({**base, **fields})

    result_df = pd.DataFrame(rows)

    out_path = _pairwise_output_path(args, output_dir)
    result_df.to_excel(out_path, index=False)
    print(f"Saved {len(result_df)} {backend_label} pairwise row(s) to {out_path}", flush=True)
    return result_df


def resolve_translation_prompt(args, construction_df, pair):

    if args.lang != "en":
        if args.lang not in LANGUAGE_NAME_BY_CODE:
            raise ValueError(
                f"No LANGUAGE_NAME_BY_CODE mapping for lang: {args.lang!r}"
            )

        target_lang_name = LANGUAGE_NAME_BY_CODE[args.lang]
        source_sentence = find_translation_source_sentence(
            construction_df,
            "en",
            pair,
        )

        return (
            f"Translate the following English sentence into {target_lang_name}.\n"
            f"English: {source_sentence}\n"
            f"{target_lang_name}:"
        )

    neg_lang = args.neg_construction_lang

    if not neg_lang or neg_lang == "en":
        raise ValueError(
            "Cannot build the English translation condition: "
            "a non-English --neg_construction_lang is required; "
            f"got {neg_lang!r}."
        )

    if neg_lang not in LANGUAGE_NAME_BY_CODE:
        raise ValueError(
            f"No LANGUAGE_NAME_BY_CODE mapping for "
            f"neg_construction_lang: {neg_lang!r}"
        )

    neg_lang_name = LANGUAGE_NAME_BY_CODE[neg_lang]
    source_sentence = find_translation_source_sentence(
        construction_df,
        neg_lang,
        pair,
    )

    return (
        f"Translate the following {neg_lang_name} sentence into English.\n"
        f"{neg_lang_name}: {source_sentence}\n"
        f"English:"
    )

def run_translation_pass(
    args, construction_df, matched_df, output_dir, use_openai, client, model, tokenizer, unigram_stats, vocab_size
):
    rows = []
    for _, pair in matched_df.iterrows():
        prompt_text = resolve_translation_prompt(args, construction_df, pair)
        base = build_pair_base_metadata(args, prompt_language="English", prompt_length="translation", pair=pair)

        if use_openai:
            fields = compute_gpt_pair_row_fields(client, args.model, pair, prompt_text)
        else:
            fields = compute_pair_row_fields(model, tokenizer, pair, prompt_text, unigram_stats, vocab_size)

        rows.append({**base, **fields})

    result_df = pd.DataFrame(rows)

    out_name = (
        f"{args.lang}_{args.pos_construction_lang}_vs_{args.neg_construction_lang}_"
        f"{args.pos_strategy}_vs_{args.neg_strategy}_translation.xlsx"
    )
    out_path = output_dir / out_name
    result_df.to_excel(out_path, index=False)
    print(f"Saved {len(result_df)} translation row(s) to {out_path}")


TRANSLATION_GENERATION_MAX_NEW_TOKENS = 128

_TRAILING_PUNCTUATION = string.punctuation + "。，！？、"


def generate_translation(model, tokenizer, prompt_text, max_new_tokens=TRANSLATION_GENERATION_MAX_NEW_TOKENS):
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).to(model_input_device(model))
    input_ids = inputs["input_ids"]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_token_id,
        )

    generated_ids = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def normalize_translation_text(text):
    if text is None:
        return ""
    normalized = str(text).strip()
    normalized = normalized.rstrip(_TRAILING_PUNCTUATION).strip()
    return normalized.casefold()


def classify_translation_generation(raw_generated_text, positive_sentence, negative_sentence):
    normalized_generated = normalize_translation_text(raw_generated_text)
    normalized_positive = normalize_translation_text(positive_sentence)
    normalized_negative = normalize_translation_text(negative_sentence)

    if normalized_generated == normalized_positive:
        classification = "positive"
    elif normalized_generated == normalized_negative:
        classification = "negative"
    else:
        classification = "other"

    return classification, normalized_generated


def _translation_generation_output_paths(args, output_dir):
    stem = (
        f"{args.lang}_{args.pos_construction_lang}_vs_{args.neg_construction_lang}_"
        f"{args.pos_strategy}_vs_{args.neg_strategy}_translation_generation"
    )
    return output_dir / f"{stem}.xlsx", output_dir / f"{stem}_summary.xlsx"


def run_translation_generation_pass(args, construction_df, matched_df, output_dir, model, tokenizer):
    rows = []
    counts = {"positive": 0, "negative": 0, "other": 0}

    for _, pair in matched_df.iterrows():
        prompt_text = resolve_translation_prompt(args, construction_df, pair)
        base = build_pair_base_metadata(
            args, prompt_language="English", prompt_length="translation_generation", pair=pair
        )

        pos_sentence = pair["Sentence_pos"]
        neg_sentence = pair["Sentence_neg"]

        raw_generated_text = generate_translation(model, tokenizer, prompt_text)
        classification, normalized_generated_text = classify_translation_generation(
            raw_generated_text, pos_sentence, neg_sentence
        )
        counts[classification] += 1

        fields = {
            "prompt": prompt_text,
            "positive_sentence": pos_sentence,
            "negative_sentence": neg_sentence,
            "generated_translation_raw": raw_generated_text,
            "generated_translation_normalized": normalized_generated_text,
            "classification": classification,
        }

        rows.append({**base, **fields})

    result_df = pd.DataFrame(rows)

    out_path, summary_path = _translation_generation_output_paths(args, output_dir)
    result_df.to_excel(out_path, index=False)
    print(f"Saved {len(result_df)} translation generation row(s) to {out_path}", flush=True)

    total = len(result_df)
    summary_rows = [
        {
            "classification": label,
            "count": counts[label],
            "proportion": (counts[label] / total) if total else float("nan"),
        }
        for label in ("positive", "negative", "other")
    ]
    summary_rows.append({
        "classification": "total",
        "count": total,
        "proportion": 1.0 if total else float("nan"),
    })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(summary_path, index=False)
    print(f"Saved translation generation summary to {summary_path}", flush=True)

    return result_df, summary_df


def run_experiment(args):
    print("[1] Loading construction data...", flush=True)
    construction_df = load_construction_data(args.construction_data)
    print(f"[1] Done. rows={len(construction_df)}", flush=True)

    print("[2] Selecting positive examples...", flush=True)
    pos_df = select_examples(
        construction_df,
        args.lang,
        args.pos_construction_lang,
        args.pos_strategy,
    )
    print(f"[2] Done. positive={len(pos_df)}", flush=True)

    if args.neg_construction_lang is None:
        raise ValueError(
            "--neg_construction_lang is required to select negative examples "
            "(Construction language filter) for this experiment run."
        )

    print("[3] Selecting negative examples...", flush=True)
    neg_df = select_examples(
        construction_df,
        args.lang,
        args.neg_construction_lang,
        args.neg_strategy,
    )
    print(f"[3] Done. negative={len(neg_df)}", flush=True)

    print("[4] Matching positive and negative examples...", flush=True)
    matched_df = match_positive_negative(pos_df, neg_df)
    print(f"[4] Done. matched={len(matched_df)}", flush=True)

    output_dir = Path("./outputs") / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[5] Checking model type...", flush=True)
    use_openai = gpt_pairwise.is_openai_model(args.model)
    print(f"[5] use_openai={use_openai}", flush=True)

    tokenizer = model = client = None
    vocab_size = unigram_stats = None

    if use_openai:
        print("[6] Creating OpenAI client...", flush=True)
        client = gpt_pairwise.get_openai_client()
        print("[6] Done.", flush=True)
    else:
        print(f"[6] Loading model and tokenizer: {args.model}", flush=True)
        tokenizer, model = load_model_and_tokenizer(args.model)
        print("[6] Model loaded.", flush=True)

        vocab_size = model.config.vocab_size
        print(f"[7] vocab_size={vocab_size}", flush=True)

        print(f"[8] Loading/building FLORES unigram counts for lang={args.lang}...", flush=True)
        unigram_stats = slor.get_unigram_token_counts(tokenizer, args.lang)
        print(
            f"[8] FLORES done. "
            f"sentences={unigram_stats['n_sentences']} "
            f"tokens={unigram_stats['total_tokens']}",
            flush=True,
        )

    print("[9] Starting experiment conditions...", flush=True)

    if use_openai:
        run_pairwise_pass(
            args,
            matched_df,
            output_dir,
            compute_row_fields=lambda pair: compute_gpt_pair_row_fields(client, args.model, pair, prompt_text=None),
            backend_label="GPT",
        )

    else:
        run_pairwise_pass(
            args,
            matched_df,
            output_dir,
            compute_row_fields=lambda pair: compute_hf_pairwise_pair_row_fields(model, tokenizer, pair),
            backend_label="Hugging Face",
        )

        if args.lang in TASK_LANGUAGE_BY_CODE:
            prompt_conditions = load_prompt_conditions(
                args.prompt_data,
                args.lang,
            )

            for condition in prompt_conditions:
                rows = []

                for _, pair in matched_df.iterrows():
                    base = build_pair_base_metadata(
                        args,
                        prompt_language=condition["prompt_language"],
                        prompt_length=condition["prompt_length"],
                        pair=pair,
                    )

                    fields = compute_pair_row_fields(
                        model,
                        tokenizer,
                        pair,
                        condition["prompt_text"],
                        unigram_stats,
                        vocab_size,
                    )

                    rows.append({**base, **fields})

                result_df = pd.DataFrame(rows)

                out_name = (
                    f"{args.lang}_{args.pos_construction_lang}_vs_"
                    f"{args.neg_construction_lang}_"
                    f"{args.pos_strategy}_vs_{args.neg_strategy}_"
                    f"{condition['prompt_language']}_"
                    f"{condition['prompt_length']}.xlsx"
                )

                out_path = output_dir / out_name
                result_df.to_excel(out_path, index=False)

                print(
                    f"Saved {len(result_df)} row(s) to {out_path}",
                    flush=True,
                )

        else:
            print(
                f"No context-prompt Task language mapping for "
                f"lang={args.lang!r}; skipping context conditions.",
                flush=True,
            )

        run_translation_pass(
            args,
            construction_df,
            matched_df,
            output_dir,
            False,
            None,
            model,
            tokenizer,
            unigram_stats,
            vocab_size,
        )

        if args.actual_translation_eval:
            run_translation_generation_pass(
                args,
                construction_df,
                matched_df,
                output_dir,
                model,
                tokenizer,
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--construction_data", type=str, required=True)
    parser.add_argument("--prompt_data", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--pos_construction_lang", type=str, required=True)
    parser.add_argument("--neg_construction_lang", type=str, required=False, default=None)
    parser.add_argument("--pos_strategy", type=str, required=True)
    parser.add_argument("--neg_strategy", type=str, required=True)
    parser.add_argument("--actual_translation_eval", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
