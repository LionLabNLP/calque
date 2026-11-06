import difflib
import json

import torch
import torch.nn.functional as F

from . import slor as slor_module


def _find_sentence_start_token_idx(tokenizer, prefix_text, full_text):
    char_start = len(prefix_text)

    try:
        encoding = tokenizer(full_text, add_special_tokens=True, return_offsets_mapping=True)
        offsets = encoding["offset_mapping"]
    except Exception:
        offsets = None

    if offsets is not None:
        for i, (start, end) in enumerate(offsets):
            if start == 0 and end == 0:
                continue
            if start >= char_start:
                return i
        return len(offsets)

    prefix_ids = tokenizer(prefix_text, add_special_tokens=True)["input_ids"]
    return len(prefix_ids)


@torch.no_grad()
def compute_sentence_logprob(model, tokenizer, sentence, prompt_text=None):
    prefix_text = f"{prompt_text} " if prompt_text else ""
    full_text = prefix_text + sentence

    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).to(model.device)
    input_ids = inputs["input_ids"]

    start_token_idx = _find_sentence_start_token_idx(tokenizer, prefix_text, full_text)
    scored_start = max(start_token_idx, 1)

    outputs = model(**inputs)
    logits = outputs.logits.float()

    shift_logits = logits[:, :-1]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    start = scored_start - 1
    sentence_logprobs = token_logprobs[:, start:]
    sentence_token_ids = input_ids[0, scored_start:].tolist()

    if sentence_logprobs.numel() == 0:
        raise ValueError(f"No sentence tokens left to score for input: {full_text!r}")

    per_token_logprobs = sentence_logprobs[0].double().tolist()
    per_token_surprisals = [-lp for lp in per_token_logprobs]
    token_strs = tokenizer.convert_ids_to_tokens(sentence_token_ids)

    return {
        "sentence": sentence,
        "prompt": prompt_text,
        "full_text": full_text,
        "n_sentence_tokens": sentence_logprobs.shape[1],
        "logprob": sentence_logprobs.double().sum().item(),
        "avg_logprob": sentence_logprobs.double().mean().item(),
        "token_ids": sentence_token_ids,
        "token_strs": token_strs,
        "token_logprobs": per_token_logprobs,
        "token_surprisals": per_token_surprisals,
    }


def align_token_ids(positive_token_ids, negative_token_ids):
    matcher = difflib.SequenceMatcher(None, positive_token_ids, negative_token_ids, autojunk=False)
    return matcher.get_opcodes()


def _mean_or_nan(values):
    return sum(values) / len(values) if values else float("nan")


def compute_diff_subset_metrics(pos_result, neg_result):
    pos_ids = pos_result["token_ids"]
    neg_ids = neg_result["token_ids"]
    pos_strs = pos_result["token_strs"]
    neg_strs = neg_result["token_strs"]
    pos_logprobs = pos_result["token_logprobs"]
    neg_logprobs = neg_result["token_logprobs"]

    opcodes = align_token_ids(pos_ids, neg_ids)

    pos_diff_tokens, neg_diff_tokens = [], []
    pos_diff_logprobs, neg_diff_logprobs = [], []
    pos_after_diff_tokens, neg_after_diff_tokens = [], []
    pos_after_diff_logprobs, neg_after_diff_logprobs = [], []

    seen_diff = False
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "equal":
            seen_diff = True
            pos_diff_tokens.extend(pos_strs[i1:i2])
            neg_diff_tokens.extend(neg_strs[j1:j2])
            pos_diff_logprobs.extend(pos_logprobs[i1:i2])
            neg_diff_logprobs.extend(neg_logprobs[j1:j2])
        elif seen_diff:
            pos_after_diff_tokens.extend(pos_strs[i1:i2])
            neg_after_diff_tokens.extend(neg_strs[j1:j2])
            pos_after_diff_logprobs.extend(pos_logprobs[i1:i2])
            neg_after_diff_logprobs.extend(neg_logprobs[j1:j2])

    positive_only_diff_tokens_avg_token_logprob = _mean_or_nan(pos_diff_logprobs)
    negative_only_diff_tokens_avg_token_logprob = _mean_or_nan(neg_diff_logprobs)
    positive_after_diff_tokens_avg_token_logprob = _mean_or_nan(pos_after_diff_logprobs)
    negative_after_diff_tokens_avg_token_logprob = _mean_or_nan(neg_after_diff_logprobs)

    return {
        "positive_only_diff_tokens_avg_token_logprob": positive_only_diff_tokens_avg_token_logprob,
        "negative_only_diff_tokens_avg_token_logprob": negative_only_diff_tokens_avg_token_logprob,
        "only_diff_tokens_tokenwise_log_prob_diff": (
            positive_only_diff_tokens_avg_token_logprob - negative_only_diff_tokens_avg_token_logprob
        ),
        "positive_after_diff_tokens_avg_token_logprob": positive_after_diff_tokens_avg_token_logprob,
        "negative_after_diff_tokens_avg_token_logprob": negative_after_diff_tokens_avg_token_logprob,
        "after_diff_tokens_tokenwise_log_prob_diff": (
            positive_after_diff_tokens_avg_token_logprob - negative_after_diff_tokens_avg_token_logprob
        ),
        "pos_diff_tokens": pos_diff_tokens,
        "neg_diff_tokens": neg_diff_tokens,
        "pos_after_diff_tokens": pos_after_diff_tokens,
        "neg_after_diff_tokens": neg_after_diff_tokens,
    }


def to_json_field(value):
    return json.dumps(value, ensure_ascii=False)


def compute_sentence_pair_metrics(
    model,
    tokenizer,
    positive_sentence,
    negative_sentence,
    prompt_text,
    unigram_stats,
    vocab_size,
):
    pos_result = compute_sentence_logprob(model, tokenizer, sentence=positive_sentence, prompt_text=prompt_text)
    neg_result = compute_sentence_logprob(model, tokenizer, sentence=negative_sentence, prompt_text=prompt_text)

    slor_pos = slor_module.compute_slor(
        pos_result["logprob"],
        pos_result["token_ids"],
        unigram_stats["token_counts"],
        unigram_stats["total_tokens"],
        vocab_size,
    )
    slor_neg = slor_module.compute_slor(
        neg_result["logprob"],
        neg_result["token_ids"],
        unigram_stats["token_counts"],
        unigram_stats["total_tokens"],
        vocab_size,
    )

    diff_metrics = compute_diff_subset_metrics(pos_result, neg_result)

    return {
        **diff_metrics,
        "positive_sentence_logprob": pos_result["logprob"],
        "negative_sentence_logprob": neg_result["logprob"],
        "positive_avg_token_logprob": pos_result["avg_logprob"],
        "negative_avg_token_logprob": neg_result["avg_logprob"],
        "token_logprob_diff": pos_result["avg_logprob"] - neg_result["avg_logprob"],
        "SLOR_positive_avg_token_logprob": slor_pos,
        "SLOR_negative_avg_token_logprob": slor_neg,
        "SLOR_token_logprob_diff": slor_pos - slor_neg,
        "positive_token_count": pos_result["n_sentence_tokens"],
        "negative_token_count": neg_result["n_sentence_tokens"],
        "positive_tokens": pos_result["token_strs"],
        "negative_tokens": neg_result["token_strs"],
        "positive_token_surprisals": pos_result["token_surprisals"],
        "negative_token_surprisals": neg_result["token_surprisals"],
    }
