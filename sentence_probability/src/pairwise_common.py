import math


def build_pairwise_prompt(sentence_a, sentence_b):
    return (
        "Which sentence is more natural?\n\n"
        f"A: {sentence_a}\n"
        f"B: {sentence_b}\n\n"
        "Answer with only A or B."
    )


def normalize_ab_label(raw_output):
    if raw_output is None:
        return None
    normalized = str(raw_output).strip().upper()
    return normalized if normalized in ("A", "B") else None


def confidence_from_logprob(logprob):
    if logprob is None:
        return 0.0
    return math.exp(logprob)


def vote_result(raw_output, chosen_label, chosen_logprob, order):
    if chosen_label not in ("A", "B"):
        return {
            "raw_output": raw_output,
            "chosen_label": None,
            "chosen_logprob": None,
            "original_chosen_label": "other",
            "confidence": 0.0,
            "signed_confidence": 0.0,
        }

    confidence = confidence_from_logprob(chosen_logprob)

    if order == "AB":
        if chosen_label == "A":
            original_chosen_label = "positive"
            signed_confidence = confidence
        else:
            original_chosen_label = "negative"
            signed_confidence = -confidence
    else:
        if chosen_label == "A":
            original_chosen_label = "negative"
            signed_confidence = -confidence
        else:
            original_chosen_label = "positive"
            signed_confidence = confidence

    return {
        "raw_output": raw_output,
        "chosen_label": chosen_label,
        "chosen_logprob": chosen_logprob,
        "original_chosen_label": original_chosen_label,
        "confidence": confidence,
        "signed_confidence": signed_confidence,
    }


def _pairwise_winner(ab_original, ba_original, signed_choice_confidence):
    if ab_original == "other" and ba_original == "other":
        return "no_valid_vote"
    if ab_original == "other" or ba_original == "other":
        return "one_valid_vote"
    if ab_original == ba_original:
        return ab_original
    if signed_choice_confidence > 0:
        return "split_vote"
    if signed_choice_confidence < 0:
        return "split_vote"
    return "balanced"


def combine_pairwise_votes(ab_vote, ba_vote):
    signed_choice_confidence = (ab_vote["signed_confidence"] + ba_vote["signed_confidence"]) / 2.0

    winner = _pairwise_winner(
        ab_vote["original_chosen_label"],
        ba_vote["original_chosen_label"],
        signed_choice_confidence,
    )

    return {
        "pairwise_ab_raw_output": ab_vote["raw_output"],
        "pairwise_ab_original_chosen_label": ab_vote["original_chosen_label"],
        "pairwise_ab_chosen_logprob": ab_vote["chosen_logprob"],
        "pairwise_ab_confidence": ab_vote["confidence"],
        "pairwise_ab_signed_confidence": ab_vote["signed_confidence"],
        "pairwise_ba_raw_output": ba_vote["raw_output"],
        "pairwise_ba_original_chosen_label": ba_vote["original_chosen_label"],
        "pairwise_ba_chosen_logprob": ba_vote["chosen_logprob"],
        "pairwise_ba_confidence": ba_vote["confidence"],
        "pairwise_ba_signed_confidence": ba_vote["signed_confidence"],
        "pairwise_winner": winner,
        "pairwise_signed_choice_confidence": signed_choice_confidence,
    }
