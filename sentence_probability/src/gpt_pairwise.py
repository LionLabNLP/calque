import os
import re

import pairwise_common as pw

_EXPLICIT_OPENAI_MODELS = {"gpt-5.1", "gpt5.1"}
_OPENAI_MODEL_PATTERN = re.compile(r"^gpt-?\d+(\.\d+)+", re.IGNORECASE)


def is_openai_model(model_identifier):
    if not model_identifier:
        return False
    normalized = model_identifier.strip().lower()
    if normalized in _EXPLICIT_OPENAI_MODELS:
        return True
    return bool(_OPENAI_MODEL_PATTERN.match(normalized))


def get_openai_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set; required for the "
            "OpenAI GPT pairwise-comparison path."
        )
    return OpenAI(api_key=api_key)


def build_pairwise_messages(condition_prompt, sentence_a, sentence_b):
    return [{"role": "user", "content": pw.build_pairwise_prompt(sentence_a, sentence_b)}]


def _find_label_logprob(choice, label):
    if label is None:
        return None
    content = choice.logprobs.content if choice.logprobs else None
    if not content:
        return None
    for token_info in content:
        if token_info.token.strip() == label:
            return token_info.logprob
    return None


def call_pairwise_choice(client, model, sentence_a, sentence_b, order):
    messages = build_pairwise_messages(None, sentence_a, sentence_b)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        reasoning_effort="none",
        max_completion_tokens=10,
        temperature=0,
        logprobs=True,
        top_logprobs=5,
    )

    choice = response.choices[0]
    raw_output = choice.message.content
    label = pw.normalize_ab_label(raw_output)
    chosen_logprob = _find_label_logprob(choice, label)

    if label is not None and chosen_logprob is None:
        label = None

    return pw.vote_result(raw_output, label, chosen_logprob, order)


def run_pairwise_comparison(client, model, condition_prompt, positive_sentence, negative_sentence, pair_metadata=None):
    ab_vote = call_pairwise_choice(client, model, positive_sentence, negative_sentence, "AB")
    ba_vote = call_pairwise_choice(client, model, negative_sentence, positive_sentence, "BA")
    return pw.combine_pairwise_votes(ab_vote, ba_vote)
