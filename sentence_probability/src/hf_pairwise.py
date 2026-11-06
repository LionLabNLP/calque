import torch

import pairwise_common as pw
from distributed_utils import model_input_device


def _top1_label_and_logprob(model, tokenizer, prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).to(model_input_device(model))

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[:, -1, :].float()
    next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)

    top1_token_id = int(torch.argmax(next_token_logprobs, dim=-1).item())
    top1_logprob = float(next_token_logprobs[0, top1_token_id].item())

    raw_output = tokenizer.decode([top1_token_id], skip_special_tokens=False)
    chosen_label = pw.normalize_ab_label(raw_output)

    return raw_output, chosen_label, top1_logprob


def call_pairwise_choice(model, tokenizer, sentence_a, sentence_b, order):
    prompt_text = pw.build_pairwise_prompt(sentence_a, sentence_b)
    raw_output, chosen_label, top1_logprob = _top1_label_and_logprob(model, tokenizer, prompt_text)
    chosen_logprob = top1_logprob if chosen_label is not None else None
    return pw.vote_result(raw_output, chosen_label, chosen_logprob, order)


def run_pairwise_comparison(model, tokenizer, positive_sentence, negative_sentence):
    ab_vote = call_pairwise_choice(model, tokenizer, positive_sentence, negative_sentence, "AB")
    ba_vote = call_pairwise_choice(model, tokenizer, negative_sentence, positive_sentence, "BA")
    return pw.combine_pairwise_votes(ab_vote, ba_vote)
