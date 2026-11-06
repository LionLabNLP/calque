from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def is_gpt_oss(model_name: str) -> bool:
    return "gpt-oss" in model_name.lower()


def should_quantize(model_name: str) -> bool:
    model_name = model_name.lower()

    if is_gpt_oss(model_name):
        return False

    return any(
        x in model_name
        for x in [
            "35b",
            "70b",
            "72b",
            "aya-23-35b",
        ]
    )


def is_rinna_japanese_gpt(model_name: str) -> bool:
    return "rinna/japanese-gpt-neox" in model_name.lower()


def safe_model_key(model_name: str) -> str:
    return model_name.split("/")[-1].replace("-", "_").replace(".", "_")


def load_model_and_tokenizer(
    model_path: str,
    cache_dir: str | None = None,
):
    tokenizer_kwargs = {
        "cache_dir": cache_dir,
    }

    if is_rinna_japanese_gpt(model_path):
        tokenizer_kwargs["use_fast"] = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        **tokenizer_kwargs,
    )

    if is_gpt_oss(model_path):
        print(f"Using native MXFP4 loading: {model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            max_memory={
                0: "42GiB",
                1: "46GiB",
                "cpu": "100GiB",
            },
            dtype="auto",
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )

    elif should_quantize(model_path):
        print(f"Using BitsAndBytes 4-bit quantization: {model_path}")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )

    else:
        print(f"Using standard loading: {model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=(
                torch.bfloat16
                if torch.cuda.is_available()
                else torch.float32
            ),
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )

    model.eval()
    return tokenizer, model