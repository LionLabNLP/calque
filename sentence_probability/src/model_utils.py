from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from distributed_utils import get_world_size, init_distributed_for_tp, is_distributed, is_main_process


def is_llama4_maverick(model_name: str) -> bool:
    return "llama-4-maverick-17b-128e" in model_name.lower()

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

    if is_llama4_maverick(model_path) and is_distributed():
        if is_main_process():
            print(f"Using BF16 Tensor Parallel loading (world_size={get_world_size()}): {model_path}")

        init_distributed_for_tp()

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            tp_plan="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )

    else:
        print(f"Using standard BF16 loading: {model_path}")

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

