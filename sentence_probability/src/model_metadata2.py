from dataclasses import asdict, dataclass

SIZE_TYPES = ("dense", "effective", "active")


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    display_name: str
    family: str
    instruction_tuned: bool
    english_centric: bool
    model_size_b: float
    total_parameters_b: float
    size_type: str
    size_source: str


def _m(model_id, display_name, family, it, ec, size, total, size_type, source):
    return ModelMetadata(model_id, display_name, family, it, ec, size, total, size_type, source)


_CARD = "model card"
_HUB = "HF safetensors total"
_CFG = "derived from config.json"

_ENTRIES = [
    _m("CohereLabs/aya-expanse-8b", "Aya Expanse 8B", "Aya Expanse", True, True, 8.028, 8.028, "dense", _HUB),
    _m("CohereLabs/aya-expanse-32b", "Aya Expanse 32B", "Aya Expanse", True, True, 32.296, 32.296, "dense", _HUB),
    _m("CohereLabs/tiny-aya-base", "Tiny Aya Base", "Tiny Aya", False, True, 3.349, 3.349, "dense", _HUB),
    _m("CohereLabs/tiny-aya-global", "Tiny Aya Global", "Tiny Aya", True, True, 3.349, 3.349, "dense", _HUB),
    _m("CohereLabs/tiny-aya-earth", "Tiny Aya Earth", "Tiny Aya", True, True, 3.349, 3.349, "dense", _HUB),
    _m("CohereLabs/tiny-aya-fire", "Tiny Aya Fire", "Tiny Aya", True, True, 3.349, 3.349, "dense", _HUB),
    _m("CohereLabs/tiny-aya-water", "Tiny Aya Water", "Tiny Aya", True, True, 3.349, 3.349, "dense", _HUB),
    _m("utter-project/EuroLLM-1.7B", "EuroLLM 1.7B", "EuroLLM", False, True, 1.657, 1.657, "dense", _CARD),
    _m("utter-project/EuroLLM-1.7B-Instruct", "EuroLLM 1.7B Instruct", "EuroLLM", True, True, 1.657, 1.657, "dense", _CARD),
    _m("utter-project/EuroLLM-9B", "EuroLLM 9B", "EuroLLM", False, True, 9.154, 9.154, "dense", _CARD),
    _m("utter-project/EuroLLM-9B-Instruct", "EuroLLM 9B Instruct", "EuroLLM", True, True, 9.154, 9.154, "dense", _CARD),
    _m("utter-project/EuroLLM-22B-2512", "EuroLLM 22B", "EuroLLM", False, True, 22.639, 22.639, "dense", _CARD),
    _m("utter-project/EuroLLM-22B-Instruct-2512", "EuroLLM 22B Instruct", "EuroLLM", True, True, 22.639, 22.639, "dense", _CARD),
    _m("google/gemma-4-E2B", "Gemma 4 E2B", "Gemma 4", False, True, 2.3, 5.1, "effective", _CARD),
    _m("google/gemma-4-E2B-it", "Gemma 4 E2B IT", "Gemma 4", True, True, 2.3, 5.1, "effective", _CARD),
    _m("google/gemma-4-E4B", "Gemma 4 E4B", "Gemma 4", False, True, 4.5, 8.0, "effective", _CARD),
    _m("google/gemma-4-E4B-it", "Gemma 4 E4B IT", "Gemma 4", True, True, 4.5, 8.0, "effective", _CARD),
    _m("google/gemma-4-12B", "Gemma 4 12B", "Gemma 4", False, True, 11.95, 11.95, "dense", _CARD),
    _m("google/gemma-4-12B-it", "Gemma 4 12B IT", "Gemma 4", True, True, 11.95, 11.95, "dense", _CARD),
    _m("google/gemma-4-26B-A4B", "Gemma 4 26B-A4B", "Gemma 4", False, True, 3.8, 25.2, "active", _CARD),
    _m("google/gemma-4-26B-A4B-it", "Gemma 4 26B-A4B IT", "Gemma 4", True, True, 3.8, 25.2, "active", _CARD),
    _m("google/gemma-4-31B", "Gemma 4 31B", "Gemma 4", False, True, 30.7, 30.7, "dense", _CARD),
    _m("google/gemma-4-31B-it", "Gemma 4 31B IT", "Gemma 4", True, True, 30.7, 30.7, "dense", _CARD),
    _m("rinna/japanese-gpt-neox-3.6b-instruction-sft-v2", "rinna Japanese GPT-NeoX 3.6B SFT v2",
       "rinna GPT-NeoX", True, False, 3.758, 3.758, "dense", _HUB),
    _m("uer/gpt2-xlarge-chinese-cluecorpussmall", "UER GPT-2 XL Chinese",
       "UER GPT-2", False, False, 1.511, 1.511, "dense", _CFG),
    _m("swiss-ai/Apertus-8B-2509", "Apertus 8B", "Apertus", False, True, 8.053, 8.053, "dense", _HUB),
    _m("swiss-ai/Apertus-70B-2509", "Apertus 70B", "Apertus", False, True, 70.600, 70.600, "dense", _HUB),
    _m("swiss-ai/Apertus-8B-Instruct-2509", "Apertus 8B Instruct", "Apertus", True, True, 8.053, 8.053, "dense", _HUB),
    _m("swiss-ai/Apertus-70B-Instruct-2509", "Apertus 70B Instruct", "Apertus", True, True, 70.600, 70.600, "dense", _HUB),
    _m("openai/gpt-oss-120b", "GPT-OSS 120B", "GPT-OSS", True, True, 5.1, 116.829, "active", _CARD),
    _m("AIInstituteLNCC/manaca-1b-instruct", "Manacá 1B Instruct", "Manacá", True, False, 1.722, 1.722, "dense", _HUB),
    _m("Aleph-Alpha/Kolibri-1-BF16", "Kolibri 1", "Kolibri", True, False, 3.458, 78.103, "active", _CARD),
    _m("llm-jp/llm-jp-3-172b", "LLM-jp-3 172B", "LLM-jp-3", False, False, 172.401, 172.401, "dense", _HUB),
]

MODEL_REGISTRY = {m.model_id: m for m in _ENTRIES}

_BY_OUTPUT_NAME = {m.model_id.replace("/", "_"): m for m in _ENTRIES}

METADATA_FIELDS = [
    "model_id",
    "display_name",
    "family",
    "instruction_tuned",
    "english_centric",
    "model_size_b",
    "total_parameters_b",
    "size_type",
]


def lookup_model(model_name):
    key = str(model_name)
    return MODEL_REGISTRY.get(key) or _BY_OUTPUT_NAME.get(key)


def model_metadata_record(model_name):
    meta = lookup_model(model_name)
    if meta is None:
        record = {f: None for f in METADATA_FIELDS}
        record["display_name"] = str(model_name)
        return record
    return {f: asdict(meta)[f] for f in METADATA_FIELDS}


def unresolved_models(model_names):
    return sorted({str(n) for n in model_names if lookup_model(n) is None})


def registry_table():
    return [asdict(m) for m in _ENTRIES]
