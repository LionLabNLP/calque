from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .cache import STATUS_FAILED, STATUS_OK, FeatureCache
from .schema import COL_EN_PROPER, COL_TL_PROPER, language_info

log = logging.getLogger(__name__)

TOKENS_NAMESPACE = "stanza_tokens"
ALIGN_NAMESPACE = "alignment"
METRICS_VERSION = "align-metrics-v1"

AWESOME_ALIGN_REFERENCE = "github.com/neulab/awesome-align@5f150d45bbe51e167daf0a84abebaeb07c3323d1"


@dataclass(frozen=True)
class StanzaSettings:
    processors: str = "tokenize"
    tokenize_no_ssplit: bool = True
    package: str = "default"
    token_unit: str = "surface_token"
    version: str = "stanza-tok-v1"


@dataclass(frozen=True)
class Checkpoint:

    name: str
    source: str
    repo_id: Optional[str] = None
    revision: Optional[str] = None
    weights_file: Optional[str] = None
    weights_sha256: Optional[str] = None
    origin: str = ""


CHECKPOINTS: Dict[str, Checkpoint] = {
    "awesome-without-co": Checkpoint(
        name="awesome-without-co",
        source="local",
        origin=("official awesome-align README, 'multilingually fine-tuned w/o --train_co, softmax': "
                "https://drive.google.com/file/d/1IcQx6t5qtv4bdcGjjVCwXnRkpr67eisJ (model_without_co.zip)"),
    ),
    "awesome-with-co": Checkpoint(
        name="awesome-with-co",
        source="hf",
        repo_id="aneuraz/awesome-align-with-co",
        revision="777756717e1fa9556e304d4d5db173ee386b9c16",
        weights_file="pytorch_model.bin",
        weights_sha256="9f9d19adbab07c9f5ec2d74fe5701a66c123919feb35d05fce29969b6a85adb9",
        origin=("HF mirror of the official 'multilingually fine-tuned w/ --train_co, softmax' model "
                "(README Drive id 1IluQED1jb0rjITJtyj4lNMPmaRFyMslg, model_with_co.zip)"),
    ),
    "mbert": Checkpoint(
        name="mbert",
        source="hf",
        repo_id="google-bert/bert-base-multilingual-cased",
        revision="3f076fdb1ab68d5b2880cb87a0886f315b8146f8",
        weights_file="model.safetensors",
        weights_sha256="876f584f15ebf14887dec17539c114bb99a032e96b9e72507a51c41e205337fc",
        origin="bert-base-multilingual-cased, the awesome-align CLI example without fine-tuning",
    ),
}
DEFAULT_CHECKPOINT = "awesome-without-co"
HF_CHECKPOINT_FILES = ["config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json"]


@dataclass(frozen=True)
class AlignmentSettings:
    checkpoint: str = DEFAULT_CHECKPOINT
    expected_weights_sha256: Optional[str] = None
    align_layer: int = 8
    extraction: str = "softmax"
    softmax_threshold: float = 1e-3
    symmetrization: str = "intersection(P_src->tgt > thr, P_tgt->src > thr)"
    special_token_mask: float = -10000.0
    subword_to_word: str = "union_over_subword_links"
    empty_word: str = "dropped (no [UNK] substitution), as in run_align.py"
    max_len: int = 512
    attention_implementation: str = "eager"
    reference: str = AWESOME_ALIGN_REFERENCE
    version: str = "awesome-align-port-v2"

    def checkpoint_spec(self) -> Checkpoint:
        if self.checkpoint not in CHECKPOINTS:
            raise ValueError(f"unknown alignment checkpoint {self.checkpoint!r}; choose from {sorted(CHECKPOINTS)}")
        return CHECKPOINTS[self.checkpoint]


def tokens_config(st: StanzaSettings) -> Dict[str, object]:
    return {"stanza": asdict(st)}


def alignment_config(st: StanzaSettings, al: AlignmentSettings) -> Dict[str, object]:
    ck = asdict(al.checkpoint_spec())
    ck.pop("origin")
    return {"stanza": asdict(st), "aligner": asdict(al), "checkpoint": ck, "metrics": METRICS_VERSION}


def runtime_versions() -> Dict[str, Optional[str]]:
    out = {}
    for pkg in ("stanza", "transformers", "torch"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def word_order_metrics(links: Iterable[Tuple[int, int]]) -> Dict[str, object]:
    links = sorted(set(links))
    n = len(links)
    if n < 2:
        return {"n_links": n, "n_link_pairs": 0 if n < 2 else None, "n_inversions": None, "wo_score": None}
    inv = 0
    for a in range(n):
        i, j = links[a]
        for b in range(a + 1, n):
            i2, j2 = links[b]
            if (i < i2 and j > j2) or (i > i2 and j < j2):
                inv += 1
    pairs = n * (n - 1) // 2
    return {"n_links": n, "n_link_pairs": pairs, "n_inversions": inv, "wo_score": inv / pairs}


def non_isomorphism_metrics(n_src: int, n_tgt: int, links: Iterable[Tuple[int, int]]) -> Dict[str, object]:
    links = set(links)
    src_partners: Dict[int, Set[int]] = {i: set() for i in range(n_src)}
    tgt_partners: Dict[int, Set[int]] = {j: set() for j in range(n_tgt)}
    for i, j in links:
        if not (0 <= i < n_src and 0 <= j < n_tgt):
            raise ValueError(f"link {(i, j)} out of range for {n_src}x{n_tgt} tokens")
        src_partners[i].add(j)
        tgt_partners[j].add(i)

    def side(own: Dict[int, Set[int]], other: Dict[int, Set[int]], n: int, prefix: str):
        unaligned = to_many = shared = non11 = 0
        for t, partners in own.items():
            u = len(partners) == 0
            m = len(partners) >= 2
            s = any(len(other[p]) >= 2 for p in partners)
            unaligned += u
            to_many += m
            shared += s
            non11 += u or m or s
        return {
            f"{prefix}_n_tokens": n,
            f"{prefix}_n_unaligned": unaligned,
            f"{prefix}_n_to_many": to_many,
            f"{prefix}_n_shared": shared,
            f"{prefix}_n_non11": non11,
            f"{prefix}_non11_prop": (non11 / n) if n > 0 else None,
        }

    out = side(src_partners, tgt_partners, n_src, "en")
    out.update(side(tgt_partners, src_partners, n_tgt, "tl"))
    return out


def pair_metrics(src_tokens: Sequence[str], tgt_tokens: Sequence[str], links) -> Dict[str, object]:
    out = word_order_metrics(links)
    out.update(non_isomorphism_metrics(len(src_tokens), len(tgt_tokens), links))
    return out


class StanzaTokenizer:
    def __init__(self, settings: StanzaSettings, use_gpu: bool, model_dir: Optional[str] = None):
        import stanza

        self.settings = settings
        self.use_gpu = use_gpu
        self.model_dir = model_dir
        self._pipelines = {}

    def _pipeline(self, stanza_lang: str):
        if stanza_lang not in self._pipelines:
            import stanza
            from stanza.pipeline.core import DownloadMethod

            kwargs = dict(
                lang=stanza_lang,
                processors=self.settings.processors,
                tokenize_no_ssplit=self.settings.tokenize_no_ssplit,
                use_gpu=self.use_gpu,
                download_method=DownloadMethod.REUSE_RESOURCES,
                logging_level="WARN",
            )
            if self.settings.package != "default":
                kwargs["package"] = self.settings.package
            if self.model_dir:
                kwargs["dir"] = self.model_dir
            self._pipelines[stanza_lang] = stanza.Pipeline(**kwargs)
        return self._pipelines[stanza_lang]

    def tokenize(self, stanza_lang: str, sentence: str) -> List[str]:
        doc = self._pipeline(stanza_lang)(sentence)
        return [t.text for s in doc.sentences for t in s.tokens]


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_weights_file(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (model_dir / name).is_file():
            return model_dir / name
    raise FileNotFoundError(f"no model.safetensors / pytorch_model.bin in {model_dir}")


def resolve_checkpoint(settings: AlignmentSettings, model_path: Optional[str]) -> Tuple[Path, str]:
    ck = settings.checkpoint_spec()
    if ck.source == "local" or model_path:
        if not model_path:
            raise ValueError(
                f"checkpoint {ck.name!r} must be supplied as a local directory with --align-model. "
                f"Source: {ck.origin}"
            )
        model_dir = Path(model_path)
        weights = find_weights_file(model_dir)
    else:
        from huggingface_hub import hf_hub_download

        for name in HF_CHECKPOINT_FILES:
            hf_hub_download(ck.repo_id, name, revision=ck.revision)
        weights = Path(hf_hub_download(ck.repo_id, ck.weights_file, revision=ck.revision))
        model_dir = weights.parent
    sha = file_sha256(weights)
    expected = settings.expected_weights_sha256 or ck.weights_sha256
    if expected and sha != expected:
        raise ValueError(f"weights SHA-256 mismatch for {weights}: {sha} != expected {expected}")
    return model_dir, sha


def load_official_tokenizer_kwargs(model_dir: Path) -> Dict[str, object]:
    kwargs: Dict[str, object] = {"do_lower_case": True, "tokenize_chinese_chars": True, "max_len": 512}
    cfg = model_dir / "tokenizer_config.json"
    if cfg.is_file():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        for k in ("do_lower_case", "tokenize_chinese_chars", "max_len", "model_max_length"):
            if k in data:
                kwargs["max_len" if k == "model_max_length" else k] = data[k]
    else:
        log.warning("%s has no tokenizer_config.json: using awesome-align's default do_lower_case=True", model_dir)
    return kwargs


def words_to_subwords(words: Sequence[str], tokenize) -> Tuple[List[str], List[int]]:
    subwords: List[str] = []
    sub2word: List[int] = []
    for wi, word in enumerate(words):
        pieces = tokenize(word)
        subwords.extend(pieces)
        sub2word.extend([wi] * len(pieces))
    return subwords, sub2word


def build_input_ids(subword_ids: Sequence[int], cls_id: int, sep_id: int, max_len: int) -> Tuple[List[int], bool]:
    keep = max(0, max_len - 2)
    truncated = len(subword_ids) > keep
    return [cls_id, *list(subword_ids)[:keep], sep_id], truncated


def extract_subword_alignment(hidden_src, hidden_tgt, special_src, special_tgt, threshold: float,
                              mask_value: float = -10000.0):
    import torch

    scores = hidden_src @ hidden_tgt.transpose(-1, -2)
    m_src = special_src.to(scores.dtype) * mask_value
    m_tgt = special_tgt.to(scores.dtype) * mask_value
    p_src = torch.softmax(scores + m_tgt.unsqueeze(0), dim=-1)
    p_tgt = torch.softmax(scores + m_src.unsqueeze(1), dim=-2)
    align = (p_src > threshold) & (p_tgt > threshold)
    return align[1:-1, 1:-1]


def subword_links_to_word_links(align_matrix, sub2word_src: Sequence[int], sub2word_tgt: Sequence[int]):
    return sorted({(sub2word_src[i], sub2word_tgt[j]) for i, j in align_matrix.nonzero().tolist()})


class AwesomeAligner:
    def __init__(self, settings: AlignmentSettings, device: str, model_dir: Path, weights_sha256: str):
        import torch
        import transformers

        self.settings = settings
        self.torch = torch
        self.device = device
        self.weights_sha256 = weights_sha256
        tk = load_official_tokenizer_kwargs(model_dir)
        self.max_len = int(tk["max_len"])
        self.tokenizer = transformers.BertTokenizer(
            vocab_file=str(model_dir / "vocab.txt"),
            do_lower_case=tk["do_lower_case"],
            tokenize_chinese_chars=tk["tokenize_chinese_chars"],
            strip_accents=None,
        )
        self.tokenizer_kwargs = tk
        model, info = transformers.BertModel.from_pretrained(
            str(model_dir), add_pooling_layer=False, attn_implementation=settings.attention_implementation,
            torch_dtype=torch.float32, output_loading_info=True,
        )
        missing = [k for k in info.get("missing_keys", []) if not k.startswith("pooler")]
        if missing:
            raise RuntimeError(f"checkpoint {model_dir} is missing encoder weights: {missing[:5]} ...")
        if settings.align_layer > model.config.num_hidden_layers:
            raise ValueError("align_layer exceeds the number of layers")
        self.model = model.to(device).eval()
        self.special_ids = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}

    def _encode(self, words: Sequence[str]):
        tok = self.tokenizer
        subwords, sub2word = words_to_subwords(words, tok.tokenize)
        ids, truncated = build_input_ids(tok.convert_tokens_to_ids(subwords), tok.cls_token_id,
                                         tok.sep_token_id, self.max_len)
        return ids, sub2word, truncated

    def _hidden(self, ids: List[int]):
        torch = self.torch
        t = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=t, attention_mask=torch.ones_like(t), token_type_ids=torch.zeros_like(t),
                         output_hidden_states=True)
        return out.hidden_states[self.settings.align_layer][0], t[0]

    def align(self, src_words: Sequence[str], tgt_words: Sequence[str]):
        ids_s, map_s, trunc_s = self._encode(src_words)
        ids_t, map_t, trunc_t = self._encode(tgt_words)
        if len(ids_s) == 2 or len(ids_t) == 2:
            return [], (trunc_s or trunc_t)
        torch = self.torch
        with torch.no_grad():
            h_s, t_s = self._hidden(ids_s)
            h_t, t_t = self._hidden(ids_t)
            special = torch.tensor(sorted(self.special_ids), device=self.device)
            matrix = extract_subword_alignment(h_s, h_t, torch.isin(t_s, special), torch.isin(t_t, special),
                                               self.settings.softmax_threshold, self.settings.special_token_mask)
        return subword_links_to_word_links(matrix.cpu(), map_s, map_t), (trunc_s or trunc_t)


ALIGN_COLUMNS = [
    "align_status", "align_error", "align_en_tokens", "align_tl_tokens", "align_links",
    "align_n_links", "wo_n_link_pairs", "wo_n_inversions", "wo_score",
    "en_n_tokens", "en_n_unaligned", "en_n_to_many", "en_n_shared", "en_n_non11", "en_non11_prop",
    "tl_n_tokens", "tl_n_unaligned", "tl_n_to_many", "tl_n_shared", "tl_n_non11", "tl_non11_prop",
    "non11_prop_self", "align_truncated", "align_runtime", "align_config_id",
]
_INT_COLS = {"align_n_links", "wo_n_link_pairs", "wo_n_inversions", "en_n_tokens", "en_n_unaligned",
             "en_n_to_many", "en_n_shared", "en_n_non11", "tl_n_tokens", "tl_n_unaligned",
             "tl_n_to_many", "tl_n_shared", "tl_n_non11"}
_FLOAT_COLS = {"wo_score", "en_non11_prop", "tl_non11_prop", "non11_prop_self"}


def _pair_key(row) -> Optional[Tuple[str, str, str, str]]:
    en, tl = row[COL_EN_PROPER], row[COL_TL_PROPER]
    if not isinstance(en, str) or not isinstance(tl, str):
        return None
    return ("en", en, row["target_language"], tl)


def _compute_missing(
    keys: List[Tuple[str, str, str, str]],
    cache: FeatureCache,
    tok_cid: str,
    align_cid: str,
    stanza_settings: StanzaSettings,
    align_settings: AlignmentSettings,
    device: str,
    stanza_dir: Optional[str],
    checkpoint_every: int,
    aligner: "AwesomeAligner",
) -> Dict[str, int]:
    runtime = runtime_versions()
    use_gpu = device.startswith("cuda")

    sent_keys = sorted({(k[0], k[1]) for k in keys} | {(k[2], k[3]) for k in keys})
    tok_cached = cache.get_many(TOKENS_NAMESPACE, tok_cid, sent_keys)
    need_tok = [k for k in sent_keys
                if not (k in tok_cached and tok_cached[k].ok and tok_cached[k].value.get("runtime_stanza") == runtime["stanza"])]
    tokens: Dict[Tuple[str, str], List[str]] = {
        k: r.value["tokens"] for k, r in tok_cached.items() if r.ok and k not in need_tok
    }
    if need_tok:
        tokenizer = StanzaTokenizer(stanza_settings, use_gpu=use_gpu, model_dir=stanza_dir)
        batch = []
        for k in need_tok:
            lang, sent = k
            try:
                toks = tokenizer.tokenize(language_info(lang).stanza, sent)
                tokens[k] = toks
                batch.append((k, STATUS_OK, {"tokens": toks, "runtime_stanza": runtime["stanza"]}, None))
            except Exception as e:
                log.exception("stanza failed for %r", k)
                batch.append((k, STATUS_FAILED, None, f"{type(e).__name__}: {e}"))
            if len(batch) >= checkpoint_every:
                cache.put_many(TOKENS_NAMESPACE, tok_cid, batch)
                batch = []
        cache.put_many(TOKENS_NAMESPACE, tok_cid, batch)

    ck = align_settings.checkpoint_spec()
    runtime_full = dict(runtime, checkpoint=ck.name, repo_id=ck.repo_id, revision=ck.revision,
                        weights_sha256=aligner.weights_sha256, tokenizer=aligner.tokenizer_kwargs)
    batch, n_ok, n_fail = [], 0, 0
    for k in keys:
        src, tgt = tokens.get((k[0], k[1])), tokens.get((k[2], k[3]))
        if src is None or tgt is None:
            batch.append((k, STATUS_FAILED, None, "tokenization failed"))
            n_fail += 1
        else:
            try:
                links, truncated = aligner.align(src, tgt)
                value = {"en_tokens": src, "tl_tokens": tgt, "links": [list(l) for l in links],
                         "truncated": truncated, "runtime": runtime_full}
                value.update(pair_metrics(src, tgt, links))
                batch.append((k, STATUS_OK, value, None))
                n_ok += 1
            except Exception as e:
                log.exception("alignment failed for %r", k)
                batch.append((k, STATUS_FAILED, None, f"{type(e).__name__}: {e}"))
                n_fail += 1
        if len(batch) >= checkpoint_every:
            cache.put_many(ALIGN_NAMESPACE, align_cid, batch)
            batch = []
    cache.put_many(ALIGN_NAMESPACE, align_cid, batch)
    return {"computed_ok": n_ok, "computed_failed": n_fail, "tokenized": len(need_tok)}


def annotate(
    table: pd.DataFrame,
    cache: FeatureCache,
    run: bool,
    stanza_settings: StanzaSettings = StanzaSettings(),
    align_settings: AlignmentSettings = AlignmentSettings(),
    device: str = "auto",
    stanza_dir: Optional[str] = None,
    checkpoint_every: int = 200,
    model_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    tok_cid = cache.register_config(TOKENS_NAMESPACE, tokens_config(stanza_settings))
    align_cid = cache.register_config(ALIGN_NAMESPACE, alignment_config(stanza_settings, align_settings))
    proper = table["sentence_role"].isin(["en_proper", "tl_proper"])

    row_keys: Dict[int, Optional[Tuple]] = {}
    unsupported: Set[int] = set()
    for idx, row in table[proper].iterrows():
        k = _pair_key(row)
        row_keys[idx] = k
        info = language_info(row["target_language"])
        if k is not None and (info is None or info.stanza is None):
            unsupported.add(idx)
    keys = sorted({k for i, k in row_keys.items() if k is not None and i not in unsupported})

    report: Dict[str, object] = {"executed": run, "config_id": align_cid, "n_pairs": len(keys)}
    cached = cache.get_many(ALIGN_NAMESPACE, align_cid, keys)
    if run:
        current = runtime_versions()
        model_dir, weights_sha = resolve_checkpoint(align_settings, model_path)
        report.update(checkpoint=align_settings.checkpoint, weights_sha256=weights_sha)

        def stale(rec):
            rt = rec.value.get("runtime", {})
            return (any(rt.get(p) != current[p] for p in ("stanza", "transformers", "torch"))
                    or rt.get("weights_sha256") != weights_sha)

        missing = [k for k in keys if not (k in cached and cached[k].ok and not stale(cached[k]))]
        report["n_to_compute"] = len(missing)
        if missing:
            if device == "auto":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            report["device"] = device
            aligner = AwesomeAligner(align_settings, device, model_dir, weights_sha)
            report.update(_compute_missing(missing, cache, tok_cid, align_cid, stanza_settings,
                                           align_settings, device, stanza_dir, checkpoint_every, aligner))
            cached = cache.get_many(ALIGN_NAMESPACE, align_cid, keys, count_stats=False)

    cols: Dict[str, List[object]] = {c: [] for c in ALIGN_COLUMNS}
    for idx, row in table.iterrows():
        vals: Dict[str, object] = {c: None for c in ALIGN_COLUMNS}
        vals["align_config_id"] = align_cid
        if idx not in row_keys:
            vals["align_status"] = "not_applicable"
        elif row_keys[idx] is None:
            vals["align_status"] = "missing_pair_sentence"
        elif idx in unsupported:
            vals["align_status"] = "unsupported_language"
        else:
            rec = cached.get(row_keys[idx])
            if rec is None:
                vals["align_status"] = "not_computed"
            elif not rec.ok:
                vals["align_status"] = "failed"
                vals["align_error"] = rec.error
            else:
                v = rec.value
                vals["align_status"] = "ok"
                vals["align_en_tokens"] = " ".join(v["en_tokens"])
                vals["align_tl_tokens"] = " ".join(v["tl_tokens"])
                vals["align_links"] = " ".join(f"{i}-{j}" for i, j in v["links"])
                vals["align_n_links"] = v["n_links"]
                vals["wo_n_link_pairs"] = v["n_link_pairs"]
                vals["wo_n_inversions"] = v["n_inversions"]
                vals["wo_score"] = v["wo_score"]
                for c in ALIGN_COLUMNS:
                    if c.startswith(("en_", "tl_")):
                        vals[c] = v.get(c)
                vals["non11_prop_self"] = v["en_non11_prop"] if row["side"] == "en" else v["tl_non11_prop"]
                vals["align_truncated"] = v.get("truncated")
                vals["align_runtime"] = str(v.get("runtime"))
                if v["n_links"] < 2:
                    vals["align_status"] = "ok_insufficient_links"
        for c in ALIGN_COLUMNS:
            cols[c].append(vals[c])

    out = table.copy()
    for c in ALIGN_COLUMNS:
        if c in _INT_COLS:
            out[c] = pd.array(cols[c], dtype="Int64")
        elif c in _FLOAT_COLS:
            out[c] = pd.array(cols[c], dtype="Float64")
        else:
            out[c] = cols[c]
    return out, report
