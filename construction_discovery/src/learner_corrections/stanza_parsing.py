from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STANZA_LANGUAGE_CODE: dict[str, str] = {
    "en": "en",
    "fr": "fr",
    "de": "de",
    "ja": "ja",
    "zh": "zh-hans",
    "pt": "pt",
    "sv": "sv",
}

REQUIRED_LANGUAGE_CODES: tuple[str, ...] = ("en", "fr", "de", "ja", "zh", "pt", "sv")

DEFAULT_PROCESSORS = "tokenize,pos,lemma,depparse"

CONLLU_FIELDS = ("id", "text", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc")


class ModelUnavailableError(RuntimeError):
    pass


@dataclass
class ParsedSentence:
    tokens: list[dict] = field(default_factory=list)
    status: str = "error:not_parsed"
    conllu: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class StanzaPipelineManager:

    def __init__(
        self,
        resources_dir: Path,
        use_gpu: bool = False,
        processors: str = DEFAULT_PROCESSORS,
    ) -> None:
        self.resources_dir = Path(resources_dir)
        self.resources_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu
        self.processors = processors
        self._pipelines: dict[str, Any] = {}

    def _ensure_downloaded(self, stanza_code: str) -> None:
        import stanza

        lang_dir = self.resources_dir / stanza_code
        if lang_dir.exists() and any(lang_dir.iterdir()):
            return
        try:
            stanza.download(
                stanza_code,
                model_dir=str(self.resources_dir),
                processors=self.processors,
                verbose=False,
            )
        except Exception as exc:
            raise ModelUnavailableError(
                f"Stanza model for language {stanza_code!r} is not cached under "
                f"{self.resources_dir} and could not be downloaded: {exc}"
            ) from exc

    def get(self, language_code: str):
        stanza_code = STANZA_LANGUAGE_CODE.get(language_code, language_code)
        if stanza_code in self._pipelines:
            return self._pipelines[stanza_code]

        import stanza

        self._ensure_downloaded(stanza_code)
        try:
            pipeline = stanza.Pipeline(
                lang=stanza_code,
                dir=str(self.resources_dir),
                processors=self.processors,
                use_gpu=self.use_gpu,
                verbose=False,
                download_method=stanza.DownloadMethod.NONE,
                tokenize_no_ssplit=True,
            )
        except Exception as exc:
            raise ModelUnavailableError(
                f"Failed to build Stanza pipeline for language {stanza_code!r}: {exc}"
            ) from exc
        self._pipelines[stanza_code] = pipeline
        return pipeline


def parse_feats(feats: str | None) -> dict[str, str]:
    if not feats or feats == "_":
        return {}
    result: dict[str, str] = {}
    for part in feats.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            result[key] = value
    return result


def _extract_tokens(doc) -> list[dict]:
    tokens: list[dict] = []
    offset = 0
    for sent in doc.sentences:
        n = len(sent.words)
        for w in sent.words:
            head = 0 if not w.head else offset + w.head
            tokens.append(
                {
                    "id": offset + w.id,
                    "text": w.text,
                    "lemma": w.lemma if w.lemma else w.text,
                    "upos": w.upos or "",
                    "xpos": w.xpos or "",
                    "feats": w.feats or "",
                    "head": head,
                    "deprel": w.deprel or "",
                    "deps": w.deps or "",
                    "misc": w.misc or "",
                }
            )
        offset += n
    return tokens


def _escape_comment_value(value: str) -> str:
    return value.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def tokens_to_conllu(tokens: list[dict], text: str, sent_id: str) -> str:
    lines = [f"# sent_id = {sent_id}", f"# text = {_escape_comment_value(text)}"]
    for t in tokens:
        lines.append(
            "\t".join(
                [
                    str(t["id"]),
                    t["text"],
                    t["lemma"] or "_",
                    t["upos"] or "_",
                    t["xpos"] or "_",
                    t["feats"] or "_",
                    str(t["head"]),
                    t["deprel"] or "_",
                    t["deps"] or "_",
                    t["misc"] or "_",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def parse_conllu_block(text: str) -> list[dict]:
    tokens: list[dict] = []
    if not text:
        return tokens
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != len(CONLLU_FIELDS):
            continue
        tok_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = fields
        tokens.append(
            {
                "id": int(tok_id),
                "text": form,
                "lemma": "" if lemma == "_" else lemma,
                "upos": "" if upos == "_" else upos,
                "xpos": "" if xpos == "_" else xpos,
                "feats": "" if feats == "_" else feats,
                "head": int(head),
                "deprel": "" if deprel == "_" else deprel,
                "deps": "" if deps == "_" else deps,
                "misc": "" if misc == "_" else misc,
            }
        )
    return tokens


def parse_sentence(
    pipeline_manager: StanzaPipelineManager, language_code: str, text: str, sent_id: str
) -> ParsedSentence:
    if not isinstance(text, str) or not text.strip():
        return ParsedSentence(status="error:empty_text")

    try:
        pipeline = pipeline_manager.get(language_code)
    except ModelUnavailableError as exc:
        return ParsedSentence(status=f"error:model_unavailable: {exc}")

    try:
        doc = pipeline(text)
    except Exception as exc:
        logger.warning("Stanza parse failed for %s (%s): %s", sent_id, language_code, exc)
        return ParsedSentence(status=f"error:parse_exception: {exc.__class__.__name__}: {exc}")

    tokens = _extract_tokens(doc)
    if not tokens:
        return ParsedSentence(status="error:no_tokens")

    conllu = tokens_to_conllu(tokens, text, sent_id)
    return ParsedSentence(tokens=tokens, status="ok", conllu=conllu)
