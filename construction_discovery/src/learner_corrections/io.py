from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

logger = logging.getLogger(__name__)


class OutputExistsError(RuntimeError):
    pass


def guard_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise OutputExistsError(
            f"Output {path} already exists. Pass --overwrite to replace it, "
            "or remove it manually."
        )


def sha256_checksum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, write_fn) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            write_fn(fh)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    _atomic_write(path, lambda fh: fh.write(text.encode(encoding)))


def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=indent, default=str)
    atomic_write_text(path, text)


def atomic_write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    def _write(fh) -> None:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, default=str)
            fh.write((line + "\n").encode("utf-8"))

    _atomic_write(path, _write)


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line %d in %s: %s", line_no, path, exc)


def append_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def atomic_write_dataframe(
    df: pd.DataFrame,
    out_dir: Path,
    stem: str,
    *,
    overwrite: bool = False,
    formats: tuple[str, ...] = ("parquet", "csv"),
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    written: dict[str, Path] = {}
    if "parquet" in formats:
        parquet_path = out_dir / f"{stem}.parquet"
        guard_overwrite(parquet_path, overwrite)

        def _write_parquet(fh, _df=df) -> None:
            _df.to_parquet(fh, index=False)

        _atomic_write(parquet_path, _write_parquet)
        written["parquet"] = parquet_path
    if "csv" in formats:
        csv_path = out_dir / f"{stem}.csv"
        guard_overwrite(csv_path, overwrite)

        def _write_csv(fh, _df=df) -> None:
            _df.to_csv(fh, index=False, encoding="utf-8")

        _atomic_write(csv_path, _write_csv)
        written["csv"] = csv_path
    return written


def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
