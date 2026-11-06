from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

STATUS_OK = "ok"
STATUS_FAILED = "failed"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_id(config: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:16]


def key_hash(key: Sequence[Any]) -> str:
    return hashlib.sha256(canonical_json(list(key)).encode("utf-8")).hexdigest()


class CacheRecord:
    __slots__ = ("key", "status", "value", "error", "updated_at")

    def __init__(self, key, status, value, error, updated_at):
        self.key = key
        self.status = status
        self.value = value
        self.error = error
        self.updated_at = updated_at

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


class FeatureCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS configs (
                namespace TEXT NOT NULL,
                config_id TEXT NOT NULL,
                config_json TEXT NOT NULL,
                PRIMARY KEY (namespace, config_id)
            );
            CREATE TABLE IF NOT EXISTS entries (
                namespace TEXT NOT NULL,
                config_id TEXT NOT NULL,
                key_hash TEXT NOT NULL,
                key_json TEXT NOT NULL,
                status TEXT NOT NULL,
                value_json TEXT,
                error TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (namespace, config_id, key_hash)
            );
            CREATE TABLE IF NOT EXISTS progress (
                namespace TEXT NOT NULL,
                progress_key TEXT NOT NULL,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (namespace, progress_key)
            );
            """
        )
        self.conn.commit()
        self.stats: Dict[str, Dict[str, int]] = {}

    def close(self):
        self.conn.close()

    def _count(self, namespace: str, what: str, n: int = 1):
        self.stats.setdefault(namespace, {"hits": 0, "misses": 0, "writes": 0})
        self.stats[namespace][what] += n

    @staticmethod
    def _now() -> str:
        return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    def register_config(self, namespace: str, config: Dict[str, Any]) -> str:
        cid = config_id(config)
        self.conn.execute(
            "INSERT OR IGNORE INTO configs VALUES (?, ?, ?)",
            (namespace, cid, canonical_json(config)),
        )
        self.conn.commit()
        return cid

    def get_many(
        self, namespace: str, cid: str, keys: Iterable[Sequence[Any]], count_stats: bool = True
    ) -> Dict[Tuple, CacheRecord]:
        keys = [tuple(k) for k in keys]
        unique = list(dict.fromkeys(keys))
        by_hash = {key_hash(k): k for k in unique}
        out: Dict[Tuple, CacheRecord] = {}
        hashes = list(by_hash)
        for i in range(0, len(hashes), 500):
            chunk = hashes[i : i + 500]
            q = (
                "SELECT key_hash, key_json, status, value_json, error, updated_at FROM entries "
                f"WHERE namespace=? AND config_id=? AND key_hash IN ({','.join('?' * len(chunk))})"
            )
            for kh, kj, status, vj, err, ts in self.conn.execute(q, [namespace, cid, *chunk]):
                key = by_hash[kh]
                if kj != canonical_json(list(key)):
                    continue
                out[key] = CacheRecord(key, status, json.loads(vj) if vj else None, err, ts)
        if count_stats:
            hits = sum(1 for k in unique if k in out and out[k].ok)
            self._count(namespace, "hits", hits)
            self._count(namespace, "misses", len(unique) - hits)
        return out

    def put_many(
        self,
        namespace: str,
        cid: str,
        items: Iterable[Tuple[Sequence[Any], str, Optional[Any], Optional[str]]],
    ) -> int:
        now = self._now()
        rows = [
            (namespace, cid, key_hash(k), canonical_json(list(k)), status,
             None if value is None else canonical_json(value), error, now)
            for k, status, value, error in items
        ]
        with self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        self._count(namespace, "writes", len(rows))
        return len(rows)

    def load_progress(self, namespace: str, progress_key: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT state_json FROM progress WHERE namespace=? AND progress_key=?",
            (namespace, progress_key),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save_progress(self, namespace: str, progress_key: str, state: Dict[str, Any]):
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO progress VALUES (?, ?, ?, ?)",
                (namespace, progress_key, canonical_json(state), self._now()),
            )

    def clear_progress(self, namespace: str, progress_key: str):
        with self.conn:
            self.conn.execute(
                "DELETE FROM progress WHERE namespace=? AND progress_key=?", (namespace, progress_key)
            )
