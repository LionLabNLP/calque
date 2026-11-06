from __future__ import annotations

import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from .io import atomic_write_json, sha256_checksum

logger = logging.getLogger(__name__)

STABLE_ID_SCHEME_VERSION = "source_row_sent_corr_v1"

TRACKED_PACKAGES = (
    "pandas",
    "pyarrow",
    "pyyaml",
    "openai",
    "stanza",
    "fastapi",
    "uvicorn",
    "jinja2",
    "tenacity",
)


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for pkg in packages:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def input_checksums(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    checksums: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        path = Path(path)
        if not path.exists():
            checksums[label] = {"path": str(path), "sha256": None, "exists": False}
            continue
        checksums[label] = {
            "path": str(path),
            "sha256": sha256_checksum(path),
            "exists": True,
            "size_bytes": path.stat().st_size,
        }
    return checksums


class RunSummary:

    def __init__(self, command: str, argv: list[str], config: dict, seed: int | None) -> None:
        self.data: dict[str, Any] = {
            "command": command,
            "argv": argv,
            "config": config,
            "random_seed": seed,
            "package_versions": package_versions(),
        }
        logger.info("Command: %s", command)
        logger.info("Args: %s", argv)
        logger.info("Random seed: %s", seed)
        for pkg, version in self.data["package_versions"].items():
            logger.info("Package version: %s=%s", pkg, version)

    def record_inputs(self, paths: dict[str, Path]) -> None:
        checksums = input_checksums(paths)
        self.data.setdefault("inputs", {}).update(checksums)
        for label, info in checksums.items():
            logger.info("Input checksum: %s (%s) sha256=%s", label, info["path"], info["sha256"])

    def record_counts(self, counts: dict[str, int]) -> None:
        self.data.setdefault("output_row_counts", {}).update(counts)
        for label, count in counts.items():
            logger.info("Output row count: %s=%d", label, count)

    def record(self, key: str, value: Any) -> None:
        self.data[key] = value

    def write(self, path: Path) -> None:
        atomic_write_json(Path(path), self.data)
        logger.info("Wrote run summary to %s", path)


def build_argv_repr() -> list[str]:
    return list(sys.argv)




class RunStateIncompatibleError(RuntimeError):
    pass


@dataclass
class ReconstructionRunState:
    input_path: str
    input_sha256: str | None
    seed: int
    model: str
    sampling_strategy_version: str
    stable_id_scheme_version: str
    allowed_pairs: list[list[str]] = field(default_factory=list)
    japanese_languages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReconstructionRunState":
        return cls(
            input_path=data["input_path"],
            input_sha256=data.get("input_sha256"),
            seed=data["seed"],
            model=data["model"],
            sampling_strategy_version=data["sampling_strategy_version"],
            stable_id_scheme_version=data["stable_id_scheme_version"],
            allowed_pairs=[list(p) for p in data.get("allowed_pairs", [])],
            japanese_languages=list(data.get("japanese_languages", [])),
        )


def load_run_state(path: Path) -> ReconstructionRunState | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return ReconstructionRunState.from_dict(json.load(fh))


def write_run_state(path: Path, state: ReconstructionRunState) -> None:
    atomic_write_json(Path(path), state.to_dict())


def validate_resume_compatibility(
    previous: ReconstructionRunState,
    current: ReconstructionRunState,
    *,
    allow_model_change: bool,
) -> None:
    mismatches: list[str] = []
    if previous.input_sha256 != current.input_sha256:
        mismatches.append(
            f"input file checksum changed ({previous.input_sha256} -> {current.input_sha256})"
        )
    if previous.seed != current.seed:
        mismatches.append(f"random seed changed ({previous.seed} -> {current.seed})")
    if previous.sampling_strategy_version != current.sampling_strategy_version:
        mismatches.append(
            "sampling strategy version changed "
            f"({previous.sampling_strategy_version!r} -> {current.sampling_strategy_version!r})"
        )
    if previous.stable_id_scheme_version != current.stable_id_scheme_version:
        mismatches.append(
            "stable ID scheme version changed "
            f"({previous.stable_id_scheme_version!r} -> {current.stable_id_scheme_version!r})"
        )
    if sorted(map(tuple, previous.allowed_pairs)) != sorted(map(tuple, current.allowed_pairs)):
        mismatches.append("allowed language pairs changed")
    if sorted(previous.japanese_languages) != sorted(current.japanese_languages):
        mismatches.append("Japanese-downsampling language set changed")

    if mismatches:
        raise RunStateIncompatibleError(
            "Cannot --resume: this invocation is incompatible with the existing checkpoint's "
            "run state (" + "; ".join(mismatches) + "). Start a fresh run with --overwrite "
            "instead, or fix the invocation to match the original run."
        )

    if previous.model != current.model:
        if not allow_model_change:
            raise RunStateIncompatibleError(
                f"Model changed ({previous.model!r} -> {current.model!r}) since the run that "
                "produced the existing checkpoint. Pass --allow-model-change to explicitly opt "
                "in to mixing models within one accumulated output."
            )
        logger.warning(
            "Model changed (%r -> %r) since the previous run; continuing because "
            "--allow-model-change was passed. Existing candidate rows keep their original "
            "api_model value.",
            previous.model,
            current.model,
        )
