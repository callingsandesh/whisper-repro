"""The contract every dataset adapter satisfies.

The whole pipeline rests on one idea: however a corpus arrives -- streamed
from HuggingFace, unpacked from a tarball, or recorded on a phone -- an
adapter reduces it to the same two artefacts:

    data/raw/<dataset>/<sample_id>.flac    16 kHz mono audio
    data/manifests/<dataset>.jsonl         one Sample per line

Downstream stages (transcribe, score, judge) read the manifest and never
learn which adapter produced it. Adding a corpus is therefore one new
module and one line in registry.py -- no changes anywhere else.

Two adapter flavours exist, differing only in where `prepare` gets its
input:

    FetchAdapter -- re-downloadable public corpora. data/raw is a cache;
                    deleting it costs bandwidth, nothing more.
    LocalAdapter -- our own recordings. The audio exists nowhere else, so
                    the adapter reads from a configured source directory
                    (kept backed up, outside this repo) and copies in.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..logging_utils import get_logger
from ..paths import audio_dir, manifest_path

log = get_logger(__name__)

TARGET_SR = 16_000  # Whisper's input rate; resample once, at prepare time.


@dataclass(frozen=True, slots=True)
class Sample:
    """One utterance. This is the unit of currency across the pipeline."""

    sample_id: str          # unique within the dataset, filesystem-safe
    audio_path: str         # relative to the dataset's audio dir
    reference: str          # gold transcript, verbatim from the source
    language: str           # ISO 639-1: "en", "ne". Drives normaliser choice.
    duration_s: float | None = None
    speaker_id: str | None = None
    # Free-form provenance: original filename, accent, recording device,
    # annotator, corpus split. Kept open deliberately -- adapters differ in
    # what they know, and slicing results by these fields is where the
    # interesting analysis lives.
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Sample":
        d = json.loads(line)
        return Sample(**d)

    def resolve_audio(self, dataset: str) -> Path:
        return audio_dir(dataset) / self.audio_path


# --------------------------------------------------------------------------
# Manifest I/O
# --------------------------------------------------------------------------

def write_manifest(dataset: str, samples: Iterable[Sample]) -> Path:
    """Write atomically: a half-written manifest is worse than none."""
    path = manifest_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.to_json() + "\n")
            n += 1
    tmp.replace(path)
    log.info("manifest written: %s (%d samples)", path, n)
    return path


def read_manifest(dataset: str, limit: int | None = None) -> list[Sample]:
    path = manifest_path(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest for '{dataset}'. Run: whisper-repro prepare --dataset {dataset}"
        )
    out: list[Sample] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit is not None and len(out) >= limit:
                break
            try:
                out.append(Sample.from_json(line))
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"{path}:{i+1} is not a valid Sample: {e}") from e
    return out


# --------------------------------------------------------------------------
# Adapter base classes
# --------------------------------------------------------------------------

class DatasetAdapter(ABC):
    """Base for every corpus.

    Subclasses implement `_iter_samples`, which does whatever that corpus
    requires (stream, unpack, walk a folder) and yields Samples whose audio
    has already been written to disk. The base class handles resume logic,
    manifest writing and logging, so adapters stay small.
    """

    name: str = "unnamed"
    language: str = "en"
    #: Human-readable provenance, surfaced by `whisper-repro datasets`.
    source: str = ""

    def __init__(self, **options: Any) -> None:
        self.options = options

    # -- to implement ------------------------------------------------------

    @abstractmethod
    def _iter_samples(self, limit: int | None) -> Iterator[Sample]:
        """Yield Samples, writing each one's audio into `self.audio_dir`."""

    # -- provided ----------------------------------------------------------

    @property
    def audio_dir(self) -> Path:
        return audio_dir(self.name)

    def prepare(self, limit: int | None = None, force: bool = False) -> Path:
        """Materialise audio + manifest. Idempotent unless `force`."""
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        mpath = manifest_path(self.name)

        if mpath.exists() and not force:
            existing = read_manifest(self.name)
            if limit is None or len(existing) >= limit:
                log.info(
                    "%s already prepared (%d samples) -- skipping. Use --force to rebuild.",
                    self.name, len(existing),
                )
                return mpath
            log.info(
                "%s has %d samples, %d requested -- rebuilding.",
                self.name, len(existing), limit,
            )

        log.info("preparing %s (limit=%s) from %s", self.name, limit, self.source or "?")
        samples = list(self._iter_samples(limit))
        if not samples:
            raise RuntimeError(f"adapter '{self.name}' produced no samples")
        return write_manifest(self.name, samples)

    def describe(self) -> dict[str, Any]:
        mpath = manifest_path(self.name)
        n = sum(1 for _ in mpath.open(encoding="utf-8")) if mpath.exists() else 0
        return {
            "name": self.name,
            "language": self.language,
            "source": self.source,
            "kind": "local" if isinstance(self, LocalAdapter) else "fetch",
            "prepared": n,
        }


class FetchAdapter(DatasetAdapter):
    """Public corpus fetched from the network. data/raw is a cache."""


class LocalAdapter(DatasetAdapter):
    """Corpus we created. Audio is irreplaceable; source lives outside the repo."""

    def source_dir(self) -> Path:
        src = self.options.get("source_dir")
        if not src:
            raise ValueError(
                f"adapter '{self.name}' needs source_dir "
                f"(set it in configs/{self.name}.yaml or pass --source-dir)"
            )
        p = Path(src).expanduser()
        if not p.is_dir():
            raise FileNotFoundError(f"source_dir does not exist: {p}")
        return p
