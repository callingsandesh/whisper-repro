"""OpenSLR SLR54 -- crowd-sourced Nepali speech (Kjartansson et al., 2018).

A low-resource corpus Whisper was not evaluated on, in Devanagari script.
The distribution is a tarball plus a tab-separated index, so this adapter
expects the archive already unpacked (download links change; keeping the
fetch manual avoids a brittle scraper).

    1. Download from https://www.openslr.org/54/
    2. Unpack somewhere, then point configs/openslr_nepali.yaml at it.

Expected after unpacking:
    <root>/utt_spk_text.tsv        utterance_id <TAB> speaker_id <TAB> transcript
    <root>/asr_nepali/data/...     .flac files, nested by id prefix
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from ..logging_utils import get_logger
from .audio_io import write_standardised
from .base import FetchAdapter, Sample

log = get_logger(__name__)


class OpenSLRNepaliAdapter(FetchAdapter):
    name = "openslr_nepali"
    language = "ne"
    source = "OpenSLR SLR54 (manual download; see module docstring)"

    def _root(self) -> Path:
        root = self.options.get("source_dir")
        if not root:
            raise ValueError(
                "openslr_nepali needs source_dir pointing at the unpacked SLR54 "
                "download -- set it in configs/openslr_nepali.yaml"
            )
        p = Path(root).expanduser()
        if not p.is_dir():
            raise FileNotFoundError(f"source_dir does not exist: {p}")
        return p

    def _find_audio(self, root: Path, utt_id: str) -> Path | None:
        # SLR54 nests files as asr_nepali/data/<2-char prefix>/<utt_id>.flac
        direct = root / "asr_nepali" / "data" / utt_id[:2] / f"{utt_id}.flac"
        if direct.exists():
            return direct
        hits = list(root.rglob(f"{utt_id}.flac"))
        return hits[0] if hits else None

    def _iter_samples(self, limit: int | None) -> Iterator[Sample]:
        root = self._root()
        index = root / "utt_spk_text.tsv"
        if not index.exists():
            raise FileNotFoundError(f"{index} not found under {root}")

        n = 0
        with index.open(encoding="utf-8", newline="") as f:
            for row in csv.reader(f, delimiter="\t"):
                if limit is not None and n >= limit:
                    break
                if len(row) < 3:
                    continue
                utt_id, spk, text = row[0], row[1], row[2]
                src = self._find_audio(root, utt_id)
                if src is None:
                    log.warning("audio missing for %s", utt_id)
                    continue
                dest = self.audio_dir / f"{utt_id}.flac"
                duration = write_standardised(src, dest)
                n += 1
                yield Sample(
                    sample_id=utt_id,
                    audio_path=dest.name,
                    reference=text.strip(),
                    language=self.language,
                    duration_s=round(duration, 3),
                    speaker_id=spk,
                    meta={"corpus": "OpenSLR SLR54", "script": "Devanagari"},
                )
