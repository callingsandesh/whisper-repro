"""The constructed dataset: Nepali speech and Nepali-accented English.

This is the one corpus in the project that exists nowhere else. Recordings
and hand-written transcripts are the project's irreplaceable artefact, so
this adapter deliberately does NOT own them: it reads from a source
directory kept outside the repo (and backed up), and copies into
data/raw/.

Expected layout of the source directory:

    <source_dir>/
      transcripts.csv       # sample_id,filename,reference,language,speaker_id,...
      audio/
        ne_001.wav          # whatever your phone/recorder produced
        ne_002.m4a

transcripts.csv is the annotation work -- the valuable part. Keep it under
version control (data/manifests/nepali_custom.jsonl is generated from it,
but the CSV is what you edit by hand).

Required CSV columns : filename, reference
Optional CSV columns : sample_id, language, speaker_id, accent, device,
                       prompt_id, session, notes
"""

from __future__ import annotations

import csv
from typing import Iterator

from ..logging_utils import get_logger
from .audio_io import write_standardised
from .base import LocalAdapter, Sample

log = get_logger(__name__)

AUDIO_SUFFIXES = {".wav", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".aac"}


class NepaliCustomAdapter(LocalAdapter):
    name = "nepali_custom"
    language = "ne"
    source = "own recordings (see configs/nepali_custom.yaml)"

    #: Default language for rows that do not specify one. Override in config
    #: to "en" for the Nepali-accented-English portion.
    default_language = "ne"

    def __init__(self, **options):
        super().__init__(**options)
        self.default_language = options.get("default_language", self.default_language)

    def _iter_samples(self, limit: int | None) -> Iterator[Sample]:
        root = self.source_dir()
        csv_path = root / "transcripts.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. It must list at least: filename,reference"
            )

        audio_root = root / "audio" if (root / "audio").is_dir() else root
        seen: set[str] = set()

        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            missing = {"filename", "reference"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path} is missing column(s): {sorted(missing)}")

            for i, row in enumerate(reader):
                if limit is not None and len(seen) >= limit:
                    break

                filename = (row.get("filename") or "").strip()
                reference = (row.get("reference") or "").strip()
                if not filename or not reference:
                    log.warning("%s:%d skipped -- blank filename or reference",
                                csv_path.name, i + 2)
                    continue

                src = audio_root / filename
                if not src.exists():  # tolerate a missing/incorrect extension
                    matches = [
                        p for p in audio_root.glob(f"{src.stem}.*")
                        if p.suffix.lower() in AUDIO_SUFFIXES
                    ]
                    if not matches:
                        log.warning("%s:%d skipped -- audio not found: %s",
                                    csv_path.name, i + 2, src)
                        continue
                    src = matches[0]

                sid = (row.get("sample_id") or "").strip() or f"np_{i:04d}"
                if sid in seen:
                    raise ValueError(f"duplicate sample_id '{sid}' in {csv_path}")
                seen.add(sid)

                dest = self.audio_dir / f"{sid}.flac"
                duration = write_standardised(src, dest)

                known = {"sample_id", "filename", "reference", "language", "speaker_id"}
                meta = {k: v for k, v in row.items() if k not in known and v}
                meta["source_file"] = src.name  # trace a result back to the original

                yield Sample(
                    sample_id=sid,
                    audio_path=dest.name,
                    reference=reference,
                    language=(row.get("language") or self.default_language).strip(),
                    duration_s=round(duration, 3),
                    speaker_id=(row.get("speaker_id") or "").strip() or None,
                    meta=meta,
                )

        log.info("constructed dataset: %d utterances from %s", len(seen), csv_path.name)

class NepaliAccentedEnglishAdapter(NepaliCustomAdapter):
    """English read by native Nepali speakers -- the same people, same room,
    same microphone as the Nepali set.

    Kept as a separate dataset rather than a language column in one set,
    because the two answer different questions: this one probes accent
    robustness on a language Whisper models well, while nepali_custom probes
    a low-resource language. Pooling them would produce one blended WER that
    answers neither. Separate datasets also keep the normaliser choice honest
    -- this one scores through EnglishTextNormalizer, as the paper does for
    English.
    """

    name = "nepali_accented_en"
    language = "en"
    default_language = "en"
    source = "own recordings -- English read by Nepali L1 speakers"