"""FLEURS -- Google's 102-language speech benchmark (Conneau et al., 2023).

Used here for Nepali (ne_np), which the Whisper paper reports but which sits
in its weakest regime: a low-resource language written in Devanagari.

This is the project's second fetch adapter, and the first in a non-English
language -- so it is also the test of whether the Sample contract holds.
The only thing that changes downstream is the `language` field, which routes
scoring to BasicTextNormalizer instead of the English one.
"""

from __future__ import annotations

from typing import Iterator

from ..logging_utils import get_logger
from .audio_io import write_standardised
from .base import FetchAdapter, Sample

log = get_logger(__name__)

HF_REPO = "google/fleurs"

#: FLEURS exposes both a normalised and a verbatim transcript, and the field
#: names have moved between dataset revisions. Prefer the raw transcript --
#: normalisation is our job, at scoring time, with Whisper's own normaliser.
TRANSCRIPT_FIELDS = ("raw_transcription", "transcription", "text", "sentence")


class FleursAdapter(FetchAdapter):
    name = "fleurs_ne"
    language = "ne"
    source = f"HuggingFace {HF_REPO} (ne_np, test)"

    config = "ne_np"   # ISO 639-1 + ISO 3166-1, lowercase. Verify with:
                       #   from datasets import get_dataset_config_names
                       #   get_dataset_config_names("google/fleurs")
    split = "test"

    def __init__(self, **options):
        super().__init__(**options)
        self.config = options.get("config", self.config)
        self.split = options.get("split", self.split)
        self.language = options.get("language", self.config.split("_")[0])

    @staticmethod
    def _transcript(ex: dict) -> str | None:
        for field in TRANSCRIPT_FIELDS:
            value = ex.get(field)
            if value:
                return str(value).strip()
        return None

    def _iter_samples(self, limit: int | None) -> Iterator[Sample]:
        from datasets import Audio, load_dataset

        ds = load_dataset(HF_REPO, self.config, split=self.split, streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))

        reported = False
        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                break

            if not reported:   # one line, first example only: what did we get?
                log.info("fleurs fields available: %s", sorted(ex))
                reported = True

            reference = self._transcript(ex)
            if not reference:
                log.warning("no transcript field on example %d -- fields: %s",
                            i, sorted(ex))
                continue

            sid = f"fl_{self.config}_{i:05d}"
            dest = self.audio_dir / f"{sid}.flac"
            duration = write_standardised(ex["audio"]["bytes"], dest)
            if i % 25 == 0:
                log.info("prepared %d utterances", i)

            yield Sample(
                sample_id=sid,
                audio_path=dest.name,
                reference=reference,
                language=self.language,
                duration_s=round(duration, 3),
                speaker_id=None,
                meta={
                    "corpus": "FLEURS",
                    "config": self.config,
                    "split": self.split,
                    "gender": ex.get("gender"),
                    "script": "Devanagari" if self.language == "ne" else None,
                },
            )


class FleursEnglishAdapter(FleursAdapter):
    """FLEURS English -- the controlled counterpart to fleurs_ne.

    Same corpus, same recording protocol, same prompt design (FLEURS is
    parallel across its 102 languages: the same sentences, translated). The
    only variable that changes is the language, which makes the fleurs_en /
    fleurs_ne pair a public-data mirror of the constructed Nepali / accented
    English comparison.
    """

    name = "fleurs_en"
    language = "en"
    config = "en_us"
    source = f"HuggingFace {HF_REPO} (en_us, test)"