"""TED-LIUM 3 -- TED talk speech (Hernandez et al., 2018).

One of the paper's own evaluation datasets, and a different speech style from
LibriSpeech: prepared but naturally delivered, single speaker, auditorium
acoustics rather than a reading booth.

SOURCE. The canonical repo (LIUM/tedlium) is gated and ships a Python loading
script that `datasets` 3.x will not run, so this adapter defaults to
HuggingFace's own ESB evaluation mirror, which is parquet-based, ungated, and
test-split only -- exactly what an evaluation needs. Columns there are
`audio`, `text`, `id`, `audio_length_s`. To use the canonical repo instead,
set hf_repo/config in configs/tedlium3.yaml.

TWO DATASET-SPECIFIC QUIRKS, handled here rather than downstream:

1. Non-speech markers. Original TED-LIUM transcripts contain {NOISE}, {COUGH},
   <sil> and (2)-style pronunciation-variant suffixes. Whisper's
   EnglishTextNormalizer strips <...> and (...) but NOT {...}, so an untreated
   {COUGH} survives normalisation as the word "cough" and counts as a reference
   word Whisper could never produce. (The ESB mirror is already cleaned, so
   this is a no-op there -- harmless, and it keeps the adapter correct against
   the canonical source too.)

2. ignore_time_segment_in_scoring. TED-LIUM marks segments the corpus authors
   exclude from scoring. Keeping them would score Whisper against text the
   dataset itself says not to score. Dropped.
"""

from __future__ import annotations

import re
from typing import Iterator

from ..logging_utils import get_logger
from .audio_io import write_standardised
from .base import FetchAdapter, Sample

log = get_logger(__name__)

#: HuggingFace's ESB evaluation mirror -- ungated, parquet, test-only.
DEFAULT_REPO = "huggingworld/open-asr-leaderboard"
DEFAULT_CONFIG = "tedlium"

IGNORE_MARKER = "ignore_time_segment_in_scoring"

_BRACED = re.compile(r"\{[^}]*\}")      # {NOISE} {COUGH} {SMACK}
_ANGLED = re.compile(r"<[^>]*>")        # <sil> <unk>
_VARIANT = re.compile(r"\((\d+)\)")     # the(2) -> the

#: Transcript field, in order of preference. The canonical repo uses "text";
#: mirrors sometimes differ.
TEXT_FIELDS = ("text", "transcription", "sentence", "raw_transcription")


def clean_transcript(text: str) -> str:
    """Strip TED-LIUM's annotation markup, leaving the spoken words."""
    t = _BRACED.sub(" ", text)
    t = _ANGLED.sub(" ", t)
    t = _VARIANT.sub("", t)
    return " ".join(t.split())


class TedLiumAdapter(FetchAdapter):
    name = "tedlium3"
    language = "en"

    config = DEFAULT_CONFIG
    split = "test"
    min_duration_s = 1.0

    def __init__(self, **options):
        super().__init__(**options)
        self.hf_repo = options.get("hf_repo", DEFAULT_REPO)
        self.config = options.get("config", self.config)
        self.split = options.get("split", self.split)
        self.min_duration_s = float(options.get("min_duration_s", self.min_duration_s))
        # Built from the resolved values, so the log line tells the truth about
        # what was actually requested -- a static f-string on the class does not.
        self.source = f"HuggingFace {self.hf_repo} ({self.config}, {self.split})"

    @staticmethod
    def _transcript(ex: dict) -> str | None:
        for field in TEXT_FIELDS:
            v = ex.get(field)
            if v:
                return str(v).strip()
        return None

    def _load(self):
        from datasets import Audio, load_dataset

        try:
            ds = load_dataset(self.hf_repo, self.config, split=self.split,
                              streaming=True)
        except Exception as e:
            msg = str(e)
            hint = ""
            if "script" in msg.lower() or "trust_remote_code" in msg:
                hint = (" -- that repo ships a loading script, which this "
                        "`datasets` version will not run; use the parquet "
                        f"mirror ({DEFAULT_REPO}, config {DEFAULT_CONFIG}).")
            elif "doesn't exist" in msg or "cannot be accessed" in msg:
                hint = (" -- the repo may be gated. Either `huggingface-cli "
                        f"login` and accept its terms, or use {DEFAULT_REPO} "
                        f"with config {DEFAULT_CONFIG}.")
            raise RuntimeError(
                f"could not load {self.hf_repo} (config={self.config}, "
                f"split={self.split}): {msg}{hint}"
            ) from None
        return ds.cast_column("audio", Audio(decode=False))

    def _iter_samples(self, limit: int | None) -> Iterator[Sample]:
        ds = self._load()

        emitted = skipped_ignored = skipped_short = seen = 0
        reported = False

        for i, ex in enumerate(ds):
            seen += 1
            if limit is not None and emitted >= limit:
                break

            if not reported:
                log.info("tedlium fields available: %s", sorted(ex))
                reported = True

            raw = self._transcript(ex)
            if not raw or IGNORE_MARKER in raw:
                skipped_ignored += 1
                continue

            reference = clean_transcript(raw)
            if not reference:
                skipped_ignored += 1
                continue

            sid = f"ted_{i:05d}"
            dest = self.audio_dir / f"{sid}.flac"
            duration = write_standardised(ex["audio"]["bytes"], dest)

            if duration < self.min_duration_s:
                dest.unlink(missing_ok=True)
                skipped_short += 1
                continue

            emitted += 1
            if emitted % 25 == 0:
                log.info("prepared %d utterances", emitted)

            yield Sample(
                sample_id=sid,
                audio_path=dest.name,
                reference=reference,
                language=self.language,
                duration_s=round(duration, 3),
                speaker_id=str(ex.get("speaker_id", "")) or None,
                meta={
                    "corpus": "TED-LIUM 3",
                    "repo": self.hf_repo,
                    "config": self.config,
                    "split": self.split,
                    "item_id": ex.get("id"),
                },
            )

        # Zero examples from the stream is a configuration error, not an empty
        # dataset. Say so here rather than letting prepare() report the vaguer
        # "adapter produced no samples".
        if seen == 0:
            raise RuntimeError(
                f"{self.hf_repo} (config={self.config}, split={self.split}) "
                f"yielded no examples. Check the config and split names:\n"
                f"  from datasets import get_dataset_config_names, get_dataset_split_names\n"
                f"  get_dataset_config_names({self.hf_repo!r})\n"
                f"  get_dataset_split_names({self.hf_repo!r}, {self.config!r})"
            )

        log.info("tedlium3: emitted %d of %d seen; skipped %d (ignored/empty), "
                 "%d (under %.1fs)", emitted, seen, skipped_ignored,
                 skipped_short, self.min_duration_s)