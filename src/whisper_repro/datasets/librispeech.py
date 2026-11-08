"""LibriSpeech -- the paper's anchor benchmark (Panayotov et al., 2015).

Read audiobook speech from LibriVox. References are UPPERCASE with no
punctuation, which is why normalisation matters so much: scored raw against
Whisper's cased, punctuated output, a near-perfect transcript can register
100% WER.

Streamed from the HuggingFace Hub so only the requested utterances are
downloaded, not the full 60 GB corpus.
"""

from __future__ import annotations

from typing import Iterator

from ..logging_utils import get_logger
from .audio_io import write_standardised
from .base import FetchAdapter, Sample

log = get_logger(__name__)

HF_REPO = "openslr/librispeech_asr"  # bare "librispeech_asr" is rejected by
                                     # newer huggingface_hub (needs namespace)


class LibriSpeechAdapter(FetchAdapter):
    name = "librispeech_test_clean"
    language = "en"
    source = f"HuggingFace {HF_REPO} (clean/test)"

    #: HF config ("clean"/"other") and split, overridable per instance.
    config = "clean"
    split = "test"

    def __init__(self, **options):
        super().__init__(**options)
        self.config = options.get("config", self.config)
        self.split = options.get("split", self.split)

    def _iter_samples(self, limit: int | None) -> Iterator[Sample]:
        from datasets import Audio, load_dataset

        ds = load_dataset(HF_REPO, self.config, split=self.split, streaming=True)
        # decode=False hands us raw bytes and keeps HF's audio decoding
        # backend (torchcodec) out of the picture -- it is tightly version
        # pinned to torch and breaks easily. soundfile decodes instead.
        ds = ds.cast_column("audio", Audio(decode=False))

        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                break
            sid = f"ls_{i:05d}"
            dest = self.audio_dir / f"{sid}.flac"
            duration = write_standardised(ex["audio"]["bytes"], dest)
            if i % 25 == 0:
                log.info("prepared %d utterances", i)
            yield Sample(
                sample_id=sid,
                audio_path=dest.name,
                reference=ex["text"],
                language=self.language,
                duration_s=round(duration, 3),
                speaker_id=str(ex.get("speaker_id", "")) or None,
                meta={
                    "corpus": "LibriSpeech",
                    "config": self.config,
                    "split": self.split,
                    "chapter_id": ex.get("chapter_id"),
                    "hf_id": ex.get("id"),
                },
            )


class LibriSpeechOtherAdapter(LibriSpeechAdapter):
    """The harder split -- noisier recordings, more accented speakers."""

    name = "librispeech_test_other"
    config = "other"
    source = f"HuggingFace {HF_REPO} (other/test)"
