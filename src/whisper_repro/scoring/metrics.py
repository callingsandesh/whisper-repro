"""Error-rate computation.

Two metrics, two aggregation levels, and the distinction between them
matters when writing up results:

  * Corpus-level: errors and reference lengths pooled across all
    utterances, then divided once. This is what the paper reports and the
    only figure comparable to its tables.
  * Per-utterance: each utterance's own rate. Never average these to get a
    corpus figure -- short utterances would carry the same weight as long
    ones. They are for finding interesting examples and plotting
    distributions.

CER is reported alongside WER because word-level rates are unreliable for
Devanagari, where orthographic variation and segmentation conventions
inflate word errors; the paper itself falls back to CER for some languages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jiwer


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return self.errors / self.ref_length if self.ref_length else 0.0

    def __add__(self, other: "ErrorCounts") -> "ErrorCounts":
        return ErrorCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.hits + other.hits,
            self.ref_length + other.ref_length,
        )


_EMPTY = ErrorCounts(0, 0, 0, 0, 0)


def word_counts(reference: str, hypothesis: str) -> ErrorCounts:
    """Word-level alignment counts for one utterance."""
    if not reference.strip():
        # jiwer rejects an empty reference; an empty ref contributes no
        # reference words, so insertions in the hypothesis are counted but
        # the rate is undefined -- return zeros and let the caller skip it.
        return _EMPTY
    out = jiwer.process_words(reference, hypothesis)
    return ErrorCounts(
        out.substitutions, out.deletions, out.insertions, out.hits,
        len(reference.split()),
    )


def char_counts(reference: str, hypothesis: str) -> ErrorCounts:
    """Character-level counts, computed with whitespace removed.

    Two reasons for stripping spaces from BOTH strings first:

      * Correctness. jiwer.process_characters COUNTS spaces as characters, so
        dividing its error count by a space-free reference length mismatches
        numerator and denominator and inflates CER. (This was a real bug here:
        it reported CER above WER on Nepali, which is never possible.)
      * Meaning. CER exists to measure character accuracy independently of word
        segmentation -- which is precisely why the paper falls back to it for
        languages where word boundaries are unreliable. Counting spaces as
        characters would reintroduce the segmentation problem CER avoids.
    """
    ref = "".join(reference.split())
    hyp = "".join(hypothesis.split())
    if not ref:
        return _EMPTY
    out = jiwer.process_characters(ref, hyp)
    return ErrorCounts(
        out.substitutions, out.deletions, out.insertions, out.hits, len(ref),
    )


def corpus_rate(counts: Sequence[ErrorCounts]) -> float:
    """Pool counts across utterances, then divide. The paper's convention."""
    total = _EMPTY
    for c in counts:
        total = total + c
    return total.rate


def pooled(counts: Sequence[ErrorCounts]) -> ErrorCounts:
    total = _EMPTY
    for c in counts:
        total = total + c
    return total