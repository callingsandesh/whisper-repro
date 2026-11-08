"""Tests for the scoring layer.

These pin the two numbers quoted in the project proposal, computed from a
real LibriSpeech utterance in the feasibility run. If a refactor ever
changes them, that is a regression in the metric, not an improvement.
"""

import pytest

from whisper_repro.scoring.metrics import char_counts, pooled, word_counts

# From the feasibility run (LibriSpeech test-clean, large-v2).
REF_NORM = ("forgetting all their weariness they seized their candles and "
            "scurried through the house finding an occasional paper tucked "
            "away in some odd corner")
HYP_NORM = ("forgetting all their weariness they seize their candles and "
            "scurry through the house finding on occasional paper tucked "
            "away in some odd corner")
REF_RAW = REF_NORM.upper()
HYP_RAW = ("Forgetting all their weariness, they seize their candles and "
           "scurry through the house, finding on occasional paper tucked "
           "away in some odd corner.")


def test_worked_example_counts():
    c = word_counts(REF_NORM, HYP_NORM)
    assert c.ref_length == 23
    assert (c.substitutions, c.deletions, c.insertions) == (3, 0, 0)
    assert c.rate == pytest.approx(3 / 23)
    assert c.rate * 100 == pytest.approx(13.04, abs=0.01)


def test_normalisation_is_what_makes_the_number_meaningful():
    """Unnormalised, every word differs by case or punctuation: 100% WER."""
    assert word_counts(REF_RAW, HYP_RAW).rate == pytest.approx(1.0)


def test_cer_is_far_lower_than_wer_for_inflection_errors():
    assert char_counts(REF_NORM, HYP_NORM).rate < word_counts(REF_NORM, HYP_NORM).rate


def test_empty_reference_is_tolerated():
    assert word_counts("", "anything").ref_length == 0


def test_pooling_is_not_averaging():
    """Corpus WER pools errors over all words — a short bad utterance must not
    outweigh a long good one, which is what averaging per-utterance rates does."""
    short = word_counts("yes", "no")                      # 1/1   = 100%
    long_ = word_counts(" ".join(["word"] * 99), " ".join(["word"] * 99))  # 0/99 = 0%
    assert pooled([short, long_]).rate == pytest.approx(1 / 100)
