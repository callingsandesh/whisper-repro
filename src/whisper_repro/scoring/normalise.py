"""Text normalisation -- dispatched by language.

Whisper ships two normalisers and using the wrong one silently corrupts
results:

    EnglishTextNormalizer  -- built around English contractions, numerals,
                              spelling variants. Required to reproduce the
                              paper's English numbers.
    BasicTextNormalizer    -- everything else. Case folding, punctuation
                              and symbol removal, whitespace collapse.

Running the English normaliser over Devanagari would mangle it, so the
choice is made from the Sample's `language` field rather than being a
global setting. This is the main reason `language` is a required field on
Sample.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol


class Normalizer(Protocol):
    def __call__(self, text: str) -> str: ...


class IdentityNormalizer:
    """No-op, for reporting raw (unnormalised) error rates alongside."""

    def __call__(self, text: str) -> str:
        return text.strip()


@lru_cache(maxsize=None)
def get_normalizer(language: str) -> Normalizer:
    """Return the normaliser Whisper itself would use for this language."""
    from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

    if (language or "").lower().startswith("en"):
        return EnglishTextNormalizer()
    # split_letters=False keeps Devanagari clusters intact; splitting would
    # inflate CER by breaking conjuncts into separate characters.
    return BasicTextNormalizer(split_letters=False)


@lru_cache(maxsize=1)
def get_identity() -> Normalizer:
    return IdentityNormalizer()
