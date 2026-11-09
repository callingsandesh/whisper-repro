"""Which writing system is a string in?

Deterministic script identification by Unicode block. This exists because
the central finding of the Nepali evaluation is a *script* failure, not a
recognition failure: 14 of 15 utterances came back as recognisable Nepali
written in Latin or Perso-Arabic characters. WER cannot distinguish that
from an unrelated hallucination -- both score 100%.

Detecting the script is cheap, exact and needs no model, so the
script-mismatch rate is a hard statistic that stands on its own. It is
reported independently of the LLM judge, and is the more defensible of the
two numbers: a reviewer can verify it by inspection, whereas a judge's
verdict rests on trusting the judge.
"""

from __future__ import annotations

import unicodedata
from collections import Counter

#: (name, inclusive codepoint ranges). Ordered so the first match wins.
_BLOCKS: list[tuple[str, tuple[tuple[int, int], ...]]] = [
    ("Devanagari", ((0x0900, 0x097F), (0xA8E0, 0xA8FF), (0x1CD0, 0x1CFF))),
    ("Latin",      ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F))),
    ("Arabic",     ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
                    (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("Bengali",    ((0x0980, 0x09FF),)),
    ("Cyrillic",   ((0x0400, 0x04FF),)),
    ("Greek",      ((0x0370, 0x03FF),)),
    ("Hebrew",     ((0x0590, 0x05FF),)),
    ("Han",        ((0x4E00, 0x9FFF), (0x3400, 0x4DBF))),
    ("Hiragana",   ((0x3040, 0x309F),)),
    ("Katakana",   ((0x30A0, 0x30FF),)),
    ("Hangul",     ((0xAC00, 0xD7AF), (0x1100, 0x11FF))),
    ("Thai",       ((0x0E00, 0x0E7F),)),
    ("Tamil",      ((0x0B80, 0x0BFF),)),
]

#: Script expected for each language tag in the manifests.
EXPECTED: dict[str, str] = {"ne": "Devanagari", "en": "Latin"}


def _script_of_char(ch: str) -> str | None:
    """Script name for one character, or None if it carries no script
    (digits, punctuation, whitespace, symbols)."""
    cat = unicodedata.category(ch)
    # Lo/Ll/Lu/Lt/Lm are letters; Mn/Mc are the combining marks that carry
    # Devanagari vowel signs -- counting them matters, since stripping them
    # is exactly the normaliser bug this project uncovered.
    if cat[0] not in ("L", "M"):
        return None
    cp = ord(ch)
    for name, ranges in _BLOCKS:
        if any(lo <= cp <= hi for lo, hi in ranges):
            return name
    return "Other"


def script_profile(text: str) -> Counter:
    """Count of script-bearing characters by script."""
    return Counter(s for s in (_script_of_char(c) for c in text) if s)


def detect_script(text: str) -> str:
    """The dominant script, or "None" for text with no script-bearing
    characters (empty output, digits only)."""
    prof = script_profile(text)
    if not prof:
        return "None"
    return prof.most_common(1)[0][0]


def script_mismatch(language: str, hypothesis: str) -> bool | None:
    """True when the hypothesis is not in the script the language expects.

    Returns None for a language with no expected script registered, so that
    an unknown language is never silently counted as a match.
    """
    expected = EXPECTED.get(language)
    if expected is None:
        return None
    got = detect_script(hypothesis)
    return got != expected and got != "None"
