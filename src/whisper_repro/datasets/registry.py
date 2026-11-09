"""Name -> adapter lookup.

The only place that knows which adapters exist. Every other module takes a
dataset name as a string and asks here, so adding a corpus means one import
and one entry below -- nothing else in the codebase changes.

Per-dataset options (source directories, speaker filters) come from
configs/<name>.yaml and are merged with any passed on the command line.
"""

from __future__ import annotations

from typing import Any, Type

from ..config import load_dataset_config
from .base import DatasetAdapter
from .l2arctic import L2ArcticAdapter
from .librispeech import LibriSpeechAdapter, LibriSpeechOtherAdapter
from .nepali_custom import NepaliAccentedEnglishAdapter, NepaliCustomAdapter
from .openslr_nepali import OpenSLRNepaliAdapter
from .fleurs import FleursAdapter, FleursEnglishAdapter
from .tedlium import TedLiumAdapter

ADAPTERS: dict[str, Type[DatasetAdapter]] = {
    LibriSpeechAdapter.name: LibriSpeechAdapter,
    LibriSpeechOtherAdapter.name: LibriSpeechOtherAdapter,
    L2ArcticAdapter.name: L2ArcticAdapter,
    OpenSLRNepaliAdapter.name: OpenSLRNepaliAdapter,
    NepaliCustomAdapter.name: NepaliCustomAdapter,
    FleursAdapter.name: FleursAdapter,
    FleursEnglishAdapter.name: FleursEnglishAdapter,
    NepaliAccentedEnglishAdapter.name: NepaliAccentedEnglishAdapter,
    TedLiumAdapter.name: TedLiumAdapter,
}


def available() -> list[str]:
    return sorted(ADAPTERS)


def get_adapter(name: str, **overrides: Any) -> DatasetAdapter:
    try:
        cls = ADAPTERS[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset '{name}'. Available: {', '.join(available())}"
        ) from None
    options = {**load_dataset_config(name), **{k: v for k, v in overrides.items() if v is not None}}
    return cls(**options)
