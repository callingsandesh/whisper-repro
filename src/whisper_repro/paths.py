"""Single source of truth for where things live on disk.

Every other module asks this one for paths. Nothing hardcodes a directory
string, so moving the data root (to an external drive, to Colab's
/content, to a scratch partition) is one environment variable:

    export WHISPER_REPRO_DATA=/media/sandesh/HDD/whisper-data
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root: src/whisper_repro/paths.py -> up three levels.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("WHISPER_REPRO_DATA", REPO_ROOT / "data"))

RAW_DIR = DATA_ROOT / "raw"          # prepared audio, one subdir per dataset
MANIFEST_DIR = DATA_ROOT / "manifests"
HYPS_DIR = DATA_ROOT / "hyps"        # <dataset>/<model>/<sample_id>.txt
RESULTS_DIR = DATA_ROOT / "results"  # scored CSVs and summaries
JUDGE_DIR = DATA_ROOT / "judge"      # LLM-as-judge verdicts
CONFIG_DIR = REPO_ROOT / "configs"


def audio_dir(dataset: str) -> Path:
    return RAW_DIR / dataset


def manifest_path(dataset: str) -> Path:
    return MANIFEST_DIR / f"{dataset}.jsonl"


def hyp_dir(dataset: str, model: str) -> Path:
    # Model names like "large-v2" are filesystem-safe; keep them verbatim
    # so the directory tree is readable.
    return HYPS_DIR / dataset / model


def hyp_path(dataset: str, model: str, sample_id: str) -> Path:
    return hyp_dir(dataset, model) / f"{sample_id}.txt"


def results_path(dataset: str, model: str) -> Path:
    return RESULTS_DIR / f"{dataset}__{model}.csv"


def summary_path(dataset: str, model: str) -> Path:
    return RESULTS_DIR / f"{dataset}__{model}.summary.json"


def judge_path(dataset: str, model: str) -> Path:
    return JUDGE_DIR / f"{dataset}__{model}.jsonl"


def ensure_dirs() -> None:
    for d in (RAW_DIR, MANIFEST_DIR, HYPS_DIR, RESULTS_DIR, JUDGE_DIR):
        d.mkdir(parents=True, exist_ok=True)
