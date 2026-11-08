"""The four stages: prepare, transcribe, score, judge.

Resumability is structural, not bolted on: the filesystem IS the checkpoint.
A unit of work is done when its output file exists and is non-empty, so
re-running any stage picks up exactly where a crash, a disconnected Colab
session or a Ctrl-C left off. No state files, no database, nothing to get
out of sync with reality.

Every write goes through a temp file and a rename, so an interruption
mid-write cannot leave a truncated file that the resume check would wrongly
accept as complete.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from .datasets import Sample, get_adapter, read_manifest
from .datasets.audio_io import read_for_model
from .logging_utils import get_logger
from .models import WhisperRunner
from .paths import (
    ensure_dirs, hyp_dir, hyp_path, results_path, summary_path,
)
from .scoring import char_counts, get_normalizer, pooled, word_counts

log = get_logger(__name__)


def _is_done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------
# Stage 1 -- prepare
# --------------------------------------------------------------------------

def prepare(dataset: str, limit: int | None = None, force: bool = False,
            **adapter_options) -> Path:
    ensure_dirs()
    adapter = get_adapter(dataset, **adapter_options)
    return adapter.prepare(limit=limit, force=force)


# --------------------------------------------------------------------------
# Stage 2 -- transcribe
# --------------------------------------------------------------------------

def transcribe(dataset: str, model: str = "base", limit: int | None = None,
               force: bool = False, detect_language: bool = False) -> int:
    """Run Whisper over prepared audio. Returns the number newly transcribed."""
    ensure_dirs()
    samples = read_manifest(dataset, limit=limit)
    out_dir = hyp_dir(dataset, model)
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [s for s in samples
            if force or not _is_done(hyp_path(dataset, model, s.sample_id))]
    if not todo:
        log.info("%s/%s: all %d hypotheses present -- nothing to do",
                 dataset, model, len(samples))
        return 0

    log.info("%s/%s: %d of %d utterances to transcribe",
             dataset, model, len(todo), len(samples))

    runner = WhisperRunner(model_name=model)
    _write_atomic(out_dir / "_environment.json",
                  json.dumps(runner.env.as_dict(), indent=2))

    done, t0 = 0, time.time()
    for i, s in enumerate(todo, 1):
        audio_file = s.resolve_audio(dataset)
        if not audio_file.exists():
            log.warning("%s: audio missing (%s) -- run prepare first",
                        s.sample_id, audio_file)
            continue
        try:
            audio = read_for_model(audio_file)
            result = runner.transcribe(
                audio, language=None if detect_language else s.language,
            )
        except Exception:
            # One bad utterance must not kill a long run; it simply stays
            # un-transcribed and the next invocation retries it.
            log.exception("%s: transcription failed -- skipping", s.sample_id)
            continue

        _write_atomic(hyp_path(dataset, model, s.sample_id),
                      (result.get("text") or "").strip())
        done += 1
        if i % 10 == 0 or i == len(todo):
            rate = (time.time() - t0) / i
            log.info("  [%d/%d] %.2fs per utterance", i, len(todo), rate)

    log.info("%s/%s: transcribed %d in %.1fs", dataset, model, done, time.time() - t0)
    return done


# --------------------------------------------------------------------------
# Stage 3 -- score
# --------------------------------------------------------------------------

def _read_hyp(dataset: str, model: str, sample_id: str) -> str | None:
    p = hyp_path(dataset, model, sample_id)
    return p.read_text(encoding="utf-8").strip() if _is_done(p) else None


def score(dataset: str, model: str = "base", limit: int | None = None) -> dict:
    """Compute WER/CER, raw and normalised. Writes a per-utterance CSV and a
    summary JSON; returns the summary."""
    import csv

    ensure_dirs()
    samples = read_manifest(dataset, limit=limit)

    rows, w_raw, w_norm, c_norm, missing = [], [], [], [], 0
    for s in samples:
        hyp = _read_hyp(dataset, model, s.sample_id)
        if hyp is None:
            missing += 1
            continue
        norm = get_normalizer(s.language)
        ref_n, hyp_n = norm(s.reference), norm(hyp)

        wr = word_counts(s.reference, hyp)       # raw
        wn = word_counts(ref_n, hyp_n)           # normalised
        cn = char_counts(ref_n, hyp_n)
        w_raw.append(wr); w_norm.append(wn); c_norm.append(cn)

        rows.append({
            "sample_id": s.sample_id,
            "language": s.language,
            "speaker_id": s.speaker_id or "",
            "duration_s": s.duration_s or "",
            "n_ref_words": wn.ref_length,
            "wer_raw": round(wr.rate * 100, 4),
            "wer_norm": round(wn.rate * 100, 4),
            "cer_norm": round(cn.rate * 100, 4),
            "sub": wn.substitutions, "del": wn.deletions, "ins": wn.insertions,
            "reference_raw": s.reference,
            "hypothesis_raw": hyp,
            "reference_norm": ref_n,
            "hypothesis_norm": hyp_n,
            **{f"meta_{k}": v for k, v in (s.meta or {}).items()},
        })

    if not rows:
        raise RuntimeError(
            f"no hypotheses found for {dataset}/{model}. "
            f"Run: whisper-repro transcribe --dataset {dataset} --model {model}"
        )

    csv_path = results_path(dataset, model)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(csv_path)

    pw_raw, pw_norm, pc_norm = pooled(w_raw), pooled(w_norm), pooled(c_norm)
    env_file = hyp_dir(dataset, model) / "_environment.json"
    summary = {
        "dataset": dataset,
        "model": model,
        "utterances": len(rows),
        "missing_hypotheses": missing,
        "total_audio_s": round(sum(s.duration_s or 0 for s in samples), 1),
        "wer_raw_pct": round(pw_raw.rate * 100, 2),
        "wer_norm_pct": round(pw_norm.rate * 100, 2),
        "cer_norm_pct": round(pc_norm.rate * 100, 2),
        "substitutions": pw_norm.substitutions,
        "deletions": pw_norm.deletions,
        "insertions": pw_norm.insertions,
        "ref_words": pw_norm.ref_length,
        "environment": json.loads(env_file.read_text()) if env_file.exists() else None,
    }
    _write_atomic(summary_path(dataset, model), json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------
# Stage 4 -- judge (LLM-as-judge; implemented in week 9+)
# --------------------------------------------------------------------------

def judge(dataset: str, model: str = "base", limit: int | None = None) -> int:
    from .scoring.llm_judge import run_judge
    return run_judge(dataset, model, limit=limit)
