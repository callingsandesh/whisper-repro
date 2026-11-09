"""Command-line entry point.

    whisper-repro datasets
    whisper-repro prepare    --dataset librispeech_test_clean --limit 50
    whisper-repro transcribe --dataset librispeech_test_clean --model base
    whisper-repro score      --dataset librispeech_test_clean --model base
    whisper-repro run        --dataset librispeech_test_clean --model base --limit 50
    whisper-repro inspect    --dataset librispeech_test_clean --model base --worst 10

Also runnable without installing: python -m whisper_repro.cli ...
"""

from __future__ import annotations

import argparse
import sys

from . import pipeline
from .datasets import available, get_adapter
from .logging_utils import configure, get_logger

log = get_logger(__name__)


def _add_common(p: argparse.ArgumentParser, model: bool = True) -> None:
    p.add_argument("--dataset", required=True, choices=available(),
                   help="which corpus to act on")
    if model:
        p.add_argument("--model", default="base",
                       help="whisper checkpoint: tiny, base, small, medium, large-v2, large-v3")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of utterances (handy for smoke tests)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="whisper-repro", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("datasets", help="list adapters and how much is prepared")

    sp = sub.add_parser("prepare", help="stage 1: fetch/convert audio + write manifest")
    _add_common(sp, model=False)
    sp.add_argument("--force", action="store_true", help="rebuild even if present")
    sp.add_argument("--source-dir", default=None,
                    help="override the source path from configs/<dataset>.yaml")

    st = sub.add_parser("transcribe", help="stage 2: run whisper over prepared audio")
    _add_common(st)
    st.add_argument("--force", action="store_true", help="re-transcribe existing")
    st.add_argument("--detect-language", action="store_true",
                    help="let whisper detect the language instead of using the manifest's")

    ss = sub.add_parser("score", help="stage 3: WER/CER, raw and normalised")
    _add_common(ss)

    sj = sub.add_parser("judge", help="stage 4: LLM-as-judge")
    _add_common(sj)
    sj.add_argument("--worst", type=int, default=None,
                    help="judge the N highest-WER utterances instead of the first N")

    sr = sub.add_parser("run", help="prepare + transcribe + score in one go")
    _add_common(sr)
    sr.add_argument("--source-dir", default=None)

    si = sub.add_parser("inspect", help="show the worst-scoring utterances")
    _add_common(si)
    si.add_argument("--worst", type=int, default=10, help="how many to show")
    return p


def _print_summary(s: dict) -> None:
    env = s.get("environment") or {}
    print("\n" + "=" * 62)
    print(f"  {s['dataset']}  /  {s['model']}")
    print("=" * 62)
    print(f"  utterances        : {s['utterances']}"
          + (f"  ({s['missing_hypotheses']} missing)" if s["missing_hypotheses"] else ""))
    print(f"  audio             : {s['total_audio_s'] / 60:.1f} min")
    print(f"  WER (raw)         : {s['wer_raw_pct']:.2f}%")
    print(f"  WER (normalised)  : {s['wer_norm_pct']:.2f}%")
    print(f"  CER (normalised)  : {s['cer_norm_pct']:.2f}%")
    print(f"  S / D / I         : {s['substitutions']} / {s['deletions']} / {s['insertions']}"
          f"  over {s['ref_words']} reference words")
    if env:
        print(f"  environment       : whisper {env.get('whisper_version')}, "
              f"torch {env.get('torch_version')}, {env.get('device')}"
              + (f" ({env.get('gpu_name')})" if env.get("gpu_name") else ""))
    print("=" * 62)
    print("  Compare WER (normalised) against the paper's reported figure.\n")


def _inspect(dataset: str, model: str, worst: int, limit: int | None) -> None:
    import csv
    from .paths import results_path

    path = results_path(dataset, model)
    if not path.exists():
        sys.exit(f"no results at {path} -- run the score stage first")
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["wer_norm"]), reverse=True)

    print(f"\nWorst {min(worst, len(rows))} utterances by normalised WER:\n")
    for r in rows[:worst]:
        print(f"[{r['sample_id']}]  WER {float(r['wer_norm']):6.2f}%   "
              f"({r['sub']}S {r['del']}D {r['ins']}I over {r['n_ref_words']} words)")
        print(f"   ref: {r['reference_norm']}")
        print(f"   hyp: {r['hypothesis_norm']}\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure(verbose=args.verbose)

    try:
        if args.command == "datasets":
            print(f"\n{'name':<28} {'lang':<5} {'kind':<14} {'prepared':>8}  source")
            print("-" * 100)
            for name in available():
                d = get_adapter(name).describe()
                print(f"{d['name']:<28} {d['language']:<5} {d['kind']:<14} "
                      f"{d['prepared']:>8}  {d['source']}")
            print()

        elif args.command == "prepare":
            pipeline.prepare(args.dataset, limit=args.limit, force=args.force,
                             source_dir=args.source_dir)

        elif args.command == "transcribe":
            pipeline.transcribe(args.dataset, model=args.model, limit=args.limit,
                                force=args.force, detect_language=args.detect_language)

        elif args.command == "score":
            _print_summary(pipeline.score(args.dataset, model=args.model, limit=args.limit))

        elif args.command == "judge":
            pipeline.judge(args.dataset, model=args.model, limit=args.limit, worst=args.worst)

        elif args.command == "run":
            pipeline.prepare(args.dataset, limit=args.limit, source_dir=args.source_dir)
            pipeline.transcribe(args.dataset, model=args.model, limit=args.limit)
            _print_summary(pipeline.score(args.dataset, model=args.model, limit=args.limit))

        elif args.command == "inspect":
            _inspect(args.dataset, args.model, args.worst, args.limit)

    except KeyboardInterrupt:
        log.warning("interrupted -- progress is on disk; re-run to resume")
        return 130
    except (FileNotFoundError, ValueError, KeyError, RuntimeError, NotImplementedError) as e:
        # Expected, actionable failures: show the message, not a traceback.
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
