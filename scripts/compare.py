#!/usr/bin/env python3
"""Compare checkpoints across datasets, from the per-utterance score CSVs.

Why not just read the summary JSONs: a pooled rate over a *subset* cannot be
derived from a pooled rate over the whole, so --intersect has to recompute
from per-utterance rows. Recomputing through the project's own metrics
functions also guarantees these numbers agree with `whisper-repro score`
rather than approximating it.

    python scripts/compare.py                         # everything scored
    python scripts/compare.py --intersect             # only utterances all models share
    python scripts/compare.py --models base large-v2 --intersect --scripts
    python scripts/compare.py --format csv > results.csv
    python scripts/compare.py --format latex          # paste into the report
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_repro.paths import RESULTS_DIR                      # noqa: E402
from whisper_repro.scoring.metrics import char_counts, word_counts  # noqa: E402

try:
    from whisper_repro.scoring.script_id import detect_script
except ImportError:                                    # script_id is optional
    detect_script = None

#: Figures reported by Radford et al. (ICML 2023), normalised WER %.
#: Fill these in FROM THE PAPER as you verify them -- leave None rather than
#: guessing. A wrong baseline is worse than a missing one.
PAPER = {
    ("librispeech_test_clean", "large-v2"): 2.7,
    ("librispeech_test_other", "large-v2"): None,
    ("tedlium3",               "large-v2"): None,
    ("fleurs_en",              "large-v2"): None,
    ("fleurs_ne",              "large-v2"): None,
}

FNAME = re.compile(r"^(?P<ds>.+)__(?P<model>.+)\.csv$")


def load() -> dict[str, dict[str, list[dict]]]:
    """{dataset: {model: [row, ...]}} from data/results/*.csv"""
    out: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for p in sorted(RESULTS_DIR.glob("*.csv")):
        m = FNAME.match(p.name)
        if not m:
            continue
        with p.open(encoding="utf-8", newline="") as f:
            out[m["ds"]][m["model"]] = list(csv.DictReader(f))
    return out


def pooled(rows: list[dict]) -> dict:
    """Recompute corpus-pooled WER/CER from the normalised columns.

    Pooled, not averaged: total errors over total reference tokens, which is
    the paper's convention. Averaging per utterance over-weights short ones.
    """
    we = wn = ce = cn = 0
    for r in rows:
        ref, hyp = r.get("reference_norm", ""), r.get("hypothesis_norm", "")
        w = word_counts(ref, hyp)
        c = char_counts(ref, hyp)
        we += w.substitutions + w.deletions + w.insertions
        wn += w.ref_length
        ce += c.substitutions + c.deletions + c.insertions
        cn += c.ref_length
    return {
        "n": len(rows),
        "words": wn,
        "wer": 100 * we / wn if wn else float("nan"),
        "cer": 100 * ce / cn if cn else float("nan"),
    }


def scripts_of(rows: list[dict]) -> str:
    if detect_script is None:
        return ""
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[detect_script(r.get("hypothesis_raw", ""))] += 1
    return " ".join(f"{k}:{v}" for k, v in
                    sorted(counts.items(), key=lambda kv: -kv[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to these checkpoints, in this order")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--intersect", action="store_true",
                    help="score only the sample_ids every selected model has")
    ap.add_argument("--scripts", action="store_true",
                    help="add a writing-system breakdown of the hypotheses")
    ap.add_argument("--format", choices=["table", "csv", "latex", "markdown"],
                    default="table")
    args = ap.parse_args()

    data = load()
    if not data:
        print(f"no score CSVs under {RESULTS_DIR}", file=sys.stderr)
        return 1
    if args.datasets:
        data = {k: v for k, v in data.items() if k in args.datasets}

    rows_out, warnings = [], []
    for ds in sorted(data):
        models = args.models or sorted(data[ds])
        present = [m for m in models if m in data[ds]]
        if not present:
            continue

        keep: set[str] | None = None
        if args.intersect and len(present) > 1:
            sets = [{r["sample_id"] for r in data[ds][m]} for m in present]
            keep = set.intersection(*sets)
            sizes = {m: len(s) for m, s in zip(present, sets)}
            if len(set(sizes.values())) > 1 or any(len(s) != len(keep) for s in sets):
                warnings.append(
                    f"{ds}: {sizes} scored, {len(keep)} shared -- comparing the shared set")

        for m in present:
            rs = data[ds][m]
            if keep is not None:
                rs = [r for r in rs if r["sample_id"] in keep]
            if not rs:
                continue
            st = pooled(rs)
            paper = PAPER.get((ds, m))
            rows_out.append({
                "dataset": ds, "model": m, "n": st["n"], "words": st["words"],
                "wer": st["wer"], "cer": st["cer"],
                "paper": paper,
                "delta": (st["wer"] - paper) if paper is not None else None,
                "scripts": scripts_of(rs) if args.scripts else "",
            })

    def fmt(v, nd=2):
        return "" if v is None else f"{v:.{nd}f}"

    if args.format == "csv":
        w = csv.DictWriter(sys.stdout, fieldnames=list(rows_out[0]))
        w.writeheader(); w.writerows(rows_out)
    elif args.format in ("latex", "markdown"):
        sep = " & " if args.format == "latex" else " | "
        end = " \\\\" if args.format == "latex" else ""
        hdr = ["Dataset", "Model", "Utts", "WER\\%" if args.format == "latex" else "WER%",
               "CER\\%" if args.format == "latex" else "CER%", "Paper", "$\\Delta$"]
        if args.format == "markdown":
            print("| " + " | ".join(hdr) + " |")
            print("|" + "---|" * len(hdr))
        for r in rows_out:
            cells = [r["dataset"].replace("_", r"\_" if args.format == "latex" else "_"),
                     r["model"], str(r["n"]), fmt(r["wer"]), fmt(r["cer"]),
                     fmt(r["paper"], 1), fmt(r["delta"])]
            line = sep.join(cells) + end
            print(("| " + line + " |") if args.format == "markdown" else line)
    else:
        head = f'{"dataset":<26}{"model":<10}{"utts":>6}{"words":>8}{"WER%":>9}{"CER%":>8}{"paper":>7}{"Δ":>8}'
        if args.scripts:
            head += "  scripts"
        print(head); print("-" * len(head))
        last = None
        for r in rows_out:
            if last and r["dataset"] != last:
                print()
            last = r["dataset"]
            line = (f'{r["dataset"]:<26}{r["model"]:<10}{r["n"]:>6}{r["words"]:>8}'
                    f'{r["wer"]:>9.2f}{r["cer"]:>8.2f}'
                    f'{fmt(r["paper"],1):>7}{fmt(r["delta"]):>8}')
            if args.scripts:
                line += "  " + r["scripts"]
            print(line)

    for w_ in warnings:
        print(f"\n[!] {w_}", file=sys.stderr)
    if not args.intersect and any(
            len({r["n"] for r in rows_out if r["dataset"] == ds}) > 1
            for ds in {r["dataset"] for r in rows_out}):
        print("\n[!] utterance counts differ between models on some datasets; "
              "re-run with --intersect for a like-for-like comparison",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
