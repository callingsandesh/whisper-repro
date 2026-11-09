#!/usr/bin/env python3
"""Aggregate data/judge/*.jsonl into a table for the report."""
import json, re, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from whisper_repro.paths import JUDGE_DIR

FN = re.compile(r"^(?P<ds>.+)__(?P<model>.+)\.jsonl$")
rows, judges, prompts = [], set(), set()

for p in sorted(JUDGE_DIR.glob("*.jsonl")):
    m = FN.match(p.name)
    if not m:
        continue
    recs = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not recs:
        continue
    judges.update(r.get("judge_model", "?") for r in recs)
    prompts.update(r.get("prompt_sha", "?") for r in recs)
    sc = [r["score"] for r in recs if isinstance(r.get("score"), int)]
    tl = sum(1 for r in recs if r.get("transliterated") is True)
    langs = defaultdict(int)
    for r in recs:
        if r.get("hypothesis_language"):
            langs[r["hypothesis_language"]] += 1
    dist = {s: sum(1 for x in sc if x == s) for s in (4, 3, 2, 1)}
    rows.append((m["ds"], m["model"], len(recs),
                 sum(sc)/len(sc) if sc else float("nan"),
                 sum(1 for x in sc if x >= 3), tl, dist,
                 ", ".join(f"{k}:{v}" for k, v in
                           sorted(langs.items(), key=lambda kv: -kv[1])[:4])))

if not rows:
    print(f"no verdicts under {JUDGE_DIR}"); raise SystemExit(1)

print(f'{"dataset":<24}{"model":<10}{"n":>5}{"mean":>7}{"n>=3":>6}{"translit":>9}  4/3/2/1')
print("-" * 82)
for ds, mdl, n, mean, ge3, tl, dist, _ in rows:
    print(f'{ds:<24}{mdl:<10}{n:>5}{mean:>7.2f}{ge3:>6}{tl:>9}  '
          f'{dist[4]}/{dist[3]}/{dist[2]}/{dist[1]}')
print("\n--- judge's language labels ---")
for ds, mdl, *_, langs in rows:
    if langs:
        print(f"  {ds} / {mdl}: {langs}")
print(f"\njudge model(s): {sorted(judges)}")
print(f"prompt sha(s) : {sorted(prompts)}   <- more than one means mixed prompts")
