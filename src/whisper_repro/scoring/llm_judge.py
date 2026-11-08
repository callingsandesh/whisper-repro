"""LLM-as-judge: does the transcript still say what the speaker said?

Motivation (and the reason this stage exists at all): WER charges the same
price for every word error. In the feasibility run, "seized"->"seize",
"scurried"->"scurry" and "an"->"on" cost 13.04% WER on a sentence whose
meaning was completely intact -- the same arithmetic weight as turning
"give 15 mg" into "give 50 mg". WER measures surface overlap, not whether
communication survived.

STATUS: scaffolding. The structure is settled (resumable JSONL, one verdict
per utterance, rationale stored for inspection) but `_call_model` is a stub.
Fill it in week 9+, once the lectures have covered the methodology.

Design notes worth keeping when you implement it:
  * Store the rationale, not just the score. An unexaminable judge is not
    evidence -- and the rationales are what you compare against the human
    judgements from your native-speaker pool.
  * Keep the scale short (e.g. 1-4) and anchor each point in the prompt;
    fine-grained scales produce noise, not signal.
  * Randomise or fix reference/hypothesis order deliberately and say which
    in the report -- position bias in LLM judges is well documented.
  * Judge the normalised pair, so the judge is not reacting to casing and
    punctuation differences that WER already handles.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..datasets import read_manifest
from ..logging_utils import get_logger
from ..paths import ensure_dirs, hyp_path, judge_path
from .normalise import get_normalizer

log = get_logger(__name__)

PROMPT_TEMPLATE = """\
You are evaluating a speech-recognition transcript.

Reference (what was actually said):
{reference}

Transcript (what the system produced):
{hypothesis}

Rate how well the transcript preserves the meaning of the reference:
4 = meaning fully preserved (errors are cosmetic or inflectional)
3 = meaning largely preserved (minor detail lost, no misleading content)
2 = meaning partly changed (a listener would be misinformed about a detail)
1 = meaning substantially changed or lost

Reply with JSON only: {{"score": <1-4>, "rationale": "<one sentence>"}}
"""


def _call_model(prompt: str) -> dict:
    """Send the prompt to a judge model and return {"score", "rationale"}.

    Not yet implemented. Options to weigh in week 9:
      * Anthropic/OpenAI API -- strongest judges, costs money, needs a key.
      * A local model via Ollama -- free, reproducible, no data leaves the
        machine (relevant: the constructed dataset is volunteers' voices).
    """
    raise NotImplementedError(
        "LLM judge not implemented yet -- scheduled for week 9. "
        "Implement _call_model in scoring/llm_judge.py."
    )


def run_judge(dataset: str, model: str = "base", limit: int | None = None) -> int:
    """Judge each transcribed utterance. Resumable: appends to a JSONL and
    skips sample_ids already present."""
    ensure_dirs()
    samples = read_manifest(dataset, limit=limit)
    out = judge_path(dataset, model)

    done: set[str] = set()
    if out.exists():
        with out.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["sample_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        log.info("%d verdicts already recorded in %s", len(done), out.name)

    n = 0
    with out.open("a", encoding="utf-8") as f:
        for s in samples:
            if s.sample_id in done:
                continue
            hp = hyp_path(dataset, model, s.sample_id)
            if not (hp.exists() and hp.stat().st_size > 0):
                continue
            norm = get_normalizer(s.language)
            verdict = _call_model(PROMPT_TEMPLATE.format(
                reference=norm(s.reference),
                hypothesis=norm(hp.read_text(encoding="utf-8").strip()),
            ))
            record = {"sample_id": s.sample_id, "language": s.language, **verdict}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # survive a crash mid-run
            n += 1

    log.info("judged %d new utterances -> %s", n, out)
    return n
