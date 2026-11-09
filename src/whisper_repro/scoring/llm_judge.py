"""LLM-as-judge: does the transcript still say what the speaker said?

Motivation (and the reason this stage exists at all): WER charges the same
price for every word error. In the feasibility run, "seized"->"seize",
"scurried"->"scurry" and "an"->"on" cost 13.04% WER on a sentence whose
meaning was completely intact -- the same arithmetic weight as turning
"give 15 mg" into "give 50 mg". WER measures surface overlap, not whether
communication survived.

For the Nepali set the point is sharper still: 14 of 15 utterances came back
as recognisable Nepali in the wrong script. WER scores a correct
transliteration and an unrelated hallucination identically at 100%. A judge
reading meaning can separate them; character arithmetic cannot.

BACKEND: Amazon Bedrock via the Converse API.

Configuration is entirely by environment variable, so no credential or model
choice is ever committed:

    AWS_BEARER_TOKEN_BEDROCK   short-term Bedrock API key (expires <=12h)
    AWS_REGION                 must be the region the key was generated in
    BEDROCK_JUDGE_MODEL        pinned model id, e.g. apac.anthropic.claude-...

BEDROCK_JUDGE_MODEL is deliberately required rather than defaulted. The judge
model is part of the experiment's definition: a verdict is meaningless
without knowing who produced it, so it gets recorded in every verdict line
alongside a hash of the prompt.

Design notes:
  * Store the rationale, not just the score. An unexaminable judge is not
    evidence -- and the rationales are what you compare against the human
    judgements from your native-speaker pool.
  * The scale is 1-4 with each point anchored in the prompt; fine-grained
    scales produce noise, not signal.
  * Reference comes first, hypothesis second, fixed, in every call. Position
    bias in LLM judges is well documented -- say so in the report.
  * Judge the RAW pair, not the normalised one. BasicTextNormalizer strips
    Unicode Mn marks, which in Devanagari are the vowel signs: the normalised
    reference is mangled text, and asking a judge whether mangled text means
    the same as something else is incoherent. Surface-form noise is handled
    by instructing the judge to ignore casing and punctuation, which a
    language model does reliably and a character-level metric cannot.
  * temperature=0 so a re-run of the same pair gives the same verdict.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

from ..datasets import read_manifest
from ..logging_utils import get_logger
from ..paths import REPO_ROOT, ensure_dirs, hyp_path, judge_path, results_path
from .script_id import EXPECTED, detect_script, script_mismatch

log = get_logger(__name__)

PROMPT_TEMPLATE = """\
You are evaluating a speech-recognition transcript.

The audio was spoken in {language_name}. The reference is what the speaker
actually said; the transcript is what the system produced from the audio.

Reference:
{reference}

Transcript:
{hypothesis}

Judge whether the meaning survives, not whether the characters match.
Disregard differences of casing, punctuation and digit formatting entirely --
references in some corpora are uppercase and unpunctuated, and those
differences are not recognition errors.

The transcript may be in a different script or alphabet from the reference --
read it phonetically and ask whether it renders the same words. A faithful
transliteration preserves the meaning fully; unrelated text does not,
however fluent it reads.

First identify what the transcript is actually written in. Be specific about
transliteration: "romanised Nepali" and "Nepali in Perso-Arabic script" are
different from "English" and "Urdu", and the distinction is the point of this
evaluation. If the transcript is unrelated to the reference, name its own
language.

Then rate how well the transcript preserves the meaning of the reference:
4 = meaning fully preserved (errors are cosmetic, inflectional, or a
    faithful transliteration into another script)
3 = meaning largely preserved (minor detail lost, no misleading content)
2 = meaning partly changed (a listener would be misinformed about a detail)
1 = meaning substantially changed or lost

Reply with JSON only:
{{"hypothesis_language": "<what the transcript is written in>",
  "transliterated": <true if it renders the reference's words in another
                     script or alphabet, false otherwise>,
  "score": <1-4>,
  "rationale": "<one sentence>"}}
"""

#: Identifies which prompt produced a verdict. Edit the template above and
#: this changes, so mixed-prompt result files are detectable after the fact.
PROMPT_SHA = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:12]

_LANGUAGE_NAMES = {"ne": "Nepali", "en": "English"}

_RETRYABLE = ("Throttling", "TooManyRequests", "ServiceUnavailable",
              "ModelNotReady", "InternalServer")
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def load_env(path: str | Path | None = None) -> Path | None:
    """Load KEY=VALUE lines from a .env file into os.environ.

    Looks in the repo root first, then the current directory. Anchoring to
    REPO_ROOT matters: a relative ".env" resolves against the shell's cwd, so
    running the CLI from anywhere but the repo root silently found nothing.

    Dependency-free and non-overriding -- a value already set in the shell
    wins, so `AWS_REGION=us-east-1 whisper-repro judge ...` behaves as
    expected. Returns the file it used, or None.
    """
    candidates = [Path(path)] if path else [REPO_ROOT / ".env", Path.cwd() / ".env"]
    p = next((c for c in candidates if c.is_file()), None)
    if p is None:
        log.warning(".env not found (looked in %s)",
                    " and ".join(str(c.parent) for c in candidates))
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
    log.info("loaded %s", p)
    return p


def judge_model_id() -> str:
    mid = os.environ.get("BEDROCK_JUDGE_MODEL", "").strip()
    if not mid:
        raise RuntimeError(
            "BEDROCK_JUDGE_MODEL is not set. Pin the judge model explicitly -- "
            "a verdict without a named judge is not evidence.\n"
            "List what your account can call:\n"
            "  aws bedrock list-inference-profiles --region $AWS_REGION \\\n"
            "    --query 'inferenceProfileSummaries[].inferenceProfileId' --output table"
        )
    return mid


@lru_cache(maxsize=1)
def _client():
    """One cached bedrock-runtime client. Built lazily so that importing this
    module never requires credentials -- `whisper-repro score` must work on a
    machine that has no AWS access at all."""
    try:
        import boto3
    except ImportError as e:
        raise RuntimeError("boto3 is not installed. pip install boto3") from e

    region = (os.environ.get("AWS_REGION")
              or os.environ.get("AWS_DEFAULT_REGION", "")).strip()
    if not region:
        raise RuntimeError(
            "AWS_REGION is not set. A short-term Bedrock API key only works in "
            "the region it was generated in -- set that region explicitly."
        )
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip():
        raise RuntimeError(
            "AWS_BEARER_TOKEN_BEDROCK is not set. Generate a short-term key in "
            "the Bedrock console (API keys -> Short-term) and put it in .env."
        )
    log.info("bedrock judge: %s in %s", judge_model_id(), region)
    return boto3.client("bedrock-runtime", region_name=region)


# --------------------------------------------------------------------------
# The judge call
# --------------------------------------------------------------------------

def _parse_verdict(text: str) -> dict:
    """Pull {"score", "rationale"} out of the model's reply.

    Tolerates markdown fences and surrounding prose, because the alternative
    is discarding an otherwise good verdict over formatting.
    """
    m = _JSON_BLOCK.search(text)
    if not m:
        raise ValueError(f"no JSON object in judge reply: {text[:200]!r}")
    d = json.loads(m.group(0))
    score = int(d["score"])
    if not 1 <= score <= 4:
        raise ValueError(f"score {score} outside the 1-4 scale")
    return {
        "score": score,
        # The judge's own reading of what the output is written in. Recorded
        # rather than trusted: off-the-shelf language ID mislabels
        # transliterated text with high confidence, so this field is evidence
        # to review against the rationale, not a measurement.
        "hypothesis_language": str(d.get("hypothesis_language", "")).strip(),
        "transliterated": bool(d["transliterated"]) if "transliterated" in d else None,
        "rationale": str(d.get("rationale", "")).strip(),
    }


def _call_model(prompt: str, max_attempts: int = 4) -> dict:
    """Send the prompt to Bedrock and return {"score", "rationale"}."""
    client = _client()
    model_id = judge_model_id()
    last: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 300, "temperature": 0.0},
            )
            return _parse_verdict(resp["output"]["message"]["content"][0]["text"])

        except Exception as e:  # noqa: BLE001 -- classified below
            last = e
            name = type(e).__name__
            if any(k in name for k in _RETRYABLE) and attempt < max_attempts:
                wait = 2 ** attempt
                log.warning("%s -- retrying in %ds (attempt %d/%d)",
                            name, wait, attempt, max_attempts)
                time.sleep(wait)
                continue
            if "ExpiredToken" in name or "UnrecognizedClient" in name:
                raise RuntimeError(
                    "Bedrock key rejected -- a short-term key lasts at most 12 "
                    "hours. Generate a new one and re-run; verdicts already in "
                    "the JSONL are kept and skipped."
                ) from e
            raise

    raise RuntimeError(f"judge call failed after {max_attempts} attempts") from last


# --------------------------------------------------------------------------
# Selecting what to judge
# --------------------------------------------------------------------------

def worst_sample_ids(dataset: str, model: str, n: int) -> list[str]:
    """The n highest-WER sample_ids, worst first, read from the score CSV.

    Judging the tail is the point for the paper's English sets: WER
    over-penalising intact meaning is a claim about the hard cases, and the
    easy 90% of LibriSpeech has nothing to say about it. `--limit` cannot do
    this -- it takes the first n lines of the manifest, which is corpus order
    and tells you nothing.
    """
    import csv

    path = results_path(dataset, model)
    if not path.exists():
        raise FileNotFoundError(
            f"--worst needs scores first. Run: "
            f"whisper-repro score --dataset {dataset} --model {model}"
        )
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "wer_norm" not in rows[0]:
        raise ValueError(f"{path} has no wer_norm column -- re-run score")

    def wer(r: dict) -> float:
        try:
            return float(r.get("wer_norm") or 0.0)
        except ValueError:
            return 0.0

    rows.sort(key=wer, reverse=True)
    return [r["sample_id"] for r in rows[:n]]


# --------------------------------------------------------------------------
# Stage 4 -- judge
# --------------------------------------------------------------------------

def run_judge(dataset: str, model: str = "base", limit: int | None = None,
              worst: int | None = None) -> int:
    """Judge transcribed utterances. Resumable: appends to a JSONL and skips
    sample_ids already present.

    `worst=n` judges the n highest-WER utterances, worst first, so an
    interrupted run has still spent its calls on the informative cases.
    """
    ensure_dirs()
    load_env()

    if worst is not None and limit is not None:
        raise ValueError(
            "--worst and --limit select differently (worst WER vs first in "
            "corpus order). Pass one or the other, not both."
        )

    if worst is not None:
        wanted = worst_sample_ids(dataset, model, worst)
        by_id = {s.sample_id: s for s in read_manifest(dataset)}
        samples = [by_id[i] for i in wanted if i in by_id]
        log.info("judging the %d worst-WER utterances of %s/%s",
                 len(samples), dataset, model)
    else:
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

    judge_id = judge_model_id()  # fail fast, before any API call
    n, failed = 0, 0

    with out.open("a", encoding="utf-8") as f:
        for s in samples:
            if s.sample_id in done:
                continue
            hp = hyp_path(dataset, model, s.sample_id)
            if not (hp.exists() and hp.stat().st_size > 0):
                continue

            raw_hyp = hp.read_text(encoding="utf-8").strip()
            # Scripts are read off the RAW text: the normaliser mangles
            # Devanagari, which would corrupt the very thing being measured.
            # Deliberately NOT passed to the judge -- the judge reports the
            # script independently, and two measurements are only a
            # cross-check while neither is fed the other's answer.
            ref_script = detect_script(s.reference)
            hyp_script = detect_script(raw_hyp)
            try:
                verdict = _call_model(PROMPT_TEMPLATE.format(
                    reference=s.reference,
                    hypothesis=raw_hyp,
                    language_name=_LANGUAGE_NAMES.get(s.language, s.language),
                ))
            except RuntimeError:
                raise  # credentials or exhausted retries: stop, don't burn the run
            except Exception:
                # One unparseable verdict must not kill a long run. It stays
                # unrecorded and the next invocation retries it.
                log.exception("%s: judging failed -- skipping", s.sample_id)
                failed += 1
                continue

            f.write(json.dumps({
                "sample_id": s.sample_id,
                "language": s.language,
                "asr_model": model,
                "judge_model": judge_id,
                "prompt_sha": PROMPT_SHA,
                "ref_script": ref_script,
                "hyp_script": hyp_script,
                "expected_script": EXPECTED.get(s.language, ""),
                "script_mismatch": script_mismatch(s.language, raw_hyp),
                **verdict,
            }, ensure_ascii=False) + "\n")
            f.flush()  # survive a crash mid-run
            n += 1

    if failed:
        log.warning("%d utterances could not be judged -- re-run to retry", failed)
    log.info("judged %d new utterances -> %s", n, out)
    return n


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def check_config() -> None:
    """Print where each piece of configuration came from, then make one real
    API call. Run this before a long judging run:

        python -c "from whisper_repro.scoring.llm_judge import check_config; check_config()"
    """
    used = load_env()
    print(f".env            : {used or 'NOT FOUND'}")
    tok = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
    print(f"bearer token    : {'set (%d chars, ends %s)' % (len(tok), tok[-4:]) if tok else 'MISSING'}")
    print(f"AWS_REGION      : {os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION') or 'MISSING'}")
    print(f"judge model     : {os.environ.get('BEDROCK_JUDGE_MODEL') or 'MISSING'}")
    try:
        import boto3
        print(f"boto3           : {boto3.__version__}")
    except ImportError:
        print("boto3           : NOT INSTALLED  ->  pip install boto3")
        return
    print("\ncalling Bedrock ...")
    try:
        v = _call_model(PROMPT_TEMPLATE.format(
            reference="the bus leaves at six", hypothesis="the bus leaves at six",
            language_name="English",
        ))
        print(f"OK -> {v}")
    except Exception as e:
        print(f"FAILED -> {type(e).__name__}: {e}")


def list_models(contains: str = "") -> None:
    """Print the models and inference profiles this account can actually call.

        python -c "from whisper_repro.scoring.llm_judge import list_models; list_models('claude')"

    Bedrock reports two separate catalogues and the distinction matters:
    most current Anthropic models are reachable only through a cross-region
    *inference profile*, whose id carries a region-group prefix (apac., us.,
    eu.). A bare model id that appears under foundation models can still be
    refused for on-demand use -- if so, call the profile instead.
    """
    import boto3
    load_env()
    region = (os.environ.get("AWS_REGION")
              or os.environ.get("AWS_DEFAULT_REGION", "")).strip()
    if not region:
        print("AWS_REGION is not set"); return
    bedrock = boto3.client("bedrock", region_name=region)
    needle = contains.lower()

    print(f"=== inference profiles in {region} (use these ids) ===")
    try:
        profiles = bedrock.list_inference_profiles()["inferenceProfileSummaries"]
        hits = [p for p in profiles if needle in p["inferenceProfileId"].lower()]
        for p in hits or profiles:
            print(f"  {p['inferenceProfileId']}")
        if not profiles:
            print("  (none)")
    except Exception as e:
        print(f"  could not list: {type(e).__name__}: {e}")

    print(f"\n=== foundation models in {region} supporting on-demand ===")
    try:
        models = bedrock.list_foundation_models(byOutputModality="TEXT")["modelSummaries"]
        for m in models:
            if needle and needle not in m["modelId"].lower():
                continue
            if "ON_DEMAND" in m.get("inferenceTypesSupported", []):
                print(f"  {m['modelId']}")
    except Exception as e:
        print(f"  could not list: {type(e).__name__}: {e}")
        print("  (a Bedrock API key may not cover control-plane calls -- "
              "read the ids off the console instead:")
        print("   Bedrock -> Model catalog, and Bedrock -> Cross-region inference)")