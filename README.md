# whisper-repro

Reproducing the evaluation of **Whisper** — *Robust Speech Recognition via
Large-Scale Weak Supervision* (Radford et al., ICML 2023) — and extending it
to accented and low-resource speech.

COMP8240 Applications of Data Science, Macquarie University.

## Scope

Whisper's training data and compute are not public, so this project
reproduces the paper's **evaluation**, not its training: run the released
checkpoints against the paper's benchmarks and check the reported word error
rates, then apply the same method to corpora the paper never touched.

Three categories of data, one uniform pipeline:

| Category | Datasets |
|---|---|
| From the paper | LibriSpeech `test-clean`, `test-other` |
| New existing | L2-ARCTIC (accented English), OpenSLR SLR54 (Nepali) |
| Newly constructed | Nepali speech and Nepali-accented English, recorded and transcribed by the author |

## Install

```bash
# 1. Install torch FIRST, matching your CUDA version — otherwise pip may
#    resolve a CPU-only wheel. See https://pytorch.org/get-started/locally/
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2. Then the project
pip install -e .
```

`ffmpeg` must be on PATH (Whisper needs it): `sudo apt install ffmpeg`.

Whisper itself is a **dependency**, installed from PyPI and used unmodified.
The reproduction's numbers are only comparable to the paper's if the decoding
path is theirs — so this repo contains no copy of their code.

## Use

```bash
whisper-repro datasets                                          # what exists, what's prepared
whisper-repro run --dataset librispeech_test_clean --model base --limit 50
```

Or stage by stage — each is independently resumable:

```bash
whisper-repro prepare    --dataset librispeech_test_clean --limit 50
whisper-repro transcribe --dataset librispeech_test_clean --model base
whisper-repro score      --dataset librispeech_test_clean --model base
whisper-repro inspect    --dataset librispeech_test_clean --model base --worst 10
```

`inspect` sorts by error rate and prints the worst utterances with their
reference/hypothesis pairs — the fastest way to find examples worth writing
about.

### The constructed dataset

```bash
cp configs/nepali_custom.example.yaml configs/nepali_custom.yaml
# edit source_dir to point at your recordings, then:
whisper-repro prepare --dataset nepali_custom
```

Your recordings live **outside this repo**, in a backed-up folder:

```
<source_dir>/
  transcripts.csv      # filename,reference[,sample_id,language,speaker_id,...]
  audio/               # whatever your recorder produced; any format
```

`transcripts.csv` is the annotation work and the irreplaceable artefact of
this project — keep it backed up. `prepare` only ever reads from here.

## Architecture

```
src/whisper_repro/
├── paths.py            where everything lives (override with WHISPER_REPRO_DATA)
├── config.py           per-dataset YAML, keeps machine paths out of git
├── pipeline.py         the four stages
├── cli.py              argument parsing and output formatting
├── datasets/
│   ├── base.py         Sample + the adapter contract
│   ├── audio_io.py     decode → 16 kHz mono FLAC
│   ├── registry.py     name → adapter (the only place that knows them all)
│   └── <corpus>.py     one module per data source
├── models/
│   └── whisper_runner.py   the ONLY module importing openai-whisper
└── scoring/
    ├── normalise.py    English vs Basic normaliser, chosen by language
    ├── metrics.py      WER and CER, corpus-pooled and per-utterance
    └── llm_judge.py    semantic-fidelity evaluation (week 9+)
```

**One contract.** However a corpus arrives, its adapter reduces it to
16 kHz mono audio in `data/raw/<dataset>/` plus a `Sample` per line in
`data/manifests/<dataset>.jsonl`. Nothing downstream knows which adapter
produced it, so adding a corpus is one new module and one registry line.

**The filesystem is the checkpoint.** A unit of work is done when its output
file exists and is non-empty. Re-run any stage and it resumes exactly where a
crash or a dropped Colab session left off — no state files to desynchronise.
Every write is a temp-file-plus-rename, so an interrupted write can never
leave a truncated file that the resume check would accept.

**Normalisation is per-language.** Whisper's `EnglishTextNormalizer` would
mangle Devanagari, so `scoring/normalise.py` dispatches on each sample's
`language` field. This is why `language` is required on `Sample`, and why WER
is always reported both raw and normalised: on LibriSpeech the gap is
100% → 13% on a single utterance, because references are uppercase and
unpunctuated while Whisper emits cased, punctuated text.

**CER alongside WER.** Word-level rates are unreliable for Devanagari, where
orthographic variation inflates word errors; the paper itself falls back to
CER for some languages.

## Reproduction target

Feasibility run (Colab T4, `openai-whisper` 20250625, `large-v2`, 50
utterances of `test-clean`): **3.01%** normalised WER against the paper's
reported **2.7%** — consistent, with the gap attributable to the subset size.

## Tests

```bash
pytest
```

`tests/test_metrics.py` pins the worked example used in the project proposal
(23 words, 3 substitutions, 13.04% WER; 100% unnormalised). If a refactor
changes those, it is a regression in the metric.

## Layout of `data/`

```
data/
├── raw/<dataset>/          prepared audio — gitignored (cache or backed up elsewhere)
├── manifests/<dataset>.jsonl   committed; for the constructed set this IS the annotation
├── hyps/<dataset>/<model>/     one .txt per utterance + _environment.json
├── results/                CSV per utterance + summary JSON
└── judge/                  LLM verdicts (JSONL)
```
