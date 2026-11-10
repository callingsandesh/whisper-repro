# whisper-repro

A pipeline for reproducing the evaluation of **Whisper** — *Robust Speech
Recognition via Large-Scale Weak Supervision* (Radford et al., ICML 2023) —
and applying the same method to accented and low-resource speech.

COMP8240 Applications of Data Science, Macquarie University.
Sandesh Timalsina.

> Results and analysis are reported separately, in the project report (PDF).
> This document covers the software: how to install it, run it, configure it,
> and extend it.

---

## What it does

Whisper's training data and compute were never released, so this project
reproduces the paper's **evaluation**, not its training: run the released
checkpoints against benchmarks, compute word and character error rates under
the paper's own normalisation, then apply the identical method to corpora the
paper never touched.

Whisper itself is an **unmodified pip dependency**. This repo contains no copy
of OpenAI's code — a reimplementation's numbers would not be comparable to
published figures, because the decoding path is part of what is being
reproduced.

The pipeline runs in four stages, each independently resumable:

```
prepare  →  transcribe  →  score  →  judge
```

| Stage | Reads | Writes |
|---|---|---|
| `prepare` | a corpus (streamed or on disk) | `raw/<ds>/*.flac` + `manifests/<ds>.jsonl` |
| `transcribe` | manifest + audio | `hyps/<ds>/<model>/<sample_id>.txt` |
| `score` | manifest + hypotheses | `results/<ds>__<model>.csv` + `.summary.json` |
| `judge` | manifest + hypotheses | `judge/<ds>__<model>.jsonl` |

---

## Install

```bash
# 1. torch FIRST, matching your CUDA version — otherwise pip may resolve a
#    CPU-only wheel, which runs but ~40x slower with no error.
#    https://pytorch.org/get-started/locally/
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2. then the project
pip install -e .

# 3. check
whisper-repro datasets
```

**Prerequisites**

| | |
|---|---|
| Python | ≥ 3.10 |
| torch | installed first, CUDA build matching your driver |
| ffmpeg | `sudo apt install ffmpeg` — required by Whisper and by the AAC fallback |

`openai-whisper` and `jiwer` are **pinned exactly** in `pyproject.toml`. They
are part of the method — one produces the transcripts, the other computes the
error rates — so version drift would silently change every number the pipeline
reports. Everything else is a compatible range.

`requirements.txt` is a one-line pointer to `pyproject.toml`; dependencies are
edited in one place only.

### Running without installing

```bash
PYTHONPATH=src python -m whisper_repro.cli datasets
```

Useful when the editable install has gone stale — which happens if the repo is
moved, since `pip install -e .` records an absolute path. Re-run
`pip install -e .` from the new location to fix it properly.

---

## Commands

### `datasets`
```bash
whisper-repro datasets
```
Lists every registered adapter with its language, source, kind, and how many
utterances are currently prepared.

### `prepare` — stage 1
```bash
whisper-repro prepare --dataset <name> [--limit N] [--force] [--source-dir PATH]
```
Fetches or reads a corpus, converts every utterance to 16 kHz mono FLAC, and
writes the manifest.

| Flag | Effect |
|---|---|
| `--limit N` | cap at N utterances |
| `--force` | rebuild even if a manifest exists |
| `--source-dir` | override `source_dir` from `configs/<name>.yaml` |

Idempotent: if the manifest exists and holds at least `--limit` samples, it
does nothing. Resumable at the *dataset* level, not per utterance — an
interrupted prepare re-decodes from scratch, which is cheap because it is CPU
work with no model loaded.

### `transcribe` — stage 2
```bash
whisper-repro transcribe --dataset <name> --model <ckpt> [--limit N] [--force] [--detect-language]
```

| Flag | Effect |
|---|---|
| `--model` | `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` |
| `--force` | re-transcribe utterances that already have a hypothesis |
| `--detect-language` | let Whisper detect the language instead of using the manifest's tag |

Resumable **per utterance** — a hypothesis is done when its file exists and is
non-empty, so a killed run redoes only what is missing. A single failing
utterance is logged and skipped rather than ending the run.

Writes `_environment.json` beside the hypotheses, recording the model, device,
precision, GPU name, and the `openai-whisper` and torch versions, so any
result traces back to the machine that produced it.

### `score` — stage 3
```bash
whisper-repro score --dataset <name> --model <ckpt> [--limit N]
```
Computes WER and CER, raw and normalised, corpus-pooled following the paper's
convention. Writes a per-utterance CSV with reference and hypothesis in both
raw and normalised form, plus a summary JSON.

Deliberately **does not cache** — it recomputes every run. A metric is the
thing you iterate on, and a cache in front of one means a fix to
`metrics.py` silently fails to take effect.

### `judge` — stage 4
```bash
whisper-repro judge --dataset <name> --model <ckpt> [--limit N | --worst N]
```
Rates whether each transcript preserved the meaning of the reference, using an
LLM. See [LLM-as-judge](#llm-as-judge) for setup.

| Flag | Effect |
|---|---|
| `--limit N` | the first N utterances in corpus order |
| `--worst N` | the N highest-WER utterances, worst first (needs `score` to have run) |

Passing both is an error — they select on different axes. Resumable by
appending to a JSONL and skipping sample_ids already present.

### `run`
```bash
whisper-repro run --dataset <name> --model <ckpt> [--limit N] [--source-dir PATH]
```
`prepare` + `transcribe` + `score` in one go. Does not include `judge`.

### `inspect`
```bash
whisper-repro inspect --dataset <name> --model <ckpt> [--worst N]
```
Prints the worst-scoring utterances with their reference/hypothesis pairs.
The fastest way to find examples worth writing about.

### Global flags
```bash
whisper-repro -v <command> ...     # debug logging
```

---

## Datasets

| Name | Language | Source | Needs `source_dir`? |
|---|---|---|---|
| `librispeech_test_clean` | en | HuggingFace `openslr/librispeech_asr` | no |
| `librispeech_test_other` | en | HuggingFace `openslr/librispeech_asr` | no |
| `tedlium3` | en | HuggingFace `huggingworld/open-asr-leaderboard`, config `tedlium` | no |
| `fleurs_en` | en | HuggingFace `google/fleurs`, `en_us` | no |
| `fleurs_ne` | ne | HuggingFace `google/fleurs` | no |
| `openslr_nepali` | ne | <https://www.openslr.org/54/> | **yes** |
| `l2arctic` | en | <https://psi.engr.tamu.edu/l2-arctic-corpus/> | **yes** |
| `nepali_custom` | ne | own recordings | **yes** |
| `nepali_accented_en` | en | own recordings | **yes** |

Streaming adapters need no setup. The four marked **yes** read from a folder
on disk, because their distribution links change or sit behind a request form.

> TED-LIUM 3's canonical repository (`LIUM/tedlium`) is gated and
> script-based, so this adapter uses a benchmark mirror. Noted here because it
> is a property of the data source, not of the code.

### Configuring a source directory

```bash
cp configs/l2arctic.example.yaml configs/l2arctic.yaml   # then edit source_dir
```

`configs/*.yaml` is gitignored and `configs/*.example.yaml` is tracked, so
machine-specific paths never enter version control. Options from the YAML are
merged with anything passed on the command line, with CLI flags winning.

**OpenSLR SLR54** — download and unpack from the link above:
```
<source_dir>/utt_spk_text.tsv          utterance_id <TAB> speaker_id <TAB> transcript
<source_dir>/asr_nepali/data/...       .flac, nested by id prefix
```

**L2-ARCTIC** — request form, then unpack the per-speaker archives into one
directory. `source_dir` points at the **parent** of the speaker folders:
```
<source_dir>/ASI/wav/arctic_a0001.wav
<source_dir>/ASI/transcript/arctic_a0001.txt
```
`annotation/` and `textgrid/` are ignored. An optional `speakers:` list
restricts the run — the adapter carries a speaker→L1 table, so results can be
sliced by first language:
```yaml
source_dir: ~/corpora/l2arctic
speakers: [ASI, RRBI, SVBI, TNI]
```

**The constructed dataset** — recordings live outside the repo, in a
backed-up folder:
```
<source_dir>/
  transcripts.csv      # filename,reference[,sample_id,language,speaker_id,...]
  audio/               # any format ffmpeg can read
```
Required columns are `filename` and `reference`; `sample_id`, `language`,
`speaker_id` are optional, and any other column is carried through into the
sample's metadata. A missing or wrong extension is tolerated — the adapter
globs for a matching stem.

`transcripts.csv` is the annotation work and the irreplaceable artefact of
this project. It is parsed as RFC 4180, so commas inside a reference are fine
when quoted, but it **must** be saved as *CSV UTF-8* — some Excel builds write
the system codepage and silently turn every Devanagari character into `?`.

`nepali_custom` and `nepali_accented_en` are separate datasets rather than one
with a language column, because they answer different questions and pooling
them would give a blended rate that answers neither. They also take different
normalisers.

### Building the constructed dataset

The order matters: **prompts first, then recordings, then the CSV.** Writing
transcripts from audio after the fact is slower and less accurate than reading
from a prepared script.

**1. Write the prompts.** One sentence per row in a spreadsheet, 10–20 seconds
of speech each (roughly 25–30 words in Nepali, 30–40 in English). Short
four-second clips give Whisper too little context and inflate error rates for
reasons that have nothing to do with the speaker. Keep the two languages on
separate sheets — they become separate datasets.

Give every row its filename up front, so the recording and the transcript can
never drift apart:

```
ne_spk01_001    Nepali, speaker 1, utterance 1
en_spk01_001    English, same speaker
ne_spk02_001    a different speaker, the same prompts
```

Reusing the same prompt list across speakers is what makes per-speaker results
comparable.

**2. Record.** Any device; phone audio is realistic and acceptable. A quiet
room matters more than good equipment. Settings that work:

| | |
|---|---|
| Format | WAV, 16-bit PCM (lossless; soundfile reads it with no ffmpeg) |
| Channels | mono |
| Sample rate | 44.1 or 48 kHz — 48 kHz downsamples to 16 kHz at an exact 3:1 ratio |

The pipeline resamples to 16 kHz mono at `prepare` time, so record at the
device's native rate and let the code convert. Keep the originals.

**3. Split into one file per utterance.** In Audacity, label each utterance
(`Ctrl+B`), type its filename into the label, then **File → Export Audio →
Multiple Files**. Audacity names each file from its label. This is worth the
setup: forty manual exports is where an off-by-one creeps in and silently
pairs audio with the wrong reference.

Trim stray *speech* — "okay, number five", a false start, a comment between
takes — because Whisper will transcribe it and every word becomes an insertion
against a reference that does not contain it. Leave background noise, breath
and a second of silence at each end; that is real recording conditions, and
removing it makes the corpus less representative, not better.

**4. Write `transcripts.csv`.** One row per recording:

```csv
filename,reference,language,speaker_id,speech_type,device,prompt_id
ne_spk01_001.wav,सगरमाथा संसारको सबैभन्दा अग्लो हिमाल हो।,ne,spk01,read,phone,ne_001
en_spk01_001.wav,"The monsoon, which arrives in June, reshapes the valley.",en,spk01,read,phone,en_001
```

`filename` and `reference` are required; everything else is optional and any
extra column is carried into the sample's metadata, where it can be used to
slice results later. Save as **CSV UTF-8** — some Excel builds write the system
codepage and turn every Devanagari character into `?` without warning.

**The reference must match the audio.** If a speaker misreads a prompt, either
re-record it (preferred — it keeps the prompt set identical across speakers) or
change the reference to what they actually said. What you must never do is
leave a mismatch: the reference is ground truth, and a wrong one silently
inflates WER in a way no amount of analysis will recover.

**5. Verify before running the full set.**

```bash
whisper-repro prepare --dataset nepali_custom --source-dir <dir> --limit 2
head -2 data/manifests/nepali_custom.jsonl
```

Check the references in the manifest are the text you wrote, not `?` or mojibake,
and that `duration_s` looks right. Two utterances takes seconds and catches an
encoding problem before it reaches forty.

**6. Back it up.** The recordings and `transcripts.csv` exist nowhere else and
cannot be regenerated. They live outside this repo deliberately — keep them in
cloud storage, and commit the derived manifest so the annotation itself is
under version control.

**Consent.** Recordings are identifiable voice data. Get the speakers' agreement
to how the recordings will be used and shared before collecting, and keep the
audio out of any public repository.

### Adding a dataset

One module and one registry line; nothing else changes.

```python
# src/whisper_repro/datasets/mycorpus.py
class MyCorpusAdapter(FetchAdapter):      # or LocalAdapter
    name = "mycorpus"
    language = "en"
    source = "where this came from"

    def _iter_samples(self, limit):
        for ...:
            dest = self.audio_dir / f"{sid}.flac"
            duration = write_standardised(src, dest)     # 16 kHz mono
            yield Sample(sample_id=sid, audio_path=dest.name,
                         reference=text, language=self.language,
                         duration_s=round(duration, 3))
```

```python
# src/whisper_repro/datasets/registry.py
ADAPTERS = { ..., MyCorpusAdapter.name: MyCorpusAdapter }
```

The base class handles resume logic, manifest writing and logging. Subclass an
existing adapter when a corpus differs only by split or config — that is how
`librispeech_test_other` and `nepali_accented_en` are defined, in five lines
each.

---

## Configuration

Three separate path concepts:

| What | Where | Default |
|---|---|---|
| **output root** — everything the pipeline produces | `WHISPER_REPRO_DATA` env var | `<repo>/data` |
| **source dirs** — where a corpus is read from | `configs/<name>.yaml` or `--source-dir` | none |
| **credentials** — Bedrock keys for the judge | `.env` in the repo root | none |

```bash
export WHISPER_REPRO_DATA=/media/<user>/HDD/whisper-data
python -c "from whisper_repro.paths import DATA_ROOT; print(DATA_ROOT)"
```

`paths.py` is the only module that knows a directory string; six other modules
import from it and nothing hardcodes a path. Relocating the data is one
environment variable.

`REPO_ROOT` is derived from `paths.py`'s own location, so moving the repo needs
no config change — only a reinstall.

---

## LLM-as-judge

WER charges the same price for every word error: `seized`→`seize` costs the
same as turning "give 15 mg" into "give 50 mg". This stage asks a language
model whether the meaning survived, and stores a rationale alongside each
score so the verdicts are examinable rather than taken on trust.

Backend is **Amazon Bedrock** via the Converse API. Configuration is entirely
by environment variable, so no credential or model choice is ever committed:

```bash
# .env in the repo root — gitignored
AWS_BEARER_TOKEN_BEDROCK=<short-term Bedrock API key; expires within 12h>
AWS_REGION=<the region the key was generated in — short-term keys are region-locked>
BEDROCK_JUDGE_MODEL=<pinned inference-profile id, e.g. apac.anthropic....>
```

```bash
# diagnostics
python -c "from whisper_repro.scoring.llm_judge import check_config; check_config()"
python -c "from whisper_repro.scoring.llm_judge import list_models; list_models('claude')"
```

`check_config()` prints where each setting came from and then makes one real
API call. `list_models()` lists what the account can actually call — most
current models are reachable only through a cross-region inference profile,
whose id carries a region prefix.

`BEDROCK_JUDGE_MODEL` is required rather than defaulted: the judge model is
part of the experiment's definition, so it is pinned explicitly and recorded
in every verdict line alongside a hash of the prompt.

Each verdict carries the judge's own reading of what the transcript is written
in, whether it is a transliteration, the score, the rationale, and — from
`scoring/script_id.py` — a deterministic script identification. **The judge is
told the spoken language and nothing else**: it is not given the detected
script, so that the two identifications stay independent. Two measurements are
only a cross-check while neither has been fed the other's answer.

---

## Architecture

```
src/whisper_repro/
├── paths.py            where everything lives (override with WHISPER_REPRO_DATA)
├── config.py           per-dataset YAML, keeps machine paths out of git
├── pipeline.py         the four stages
├── cli.py              argument parsing and output formatting
├── logging_utils.py    consistent logging to console and data/run.log
├── datasets/
│   ├── base.py         Sample, the adapter contract, manifest I/O
│   ├── audio_io.py     decode → 16 kHz mono FLAC (ffmpeg fallback for m4a/aac)
│   ├── registry.py     name → adapter (the only place that knows them all)
│   └── <corpus>.py     one module per data source
├── models/
│   └── whisper_runner.py   the ONLY module importing openai-whisper
└── scoring/
    ├── normalise.py    English vs Basic normaliser, chosen by language
    ├── metrics.py      WER and CER, corpus-pooled and per-utterance
    ├── script_id.py    writing-system detection by Unicode block
    └── llm_judge.py    semantic fidelity via Amazon Bedrock
```

**One contract.** However a corpus arrives — streamed from HuggingFace,
unpacked from an archive, recorded on a phone — its adapter reduces it to
16 kHz mono audio plus one `Sample` per line of JSONL. Past that line nothing
downstream knows where the data came from, so `transcribe` and `score` are
written once and run on every dataset.

**The manifest is immutable input, not a progress ledger.** It is written
once, atomically, only after every audio file it references exists — so its
presence *means* "all of this audio is on disk at 16 kHz mono", and downstream
stages can trust it without checking. Stages never write status back to it:
status belongs to a *(dataset, model)* pair, which the directory tree
`hyps/<dataset>/<model>/` already encodes for free, and which a column in a
per-dataset file could not express.

**The filesystem is the checkpoint.** A unit of work is done when its output
file exists and is non-empty — no state file to desynchronise from reality.
Every write is temp-file-plus-rename (`os.replace`, atomic on POSIX), so an
interrupted write can never leave a truncated file that the resume check would
accept as complete. Delete an output and that one unit is redone.

**One module imports Whisper.** `models/whisper_runner.py` is the only place,
so the dependency is swappable and the rest of the codebase is testable
without a GPU.

**Normalisation is per-language.** `EnglishTextNormalizer` strips Devanagari
combining marks, so `normalise.py` dispatches on each sample's `language`
field. That is why `language` is required on `Sample`. WER is always reported
raw *and* normalised, because the gap between them is large and meaningful:
LibriSpeech references are uppercase and unpunctuated while Whisper emits
cased, punctuated text.

**CER alongside WER**, because word-level rates are unreliable for Devanagari,
where orthographic variation inflates word errors. The paper itself falls back
to CER for some languages.

**Scripts are detected from raw text**, before normalisation — normalising
first would corrupt the very property being measured.

---

## Folder structure

```
whisper-repro/
├── pyproject.toml          dependencies and the console-script entry point
├── requirements.txt        a pointer to pyproject.toml
├── configs/
│   ├── *.example.yaml      tracked templates
│   └── *.yaml              your machine's paths — gitignored
├── src/whisper_repro/      the package (see Architecture)
├── tests/
└── data/                   output root, unless WHISPER_REPRO_DATA says otherwise
    ├── raw/<dataset>/              prepared audio — gitignored
    ├── manifests/<dataset>.jsonl   committed
    ├── hyps/<dataset>/<model>/     one .txt per utterance + _environment.json
    ├── results/                    <dataset>__<model>.csv + .summary.json
    └── judge/                      <dataset>__<model>.jsonl
```

`data/raw/` is gitignored because it is either re-downloadable or backed up
elsewhere. Manifests and results are committed: manifests for the constructed
set *are* the annotation work, and results are small and expensive to
regenerate.

---

## Running on Colab

```python
from google.colab import drive; drive.mount('/content/drive')
%env WHISPER_REPRO_DATA=/content/drive/MyDrive/whisper-repro-data
```

`paths.py` reads that variable, so everything lands on Drive and a dropped
session costs nothing — re-run the cells and each stage skips what is already
done. Set it before importing the package: `DATA_ROOT` is resolved at import
time. See `colab_quickstart.ipynb`.

For a corpus that needs a `source_dir`, upload the **prepared** audio and
manifest rather than the source archive. `transcribe` reads only
`data/raw/<ds>/` and `data/manifests/<ds>.jsonl`, so the same utterances are
guaranteed across machines by construction rather than by assumption — and
16 kHz mono FLAC is a fraction of the size of the original corpus.

---

## Tests

```bash
pytest
```

`tests/test_metrics.py` pins a worked example (23 words, 3 substitutions,
13.04% WER; 100% unnormalised). If a refactor changes those, it is a
regression in the metric, not a new result.

---

## Troubleshooting

**`whisper-repro: command not found`** — the editable install records an
absolute path to the source. Moving the repo, or rebuilding the conda
environment, breaks it. Re-run `pip install -e .` from the current location,
or use `PYTHONPATH=src python -m whisper_repro.cli ...`.

**All logits are `NaN` on a GTX 16-series GPU** — a known hardware defect in
half precision on that generation. `whisper_runner.py` detects the GPU by name
and falls back to fp32 with a warning. Nothing to do, but note that fp32 is
slower and uses roughly double the VRAM.

**`Could not load libtorchcodec_core4.so`** — HuggingFace `datasets` decodes
audio through torchcodec, which breaks against several torch builds. Every
fetch adapter uses `Audio(decode=False)` and decodes with soundfile instead,
so this should not arise; if it does, uninstall torchcodec.

**`Format not recognised` on a `.m4a` file** — libsndfile has no AAC decoder.
`audio_io.load_any` falls back to an ffmpeg subprocess, so check ffmpeg is on
PATH.

**Process killed (SIGKILL) during a long run** — system RAM, not VRAM. Close
other applications; the pipeline itself is frugal, holding one utterance in
memory at a time.

**`adapter '<name>' produced no samples`** — the source directory is wrong, or
a HuggingFace repo/config pair does not match. Run with `-v` to see what the
adapter looked at.

**`judge` fails with `AccessDeniedException`** — the model id is not callable
by the account. Run `list_models('claude')`; most current models need a
cross-region inference profile id with a region prefix (`apac.`, `us.`, `eu.`),
and model access must be granted in the Bedrock console.

---

## Citation

```bibtex
@InProceedings{pmlr-v202-radford23a,
  title     = {Robust Speech Recognition via Large-Scale Weak Supervision},
  author    = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg
               and McLeavey, Christine and Sutskever, Ilya},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages     = {28492--28518},
  year      = {2023},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  publisher = {PMLR}
}
```

Corpora: LibriSpeech (Panayotov et al., 2015), TED-LIUM 3 (Hernandez et al.,
2018), FLEURS (Conneau et al., 2023), OpenSLR SLR54 (Kjartansson et al., 2018),
L2-ARCTIC (Zhao et al., 2018 — CC BY-NC 4.0).
