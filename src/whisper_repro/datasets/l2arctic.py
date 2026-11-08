"""L2-ARCTIC -- non-native English speech (Zhao et al., 2018).

Accented English from speakers with six L1 backgrounds, including Hindi.
Tests the paper's robustness claim on exactly the population it is most
likely to under-serve.

Distributed as a per-speaker zip after a request form, so like SLR54 the
download stays manual.

Expected after unpacking:
    <root>/<SPEAKER>/wav/arctic_a0001.wav
    <root>/<SPEAKER>/transcript/arctic_a0001.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..logging_utils import get_logger
from .audio_io import write_standardised
from .base import FetchAdapter, Sample

log = get_logger(__name__)

#: L1 background per speaker code, from the corpus documentation. Used to
#: slice results by first language -- the comparison that matters here.
L1_BY_SPEAKER = {
    "ABA": "Arabic", "SKA": "Arabic", "YBAA": "Arabic", "ZHAA": "Arabic",
    "BWC": "Mandarin", "LXC": "Mandarin", "NCC": "Mandarin", "TXHC": "Mandarin",
    "ASI": "Hindi", "RRBI": "Hindi", "SVBI": "Hindi", "TNI": "Hindi",
    "HJK": "Korean", "HKK": "Korean", "YDCK": "Korean", "YKWK": "Korean",
    "EBVS": "Spanish", "ERMS": "Spanish", "MBMPS": "Spanish", "NJS": "Spanish",
    "HQTV": "Vietnamese", "PNV": "Vietnamese", "THV": "Vietnamese", "TLV": "Vietnamese",
}


class L2ArcticAdapter(FetchAdapter):
    name = "l2arctic"
    language = "en"
    source = "L2-ARCTIC (manual download; see module docstring)"

    def _root(self) -> Path:
        root = self.options.get("source_dir")
        if not root:
            raise ValueError(
                "l2arctic needs source_dir pointing at the unpacked corpus "
                "-- set it in configs/l2arctic.yaml"
            )
        p = Path(root).expanduser()
        if not p.is_dir():
            raise FileNotFoundError(f"source_dir does not exist: {p}")
        return p

    def _iter_samples(self, limit: int | None) -> Iterator[Sample]:
        root = self._root()
        speakers = self.options.get("speakers")  # e.g. ["ASI", "RRBI"] for Hindi L1
        n = 0

        for spk_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            spk = spk_dir.name.upper()
            if speakers and spk not in {s.upper() for s in speakers}:
                continue
            wav_dir, txt_dir = spk_dir / "wav", spk_dir / "transcript"
            if not wav_dir.is_dir():
                continue

            for wav in sorted(wav_dir.glob("*.wav")):
                if limit is not None and n >= limit:
                    return
                txt = txt_dir / f"{wav.stem}.txt"
                if not txt.exists():
                    log.warning("no transcript for %s/%s", spk, wav.name)
                    continue
                reference = txt.read_text(encoding="utf-8").strip()
                if not reference:
                    continue

                sid = f"{spk}_{wav.stem}"
                dest = self.audio_dir / f"{sid}.flac"
                duration = write_standardised(wav, dest)
                n += 1
                yield Sample(
                    sample_id=sid,
                    audio_path=dest.name,
                    reference=reference,
                    language=self.language,
                    duration_s=round(duration, 3),
                    speaker_id=spk,
                    meta={
                        "corpus": "L2-ARCTIC",
                        "l1": L1_BY_SPEAKER.get(spk, "unknown"),
                        "prompt_id": wav.stem,
                    },
                )
