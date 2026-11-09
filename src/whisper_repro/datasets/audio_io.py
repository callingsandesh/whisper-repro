"""Audio decoding and writing, shared by every adapter.

One rule, enforced here rather than in each adapter: everything lands on
disk as 16 kHz mono FLAC. Whisper consumes 16 kHz mono, so resampling once
at prepare time means the transcribe stage never has to. FLAC because it is
lossless and roughly half the size of WAV; soundfile reads it transparently.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from ..logging_utils import get_logger
from .base import TARGET_SR

log = get_logger(__name__)


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim > 1 else audio


def resample(audio: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample to target rate.

    Prefers soxr (fast, high quality), falls back to scipy, then to naive
    linear interpolation so the pipeline still runs on a bare environment.
    """
    if sr == target_sr:
        return audio
    try:
        import soxr
        return soxr.resample(audio, sr, target_sr)
    except ImportError:
        pass
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        return resample_poly(audio, target_sr // g, sr // g)
    except ImportError:
        log.warning("no soxr/scipy; using linear interpolation (install soxr for quality)")
        n_out = int(round(len(audio) * target_sr / sr))
        return np.interp(
            np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio
        ).astype(np.float32)


def _load_via_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """Decode with ffmpeg, for formats libsndfile cannot read.

    libsndfile handles wav/flac/ogg natively but has no AAC decoder, so .m4a
    -- what most phone recorders produce -- fails with "Format not recognised".
    ffmpeg decodes it and writes raw float32 to stdout, already downmixed and
    resampled, so the mono/resample steps downstream become no-ops.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(TARGET_SR),
        "-loglevel", "error", "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError:
        raise RuntimeError(
            f"cannot decode {path.name}: soundfile does not support this format "
            f"and ffmpeg is not installed. Install it with "
            f"'sudo apt install ffmpeg' (or 'sudo dnf install ffmpeg')."
        ) from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed on {path}: {e.stderr.decode(errors='replace').strip()}"
        ) from None
    audio = np.frombuffer(out, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg produced no audio for {path}")
    return audio, TARGET_SR


def load_any(src: Path | bytes) -> tuple[np.ndarray, int]:
    """Decode a file path or raw bytes into (float32 samples, sample rate).

    Tries libsndfile first -- fast, no subprocess -- and falls back to ffmpeg
    for anything it cannot read (.m4a/AAC above all).
    """
    if isinstance(src, bytes):
        return sf.read(io.BytesIO(src), dtype="float32")
    try:
        return sf.read(str(src), dtype="float32")
    except Exception:
        log.debug("soundfile could not read %s -- trying ffmpeg", src)
        return _load_via_ffmpeg(Path(src))


def write_standardised(src: Path | bytes, dest: Path) -> float:
    """Decode, convert to 16 kHz mono, write FLAC. Returns duration in seconds.

    Written via a temp file and renamed, so an interrupted run never leaves a
    truncated audio file that the resume check would accept as complete.
    """
    audio, sr = load_any(src)
    audio = resample(to_mono(audio), sr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    sf.write(tmp, audio, TARGET_SR, format="FLAC")
    tmp.replace(dest)
    return len(audio) / TARGET_SR


def read_for_model(path: Path) -> np.ndarray:
    """Read prepared audio for inference. Asserts the prepare-time contract."""
    audio, sr = sf.read(str(path), dtype="float32")
    if sr != TARGET_SR:
        # Should be impossible if the file came through write_standardised.
        log.warning("%s is %d Hz, expected %d -- resampling", path, sr, TARGET_SR)
        audio = resample(to_mono(audio), sr)
    return to_mono(audio)