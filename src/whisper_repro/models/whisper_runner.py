"""The only module that touches OpenAI's whisper package.

Everything else in the pipeline talks to this interface. That isolation is
deliberate: swapping in faster-whisper or a HuggingFace checkpoint for a
comparison means writing one more class here, not editing the pipeline.

The reference implementation is used unmodified and installed from PyPI --
the reproduction's numbers are only comparable to the paper's if the
decoding path is theirs, not ours.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_utils import get_logger

log = get_logger(__name__)

#: GPUs that report fp16 support but compute it incorrectly. The GTX 16-series
#: (1650/1660/Ti/Super) has a hardware defect in half precision: inference runs
#: but every logit is NaN. Detected by name so the pipeline just works on them.
FP16_BROKEN_GPUS = ("GTX 16", "GTX 15")


@dataclass(frozen=True)
class RunEnvironment:
    """Captured once per run and written into the results summary.

    Reproducibility is the unit's subject, so the environment that produced
    a number is part of the result, not incidental.
    """

    model: str
    device: str
    fp16: bool
    whisper_version: str
    torch_version: str
    gpu_name: str | None
    platform: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class WhisperRunner:
    """Thin wrapper over whisper.load_model / model.transcribe."""

    def __init__(
        self,
        model_name: str = "base",
        device: str | None = None,
        download_root: str | Path | None = None,
        fp16: bool | None = None,
    ) -> None:
        import torch
        import whisper

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        on_cuda = self.device.startswith("cuda")
        gpu = torch.cuda.get_device_name(0) if on_cuda else None

        # fp16 halves VRAM, but the GTX 16-series advertises half precision it
        # cannot actually compute: every logit comes back NaN. Force fp32 there.
        if fp16 is None:
            fp16 = on_cuda and not (gpu and any(m in gpu for m in FP16_BROKEN_GPUS))
            if on_cuda and not fp16:
                log.warning("%s has unreliable fp16 -- running in fp32 "
                            "(slower, roughly double the VRAM)", gpu)
        self.fp16 = bool(fp16)

        log.info("loading whisper '%s' on %s%s, %s", model_name, self.device,
                 f" ({gpu})" if gpu else "", "fp16" if self.fp16 else "fp32")
        self.model = whisper.load_model(
            model_name, device=self.device,
            download_root=str(download_root) if download_root else None,
        )

        self.env = RunEnvironment(
            model=model_name,
            device=self.device,
            fp16=self.fp16,
            whisper_version=getattr(whisper, "__version__", "unknown"),
            torch_version=torch.__version__,
            gpu_name=gpu,
            platform=platform.platform(),
        )

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        task: str = "transcribe",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Transcribe one utterance.

        `language=None` lets Whisper detect it -- useful for measuring
        language identification on the Nepali data, but for WER reproduction
        the language is always passed explicitly so a detection error cannot
        masquerade as a recognition error.
        """
        return self.model.transcribe(
            audio, language=language, task=task, fp16=self.fp16, **kwargs
        )
