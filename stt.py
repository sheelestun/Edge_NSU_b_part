from pathlib import Path

import numpy as np
from pywhispercpp.model import Model


BASE_DIR = Path(__file__).resolve().parent

WHISPER_MODEL = BASE_DIR / "whisper.cpp" / "ggml-base.bin"

model = Model(
    str(WHISPER_MODEL),
    language="ru",
    print_realtime=False,
    print_progress=False,
)


def transcribe(audio):
    audio = audio.astype(np.float32) / 32768.0

    segments = model.transcribe(audio)

    return " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    )