from pathlib import Path
import os

import numpy as np
from pywhispercpp.model import Model


BASE_DIR = Path(__file__).resolve().parent

WHISPER_MODEL = BASE_DIR / "whisper.cpp" / "ggml-base.bin"


def load_model():
    stderr_fd = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        os.dup2(null_fd, 2)

        return Model(
            str(WHISPER_MODEL),
            language="ru",
            print_realtime=False,
            print_progress=False,
        )
    finally:
        os.dup2(stderr_fd, 2)
        os.close(null_fd)
        os.close(stderr_fd)


model = load_model()


def transcribe(audio):
    audio = audio.astype(np.float32) / 32768.0

    segments = model.transcribe(audio)

    return " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    )