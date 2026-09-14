from pathlib import Path
import io
import wave

from piper import PiperVoice


import config as cfg

BASE_DIR = Path(__file__).resolve().parent

PIPER_MODEL = BASE_DIR / "ru_RU-dmitri-medium.onnx"

voice = PiperVoice.load(str(PIPER_MODEL))


def synthesize(text):
    output = io.BytesIO()

    with wave.open(output, "wb") as wav:
        wav.setnchannels(cfg.CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(cfg.SAMPLE_RATE)
        voice.synthesize(text, wav)

    return output.getvalue()