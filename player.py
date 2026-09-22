import io
from math import gcd

import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

import config as cfg


def _output_rate(rate, channels):
    # Piper outputs 22050, which the board's codecs don't support.
    # Prefer 44100 (exact x2), then 48000.
    for candidate in (rate, 44100, 48000):
        try:
            sd.check_output_settings(device=cfg.OUTPUT_DEVICE_NUMBER, samplerate=candidate, channels=channels)
            return candidate
        except sd.PortAudioError:
            pass
    raise RuntimeError(f"Output device {cfg.OUTPUT_DEVICE_NUMBER} supports none of {rate}/44100/48000 Hz")


def play_audio(audio_bytes):
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    target = _output_rate(sample_rate, channels)
    if target != sample_rate:
        g = gcd(sample_rate, target)
        audio = resample_poly(audio, target // g, sample_rate // g, axis=0).astype("float32")
    sd.play(audio, target, device=cfg.OUTPUT_DEVICE_NUMBER)
    sd.wait()

