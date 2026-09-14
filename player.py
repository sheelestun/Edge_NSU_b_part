import io

import sounddevice as sd
import soundfile as sf


def play_audio(audio_bytes):
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    sd.play(audio, sample_rate)
    sd.wait()