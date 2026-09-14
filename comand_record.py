import wave

import numpy as np
import sounddevice as sd

import config as cfg


def record_command(output_path="command.wav"):
    audio_chunks = []
    speech_started = False
    silence_chunks = 0
    finished = False

    def callback(indata, frames, time, status):
        nonlocal speech_started
        nonlocal silence_chunks
        nonlocal finished

        audio = indata[:, 0]
        audio_chunks.append(audio.copy())

        volume = np.abs(audio).mean()

        if volume > cfg.SILENCE_THRESHOLD:
            speech_started = True
            silence_chunks = 0

        elif speech_started:
            silence_chunks += 1

            if silence_chunks >= cfg.SILENCE_CHUNKS:
                finished = True

    print("Tell me your command ;)")

    with sd.InputStream(
        samplerate=cfg.SAMPLE_RATE,
        blocksize=cfg.CHUNK_SIZE,
        channels=cfg.CHANNELS,
        dtype="int16",
        callback=callback,
        device=cfg.DEVICE_NUMBER
    ):
        while not finished:
            sd.sleep(100)

    audio = np.concatenate(audio_chunks)

    with wave.open(output_path, "wb") as wav:
        wav.setnchannels(cfg.CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(cfg.SAMPLE_RATE)
        wav.writeframes(audio.tobytes())

    return output_path
