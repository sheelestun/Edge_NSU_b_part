import sounddevice as sd

import config as cfg


def start_audio(callback):
    return sd.InputStream(
        samplerate=cfg.SAMPLE_RATE,
        blocksize=cfg.CHUNK_SIZE,
        channels=cfg.CHANNELS,
        dtype="int16",
        callback=callback,
        device=cfg.DEVICE_NUMBER
    )