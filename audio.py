import numpy as np
import sounddevice as sd
from scipy.signal import firwin, lfilter

import config as cfg


def _input_rate():
    # USB camera mics often only do 48k. Pick a rate that's an integer
    # multiple of SAMPLE_RATE so we can decimate cleanly.
    for rate in (cfg.SAMPLE_RATE, 48000, 32000, 96000):
        if rate % cfg.SAMPLE_RATE:
            continue
        try:
            sd.check_input_settings(device=cfg.DEVICE_NUMBER, samplerate=rate,
                                    channels=cfg.CHANNELS, dtype="int16")
            return rate
        except sd.PortAudioError:
            pass
    raise RuntimeError(f"Input device {cfg.DEVICE_NUMBER!r} supports no multiple of {cfg.SAMPLE_RATE} Hz")


def start_audio(callback):
    rate = _input_rate()
    factor = rate // cfg.SAMPLE_RATE

    if factor == 1:
        wrapped = callback
    else:
        # Anti-alias low-pass, then keep every `factor`-th sample. Filter state
        # carries across blocks, so there are no clicks at block boundaries.
        taps = firwin(63, 0.45 * cfg.SAMPLE_RATE, fs=rate)
        state = [np.zeros((len(taps) - 1, cfg.CHANNELS))]

        def wrapped(indata, frames, time, status):
            y, state[0] = lfilter(taps, 1.0, indata.astype(np.float32), axis=0, zi=state[0])
            out = np.clip(y[::factor], -32768, 32767).astype(np.int16)
            callback(out, frames // factor, time, status)

    return sd.InputStream(
        samplerate=rate,
        blocksize=cfg.CHUNK_SIZE * factor,
        channels=cfg.CHANNELS,
        dtype="int16",
        callback=wrapped,
        device=cfg.DEVICE_NUMBER,
    )
