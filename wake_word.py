from openwakeword.model import Model
import sounddevice as sd

import config as cfg

model = Model(inference_framework="onnx")


def wait_for_wake_word():
    detected = False

    def callback(indata, frames, time, status):
        nonlocal detected

        audio = indata[:, 0]
        prediction = model.predict(audio)

        if prediction["hey_jarvis"] > 0.5:
            detected = True

    with sd.InputStream(
        samplerate=cfg.SAMPLE_RATE,
        blocksize=cfg.CHUNK_SIZE,
        channels=cfg.CHANNELS,
        dtype="int16",
        callback=callback,
        device=cfg.DEVICE_NUMBER
    ):
        print("Waiting for wake word...")

        while not detected:
            sd.sleep(100)