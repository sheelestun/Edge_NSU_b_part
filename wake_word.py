from openwakeword.model import Model

import config as cfg

model = Model(inference_framework="onnx")


def detect(audio):
    prediction = model.predict(audio)
    return prediction["hey_jarvis"] > cfg.WAKE_WORD_THRESHOLD