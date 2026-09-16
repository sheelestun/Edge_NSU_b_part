from openwakeword.model import Model

import config as cfg

# Без wakeword_models openWakeWord поднимает ONNX-сессию на КАЖДУЮ
# предобученную модель (alexa, hey_mycroft, hey_rhasspy, timer, weather),
# хотя используется ровно одна. Замерено: 142 МБ RSS против 121 МБ.
#
# Важнее другое: при отсутствии любого из шести файлов конструктор падает
# с NO_SUCHFILE. То есть на плате пришлось бы тянуть все модели
# (openwakeword.utils.download_models() без аргументов), а не одну:
#   python -c "import openwakeword.utils as u; u.download_models(['hey_jarvis'])"
model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")


def detect(audio):
    prediction = model.predict(audio)
    return prediction["hey_jarvis"] > cfg.WAKE_WORD_THRESHOLD