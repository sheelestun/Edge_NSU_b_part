from pathlib import Path
import io
import wave

from piper import PiperVoice

BASE_DIR = Path(__file__).resolve().parent

PIPER_MODEL = BASE_DIR / "ru_RU-dmitri-medium.onnx"

voice = PiperVoice.load(str(PIPER_MODEL))


def synthesize(text):
    output = io.BytesIO()

    # synthesize_wav(), не synthesize(): начиная с piper-tts 1.3 вторым
    # аргументом идёт syn_config, а сам synthesize() возвращает генератор
    # чанков. Старый вызов не писал в wave вообще — на выходе был WAV из
    # одного заголовка (44 байта) и полная тишина.
    #
    # Параметры WAV ставит сам Piper: частота дискретизации берётся из
    # модели голоса (22050 Гц), а не из config.SAMPLE_RATE (16000 Гц),
    # который относится ко входу с микрофона. С 16000 в заголовке ответ
    # звучал бы на треть медленнее и ниже.
    with wave.open(output, "wb") as wav:
        voice.synthesize_wav(text, wav)

    return output.getvalue()