# Repo for part B realization of edge_controller NSU project

# Jarvis — локальный голосовой ассистент

Локальный голосовой ассистент с распознаванием ключевой фразы, идентификацией пользователя, распознаванием речи и синтезом ответа.

## Возможности

* обнаружение wake word `Hey Jarvis`;
* запись голосовой команды;
* идентификация пользователя по голосу;
* распознавание речи на русском языке;
* синтез речи;
* полностью локальная обработка аудио.

## Архитектура

```text
Микрофон
   ↓
openWakeWord
   ↓
Запись команды
   ↓
WeSpeaker
   ↓
Whisper
   ↓
Обработка команды
   ↓
Piper TTS
   ↓
Динамик
```

Основной запуск выполняется через `pipeline.py`.

## Структура проекта

```text
edge_project/
├── audio.py
├── command_record.py
├── config.py
├── pipeline.py
├── player.py
├── speaker.py
├── stt.py
├── tts.py
├── wake_word.py
├── requirements.txt
│
├── openWakeWord/
├── wespeaker/
├── whisper.cpp/
│
├── audio_dataset/
│   └── references/
│
├── ru_RU-dmitri-medium.onnx
├── ru_RU-dmitri-medium.onnx.json
└── whisper.cpp/
    └── ggml-base.bin
```

## Требования

* Python 3.13
* микрофон
* Windows / Linux
* CMake и компилятор C/C++ для сборки `whisper.cpp`

## Установка Python-зависимостей

Создать виртуальное окружение:

```bash
python -m venv .venv
```

Активировать его в Windows:

```powershell
.venv\Scripts\activate
```

Установить зависимости:

```bash
python -m pip install -r requirements.txt
```

## Компоненты

### openWakeWord

Используется для обнаружения ключевой фразы.

Официальный репозиторий:

https://github.com/dscripka/openWakeWord

Установка и дополнительная информация находятся в README проекта.

В текущей реализации используется модель `hey_jarvis`.

### WeSpeaker

Используется для идентификации пользователя по голосу.

Официальный репозиторий:

https://github.com/wenet-e2e/wespeaker

WeSpeaker предоставляет Python API и поддерживает получение speaker embeddings и вычисление сходства между голосами.

Эталонные записи пользователей находятся в:

```text
audio_dataset/references/
```

Для каждого пользователя используется отдельная директория:

```text
audio_dataset/references/
├── shelestov/
│   ├── reference1.wav
│   └── reference2.wav
└── puchkov/
    ├── reference1.wav
    └── reference2.wav
```

### whisper.cpp

Используется для локального Speech-to-Text.

Официальный репозиторий:

https://github.com/ggml-org/whisper.cpp

Проект поддерживает Windows и CPU-only inference. Сборка выполняется через CMake.

Клонирование:

```powershell
git clone https://github.com/ggml-org/whisper.cpp.git
```

Сборка:

```powershell
cd whisper.cpp
cmake -B build
cmake --build build --config Release
```

Модель Whisper должна находиться в:

```text
whisper.cpp/ggml-base.bin
```

Подробнее о моделях и сборке:

https://github.com/ggml-org/whisper.cpp#quick-start

### Piper

Используется для локального Text-to-Speech.

Официальный репозиторий:

https://github.com/OHF-Voice/piper1-gpl

Установка Python-пакета:

```bash
pip install piper-tts
```

Piper предоставляет Python API для синтеза речи.

В проекте используется русская модель:

```text
ru_RU-dmitri-medium.onnx
ru_RU-dmitri-medium.onnx.json
```

## Настройка

Основные параметры находятся в `config.py`:

```python
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1280

SILENCE_THRESHOLD = 500
SILENCE_DURATION = 0.5

WAKE_WORD_THRESHOLD = 0.5
SPEAKER_THRESHOLD = 0.6

AUDIO_DEVICE = 1
```

При необходимости нужно изменить `AUDIO_DEVICE` на индекс используемого микрофона.

## Запуск

После установки зависимостей и подготовки моделей:

```powershell
python pipeline.py
```

После запуска программа ожидает wake word:

```text
Waiting for wake word...
```

После обнаружения:

```text
Tell me your command ;)
```

После окончания команды выполняются:

1. идентификация пользователя;
2. распознавание речи;
3. обработка команды;
4. синтез ответа;
5. воспроизведение ответа.

## Тестирование отдельных компонентов

Для проверки отдельных частей проекта используются тестовые файлы:

```text
audio_test.py
wake_word_test.py
stt_test.py
tts_test.py
player_test.py
module_test.py
```

Основной end-to-end тест:

```powershell
python pipeline.py
```

## Основные файлы

| Файл                | Назначение                   |
| ------------------- | ---------------------------- |
| `config.py`         | Настройки проекта            |
| `audio.py`          | Работа с микрофоном          |
| `wake_word.py`      | Обнаружение `Hey Jarvis`     |
| `command_record.py` | Запись голосовой команды     |
| `speaker.py`        | Идентификация пользователя   |
| `stt.py`            | Speech-to-Text               |
| `tts.py`            | Text-to-Speech               |
| `player.py`         | Воспроизведение аудио        |
| `pipeline.py`       | Основной pipeline приложения |

## Принцип работы

Микрофон постоянно передаёт небольшие аудиоблоки в общий `InputStream`.

`openWakeWord` проверяет каждый блок на наличие ключевой фразы. После её обнаружения запускается запись команды.

Когда определяется пауза в речи, запись завершается. Полученный PCM-аудиопоток передаётся одновременно в WeSpeaker и Whisper.

WeSpeaker определяет наиболее похожего зарегистрированного пользователя, а Whisper преобразует речь в текст.

После обработки команды текст передаётся в Piper, который генерирует WAV-аудио. Полученный звук воспроизводится через выбранное устройство.

Все основные модели загружаются один раз при запуске приложения, поэтому они не загружаются заново для каждой команды.

## Использованные проекты

* [openWakeWord](https://github.com/dscripka/openWakeWord)
* [WeSpeaker](https://github.com/wenet-e2e/wespeaker)
* [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
* [Piper](https://github.com/OHF-Voice/piper1-gpl)
