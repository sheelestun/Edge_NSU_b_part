from openwakeword.model import Model
import sounddevice as sd
import wave
import numpy as np

import config as cfg

model = Model(inference_framework="onnx")

state = cfg.WAITING
audio_chunks = []
speech_started = False
silence_chunks = 0

def callback(indata, frames, time, status):
    global state
    global speech_started
    global silence_chunks

    indata = indata[:, 0]
    volume = np.abs(indata).mean()
    prediction = model.predict(indata)
    # print(volume)
    wake_word_variacy = prediction['hey_jarvis']
    if state == cfg.WAITING and wake_word_variacy > 0.5:
        state = cfg.RECORDING
        print("Tell me your command ;)")
        return

    if state == cfg.RECORDING:
        audio_chunks.append(indata.copy())

        if volume > cfg.SILENCE_THRESHOLD:
            speech_started = True
            silence_chunks = 0

        elif speech_started:
            silence_chunks += 1

            if silence_chunks >= cfg.SILENCE_CHUNKS:
                state = cfg.SAVED

with sd.InputStream(
    samplerate=cfg.SAMPLE_RATE, 
    blocksize=cfg.CHUNK_SIZE, 
    channels=cfg.CHANNELS,
    dtype="int16",
    callback=callback,
    device=1
):
    print("Waiting for wake word")
    while state != cfg.SAVED:
        sd.sleep(100)


audio = np.concatenate(audio_chunks)

with wave.open("command.wav", "wb") as wav:
    wav.setnchannels(cfg.CHANNELS)
    wav.setsampwidth(2)
    wav.setframerate(cfg.SAMPLE_RATE)
    wav.writeframes(audio.tobytes())

print("Command saved")