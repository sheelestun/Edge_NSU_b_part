import threading
import wave

import config as cfg
from audio import start_audio
from comand_record import CommandRecorder
from wake_word import detect
from speaker import identify_speaker
from stt import transcribe
from tts import synthesize
from player import play_audio


def save_audio(audio, output_path="command.wav"):
    with wave.open(output_path, "wb") as wav:
        wav.setnchannels(cfg.CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(cfg.SAMPLE_RATE)
        wav.writeframes(audio.tobytes())

    return output_path


def main():
    recorder = None
    command_path = None
    finished = threading.Event()
    state = "waiting"

    def callback(indata, frames, time, status):
        nonlocal state
        nonlocal recorder
        nonlocal command_path

        audio = indata[:, 0].copy()

        if state == "waiting":
            if detect(audio):
                print("Tell me your command ;)")
                recorder = CommandRecorder()
                state = "recording"

        elif state == "recording":
            recorder.process(audio)

            if recorder.finished:
                command_path = save_audio(
                    recorder.get_audio()
                )
                state = "finished"
                finished.set()

    with start_audio(callback):
        print("Waiting for wake word...")
        finished.wait()

    user = identify_speaker(
        recorder.get_audio()
    )

    text = transcribe(command_path)

    print("User:", user)
    print("Command:", text)

    response = "Я получил твою команду."

    audio = synthesize(response)
    play_audio(audio)


if __name__ == "__main__":
    main()