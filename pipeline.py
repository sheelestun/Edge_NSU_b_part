import threading

from audio import start_audio
from comand_record import CommandRecorder
from wake_word import detect
from speaker import identify_speaker
from stt import transcribe
from tts import synthesize
from player import play_audio


def main():
    recorder = None
    finished = threading.Event()
    state = "waiting"

    def callback(indata, frames, time, status):
        nonlocal state
        nonlocal recorder

        audio = indata[:, 0].copy()

        if state == "waiting":
            if detect(audio):
                print("Tell me your command ;)")
                recorder = CommandRecorder()
                state = "recording"

        elif state == "recording":
            recorder.process(audio)

            if recorder.finished:
                state = "finished"
                finished.set()

    with start_audio(callback):
        print("Waiting for wake word...")
        finished.wait()

    audio = recorder.get_audio()

    user = identify_speaker(audio)
    text = transcribe(audio)

    print("User:", user)
    print("Command:", text)

    response = "Я получил твою команду."

    audio = synthesize(response)
    play_audio(audio)


if __name__ == "__main__":
    main()