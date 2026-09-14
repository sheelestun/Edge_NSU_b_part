from wake_word import wait_for_wake_word
from comand_record import record_command
from speaker import identify_speaker
from stt import transcribe
from tts import synthesize
from player import play_audio


def main():
    wait_for_wake_word()

    audio_path = record_command()

    user = identify_speaker(audio_path)
    text = transcribe(audio_path)

    print("User:", user)
    print("Command:", text)

    response = "Я получил твою команду."

    audio = synthesize(response)

    play_audio(audio)


if __name__ == "__main__":
    main()