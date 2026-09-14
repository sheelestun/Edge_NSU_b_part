from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).resolve().parent
WHISPER_CLI = BASE_DIR / "whisper.cpp" / "build" / "bin" / "Release" / "whisper-cli.exe"
WHISPER_MODEL = BASE_DIR / "whisper.cpp" / "ggml-base.bin"


def transcribe(audio_path):
    result = subprocess.run(
        [
            str(WHISPER_CLI),
            "-m",
            str(WHISPER_MODEL),
            "-f",
            str(audio_path),
            "-l",
            "ru",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    for line in result.stdout.splitlines():
        if "]" in line:
            return line.split("]", 1)[1].strip()

    return ""