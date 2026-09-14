from pathlib import Path
import subprocess
import tempfile

BASE_DIR = Path(__file__).resolve().parent

PIPER_MODEL = BASE_DIR / "ru_RU-dmitri-medium.onnx"
PIPER_CONFIG = BASE_DIR / "ru_RU-dmitri-medium.onnx.json"

def synthesize(text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        output_path = Path(temp.name)

    try:
        result = subprocess.run(
            [
                "piper",
                "-m",
                str(PIPER_MODEL),
                "-c",
                str(PIPER_CONFIG),
                "-f",
                str(output_path),
            ],
            input=text,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="ignore"))

        return output_path.read_bytes()

    finally:
        output_path.unlink(missing_ok=True)