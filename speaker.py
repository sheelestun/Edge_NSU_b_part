import os
import contextlib
import io
import logging
import warnings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings(
    "ignore",
    message="torchaudio._backend.set_audio_backend has been deprecated"
)

import torch

import torchaudio
# s3prl (pulled in by wespeaker) calls set_audio_backend at import time;
# removed in newer torchaudio, and a no-op there anyway.
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *a, **k: None
import sys
import types

# wespeaker imports its s3prl frontend at import time, even though the pretrained
# models we use never touch it. s3prl can't import with modern torchaudio
# (sox_effects, set_audio_backend were removed), so stub it out.
if "s3prl" not in sys.modules:
    class _S3prlUnavailable:
        def __init__(self, *args, **kwargs):
            raise ImportError("s3prl is stubbed out in speaker.py")

    _s3prl = types.ModuleType("s3prl")
    _s3prl_nn = types.ModuleType("s3prl.nn")
    _s3prl_nn.Featurizer = _s3prl_nn.S3PRLUpstream = _S3prlUnavailable
    _s3prl.nn = _s3prl_nn
    sys.modules["s3prl"] = _s3prl
    sys.modules["s3prl.nn"] = _s3prl_nn

import wespeaker

import paths as ph
import config as cfg

log = logging.getLogger(__name__)


with contextlib.redirect_stdout(io.StringIO()):
    model = wespeaker.load_model("english")


def get_embedding(audio):
    pcm = torch.from_numpy(audio).unsqueeze(0)
    return model.extract_embedding_from_pcm(
        pcm,
        cfg.SAMPLE_RATE
    )


def compare_embeddings(emb1, emb2):
    return torch.nn.functional.cosine_similarity(
        emb1.unsqueeze(0),
        emb2.unsqueeze(0)
    ).item()


EMBEDDINGS_DIR = ph.REFERENCE_DIR / "embeddings"


def load_reference_embeddings():
    references = {}

    # Precomputed: embeddings/<user_id>/*.pt
    if EMBEDDINGS_DIR.is_dir():
        for user_dir in EMBEDDINGS_DIR.iterdir():
            if not user_dir.is_dir():
                continue

            embeddings = [
                torch.load(f, map_location="cpu")
                for f in sorted(user_dir.glob("*.pt"))
            ]

            if embeddings:
                references[user_dir.name] = embeddings
                log.info("speaker %r: %d precomputed embeddings", user_dir.name, len(embeddings))

    # Fallback: <user_id>/*.wav, only for users without precomputed embeddings
    for user_dir in ph.REFERENCE_DIR.iterdir():
        if (
            not user_dir.is_dir()
            or user_dir == EMBEDDINGS_DIR
            or user_dir.name in references
        ):
            continue

        wav_files = sorted(user_dir.glob("*.wav"))
        if not wav_files:
            continue

        log.info("speaker %r: computing embeddings from %d wav files (slow)", user_dir.name, len(wav_files))
        references[user_dir.name] = [
            model.extract_embedding(str(f)) for f in wav_files
        ]

    return references


REFERENCE_EMBEDDINGS = load_reference_embeddings()


def identify_speaker(audio):
    audio_embedding = get_embedding(audio)

    best_user = None
    best_score = -1

    for user_name, embeddings in REFERENCE_EMBEDDINGS.items():
        user_best_score = -1

        for reference_embedding in embeddings:
            similarity = compare_embeddings(
                audio_embedding,
                reference_embedding
            )

            if similarity > user_best_score:
                user_best_score = similarity

        if user_best_score > best_score:
            best_score = user_best_score
            best_user = user_name

    if best_score < cfg.SPEAKER_THRESHOLD:
        return {
            "user_id": None,
            "user_name": None,
            "confidence": best_score
        }

    return {
        "user_id": best_user,
        "user_name": best_user,
        "confidence": best_score
    }
