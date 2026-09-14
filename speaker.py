import os
import contextlib
import io
import warnings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings(
    "ignore",
    message="torchaudio._backend.set_audio_backend has been deprecated"
)

import torch
import wespeaker

import paths as ph
import config as cfg


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


def load_reference_embeddings():
    references = {}

    for user_dir in ph.REFERENCE_DIR.iterdir():
        if not user_dir.is_dir():
            continue

        embeddings = []

        for reference_file in user_dir.glob("*.wav"):
            embedding = model.extract_embedding(str(reference_file))
            embeddings.append(embedding)

        if embeddings:
            references[user_dir.name] = embeddings

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