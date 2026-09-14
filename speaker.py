import wespeaker
import torch

import paths as ph
import config as cfg

model = wespeaker.load_model("english")

def get_embedding(audio_path):
    return model.extract_embedding(audio_path)

def compare_embeddings(emb1, emb2):
    return torch.nn.functional.cosine_similarity(
        emb1.unsqueeze(0), 
        emb2.unsqueeze(0)
    ).item()


def identify_speaker(audio_path):
    audio_embedding = get_embedding(audio_path)

    best_user = None
    best_score = -1

    for user_dir in ph.REFERENCE_DIR.iterdir():
        if not user_dir.is_dir():
            continue

        reference_files = list(user_dir.glob("*.wav"))

        if not reference_files:
            continue

        user_best_score = -1

        for reference_file in reference_files:

            reference_embedding = get_embedding(reference_file)
            similarity = compare_embeddings(
                audio_embedding,
                reference_embedding
            )

            if similarity > user_best_score:
                user_best_score = similarity

        if user_best_score > best_score:
            best_score = user_best_score
            best_user = user_dir.name

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