import os
import torch
from .config import GENRES, CHECKPOINT_PATH, DURATION_SEC
from .features import load_audio, waveform_to_melspec
from .model import GenreCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_checkpoint(path: str = None):
    ckpt_path = str(CHECKPOINT_PATH if path is None else path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}.\n"
            f"Train first: python -m src.train"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    genres = ckpt.get("genres", GENRES)
    model = GenreCNN(num_classes=len(genres)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, genres

@torch.no_grad()
def predict_file(model, genres, audio_path: str):
    wav = load_audio(audio_path, duration_sec=DURATION_SEC)
    mel = waveform_to_melspec(wav)          # [1, N_MELS, T]
    x = mel.unsqueeze(0).to(DEVICE)         # [1, 1, N_MELS, T]
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    out = {genres[i]: float(probs[i]) for i in range(len(genres))}
    topk = torch.topk(probs, k=min(3, len(genres)))
    top = [(genres[i], float(s)) for i, s in zip(topk.indices.tolist(), topk.values.tolist())]
    return out, top
