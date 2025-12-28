import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .config import GENRES, DATA_DIR, DURATION_SEC
from .features import load_audio, waveform_to_melspec

class FolderGenreDataset(Dataset):
    '''
    Expects:
      data/genres/<genre>/*.wav
    where <genre> folder names match GENRES in config.py
    '''
    def __init__(self, root: Path = DATA_DIR):
        self.root = Path(root)
        self.samples = []
        missing = [g for g in GENRES if not (self.root / g).exists()]
        if missing:
            raise FileNotFoundError(
                f"Dataset not found or incomplete. Missing genre folders under {self.root}: {missing}\n"
                f"Expected structure: data/genres/<genre>/*.wav"
            )

        for idx, genre in enumerate(GENRES):
            genre_dir = self.root / genre
            for fname in os.listdir(genre_dir):
                if fname.lower().endswith(".wav"):
                    self.samples.append((str(genre_dir / fname), idx))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No .wav files found under {self.root}. Expected data/genres/<genre>/*.wav"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav = load_audio(path, duration_sec=DURATION_SEC)  # [1, T]
        mel = waveform_to_melspec(wav)                     # [1, N_MELS, Time]
        x = mel                               # [1, 1, N_MELS, Time]
        y = torch.tensor(label, dtype=torch.long)
        return x, y
