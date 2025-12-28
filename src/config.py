# Project-wide configuration

from pathlib import Path

# Audio preprocessing
TARGET_SR = 22050
DURATION_SEC = 30.0

# Mel-spectrogram
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Dataset (you will place the dataset here):
# data/genres/<genre>/*.wav
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "genres"

# 10-class GTZAN genre list (folder names must match)
GENRES = [
    "blues","classical","country","disco","hiphop",
    "jazz","metal","pop","reggae","rock"
]

# Outputs
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINT_PATH = ARTIFACTS_DIR / "best_model.pt"
