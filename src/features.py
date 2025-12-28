import subprocess
import tempfile
from pathlib import Path

import torch
import torchaudio
import soundfile as sf

from .config import TARGET_SR, N_MELS, N_FFT, HOP_LENGTH

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)
_db = torchaudio.transforms.AmplitudeToDB(stype="power")


def _read_audio_any(path: str):
    # 1) Try direct read (works for wav/flac etc.)
    try:
        audio, sr = sf.read(path, always_2d=True)  # [T, C] numpy
        return audio, sr
    except Exception:
        # 2) Convert to wav via ffmpeg (for webm/ogg/m4a etc.)
        tmp_wav = Path(tempfile.mkstemp(suffix=".wav")[1])
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, str(tmp_wav)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            audio, sr = sf.read(str(tmp_wav), always_2d=True)
            return audio, sr
        finally:
            tmp_wav.unlink(missing_ok=True)


def load_audio(path: str, duration_sec: float) -> torch.Tensor:
    """
    Returns waveform tensor: [1, T] (mono, TARGET_SR, fixed duration)
    """
    audio_np, sr = _read_audio_any(path)  # [T, C]
    audio = torch.from_numpy(audio_np).float().t()  # [C, T]

    # mono
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # resample
    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(audio, sr, TARGET_SR)

    # pad/crop
    num_samples = int(TARGET_SR * duration_sec)
    if audio.size(1) < num_samples:
        audio = torch.nn.functional.pad(audio, (0, num_samples - audio.size(1)))
    else:
        audio = audio[:, :num_samples]

    return audio


@torch.no_grad()
def waveform_to_melspec(wav: torch.Tensor) -> torch.Tensor:
    mel = _mel(wav)          # [1, N_MELS, Time]
    mel_db = _db(mel)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db