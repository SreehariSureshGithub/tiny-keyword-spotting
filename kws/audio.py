"""Audio loading and feature extraction."""

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from .config import CLIP_LEN, HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE


def load_wav(path) -> torch.Tensor:
    """Read an audio file and return a mono 16 kHz tensor of shape [1, samples].

    soundfile is used for decoding instead of torchaudio.load, which depends on
    an FFmpeg/torchcodec backend that is unreliable on Windows.
    """
    wav, sr = sf.read(str(path), dtype="float32")
    wav = torch.from_numpy(wav)

    if wav.ndim > 1:              # stereo -> mono
        wav = wav.mean(dim=1)

    wav = wav.unsqueeze(0)

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    return wav


def fix_length(wav: torch.Tensor, length: int = CLIP_LEN) -> torch.Tensor:
    """Zero-pad or trim the waveform to exactly `length` samples."""
    if wav.shape[1] < length:
        return F.pad(wav, (0, length - wav.shape[1]))
    return wav[:, :length]


class LogMel(torch.nn.Module):
    """Waveform [1, samples] -> log-mel spectrogram [1, N_MELS, frames] in dB."""

    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.to_db(self.mel(wav))
