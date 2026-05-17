"""Audio loading, mel-spectrogram extraction, and dataset preparation."""

import os
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from tqdm import tqdm

from audiovec.config import SAMPLE_RATE, N_MELS, FMAX, MAX_PAD_LEN


def load_and_process_audio(
    file_path: str | Path,
    max_pad_len: Optional[int] = None,
) -> np.ndarray:
    """Load a WAV file and return a normalized mel-spectrogram.

    Steps:
        1. Load audio at the configured sample rate.
        2. Compute a mel-spectrogram (128 bands, 8 kHz max freq).
        3. Convert to dB scale.
        4. Pad or trim to ``max_pad_len`` time-steps.
        5. Add a channel dimension.
    """
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Mel-spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, fmax=FMAX)

    # Convert to dB scale
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Normalise to [0, 1] so that ReLU activations retain signal
    # power_to_db with default top_db=80 yields values in [-80, 0]
    S_dB = (S_dB + 80.0) / 80.0

    # Pad or trim to a fixed length
    if max_pad_len is not None:
        pad_width = max_pad_len - S_dB.shape[1]
        if pad_width > 0:
            S_dB = np.pad(S_dB, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            S_dB = S_dB[:, :max_pad_len]

    # Add channel dimension: (n_mels, time, 1)
    S_dB = S_dB[..., np.newaxis]
    return S_dB


def parse_emotion_from_filename(filename: str) -> int:
    """Extract the emotion label from a RAVDESS filename.

    The RAVDESS filename format is::

        Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav

    The emotion is the third field (1 = neutral, 2 = calm, … 8 = surprised).
    """
    return int(filename.split("-")[2])


def load_ravdess_dataset(
    data_dir: str | Path,
    max_pad_len: Optional[int] = MAX_PAD_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk through *data_dir/* and load all ``.wav`` files.

    Returns
    -------
    X : ndarray, shape (n_samples, n_mels, max_pad_len, 1)
        Mel-spectrograms.
    y : ndarray, shape (n_samples,)
        Integer emotion labels (1–8).
    """
    data_dir = Path(data_dir)
    data: list[np.ndarray] = []
    labels: list[int] = []

    actor_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not actor_dirs:
        # Maybe the dataset is flat — try looking for wav files directly
        wav_files = sorted(data_dir.glob("*.wav"))
        if wav_files:
            for file_path in tqdm(wav_files, desc="Loading audio"):
                spectrogram = load_and_process_audio(file_path, max_pad_len)
                emotion = parse_emotion_from_filename(file_path.name)
                data.append(spectrogram)
                labels.append(emotion)
            X = np.array(data)
            y = np.array(labels)
            return X, y

    for subdir in tqdm(actor_dirs, desc="Loading audio"):
        for filename in os.listdir(subdir):
            file_path = subdir / filename
            if file_path.suffix.lower() != ".wav":
                continue
            try:
                spectrogram = load_and_process_audio(file_path, max_pad_len)
                emotion = parse_emotion_from_filename(filename)
                data.append(spectrogram)
                labels.append(emotion)
            except Exception as exc:
                print(f"  ⚠ Skipping {file_path}: {exc}")

    X = np.array(data)
    y = np.array(labels)
    return X, y
