"""Shared fixtures for audiovec tests."""

from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from audiovec.model import build_transformer_model

# ── Constants ─────────────────────────────────────────────────────────────────

N_MELS = 128
MAX_PAD_LEN = 228
EMBEDDING_DIM = 256
NUM_CLASSES = 8
SAMPLE_RATE = 22050


# ── Model fixture (session-scoped, architecture only — no weights loaded) ────


@pytest.fixture(scope="session")
def model() -> torch.nn.Module:
    """A bare CRNN-Transformer with random weights — no training.

    Only used to verify forward-pass shapes and return types.
    """
    m = build_transformer_model(
        input_shape=(N_MELS, MAX_PAD_LEN, 1),
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
    )
    m.eval()
    return m


# ── Dummy spectrogram ────────────────────────────────────────────────────────


@pytest.fixture
def dummy_spectrogram() -> np.ndarray:
    """Random noise spectrogram, shape (128, 228, 1)."""
    return np.random.randn(N_MELS, MAX_PAD_LEN, 1).astype(np.float32)


@pytest.fixture
def dummy_spectrogram_batch() -> np.ndarray:
    """Batch of 4 random spectrograms, shape (4, 128, 228, 1)."""
    return np.random.randn(4, N_MELS, MAX_PAD_LEN, 1).astype(np.float32)


# ── Synthetic WAV file ───────────────────────────────────────────────────────


@pytest.fixture
def wav_file() -> str:
    """Create a temporary 1-second 440 Hz sine WAV file.

    The file is deleted after the test.
    """
    sr = SAMPLE_RATE
    duration = 1.0
    freq = 440.0
    t = np.arange(int(sr * duration)) / sr
    samples = (np.sin(2 * np.pi * freq * t) * 32767 * 0.9).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_name: str = tmp.name
    tmp.close()

    with wave.open(tmp_name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())

    yield tmp_name

    try:
        os.unlink(tmp_name)
    except Exception:
        pass


# ── RAVDESS-style metadata ───────────────────────────────────────────────────


@pytest.fixture
def ravdess_embeddings() -> np.ndarray:
    """Simulated RAVDESS reference set — 20 unit-norm 256-d vectors."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((20, 256))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


@pytest.fixture
def ravdess_metadata() -> list[dict]:
    """Simulated metadata parallel to :func:`ravdess_embeddings`."""
    return [
        {
            "path": f"/fake/ravdess/Actor_{a:02d}/{i:02d}.wav",
            "emotion": ["neutral", "calm", "happy", "sad", "angry"][i % 5],
            "emotion_code": (i % 5) + 1,
            "actor": a,
            "intensity": 1,
        }
        for i, a in enumerate(range(1, 21))
    ]
