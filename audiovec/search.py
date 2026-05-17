"""Similarity search: find nearest RAVDESS samples for a query embedding.

Pre-computed embeddings are cached to disk so they only need to be computed
once.  Cosine similarity is used as the distance metric between embedding vectors.

Usage (pre-compute once):
    uv run python -c "from audiovec.search import precompute_ravdess_embeddings; precompute_ravdess_embeddings()"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from audiovec.config import EMOTION_MAPPING, MAX_PAD_LEN
from audiovec.data import load_and_process_audio
from audiovec.predict import load_trained_model

# ── Cache paths ──────────────────────────────────────────────────────────────

CACHE_DIR = Path("models")
EMBEDDINGS_CACHE = CACHE_DIR / "ravdess_embeddings.npy"
METADATA_CACHE = CACHE_DIR / "ravdess_metadata.json"
RAVDESS_DIR = Path("ravdess")


# ── RAVDESS filename helpers ─────────────────────────────────────────────────


def parse_ravdess_filename(filename: str) -> dict:
    """Extract metadata from a RAVDESS filename.

    Format: ``Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav``

    Returns
    -------
    dict with keys: emotion, emotion_code, intensity, statement, repetition, actor
    """
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("-")
    emotion_code = int(parts[2])
    return {
        "emotion": EMOTION_MAPPING.get(emotion_code, "unknown"),
        "emotion_code": emotion_code,
        "intensity": int(parts[3]),
        "statement": int(parts[4]),
        "repetition": int(parts[5]),
        "actor": int(parts[6]),
    }


def get_ravdess_wav_files(ravdess_dir: str | Path = RAVDESS_DIR) -> list[Path]:
    """Collect all WAV file paths from a RAVDESS dataset directory.

    Supports both nested (``Actor_01/…``) and flat directory layouts.
    """
    ravdess_dir = Path(ravdess_dir)
    if not ravdess_dir.exists():
        raise FileNotFoundError(f"RAVDESS directory not found: {ravdess_dir}")

    # Actor subdirectories
    actor_dirs = sorted(
        d for d in ravdess_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if actor_dirs:
        wav_files: list[Path] = []
        for d in actor_dirs:
            wav_files.extend(sorted(d.glob("*.wav")))
    else:
        # Flat layout fallback
        wav_files = sorted(ravdess_dir.glob("*.wav"))

    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {ravdess_dir}")

    return wav_files


# ── Pre-computation ──────────────────────────────────────────────────────────


def precompute_ravdess_embeddings(
    model_path: str | Path = "models/audiovec_model.pt",
    ravdess_dir: str | Path = RAVDESS_DIR,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Compute embeddings for every RAVDESS WAV file and cache to disk.

    Parameters
    ----------
    model_path :
        Path to the trained ``.pt`` model file.
    ravdess_dir :
        Path to the RAVDESS dataset directory.
    progress_callback :
        Optional ``(current, total)`` callback — useful for Streamlit progress bars.

    Returns
    -------
    (embeddings, metadata)
        ``embeddings`` shape ``(N, 256)``, normalised to unit length.
        ``metadata`` is a list of dicts with parsed filename info + resolved path.
    """
    model = load_trained_model(model_path)
    device = next(model.parameters()).device
    model.eval()

    wav_files = get_ravdess_wav_files(ravdess_dir)
    total = len(wav_files)

    all_embeddings: list[np.ndarray] = []
    all_metadata: list[dict] = []

    for i, wav_path in enumerate(wav_files):
        if progress_callback:
            progress_callback(i + 1, total)

        # Mel-spectrogram → embedding via model
        spectrogram = load_and_process_audio(wav_path, max_pad_len=MAX_PAD_LEN)
        batch = torch.from_numpy(spectrogram[np.newaxis, ...]).float().to(device)
        with torch.no_grad():
            emb = model(batch, return_embedding=True)[0].cpu().numpy()

        all_embeddings.append(emb)

        meta = parse_ravdess_filename(wav_path.name)
        meta["path"] = str(wav_path.resolve())  # absolute path for file reading
        all_metadata.append(meta)

    embeddings = np.array(all_embeddings)  # (N, 256)

    # L2-normalise so dot-product = cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # Cache to disk
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(str(EMBEDDINGS_CACHE), embeddings)
    with open(METADATA_CACHE, "w") as f:
        json.dump(all_metadata, f, indent=2)

    return embeddings, all_metadata


# ── Load cached embeddings ───────────────────────────────────────────────────


def embeddings_cached() -> bool:
    """Check whether pre-computed RAVDESS embeddings exist on disk."""
    return EMBEDDINGS_CACHE.exists() and METADATA_CACHE.exists()


def load_ravdess_embeddings() -> tuple[np.ndarray, list[dict]]:
    """Load pre-computed RAVDESS embeddings and metadata from disk cache.

    Raises ``FileNotFoundError`` if the cache does not exist — call
    :func:`precompute_ravdess_embeddings` first.
    """
    if not embeddings_cached():
        raise FileNotFoundError(
            "RAVDESS embeddings not found. "
            "Run `precompute_ravdess_embeddings()` first."
        )

    embeddings = np.load(str(EMBEDDINGS_CACHE))
    with open(METADATA_CACHE) as f:
        metadata = json.load(f)

    return embeddings, metadata


# ── Similarity search ────────────────────────────────────────────────────────


def find_similar(
    query_embedding: np.ndarray,
    ravdess_embeddings: np.ndarray,
    ravdess_metadata: list[dict],
    k: int = 5,
) -> list[dict]:
    """Return top-*k* most similar RAVDESS samples by cosine similarity.

    Parameters
    ----------
    query_embedding :
        Query embedding vector, shape ``(256,)``.
    ravdess_embeddings :
        Pre-computed RAVDESS embeddings, shape ``(N, 256)`` (pre-normalised).
    ravdess_metadata :
        List of metadata dicts parallel to ``ravdess_embeddings``.
    k :
        Number of nearest neighbours to return.

    Returns
    -------
    list[dict]
        Each dict contains the original metadata keys plus a ``similarity`` float.
        Sorted by similarity descending.
    """
    # Normalise query
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

    # Cosine similarity = dot product of unit vectors
    similarities = ravdess_embeddings @ q_norm  # (N,)

    # Top-K indices
    top_k = int(min(k, len(similarities)))
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    # Sort this subset descending
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    results = []
    for idx in top_indices:
        entry = dict(ravdess_metadata[idx])  # shallow copy
        entry["similarity"] = float(similarities[idx])
        results.append(entry)

    return results
