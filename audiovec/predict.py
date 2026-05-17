"""Inference utilities: predict emotion and extract embeddings from audio files (PyTorch)."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from audiovec.config import MAX_PAD_LEN, INPUT_SHAPE, EMOTION_MAPPING
from audiovec.data import load_and_process_audio
from audiovec.model import CRNNTransformer, build_transformer_model

HF_REPO_ID = "lothnic/audiovec"
HF_MODEL_PATH = "models/audiovec_model.pt"


def ensure_model(model_path: str | Path = HF_MODEL_PATH) -> Path:
    """Ensure the trained model exists locally, downloading from HF Hub if needed.

    In a Hugging Face Space, the model is stored in the Space repo (uploaded via
    the API). This function downloads it on first run if it's not already present.
    """
    model_path = Path(model_path)
    if model_path.exists():
        return model_path

    # Download from Hugging Face Hub (the model is stored in the Space repo)
    try:
        from huggingface_hub import hf_hub_download

        print(f"Downloading model from {HF_REPO_ID}...")
        # Strip the models/ prefix if present since the HF repo stores the file at root
        hf_filename = HF_MODEL_PATH
        if hf_filename.startswith("models/"):
            hf_filename = hf_filename[len("models/"):]
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_filename,
            repo_type="space",
            local_dir=model_path.parent,
            local_dir_use_symlinks=False,
        )
        print("Model downloaded successfully.")
        return Path(downloaded)
    except Exception as e:
        raise FileNotFoundError(
            f"Model not found at {model_path} and download failed: {e}. "
            f"Please train the model first with `uv run python train_crnn.py`"
        ) from e


def load_trained_model(
    model_path: str | Path,
    device: torch.device | str | None = None,
) -> CRNNTransformer:
    """Load a trained model from disk, downloading first if needed."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = ensure_model(model_path)

    # Build model architecture — CRNN-Transformer (default)
    model = build_transformer_model(
        input_shape=INPUT_SHAPE,
        embedding_dim=256,
        num_classes=8,
    )

    # Load state dict
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_emotion(
    model: CRNNTransformer,
    spectrogram: np.ndarray,
    device: torch.device | str | None = None,
) -> tuple[str, float, np.ndarray]:
    """Predict emotion from a single spectrogram sample.

    Returns
    -------
    emotion_label : str
        Predicted emotion name (e.g. "happy").
    confidence : float
        Confidence score for the predicted emotion.
    all_probs : np.ndarray, shape (num_classes,)
        Probability distribution over all emotion classes.
    """
    if device is None:
        device = next(model.parameters()).device

    # Add batch dimension: (1, 128, 228, 1)
    batch = torch.from_numpy(spectrogram[np.newaxis, ...]).float().to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=-1)[0]  # shape (num_classes,)

    probs_np = probs.cpu().numpy()
    pred_idx = int(np.argmax(probs_np))
    confidence = float(probs_np[pred_idx])

    # Map index back to emotion label (indices 0-7 → mapping 1-8)
    emotion_label = EMOTION_MAPPING.get(pred_idx + 1, "unknown")
    return emotion_label, confidence, probs_np


def extract_embedding(
    model: CRNNTransformer,
    spectrogram: np.ndarray,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Extract the 256-dimensional embedding vector from a spectrogram.

    Returns
    -------
    np.ndarray, shape (256,)
    """
    if device is None:
        device = next(model.parameters()).device

    batch = torch.from_numpy(spectrogram[np.newaxis, ...]).float().to(device)

    with torch.no_grad():
        embedding = model(batch, return_embedding=True)[0]

    return embedding.cpu().numpy()


def predict_from_file(
    model_path: str | Path,
    audio_path: str | Path,
) -> dict:
    """Full inference pipeline: load model, process audio, predict + embed.

    Returns
    -------
    dict with keys:
        emotion, confidence, probabilities, embedding, spectrogram
    """
    model = load_trained_model(model_path)
    spectrogram = load_and_process_audio(audio_path, max_pad_len=MAX_PAD_LEN)

    emotion, confidence, probs = predict_emotion(model, spectrogram)
    embedding = extract_embedding(model, spectrogram)

    return {
        "emotion": emotion,
        "confidence": confidence,
        "probabilities": probs,
        "embedding": embedding,
        "spectrogram": spectrogram,
    }
