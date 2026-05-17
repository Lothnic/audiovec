#!/usr/bin/env python3
"""Train the audiovec CRNN emotion classifier — fully self-contained.

Edit the constants at the top and run:

    uv run python train_crnn.py

Set VISUALIZE = False to skip t-SNE plots after training.
"""

from __future__ import annotations

import os
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from audiovec.model import CRNNTransformer


# ═══════════════════════════════════════════════════════════════════════
#  Configuration  — edit freely
# ═══════════════════════════════════════════════════════════════════════

# Audio processing
SAMPLE_RATE  = 22050
N_MELS       = 128
FMAX         = 8000
MAX_PAD_LEN  = 228

# Model architecture
INPUT_SHAPE    = (N_MELS, MAX_PAD_LEN, 1)  # (H, W, C) channels-last
EMBEDDING_DIM  = 256

# Training
EPOCHS         = 50
BATCH_SIZE     = 32
LEARNING_RATE  = 0.001
WEIGHT_DECAY   = 1e-4
VAL_SPLIT      = 0.2

# Regularisation  (3 conv + 1 rnn + 1 fc)
DROPOUT_RATES  = (0.05, 0.05, 0.1, 0.3, 0.3)

# Paths & flags
DATA_DIR       = "ravdess"
SAVE_PATH      = "models/audiovec_model.pt"
VISUALIZE      = True

# RAVDESS emotion mapping  (3rd field in filename)
EMOTION_MAPPING: dict[int, str] = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}


# ═══════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_and_process_audio(
    file_path: str | Path,
    max_pad_len: int | None = None,
) -> np.ndarray:
    """Load a WAV file → normalised mel-spectrogram → (n_mels, time, 1)."""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, fmax=FMAX)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = (S_dB + 80.0) / 80.0  # normalise to [0, 1]

    if max_pad_len is not None:
        pad = max_pad_len - S_dB.shape[1]
        if pad > 0:
            S_dB = np.pad(S_dB, ((0, 0), (0, pad)), mode="constant")
        else:
            S_dB = S_dB[:, :max_pad_len]

    return S_dB[..., np.newaxis]  # (n_mels, time, 1)


def parse_emotion(filename: str) -> int:
    """Extract emotion label (1-8) from a RAVDESS filename."""
    return int(filename.split("-")[2])


def load_ravdess_dataset(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Walk *data_dir/*, load all WAV files, return (X, y)."""
    data_dir = Path(data_dir)
    data: list[np.ndarray] = []
    labels: list[int] = []

    actor_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not actor_dirs:
        wav_files = sorted(data_dir.glob("*.wav"))
        if wav_files:
            for fp in tqdm(wav_files, desc="Loading audio"):
                data.append(load_and_process_audio(fp, MAX_PAD_LEN))
                labels.append(parse_emotion(fp.name))
            return np.array(data), np.array(labels)

    for subdir in tqdm(actor_dirs, desc="Loading audio"):
        for fname in os.listdir(subdir):
            fp = subdir / fname
            if fp.suffix.lower() != ".wav":
                continue
            try:
                data.append(load_and_process_audio(fp, MAX_PAD_LEN))
                labels.append(parse_emotion(fname))
            except Exception as exc:
                print(f"  Skipping {fp}: {exc}")

    return np.array(data), np.array(labels)





# ═══════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════

def train_model(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[CRNNTransformer, LabelEncoder, np.ndarray]:
    """Full train loop with per-epoch validation metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Encode labels 1–8 → 0–7
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)

    # Build model
    model = CRNNTransformer(INPUT_SHAPE, EMBEDDING_DIM, num_classes, DROPOUT_RATES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Stratified split
    X_tr, X_va, y_tr, y_va = sk_train_test_split(
        X, y_enc, test_size=VAL_SPLIT, stratify=y_enc, random_state=42,
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()),
        BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).long()),
        BATCH_SIZE, shuffle=False,
    )

    print(f"  Train samples: {len(X_tr)}  Val samples: {len(X_va)}")

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        tr_loss = tr_correct = tr_total = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * bx.size(0)
            tr_correct += (model(bx).argmax(1) == by).sum().item()
            tr_total += by.size(0)

        # --- Validation ---
        model.eval()
        va_loss = va_correct = va_total = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                loss = criterion(model(bx), by)
                va_loss += loss.item() * bx.size(0)
                va_correct += (model(bx).argmax(1) == by).sum().item()
                va_total += by.size(0)

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"train_loss={tr_loss/tr_total:.4f}  train_acc={tr_correct/tr_total:.4f}  "
            f"val_loss={va_loss/va_total:.4f}  val_acc={va_correct/va_total:.4f}"
        )

    model.eval()
    return model, encoder, y_enc


# ═══════════════════════════════════════════════════════════════════════
#  Visualisation  (t-SNE)
# ═══════════════════════════════════════════════════════════════════════

def compute_embeddings(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Extract embeddings in batches to avoid OOM."""
    device = next(model.parameters()).device
    model.eval()
    embeddings: list[np.ndarray] = []
    for start in range(0, X.shape[0], 64):
        batch = torch.from_numpy(X[start:start + 64]).float().to(device)
        with torch.no_grad():
            embeddings.append(model(batch, return_embedding=True).cpu().numpy())
    return np.concatenate(embeddings)


def visualize(model: nn.Module, X: np.ndarray, y_enc: np.ndarray) -> None:
    """2D + 3D t-SNE of the embedding space."""
    labels = [EMOTION_MAPPING[i + 1] for i in sorted(np.unique(y_enc))]

    print("Extracting embeddings...")
    emb = compute_embeddings(model, X)
    print(f"  Embedding shape: {emb.shape}")

    # 2D
    print("Computing 2D t-SNE...")
    tsne2 = TSNE(n_components=2, random_state=42).fit_transform(emb)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(tsne2[:, 0], tsne2[:, 1], c=y_enc, cmap="viridis")
    cbar = plt.colorbar(sc, ticks=sorted(np.unique(y_enc)))
    cbar.ax.set_yticklabels([EMOTION_MAPPING[i + 1] for i in sorted(np.unique(y_enc))])
    plt.title("2D t-SNE projection of audio emotion embeddings")
    plt.tight_layout()
    plt.savefig("embeddings_2d.png", dpi=150)
    print("  Saved -> embeddings_2d.png")
    plt.show()

    # 3D
    print("Computing 3D t-SNE...")
    import plotly.express as px
    tsne3 = TSNE(n_components=3, random_state=42).fit_transform(emb)
    fig = px.scatter_3d(
        x=tsne3[:, 0], y=tsne3[:, 1], z=tsne3[:, 2],
        color=[labels[i] for i in y_enc],
        title="3D t-SNE projection of audio emotion embeddings",
    )
    fig.update_traces(marker=dict(size=5))
    fig.write_html("embeddings_3d.html")
    print("  Saved -> embeddings_3d.html")
    fig.show()


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"Loading audio from {DATA_DIR} ...")
    X, y = load_ravdess_dataset(DATA_DIR)
    print(f"  X: {X.shape}   y: {y.shape}")
    print(f"  Emotions: {sorted(np.unique(y))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device} ({EPOCHS} epochs, batch {BATCH_SIZE}) ...")
    model, encoder, y_enc = train_model(X, y)

    Path(SAVE_PATH).parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nModel saved -> {SAVE_PATH}")

    if VISUALIZE:
        print()
        visualize(model, X, y_enc)

    print("\nDone.")


if __name__ == "__main__":
    main()
