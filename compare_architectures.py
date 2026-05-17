#!/usr/bin/env python3
"""Compare 5 architectures for audio emotion classification.

Trains 5 architectures on the same train/val split and prints a summary table:

    1. CNN               — 3× Conv blocks → global avg pool → FC (no temporal RNN)
    2. CRNN-GRU (BiGRU)  — 3× Conv → BiGRU (2 layers) → FC
    3. CRNN-LSTM (BiLSTM) — 3× Conv → BiLSTM (2 layers) → FC
    4. CRNN-uniGRU        — 3× Conv → uni-GRU (2 layers) → FC
    5. CRNN-Transformer   — 3× Conv → Transformer Encoder (2 layers, 4 heads) → FC  🏆 default

Usage:
    uv run python compare_architectures.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

SAMPLE_RATE  = 22050
N_MELS       = 128
FMAX         = 8000
MAX_PAD_LEN  = 228
INPUT_SHAPE  = (N_MELS, MAX_PAD_LEN, 1)  # (H, W, C) channels-last
EMBEDDING_DIM = 256
EPOCHS       = 50
BATCH_SIZE   = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.2
DATA_DIR     = "ravdess"
DROPOUT_RATES = (0.05, 0.05, 0.1, 0.3, 0.3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")


# ═══════════════════════════════════════════════════════════════════════
#  Data loading  (same as train_crnn.py)
# ═══════════════════════════════════════════════════════════════════════

def load_and_process_audio(file_path: str | Path, max_pad_len: int | None = None) -> np.ndarray:
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, fmax=FMAX)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = (S_dB + 80.0) / 80.0
    if max_pad_len is not None:
        pad = max_pad_len - S_dB.shape[1]
        if pad > 0:
            S_dB = np.pad(S_dB, ((0, 0), (0, pad)), mode="constant")
        else:
            S_dB = S_dB[:, :max_pad_len]
    return S_dB[..., np.newaxis]


def parse_emotion(filename: str) -> int:
    return int(filename.split("-")[2])


def load_ravdess_dataset(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    data: list[np.ndarray] = []
    labels: list[int] = []
    actor_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
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
#  Shared ConvBlock
# ═══════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: int = 2, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pool(F.relu(self.bn(self.conv(x)))))


def _conv_stack(x: torch.Tensor, blocks: list[ConvBlock]) -> torch.Tensor:
    """Apply a list of ConvBlock sequentially."""
    for b in blocks:
        x = b(x)
    return x


# ═══════════════════════════════════════════════════════════════════════
#  Architecture 1:  CNN  (no temporal RNN)
# ═══════════════════════════════════════════════════════════════════════

class CNNModel(nn.Module):
    """3× Conv blocks → global avg pool → embedding → classifier."""

    def __init__(self, input_shape: tuple[int, int, int], embedding_dim: int = 256,
                 num_classes: int = 8, dropout_rates: tuple[float, ...] = DROPOUT_RATES):
        super().__init__()
        h, w, c = input_shape
        dr = dropout_rates if len(dropout_rates) >= 5 else dropout_rates + (0.0,) * (5 - len(dropout_rates))

        self.blocks = nn.ModuleList([
            ConvBlock(c, 32, dropout=dr[0]),
            ConvBlock(32, 64, dropout=dr[1]),
            ConvBlock(64, 128, dropout=dr[2]),
        ])

        # Compute output channels after conv blocks
        with torch.no_grad():
            x = torch.zeros(1, c, h, w)
            x = _conv_stack(x, self.blocks)
            _, c_out, _, _ = x.shape

        self.drop = nn.Dropout(dr[4]) if dr[4] > 0 else nn.Identity()
        self.fc_embed = nn.Linear(c_out, embedding_dim)  # after global avg pool
        self.fc_cls = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        # (N, H, W, C) → (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = _conv_stack(x, self.blocks)
        # Global average pool over spatial dims → (N, C)
        x = x.mean(dim=(-2, -1))
        x = self.drop(x)
        emb = F.relu(self.fc_embed(x))
        return emb if return_embedding else self.fc_cls(emb)

    @torch.no_grad()
    def predict_embedding(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x, return_embedding=True)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return F.softmax(self.forward(x), dim=-1)


# ═══════════════════════════════════════════════════════════════════════
#  Architecture 2:  CRNN-GRU  (current baseline — BiGRU)
# ═══════════════════════════════════════════════════════════════════════

class CRNN_GRU(nn.Module):
    """3× Conv → BiGRU (2 layers) → mean-pool → embedding → classifier."""

    def __init__(self, input_shape: tuple[int, int, int], embedding_dim: int = 256,
                 num_classes: int = 8, dropout_rates: tuple[float, ...] = DROPOUT_RATES,
                 bidirectional: bool = True):
        super().__init__()
        h, w, c = input_shape
        dr = dropout_rates if len(dropout_rates) >= 5 else dropout_rates + (0.0,) * (5 - len(dropout_rates))

        self.blocks = nn.ModuleList([
            ConvBlock(c, 32, dropout=dr[0]),
            ConvBlock(32, 64, dropout=dr[1]),
            ConvBlock(64, 128, dropout=dr[2]),
        ])

        with torch.no_grad():
            x = torch.zeros(1, c, h, w)
            x = _conv_stack(x, self.blocks)
            _, c_out, h_out, w_out = x.shape
            self._seq_len  = w_out
            self._feat_dim = c_out * h_out

        rnn_out = 2 * 128 if bidirectional else 128
        self.gru = nn.GRU(self._feat_dim, 128, num_layers=2, bidirectional=bidirectional,
                          batch_first=True, dropout=dr[3] if dr[3] > 0 else 0)
        self.drop = nn.Dropout(dr[4]) if dr[4] > 0 else nn.Identity()
        self.fc_embed = nn.Linear(rnn_out, embedding_dim)
        self.fc_cls = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = _conv_stack(x, self.blocks)
        N, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).reshape(N, W, C * H)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        x = self.drop(x)
        emb = F.relu(self.fc_embed(x))
        return emb if return_embedding else self.fc_cls(emb)

    @torch.no_grad()
    def predict_embedding(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x, return_embedding=True)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return F.softmax(self.forward(x), dim=-1)


# ═══════════════════════════════════════════════════════════════════════
#  Architecture 3:  CRNN-LSTM  (BiLSTM)
# ═══════════════════════════════════════════════════════════════════════

class CRNN_LSTM(nn.Module):
    """3× Conv → BiLSTM (2 layers) → mean-pool → embedding → classifier."""

    def __init__(self, input_shape: tuple[int, int, int], embedding_dim: int = 256,
                 num_classes: int = 8, dropout_rates: tuple[float, ...] = DROPOUT_RATES,
                 bidirectional: bool = True):
        super().__init__()
        h, w, c = input_shape
        dr = dropout_rates if len(dropout_rates) >= 5 else dropout_rates + (0.0,) * (5 - len(dropout_rates))

        self.blocks = nn.ModuleList([
            ConvBlock(c, 32, dropout=dr[0]),
            ConvBlock(32, 64, dropout=dr[1]),
            ConvBlock(64, 128, dropout=dr[2]),
        ])

        with torch.no_grad():
            x = torch.zeros(1, c, h, w)
            x = _conv_stack(x, self.blocks)
            _, c_out, h_out, w_out = x.shape
            self._feat_dim = c_out * h_out

        rnn_out = 2 * 128 if bidirectional else 128
        self.lstm = nn.LSTM(self._feat_dim, 128, num_layers=2, bidirectional=bidirectional,
                            batch_first=True, dropout=dr[3] if dr[3] > 0 else 0)
        self.drop = nn.Dropout(dr[4]) if dr[4] > 0 else nn.Identity()
        self.fc_embed = nn.Linear(rnn_out, embedding_dim)
        self.fc_cls = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = _conv_stack(x, self.blocks)
        N, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).reshape(N, W, C * H)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        x = self.drop(x)
        emb = F.relu(self.fc_embed(x))
        return emb if return_embedding else self.fc_cls(emb)

    @torch.no_grad()
    def predict_embedding(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x, return_embedding=True)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return F.softmax(self.forward(x), dim=-1)


# ═══════════════════════════════════════════════════════════════════════
#  Architecture 5:  CRNN-Transformer  (default)
# ═══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Learned positional encoding for the transformer."""
    def __init__(self, d_model: int, max_len: int = 60):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class CRNN_Transformer(nn.Module):
    """3× Conv → Transformer Encoder (2 layers, 4 heads) → mean-pool → embedding → classifier."""

    def __init__(self, input_shape: tuple[int, int, int], embedding_dim: int = 256,
                 num_classes: int = 8, dropout_rates: tuple[float, ...] = DROPOUT_RATES):
        super().__init__()
        h, w, c = input_shape
        dr = dropout_rates if len(dropout_rates) >= 5 else dropout_rates + (0.0,) * (5 - len(dropout_rates))

        self.blocks = nn.ModuleList([
            ConvBlock(c, 32, dropout=dr[0]),
            ConvBlock(32, 64, dropout=dr[1]),
            ConvBlock(64, 128, dropout=dr[2]),
        ])

        with torch.no_grad():
            x = torch.zeros(1, c, h, w)
            x = _conv_stack(x, self.blocks)
            _, c_out, h_out, w_out = x.shape
            self._seq_len  = w_out
            self._feat_dim = c_out * h_out

        d_model = 128  # matches GRU hidden dim
        nhead = 4
        num_layers = 2

        self.input_proj = nn.Linear(self._feat_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=self._seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dr[3],
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dr[4]) if dr[4] > 0 else nn.Identity()
        self.fc_embed = nn.Linear(d_model, embedding_dim)
        self.fc_cls = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = _conv_stack(x, self.blocks)
        N, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).reshape(N, W, C * H)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.drop(x)
        emb = F.relu(self.fc_embed(x))
        return emb if return_embedding else self.fc_cls(emb)

    @torch.no_grad()
    def predict_embedding(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x, return_embedding=True)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return F.softmax(self.forward(x), dim=-1)


# ═══════════════════════════════════════════════════════════════════════
#  Training  (generic — works for any nn.Module)
# ═══════════════════════════════════════════════════════════════════════

def count_params(m: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_one_architecture(
    name: str,
    model_factory,
    X: np.ndarray,
    y_enc: np.ndarray,
    num_classes: int,
) -> tuple[dict, nn.Module]:
    """Train a single architecture and return (metrics_dict, trained_model)."""
    print(f"\n{'═' * 60}")
    print(f"  {name}")
    print(f"{'═' * 60}")

    model = model_factory(INPUT_SHAPE, EMBEDDING_DIM, num_classes).to(device)
    n_params = count_params(model)
    print(f"  Parameters: {n_params:,}")

    # Split (same seed for all architectures)
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"  Train: {len(X_tr)}  Val: {len(X_va)}")

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
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

        model.eval()
        va_loss = va_correct = va_total = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                loss = criterion(model(bx), by)
                va_loss += loss.item() * bx.size(0)
                va_correct += (model(bx).argmax(1) == by).sum().item()
                va_total += by.size(0)

        tr_acc = tr_correct / tr_total
        va_acc = va_correct / va_total
        best_val_acc = max(best_val_acc, va_acc)

        print(
            f"    Epoch {epoch:3d}/{EPOCHS}  "
            f"train_loss={tr_loss/tr_total:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={va_loss/va_total:.4f}  val_acc={va_acc:.4f}"
        )

    elapsed = time.time() - start_time
    model.eval()

    metrics = {
        "name": name,
        "params": n_params,
        "time_s": round(elapsed, 1),
        "train_loss": round(tr_loss / tr_total, 4),
        "val_loss": round(va_loss / va_total, 4),
        "train_acc": round(tr_correct / tr_total, 4),
        "val_acc": round(va_correct / va_total, 4),
        "best_val_acc": round(best_val_acc, 4),
    }
    return metrics, model


# ═══════════════════════════════════════════════════════════════════════
#  Main  —  run all architectures
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Load data once ──
    print("Loading audio ...")
    X, y = load_ravdess_dataset(DATA_DIR)
    print(f"  X: {X.shape}   y: {y.shape}")

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)

    # ── Architectures to compare ──
    # CRNN-uniGRU reuses CRNN_GRU with bidirectional=False
    architectures = [
        ("CNN (no RNN)", lambda ish, edim, ncls: CNNModel(ish, edim, ncls)),
        ("CRNN-GRU (BiGRU)", lambda ish, edim, ncls: CRNN_GRU(ish, edim, ncls, bidirectional=True)),
        ("CRNN-LSTM (BiLSTM)", lambda ish, edim, ncls: CRNN_LSTM(ish, edim, ncls, bidirectional=True)),
        ("CRNN-uniGRU (uni-GRU)", lambda ish, edim, ncls: CRNN_GRU(ish, edim, ncls, bidirectional=False)),
        ("CRNN-Transformer", lambda ish, edim, ncls: CRNN_Transformer(ish, edim, ncls)),
    ]

    safename = lambda n: n.split("(")[0].strip().replace(" ", "_").replace("/", "_").lower()
    results: list[dict] = []
    for name, factory in architectures:
        result, model = train_one_architecture(name, factory, X, y_enc, num_classes)
        results.append(result)

        # Save trained model
        model_path = Path("models") / f"{safename(name)}.pt"
        model_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"  Saved -> {model_path}")

    # ── Summary table ──
    print(f"\n{'═' * 83}")
    print("  COMPARISON SUMMARY")
    print(f"{'═' * 75}")
    header = f"  {'Architecture':<28} {'Params':>8} {'Time(s)':>7}  {'Train Loss':>10} {'Val Loss':>10}  {'Train Acc':>9} {'Val Acc':>9} {'Best Val':>9}"
    print(header)
    print("  " + "-" * 99)
    for r in results:
        print(
            f"  {r['name']:<28} {r['params']:>8,} {r['time_s']:>7.1f}  "
            f"{r['train_loss']:>10.4f} {r['val_loss']:>10.4f}  "
            f"{r['train_acc']:>9.4f} {r['val_acc']:>9.4f} {r['best_val_acc']:>9.4f}"
        )

    # Best model
    best = max(results, key=lambda r: r["best_val_acc"])
    print(f"\n  🏆  Best architecture: {best['name']}  (best val acc: {best['best_val_acc']:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
