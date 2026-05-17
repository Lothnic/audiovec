"""Configuration constants for the audiovec embedding model."""

from pathlib import Path

# ── Audio processing ──────────────────────────────────────────────────────────
SAMPLE_RATE = 22050          # Target sample rate for loading audio
N_MELS = 128                 # Number of mel bands
FMAX = 8000                  # Max frequency for mel filterbank
MAX_PAD_LEN = 228            # Padded/trimmed time-steps for mel spectrogram

# ── Model architecture ────────────────────────────────────────────────────────
INPUT_SHAPE = (N_MELS, MAX_PAD_LEN, 1)  # (height, width, channels)
EMBEDDING_DIM = 256          # Dimensionality of the embedding space

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4        # L2 regularization for Adam optimizer
VALIDATION_SPLIT = 0.2  # Fraction of training data held out for validation

# ── Regularisation ───────────────────────────────────────────────────────────
DROPOUT_RATES: tuple[float, ...] = (0.05, 0.05, 0.1, 0.3, 0.3)  # 3 conv + 1 rnn + 1 fc

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "ravdess"

# ── RAVDESS emotion mapping (3rd field in filename) ──────────────────────────
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
