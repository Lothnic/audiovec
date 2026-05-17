"""Sliding-window inference — per-time-step emotion probability curves for long audio."""

from __future__ import annotations

from typing import Callable

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from audiovec.config import EMOTION_MAPPING, FMAX, MAX_PAD_LEN, N_MELS, SAMPLE_RATE

# ── Emotion colors (matching app.py) ─────────────────────────────────────────

COLORS = [
    "#A78BFA",  # neutral
    "#60A5FA",  # calm
    "#34D399",  # happy
    "#FBBF24",  # sad
    "#F87171",  # angry
    "#FB923C",  # fearful
    "#E879F9",  # disgust
    "#22D3EE",  # surprised
]

EMOTION_LABELS = [EMOTION_MAPPING[i].title() for i in range(1, 9)]


# ── Sliding-window inference ─────────────────────────────────────────────────


def compute_sliding_predictions(
    model: torch.nn.Module,
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    step_frames: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Run sliding-window emotion inference over a long audio signal.

    Parameters
    ----------
    model :
        Trained CRNN-Transformer model (in ``.eval()`` mode).
    audio :
        1-D audio signal.
    sr :
        Sample rate of ``audio``.
    step_frames :
        Number of spectrogram frames between successive windows.
        Defaults to ``MAX_PAD_LEN // 2`` (50% overlap, ~2.6 s steps).
    progress_callback :
        Optional ``(current, total)`` callback for progress reporting.

    Returns
    -------
    dict with keys:

        - **timeline** *(np.ndarray)* — center time (seconds) of each window.
        - **probs** *(np.ndarray, shape (n_windows, 8))* — softmax probabilities
          per emotion class for each window.
        - **emotion_labels** *(list[str])* — the 8 emotion class names.
        - **single** *(bool)* — ``True`` if the audio was too short for sliding
          (single global prediction returned).
    """
    device = next(model.parameters()).device

    # ── Compute full mel-spectrogram ────────────────────────────────────────
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, fmax=FMAX)
    total_frames = S.shape[1]

    # ── If audio fits in one window, return a single prediction ──────────────
    if total_frames <= MAX_PAD_LEN:
        S_dB = librosa.power_to_db(S, ref=np.max)
        S_dB = (S_dB + 80.0) / 80.0
        # Pad to MAX_PAD_LEN
        pad_w = MAX_PAD_LEN - S_dB.shape[1]
        if pad_w > 0:
            S_dB = np.pad(S_dB, ((0, 0), (0, pad_w)), mode="constant")
        S_dB = S_dB[np.newaxis, ..., np.newaxis]  # (1, n_mels, MAX_PAD_LEN, 1)

        batch = torch.from_numpy(S_dB).float().to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # (8,)

        center_time = len(audio) / sr / 2
        return {
            "timeline": np.array([center_time]),
            "probs": probs[np.newaxis, :],
            "emotion_labels": EMOTION_LABELS,
            "single": True,
        }

    # ── Slide window over the spectrogram ───────────────────────────────────
    step_frames = step_frames or MAX_PAD_LEN // 2
    windows = list(range(0, total_frames - MAX_PAD_LEN + 1, step_frames))
    # Ensure we always include the last possible position
    if windows[-1] < total_frames - MAX_PAD_LEN:
        windows.append(total_frames - MAX_PAD_LEN)

    n_windows = len(windows)

    # If only 1 window, treat as single prediction (avoids a flat-line chart)
    if n_windows <= 1:
        start = windows[0]
        window = S[:, start : start + MAX_PAD_LEN]
        S_dB = librosa.power_to_db(window, ref=np.max)
        S_dB = (S_dB + 80.0) / 80.0
        S_dB = S_dB[np.newaxis, ..., np.newaxis]
        batch = torch.from_numpy(S_dB).float().to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        hop_length = librosa.frames_to_time(1, sr=sr, hop_length=512)
        center_time = (start + MAX_PAD_LEN // 2) * hop_length
        return {
            "timeline": np.array([center_time]),
            "probs": probs[np.newaxis, :],
            "emotion_labels": EMOTION_LABELS,
            "single": True,
        }

    all_probs = np.empty((n_windows, 8), dtype=np.float32)
    hop_length = librosa.frames_to_time(1, sr=sr, hop_length=512)  # seconds per frame

    for i, start in enumerate(windows):
        if progress_callback is not None:
            progress_callback(i + 1, n_windows)

        # Extract window: (n_mels, MAX_PAD_LEN)
        window = S[:, start : start + MAX_PAD_LEN]

        # Normalize this window independently (same as training pipeline)
        S_dB = librosa.power_to_db(window, ref=np.max)
        S_dB = (S_dB + 80.0) / 80.0
        S_dB = S_dB[np.newaxis, ..., np.newaxis]  # (1, n_mels, MAX_PAD_LEN, 1)

        batch = torch.from_numpy(S_dB).float().to(device)
        with torch.no_grad():
            logits = model(batch)
            all_probs[i] = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # Center time of each window in seconds
    center_frames = np.array(windows) + MAX_PAD_LEN // 2
    timeline = center_frames * hop_length

    return {
        "timeline": timeline,
        "probs": all_probs,
        "emotion_labels": EMOTION_LABELS,
        "single": False,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_emotion_curves(
    timeline: np.ndarray,
    probs: np.ndarray,
    emotion_labels: list[str] | None = None,
    figsize: tuple[float, float] = (10, 4.5),
    dark_bg: str = "#0B0E14",
) -> matplotlib.figure.Figure:
    """Stacked area chart showing emotion probabilities over time.

    Parameters
    ----------
    timeline :
        1-D array of time centers (seconds).
    probs :
        Shape ``(n_windows, 8)`` — probability per emotion per window.
    emotion_labels :
        Optional; defaults to the 8 standard emotion names.
    figsize :
        Figure dimensions.
    dark_bg :
        Background color for dark theme.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if emotion_labels is None:
        emotion_labels = EMOTION_LABELS

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(dark_bg)
    ax.set_facecolor(dark_bg)

    ax.stackplot(
        timeline,
        probs.T,
        labels=emotion_labels,
        colors=COLORS,
        alpha=0.85,
        edgecolor="none",
    )

    ax.set_xlim(timeline[0] if len(timeline) > 0 else 0, timeline[-1] if len(timeline) > 0 else 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Time (s)", color="#8B95A8", fontsize=8)
    ax.set_ylabel("Probability", color="#8B95A8", fontsize=8)
    ax.tick_params(colors="#8B95A8", labelsize=7)

    legend = ax.legend(
        loc="upper left",
        framealpha=0.7,
        fontsize=7,
        facecolor="#141A24",
        edgecolor="#1E293B",
        labelcolor="#CBD5E1",
    )
    for text in legend.get_texts():
        text.set_color("#CBD5E1")

    # Minimal spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Ticks on bottom and left only
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig
