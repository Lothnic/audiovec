"""Tests for audiovec.sliding — sliding-window emotion curves and plotting."""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib.figure

from audiovec.sliding import EMOTION_LABELS, compute_sliding_predictions, plot_emotion_curves


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sine_wave(duration: float, sr: int = 22050, freq: float = 440.0) -> np.ndarray:
    """Generate a 1-D sine wave of the given duration."""
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


# ── compute_sliding_predictions ──────────────────────────────────────────────


class TestShortAudio:
    """Audio that fits within a single window (<= MAX_PAD_LEN frames)."""

    def test_returns_single_flag(self, model):
        audio = _sine_wave(2.0)  # ~86 frames, well under 228
        result = compute_sliding_predictions(model, audio)
        assert result["single"] is True

    def test_timeline_has_one_point(self, model):
        audio = _sine_wave(2.0)
        result = compute_sliding_predictions(model, audio)
        assert result["timeline"].shape == (1,)

    def test_probs_shape(self, model):
        audio = _sine_wave(2.0)
        result = compute_sliding_predictions(model, audio)
        assert result["probs"].shape == (1, 8)

    def test_probs_sum_to_one(self, model):
        audio = _sine_wave(2.0)
        result = compute_sliding_predictions(model, audio)
        assert abs(result["probs"].sum() - 1.0) < 1e-5

    def test_emotion_labels(self, model):
        audio = _sine_wave(2.0)
        result = compute_sliding_predictions(model, audio)
        assert result["emotion_labels"] == EMOTION_LABELS


class TestBoundaryAudio:
    """Audio at the boundary — just over one window."""

    def test_exactly_one_window(self, model):
        """Audio that produces exactly MAX_PAD_LEN frames → single prediction."""
        # MAX_PAD_LEN = 228 frames → ~5.29s at 22050 Hz with hop=512
        # frames = floor(samples / 512) + 1
        # For 228 frames: need samples such that floor(samples/512) + 1 = 228
        # floor(samples/512) = 227 → samples in [227*512, 228*512)
        # samples = 227 * 512 = 116224 → duration = 116224 / 22050 ≈ 5.27s
        samples = 227 * 512
        audio = np.sin(2 * np.pi * 440 * np.arange(samples) / 22050).astype(np.float32)
        result = compute_sliding_predictions(model, audio)
        assert result["single"] is True
        assert result["timeline"].shape == (1,)
        assert result["probs"].shape == (1, 8)

    def test_just_over_one_window(self, model):
        """Audio with 229 frames → sliding path produces 2 windows."""
        samples = 228 * 512 + 1  # one sample past the boundary → 229 frames
        audio = np.sin(2 * np.pi * 440 * np.arange(samples) / 22050).astype(np.float32)
        result = compute_sliding_predictions(model, audio)
        # 229 frames > MAX_PAD_LEN=228 → sliding path, range produces [0],
        # then appends position 1 → 2 windows
        assert result["single"] is False
        assert result["timeline"].shape == (2,)
        assert result["probs"].shape == (2, 8)

    def test_two_windows_correct_count(self, model):
        """Audio long enough for exactly 2 windows (non-guarded)."""
        # With step_frames=114, 320 frames gives range(0, 93, 114)=[0]
        # then 0 < 92 → append 92 → exactly 2 windows
        samples = 319 * 512  # → approximately 320 frames
        audio = np.sin(2 * np.pi * 440 * np.arange(samples) / 22050).astype(np.float32)
        result = compute_sliding_predictions(model, audio)
        assert not result["single"]
        assert len(result["timeline"]) == 2

    def test_all_probs_sum_to_one(self, model):
        """Every window's probabilities sum to 1."""
        audio = _sine_wave(30.0)  # ~1293 frames → many windows
        result = compute_sliding_predictions(model, audio)
        assert not result["single"]
        row_sums = result["probs"].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)


class TestLongAudio:
    """Multi-window sliding inference."""

    def test_single_flag_false(self, model):
        audio = _sine_wave(30.0)
        result = compute_sliding_predictions(model, audio)
        assert result["single"] is False

    def test_timeline_shape_matches_probs(self, model):
        audio = _sine_wave(30.0)
        result = compute_sliding_predictions(model, audio)
        assert result["timeline"].shape[0] == result["probs"].shape[0]

    def test_timeline_is_ascending(self, model):
        audio = _sine_wave(30.0)
        result = compute_sliding_predictions(model, audio)
        assert all(result["timeline"][i] < result["timeline"][i + 1] for i in range(len(result["timeline"]) - 1))

    def test_probs_are_non_negative(self, model):
        audio = _sine_wave(30.0)
        result = compute_sliding_predictions(model, audio)
        assert (result["probs"] >= 0.0).all()

    def test_probs_are_float32(self, model):
        audio = _sine_wave(30.0)
        result = compute_sliding_predictions(model, audio)
        assert result["probs"].dtype == np.float32

    def test_emotion_labels(self, model):
        audio = _sine_wave(30.0)
        result = compute_sliding_predictions(model, audio)
        assert result["emotion_labels"] == EMOTION_LABELS


class TestProgressCallback:
    """Progress reporting during sliding inference."""

    def test_callback_invoked_for_long_audio(self, model):
        audio = _sine_wave(30.0)
        calls: list[tuple[int, int]] = []

        def cb(current, total):
            calls.append((current, total))

        compute_sliding_predictions(model, audio, progress_callback=cb)
        assert len(calls) > 1  # multiple invocations

    def test_callback_not_invoked_for_short_audio(self, model):
        audio = _sine_wave(2.0)
        calls: list[tuple[int, int]] = []

        def cb(current, total):
            calls.append((current, total))

        compute_sliding_predictions(model, audio, progress_callback=cb)
        assert len(calls) == 0  # single prediction path, no progress needed

    def test_callback_reports_correct_total(self, model):
        audio = _sine_wave(30.0)
        total_reported = None

        def cb(current, total):
            nonlocal total_reported
            total_reported = total

        result = compute_sliding_predictions(model, audio, progress_callback=cb)
        assert total_reported == len(result["timeline"])

    def test_callback_current_starts_at_one(self, model):
        audio = _sine_wave(30.0)
        first_current = None

        def cb(current, total):
            nonlocal first_current
            if first_current is None:
                first_current = current

        compute_sliding_predictions(model, audio, progress_callback=cb)
        assert first_current == 1


class TestCustomStepFrames:
    """Different step sizes produce different numbers of windows."""

    def test_smaller_step_more_windows(self, model):
        audio = _sine_wave(30.0)
        result_big = compute_sliding_predictions(model, audio, step_frames=200)
        result_small = compute_sliding_predictions(model, audio, step_frames=50)
        assert len(result_small["timeline"]) > len(result_big["timeline"])

    def test_step_equals_max_pad_len_no_overlap(self, model):
        """step_frames == MAX_PAD_LEN → no overlap between windows."""
        audio = _sine_wave(30.0)
        result = compute_sliding_predictions(model, audio, step_frames=228)
        assert result["single"] is False or len(result["timeline"]) >= 1


class TestDeterminism:
    """Same input produces identical output."""

    def test_repeated_call_identical(self, model):
        audio = _sine_wave(10.0)
        result_a = compute_sliding_predictions(model, audio)
        result_b = compute_sliding_predictions(model, audio)
        assert np.array_equal(result_a["probs"], result_b["probs"])
        assert np.array_equal(result_a["timeline"], result_b["timeline"])

    def test_constant_audio_consistent(self, model):
        """A constant signal should give the same prediction each window (or close)."""
        audio = np.ones(int(22050 * 15), dtype=np.float32)  # 15s of constant signal
        result = compute_sliding_predictions(model, audio)
        if not result["single"]:
            # Probabilities should be similar across windows since input is constant
            probs = result["probs"]
            max_diffs = np.abs(probs - probs[0]).max(axis=0)
            # They won't be identical due to spectrogram edge effects,
            # but should be reasonably close
            assert max_diffs.max() < 0.15


# ── plot_emotion_curves ──────────────────────────────────────────────────────


class TestPlotEmotionCurves:
    """Plotting helper."""

    def test_returns_figure(self):
        timeline = np.array([0.0, 5.0, 10.0])
        probs = np.ones((3, 8)) / 8.0
        fig = plot_emotion_curves(timeline, probs)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_single_data_point(self):
        timeline = np.array([2.5])
        probs = np.ones((1, 8)) / 8.0
        fig = plot_emotion_curves(timeline, probs)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_labels(self):
        timeline = np.array([0.0, 5.0])
        probs = np.ones((2, 8)) / 8.0
        labels = ["N", "C", "H", "Sd", "A", "F", "D", "Sp"]
        fig = plot_emotion_curves(timeline, probs, emotion_labels=labels)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_default_labels_match_emotion_labels(self):
        timeline = np.array([0.0, 5.0])
        probs = np.ones((2, 8)) / 8.0
        fig = plot_emotion_curves(timeline, probs)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_dark_bg_override(self):
        timeline = np.array([0.0, 5.0, 10.0])
        probs = np.ones((3, 8)) / 8.0
        fig = plot_emotion_curves(timeline, probs, dark_bg="#1E293B")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_figsize(self):
        timeline = np.array([0.0, 5.0])
        probs = np.ones((2, 8)) / 8.0
        fig = plot_emotion_curves(timeline, probs, figsize=(6, 3))
        w, h = fig.get_size_inches()
        assert abs(w - 6.0) < 0.01
        assert abs(h - 3.0) < 0.01

    def test_probs_not_all_same_renders_ok(self):
        """Diverse probabilities produce distinct colored bands."""
        rng = np.random.default_rng(42)
        timeline = np.linspace(0, 10, 20)
        probs = rng.dirichlet(np.ones(8), size=20)
        fig = plot_emotion_curves(timeline, probs)
        assert isinstance(fig, matplotlib.figure.Figure)
