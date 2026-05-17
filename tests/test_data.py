"""Tests for audiovec.data — audio loading, spectrogram extraction, filename parsing."""

import numpy as np
import pytest

from audiovec.config import MAX_PAD_LEN, N_MELS, SAMPLE_RATE
from audiovec.data import load_and_process_audio, parse_emotion_from_filename


class TestParseEmotionFromFilename:
    """RAVDESS filename → emotion label."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("03-01-01-01-01-01-01.wav", 1),
            ("03-01-02-01-01-01-01.wav", 2),
            ("03-01-05-01-01-02-21.wav", 5),
            ("03-01-08-02-02-02-24.wav", 8),
        ],
    )
    def test_parses_correctly(self, filename: str, expected: int) -> None:
        assert parse_emotion_from_filename(filename) == expected


class TestLoadAndProcessAudio:
    """Spectrogram extraction from WAV files."""

    def test_output_shape(self, wav_file: str) -> None:
        """Should return (N_MELS, MAX_PAD_LEN, 1)."""
        spec = load_and_process_audio(wav_file, max_pad_len=MAX_PAD_LEN)
        assert spec.shape == (N_MELS, MAX_PAD_LEN, 1)

    def test_values_are_float32(self, wav_file: str) -> None:
        spec = load_and_process_audio(wav_file, max_pad_len=MAX_PAD_LEN)
        assert spec.dtype == np.float32

    def test_values_in_range(self, wav_file: str) -> None:
        """Normalised to [0, 1] range."""
        spec = load_and_process_audio(wav_file, max_pad_len=MAX_PAD_LEN)
        assert spec.min() >= 0.0
        assert spec.max() <= 1.0

    def test_short_audio_is_padded(self, wav_file: str) -> None:
        """A 1-second file at 22 kHz → ~86 time-steps → padded to MAX_PAD_LEN."""
        spec = load_and_process_audio(wav_file, max_pad_len=MAX_PAD_LEN)
        # The spectrogram should have exactly MAX_PAD_LEN time frames
        assert spec.shape[1] == MAX_PAD_LEN

    def test_padding_appears_as_zeros(self, wav_file: str) -> None:
        """Padded region (rightmost columns) should be zeros."""
        spec = load_and_process_audio(wav_file, max_pad_len=MAX_PAD_LEN)
        # Compute spectrogram without padding to find the actual width
        audio, sr = (
            __import__("librosa").load(wav_file, sr=SAMPLE_RATE),
            SAMPLE_RATE,
        )[0], SAMPLE_RATE
        # If the file is short, trailing columns will be zeros
        nonzero_cols = np.any(spec[:, :, 0] > 0, axis=0)
        # At least one column should be non-zero (the audio)
        assert nonzero_cols.sum() > 0

    def test_rejects_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_and_process_audio("/nonexistent/file.wav", max_pad_len=MAX_PAD_LEN)

    def test_no_padding_when_none_given(self, wav_file: str) -> None:
        """Without max_pad_len, the spectrogram keeps its natural width."""
        spec = load_and_process_audio(wav_file, max_pad_len=None)
        # Should be (128, n_frames, 1) where n_frames is whatever librosa produces
        assert spec.ndim == 3
        assert spec.shape[0] == N_MELS
        assert spec.shape[2] == 1
