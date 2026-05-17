"""Tests for audiovec.config — constants and emotion mapping."""

from audiovec.config import (
    SAMPLE_RATE,
    N_MELS,
    FMAX,
    MAX_PAD_LEN,
    EMBEDDING_DIM,
    EPOCHS,
    BATCH_SIZE,
    EMOTION_MAPPING,
)


class TestAudioProcessingConstants:
    """Sample rate, mel bands, frequency limits."""

    def test_sample_rate(self) -> None:
        assert SAMPLE_RATE == 22050

    def test_n_mels(self) -> None:
        assert N_MELS == 128

    def test_fmax(self) -> None:
        assert FMAX == 8000

    def test_max_pad_len(self) -> None:
        assert MAX_PAD_LEN == 228


class TestModelConstants:
    """Embedding dimension and input shape."""

    def test_embedding_dim(self) -> None:
        assert EMBEDDING_DIM == 256


class TestTrainingConstants:
    """Training hyperparameters."""

    def test_epochs(self) -> None:
        assert EPOCHS == 30

    def test_batch_size(self) -> None:
        assert BATCH_SIZE == 32


class TestEmotionMapping:
    """RAVDESS emotion code → label mapping."""

    def test_has_all_eight_emotions(self) -> None:
        assert len(EMOTION_MAPPING) == 8

    def test_emotion_codes_are_contiguous(self) -> None:
        assert list(EMOTION_MAPPING.keys()) == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_emotion_names(self) -> None:
        expected = {
            1: "neutral",
            2: "calm",
            3: "happy",
            4: "sad",
            5: "angry",
            6: "fearful",
            7: "disgust",
            8: "surprised",
        }
        assert EMOTION_MAPPING == expected
