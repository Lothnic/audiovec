"""Tests for audiovec.predict — prediction and embedding extraction."""

import numpy as np
import pytest

from audiovec.predict import extract_embedding, predict_emotion


class TestPredictEmotion:
    """Emotion prediction from a spectrogram."""

    def test_returns_emotion_label(self, model, dummy_spectrogram):
        emotion, confidence, probs = predict_emotion(model, dummy_spectrogram)
        assert isinstance(emotion, str)
        assert len(emotion) > 0

    def test_confidence_is_probability(self, model, dummy_spectrogram):
        _, confidence, _ = predict_emotion(model, dummy_spectrogram)
        assert 0.0 <= confidence <= 1.0

    def test_probabilities_shape(self, model, dummy_spectrogram):
        _, _, probs = predict_emotion(model, dummy_spectrogram)
        assert probs.shape == (8,)

    def test_probabilities_sum_to_one(self, model, dummy_spectrogram):
        _, _, probs = predict_emotion(model, dummy_spectrogram)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_max_probability_matches_confidence(self, model, dummy_spectrogram):
        _, confidence, probs = predict_emotion(model, dummy_spectrogram)
        assert abs(probs.max() - confidence) < 1e-6

    def test_all_probabilities_non_negative(self, model, dummy_spectrogram):
        _, _, probs = predict_emotion(model, dummy_spectrogram)
        assert (probs >= 0.0).all()


class TestExtractEmbedding:
    """Embedding vector extraction."""

    def test_shape(self, model, dummy_spectrogram):
        embedding = extract_embedding(model, dummy_spectrogram)
        assert embedding.shape == (256,)

    def test_dtype(self, model, dummy_spectrogram):
        embedding = extract_embedding(model, dummy_spectrogram)
        assert embedding.dtype == np.float32 or embedding.dtype == np.float64

    def test_not_all_zeros(self, model, dummy_spectrogram):
        embedding = extract_embedding(model, dummy_spectrogram)
        assert not np.allclose(embedding, 0.0)

    def test_different_inputs_different_embeddings(self, model):
        spec_a = np.random.randn(128, 228, 1).astype(np.float32)
        spec_b = np.random.randn(128, 228, 1).astype(np.float32)
        emb_a = extract_embedding(model, spec_a)
        emb_b = extract_embedding(model, spec_b)
        # Very unlikely to be identical with random inputs
        assert not np.allclose(emb_a, emb_b)
