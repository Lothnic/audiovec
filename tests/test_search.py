"""Tests for audiovec.search — RAVDESS filename parsing, cosine similarity search."""

import numpy as np
import pytest

from audiovec.search import find_similar, parse_ravdess_filename


class TestParseRavdessFilename:
    """RAVDESS filename → metadata extraction."""

    @pytest.mark.parametrize(
        "filename,expected_emotion,expected_code,expected_actor",
        [
            ("03-01-01-01-01-01-01.wav", "neutral", 1, 1),
            ("03-01-02-01-01-01-01.wav", "calm", 2, 1),
            ("03-01-03-02-02-02-12.wav", "happy", 3, 12),
            ("03-01-05-01-01-02-21.wav", "angry", 5, 21),
            ("03-01-08-02-02-02-24.wav", "surprised", 8, 24),
        ],
    )
    def test_parses_correctly(self, filename, expected_emotion, expected_code, expected_actor):
        meta = parse_ravdess_filename(filename)
        assert meta["emotion"] == expected_emotion
        assert meta["emotion_code"] == expected_code
        assert meta["actor"] == expected_actor

    def test_parses_intensity(self):
        meta = parse_ravdess_filename("03-01-05-01-01-02-21.wav")
        assert meta["intensity"] == 1

    def test_parses_strong_intensity(self):
        meta = parse_ravdess_filename("03-01-05-02-01-02-21.wav")
        assert meta["intensity"] == 2

    def test_statement_and_repetition(self):
        meta = parse_ravdess_filename("03-01-03-02-02-01-11.wav")
        assert meta["statement"] == 2
        assert meta["repetition"] == 1

    def test_unknown_emotion_code(self):
        meta = parse_ravdess_filename("03-01-99-01-01-01-01.wav")
        assert meta["emotion"] == "unknown"
        assert meta["emotion_code"] == 99


class TestFindSimilar:
    """Cosine similarity search against a reference set."""

    def test_returns_correct_number_of_results(self, ravdess_embeddings, ravdess_metadata):
        query = ravdess_embeddings[0]
        results = find_similar(query, ravdess_embeddings, ravdess_metadata, k=3)
        assert len(results) == 3

    def test_top_result_is_identical_vector(self, ravdess_embeddings, ravdess_metadata):
        query = ravdess_embeddings[0]
        results = find_similar(query, ravdess_embeddings, ravdess_metadata, k=5)
        assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_results_are_sorted_descending(self, ravdess_embeddings, ravdess_metadata):
        query = ravdess_embeddings[0]
        results = find_similar(query, ravdess_embeddings, ravdess_metadata, k=5)
        sims = [r["similarity"] for r in results]
        assert all(sims[i] >= sims[i + 1] for i in range(len(sims) - 1))

    def test_orthogonal_vectors_return_zero_similarity(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        meta = [{"path": "/a"}, {"path": "/b"}]
        # Normalize
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        query = np.array([1.0, 0.0])
        results = find_similar(query, emb, meta, k=2)
        # /a should have similarity 1.0, /b should have 0.0
        assert results[1]["similarity"] == pytest.approx(0.0, abs=1e-6)

    def test_k_larger_than_dataset(self, ravdess_embeddings, ravdess_metadata):
        query = ravdess_embeddings[0]
        results = find_similar(query, ravdess_embeddings, ravdess_metadata, k=100)
        assert len(results) == 20  # only 20 in the fixture

    def test_returns_metadata_fields(self, ravdess_embeddings, ravdess_metadata):
        query = ravdess_embeddings[5]
        results = find_similar(query, ravdess_embeddings, ravdess_metadata, k=1)
        r = results[0]
        assert "path" in r
        assert "emotion" in r
        assert "actor" in r
        assert "similarity" in r

    def test_empty_reference_set(self):
        emb = np.empty((0, 256))
        meta: list[dict] = []
        query = np.random.randn(256)
        results = find_similar(query, emb, meta, k=5)
        assert results == []

    def test_different_query_finds_different_top(self, ravdess_embeddings, ravdess_metadata):
        query_a = ravdess_embeddings[0]
        query_b = ravdess_embeddings[5]
        top_a = find_similar(query_a, ravdess_embeddings, ravdess_metadata, k=1)[0]
        top_b = find_similar(query_b, ravdess_embeddings, ravdess_metadata, k=1)[0]
        # Extremely unlikely with seeded rng and different embeddings
        assert top_a["path"] != top_b["path"]
