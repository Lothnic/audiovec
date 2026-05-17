"""Tests for audiovec.model — forward-pass shapes, embedding extraction."""

import numpy as np
import pytest
import torch

from audiovec.model import (
    CRNNTransformer,
    ConvBlock,
    PositionalEncoding,
    build_model,
    build_transformer_model,
)


# ── ConvBlock ─────────────────────────────────────────────────────────────────


class TestConvBlock:
    """Individual convolutional block."""

    def test_output_shape(self) -> None:
        block = ConvBlock(1, 32, dropout=0.05)
        x = torch.randn(2, 1, 128, 228)
        out = block(x)
        # MaxPool2d(2) halves H and W
        assert out.shape == (2, 32, 64, 114)

    def test_identity_dropout_when_zero(self) -> None:
        block = ConvBlock(1, 32, dropout=0.0)
        assert isinstance(block.dropout, torch.nn.Identity)


# ── PositionalEncoding ───────────────────────────────────────────────────────


class TestPositionalEncoding:
    """Learned positional encoding."""

    def test_output_shape(self) -> None:
        pe = PositionalEncoding(d_model=128, max_len=60)
        x = torch.randn(2, 30, 128)
        out = pe(x)
        assert out.shape == (2, 30, 128)

    def test_adds_to_input(self) -> None:
        pe = PositionalEncoding(d_model=128, max_len=60)
        x = torch.zeros(1, 30, 128)
        out = pe(x)
        # After adding PE, the output should not be zero
        assert not torch.allclose(out, torch.zeros_like(out))


# ── CRNNTransformer (default architecture) ───────────────────────────────────


class TestCRNNTransformer:
    """Forward pass through the default model."""

    def test_logits_shape(self, model: CRNNTransformer, dummy_spectrogram: np.ndarray) -> None:
        batch = torch.from_numpy(dummy_spectrogram[np.newaxis, ...]).float()
        logits = model(batch)
        assert logits.shape == (1, 8)

    def test_embedding_shape(self, model: CRNNTransformer, dummy_spectrogram: np.ndarray) -> None:
        batch = torch.from_numpy(dummy_spectrogram[np.newaxis, ...]).float()
        emb = model(batch, return_embedding=True)
        assert emb.shape == (1, 256)

    def test_batch_forward(self, model: CRNNTransformer, dummy_spectrogram_batch: np.ndarray) -> None:
        batch = torch.from_numpy(dummy_spectrogram_batch).float()
        logits = model(batch)
        assert logits.shape == (4, 8)

    def test_logits_and_embedding_differ(self, model: CRNNTransformer, dummy_spectrogram: np.ndarray) -> None:
        batch = torch.from_numpy(dummy_spectrogram[np.newaxis, ...]).float()
        logits = model(batch)
        emb = model(batch, return_embedding=True)
        assert logits.shape[1] == 8 and emb.shape[1] == 256
        # Embedding should not simply be the logits padded with zeros
        assert not torch.allclose(emb[:, :8], logits)

    def test_predict_embedding_convenience(self, model: CRNNTransformer, dummy_spectrogram: np.ndarray) -> None:
        batch = torch.from_numpy(dummy_spectrogram[np.newaxis, ...]).float()
        emb = model.predict_embedding(batch)
        assert emb.shape == (1, 256)

    def test_predict_proba_sums_to_one(self, model: CRNNTransformer, dummy_spectrogram: np.ndarray) -> None:
        batch = torch.from_numpy(dummy_spectrogram[np.newaxis, ...]).float()
        probs = model.predict_proba(batch)
        assert probs.shape == (1, 8)
        assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5)

    def test_gradient_not_tracked_during_inference(self, model: CRNNTransformer, dummy_spectrogram: np.ndarray) -> None:
        """Eval mode alone does NOT disable autograd; torch.no_grad() is required."""
        batch = torch.from_numpy(dummy_spectrogram[np.newaxis, ...]).float()
        # Without no_grad: output tracks gradients (eval doesn't disable autograd)
        out = model(batch)
        assert out.requires_grad
        # With no_grad: output does not track gradients
        with torch.no_grad():
            out_no_grad = model(batch)
        assert not out_no_grad.requires_grad


# ── Factory functions ────────────────────────────────────────────────────────


class TestBuildModel:
    """Factory functions produce usable models."""

    INPUT_SHAPE = (128, 228, 1)

    def test_build_transformer_model(self) -> None:
        model = build_transformer_model(self.INPUT_SHAPE, embedding_dim=256, num_classes=8)
        assert isinstance(model, CRNNTransformer)
        assert model.embedding_dim == 256
        assert model.num_classes == 8

    def test_build_model_legacy(self) -> None:
        model = build_model(self.INPUT_SHAPE, embedding_dim=256, num_classes=8)
        from audiovec.model import AudiovecModel
        assert isinstance(model, AudiovecModel)
        assert model.embedding_dim == 256
        assert model.num_classes == 8

    @pytest.mark.parametrize("factory", ["build_transformer_model", "build_model"])
    def test_both_architectures_forward(self, factory: str) -> None:
        """Both architectures produce correct logits shape on a dummy input."""
        import audiovec.model as M
        fn = getattr(M, factory)
        m = fn(self.INPUT_SHAPE, embedding_dim=256, num_classes=8)
        m.eval()
        x = torch.randn(1, 128, 228, 1)
        out = m(x)
        assert out.shape == (1, 8)
