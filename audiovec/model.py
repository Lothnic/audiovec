"""Audio emotion embedding models (PyTorch).

Architecture
------------
Two architectures are available:

**Default — CRNN-Transformer** (recommended, 659K params):
    3× Conv blocks → input projection → positional encoding →
    Transformer Encoder (2 layers, 4 heads) → mean-pool → embedding → classifier

**Legacy — AudiovecModel (BiGRU)** (2.1M params):
    3× Conv blocks → BiGRU (2 layers) → mean-pool → embedding → classifier

Input shape convention
----------------------
Training data arrives as numpy arrays in channels-last format (N, H, W, C).
PyTorch expects channels-first (N, C, H, W).  The :meth:`forward` method
handles the permutation automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → MaxPool2d → Dropout2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.pool(F.relu(self.bn(self.conv(x)))))


class AudiovecModel(nn.Module):
    """CRNN that maps mel-spectrograms to emotion logits or embeddings.

    Parameters
    ----------
    input_shape :
        (height, width, channels) in channels-last convention, e.g. (128, 228, 1).
    embedding_dim :
        Size of the embedding vector (default 256).
    num_classes :
        Number of emotion classes in the classifier head.
    dropout_rates :
        Dropout probabilities: (conv1, conv2, conv3, rnn_dropout, fc_dropout).
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        embedding_dim: int = 256,
        num_classes: int = 8,
        dropout_rates: tuple[float, ...] = (0.05, 0.05, 0.1, 0.3, 0.3),
    ):
        super().__init__()
        h, w, c = input_shape

        # Pad dropout_rates to at least 5 entries
        if len(dropout_rates) < 5:
            dropout_rates = dropout_rates + (0.0,) * (5 - len(dropout_rates))

        # ── Convolutional feature extractor ────────────────────────────────
        self.block1 = ConvBlock(c, 32, dropout=dropout_rates[0])
        self.block2 = ConvBlock(32, 64, dropout=dropout_rates[1])
        self.block3 = ConvBlock(64, 128, dropout=dropout_rates[2])

        # Compute RNN input dimensions via dummy forward
        with torch.no_grad():
            x = torch.zeros(1, c, h, w)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            _, c_out, h_out, w_out = x.shape
            rnn_input_size = c_out * h_out  # features per time step
            self._seq_len = w_out

        # ── Bidirectional GRU ──────────────────────────────────────────────
        rnn_drop = dropout_rates[3]
        self.gru = nn.GRU(
            rnn_input_size,
            128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=rnn_drop if rnn_drop > 0 else 0,
        )

        # ── Embedding + classifier head ────────────────────────────────────
        fc_drop = dropout_rates[4]
        self.fc_dropout = nn.Dropout(fc_drop) if fc_drop > 0 else nn.Identity()
        self.fc_embed = nn.Linear(256, embedding_dim)  # BiGRU output = 128*2
        self.fc_classifier = nn.Linear(embedding_dim, num_classes)

        # Metadata
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x :
            Input tensor of shape (N, H, W, C) in channels-last convention.
        return_embedding :
            If True, return the embedding instead of class logits.

        Returns
        -------
        torch.Tensor
            Classification logits (N, num_classes) or embeddings (N, embedding_dim).
        """
        # Channels-last → channels-first
        x = x.permute(0, 3, 1, 2)

        # Convolutional blocks
        x = self.block1(x)  # (N,  32, H/2, W/2)
        x = self.block2(x)  # (N,  64, H/4, W/4)
        x = self.block3(x)  # (N, 128, H/8, W/8)

        # Reshape for RNN: (N, C, H', W') → (N, W', C*H')
        N, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).reshape(N, W, C * H)  # (N, seq_len, features)

        # Bidirectional GRU
        x, _ = self.gru(x)  # (N, seq_len, 256)

        # Mean-pool over the time axis
        x = x.mean(dim=1)   # (N, 256)

        # Dropout → embedding → classifier
        x = self.fc_dropout(x)
        embedding = F.relu(self.fc_embed(x))

        if return_embedding:
            return embedding
        return self.fc_classifier(embedding)

    @torch.no_grad()
    def predict_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: forward pass returning the embedding vector."""
        self.eval()
        return self.forward(x, return_embedding=True)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: forward pass returning softmax probabilities."""
        self.eval()
        logits = self.forward(x, return_embedding=False)
        return F.softmax(logits, dim=-1)


# ═══════════════════════════════════════════════════════════════════════
#  CRNN-Transformer  (default architecture)
# ═══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Learned positional encoding for the transformer."""
    def __init__(self, d_model: int, max_len: int = 60):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class CRNNTransformer(nn.Module):
    """3× Conv → Transformer Encoder (2 layers, 4 heads) → mean-pool → embedding → classifier.

    This is the default architecture — best accuracy (72.92%) with only 659K params.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        embedding_dim: int = 256,
        num_classes: int = 8,
        dropout_rates: tuple[float, ...] = (0.05, 0.05, 0.1, 0.3, 0.3),
    ):
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
            for b in self.blocks:
                x = b(x)
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
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        for b in self.blocks:
            x = b(x)
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


def build_model(
    input_shape: tuple[int, int, int],
    embedding_dim: int = 256,
    num_classes: int = 8,
    dropout_rates: tuple[float, ...] = (0.05, 0.05, 0.1, 0.3, 0.3),
) -> AudiovecModel:
    """Build the audiovec CRNN model (factory function).

    Parameters
    ----------
    input_shape :
        (n_mels, time_steps, 1) — channels-last, e.g. (128, 228, 1).
    embedding_dim :
        Size of the embedding vector (default 256).
    num_classes :
        Number of emotion classes in the classifier head.
    dropout_rates :
        Dropout probabilities (3 conv + 1 rnn + 1 fc).

    Returns
    -------
    AudiovecModel
    """
    return AudiovecModel(
        input_shape=input_shape,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        dropout_rates=dropout_rates,
    )


def build_transformer_model(
    input_shape: tuple[int, int, int],
    embedding_dim: int = 256,
    num_classes: int = 8,
    dropout_rates: tuple[float, ...] = (0.05, 0.05, 0.1, 0.3, 0.3),
) -> CRNNTransformer:
    """Build the CRNN-Transformer model (factory function).

    This is the default architecture — best accuracy (72.92%) with 659K params.
    """
    return CRNNTransformer(
        input_shape=input_shape,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        dropout_rates=dropout_rates,
    )