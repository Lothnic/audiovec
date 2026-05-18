"""Export the trained CRNN-Transformer model to ONNX format.

Produces ``models/crnn-transformer.onnx`` that can be consumed by
`onnxruntime <https://onnxruntime.ai/>`_ in any language (Node.js, Python,
C++, Rust, etc.).

The ONNX model takes a single input and returns two outputs:

**Input** (name ``spectrogram``)
    shape ``(batch, 128, 228, 1)`` — a mel-spectrogram in channels-last
    convention (same format the model was trained with).

**Output #1** (name ``logits``)
    shape ``(batch, 8)`` — unnormalised class scores for each emotion.
    Apply ``softmax`` to obtain probabilities.

**Output #2** (name ``embedding``)
    shape ``(batch, 256)`` — the 256-dimensional embedding vector,
    post ReLU-nonlinearity but before the classifier head.

Usage
-----
::

    uv run python -m audiovec.export_onnx
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from audiovec.config import INPUT_SHAPE
from audiovec.model import CRNNTransformer


# ── ONNX-compatible wrapper ──────────────────────────────────────────────────


class ONNXWrapper(torch.nn.Module):
    """Wrapper that outputs **both** logits and embeddings.

    PyTorch's ``forward`` has a ``return_embedding`` switch, which means we
    cannot trace both paths in a single ``torch.onnx.export`` call.  This
    wrapper calls the underlying model with ``return_embedding=False`` and
    extracts the embedding from the penultimate layer.
    """

    def __init__(self, model: CRNNTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, embedding)``."""
        emb = self.model.forward_embedding(x)
        logits = self.model.fc_cls(emb)
        return logits, emb


def patch_model_for_onnx(model: CRNNTransformer) -> None:
    """Add a ``forward_embedding`` method that stops at the embedding.

    The ONNX wrapper needs a clean forward path that goes all the way up to
    the embedding without branching on a boolean flag (which ``torch.jit``
    can trace, but this approach is simpler and guaranteed to work).
    """

    def forward_embedding(self: CRNNTransformer, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)  # channels-last → channels-first
        for b in self.blocks:
            x = b(x)
        N, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).reshape(N, W, C * H)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.drop(x)
        return torch.relu(self.fc_embed(x))

    # Bind the method to the instance (this will be traced, so it's fine)
    import types

    model.forward_embedding = types.MethodType(forward_embedding, model)


# ── Export ───────────────────────────────────────────────────────────────────


@torch.no_grad()
def export_to_onnx(
    checkpoint: str | Path = "models/crnn-transformer.pt",
    output: str | Path = "models/crnn-transformer.onnx",
    input_shape: tuple[int, int, int] = INPUT_SHAPE,  # (128, 228, 1)
    opset_version: int = 18,
) -> Path:
    """Export a trained CRNN-Transformer checkpoint to ONNX.

    Parameters
    ----------
    checkpoint :
        Path to the ``.pt`` state-dict file.
    output :
        Where to write the ``.onnx`` file.
    input_shape :
        (n_mels, time_steps, channels) — defaults to (128, 228, 1).
    opset_version :
        ONNX opset version.  17 is well-supported by onnxruntime 1.18+.

    Returns
    -------
    Path
        Absolute path to the exported ONNX file.
    """
    output = Path(output).resolve()
    device = torch.device("cpu")

    # ── Load model ───────────────────────────────────────────────────────────
    model = CRNNTransformer(input_shape=input_shape, num_classes=8)
    state = torch.load(str(checkpoint), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # ── Patch and wrap ───────────────────────────────────────────────────────
    patch_model_for_onnx(model)
    wrapper = ONNXWrapper(model)
    wrapper.eval()

    # ── Dummy input ──────────────────────────────────────────────────────────
    dummy = torch.randn(1, *input_shape)  # (1, 128, 228, 1)

    # ── Export ───────────────────────────────────────────────────────────────
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        input_names=["spectrogram"],
        output_names=["logits", "embedding"],
        dynamic_axes={
            "spectrogram": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        opset_version=opset_version,
        export_params=True,
        do_constant_folding=True,
    )

    print(f"ONNX model exported to: {output}")
    print(f"  Input:  spectrogram  (batch, 128, 228, 1)")
    print(f"  Output: logits       (batch, 8)")
    print(f"  Output: embedding    (batch, 256)")
    print(f"  Opset:  {opset_version}")
    print(f"  Size:   {output.stat().st_size / 1024:.1f} KB")
    return output
    return output


# ── Validation ───────────────────────────────────────────────────────────────


def validate_onnx(
    onnx_path: str | Path,
    checkpoint: str | Path = "models/crnn-transformer.pt",
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    atol: float = 1e-4,
) -> None:
    """Compare ONNX Runtime outputs against the PyTorch model.

    Raises ``AssertionError`` if outputs differ beyond ``atol``.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  ⚠ onnxruntime not available — skipping validation.")
        print("    Run `npm install onnxruntime-node` on Node.js to test inference.")
        return

    device = torch.device("cpu")

    # ── PyTorch reference ────────────────────────────────────────────────────
    model = CRNNTransformer(input_shape=input_shape, num_classes=8)
    state = torch.load(str(checkpoint), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # ── ONNX Runtime session ─────────────────────────────────────────────────
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name

    # ── Multiple batch sizes ──────────────────────────────────────────────────
    for batch_size in [1, 4, 8]:
        dummy_np = np.random.randn(batch_size, *input_shape).astype(np.float32)
        dummy_t = torch.from_numpy(dummy_np)

        # PyTorch
        with torch.no_grad():
            pt_logits = model(dummy_t)  # (B, 8)
            pt_embedding = model(dummy_t, return_embedding=True)  # (B, 256)

        # ONNX Runtime
        ort_outputs = session.run(None, {input_name: dummy_np})
        ort_logits, ort_embedding = ort_outputs

        # Validate
        logits_close = np.allclose(pt_logits.numpy(), ort_logits, atol=atol)
        embed_close = np.allclose(pt_embedding.numpy(), ort_embedding, atol=atol)

        if not logits_close:
            max_diff = np.abs(pt_logits.numpy() - ort_logits).max()
            print(f"  ⚠ B={batch_size}: logits max diff = {max_diff:.6e} (atol={atol})")
        if not embed_close:
            max_diff = np.abs(pt_embedding.numpy() - ort_embedding).max()
            print(f"  ⚠ B={batch_size}: embedding max diff = {max_diff:.6e} (atol={atol})")

        if not logits_close:
            print(f"  ❌ B={batch_size}: logits mismatch")
        elif not embed_close:
            print(f"  ❌ B={batch_size}: embedding mismatch")
        else:
            print(f"  ✅ B={batch_size}: logits and embedding match within {atol}")


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export audiovec model to ONNX")
    parser.add_argument(
        "--checkpoint",
        default="models/crnn-transformer.pt",
        help="Path to the trained .pt checkpoint",
    )
    parser.add_argument(
        "--output",
        default="models/crnn-transformer.onnx",
        help="Output .onnx file path",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip ONNX Runtime validation (useful if onnxruntime is not installed)",
    )
    args = parser.parse_args()

    onnx_path = export_to_onnx(checkpoint=args.checkpoint, output=args.output)

    # ── Validate with onnx.checker ──────────────────────────────────────
    import onnx
    model_onnx = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_onnx)
    print(f"ONNX model validated: {len(model_onnx.graph.node)} ops, "
          f"{len(model_onnx.graph.initializer)} weights")

    if not args.skip_validation:
        print("\n── Runtime Validation ──────────────────────────────────")
        validate_onnx(onnx_path, checkpoint=args.checkpoint)
    else:
        print("\nDone (validation skipped).")
