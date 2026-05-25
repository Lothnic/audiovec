#!/usr/bin/env python3
"""
Unified retrain → export → upload → redeploy pipeline for audiovec.

Usage
-----
    # Full pipeline — train, export, upload to HF Hub, trigger Vercel redeploy
    uv run python deploy_pipeline.py

    # Resume from a previously trained checkpoint (skip training)
    uv run python deploy_pipeline.py --skip-train

    # Only export ONNX from an existing checkpoint (no upload, no redeploy)
    uv run python deploy_pipeline.py --skip-train --skip-upload --skip-redeploy

    # Only trigger a Vercel redeploy (expects files already pushed)
    uv run python deploy_pipeline.py --skip-train --skip-export

Environment variables
---------------------
HF_TOKEN                Hugging Face token with write access to lothnic/audiovec
                        (https://huggingface.co/settings/tokens)
VERCEL_DEPLOY_HOOK_URL  Vercel deploy hook URL (Settings → Git → Deploy Hooks)
GITHUB_TOKEN            (optional) GitHub token for committing ONNX files
                        via the API (https://github.com/settings/tokens)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
PT_CHECKPOINT = PROJECT_ROOT / "models" / "audiovec_model.pt"
ONNX_DIR = PROJECT_ROOT / "models"
ONNX_FILES = ["crnn-transformer.onnx", "crnn-transformer.onnx.data"]
WEB_MODELS_DIR = PROJECT_ROOT / "web" / "app" / "api" / "predict" / "models"

# HF Hub
HF_REPO_ID = "lothnic/audiovec"
HF_FILENAME = "models/audiovec_model.pt"  # stored under models/ in the repo

# Vercel
VERCEL_DEPLOY_HOOK_URL = os.environ.get("VERCEL_DEPLOY_HOOK_URL", "")

# ═══════════════════════════════════════════════════════════════════════
#  Model versioning
# ═══════════════════════════════════════════════════════════════════════


def resolve_model_version(custom_version: str | None) -> str:
    """Return a version string: custom value, git describe, or date-based."""
    if custom_version:
        return custom_version

    # Try git describe — gives the most meaningful label
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            capture_output=True, text=True, check=True, cwd=PROJECT_ROOT,
        )
        version = result.stdout.strip()
        if version:
            return version
    except Exception:
        pass

    # Fallback: date-based version
    return time.strftime("%Y%m%d.%H%M%S")


MODEL_VERSION: str = ""  # resolved in main() from CLI arg

# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def log(step: str, message: str) -> None:
    """Print a coloured pipeline log line."""
    print(f"\n  [{step}] {message}")


def run(cmd: list[str], cwd: str | Path | None = None) -> None:
    """Run a subprocess, raising on failure."""
    sys.stdout.flush()
    subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, check=True)


# ═══════════════════════════════════════════════════════════════════════
#  Steps
# ═══════════════════════════════════════════════════════════════════════


def step_train() -> None:
    """Train the PyTorch CRNN-Transformer model."""
    log("TRAIN", "Starting training (this may take a while)…")
    t0 = time.time()
    run(["uv", "run", "python", "train_crnn.py"])
    elapsed = time.time() - t0

    if PT_CHECKPOINT.exists():
        size_mb = PT_CHECKPOINT.stat().st_size / 1024 / 1024
        log("TRAIN", f"Checkpoint saved → {PT_CHECKPOINT} ({size_mb:.1f} MB, {elapsed:.0f}s)")
    else:
        print(f"  ⚠ Expected checkpoint not found at {PT_CHECKPOINT}", file=sys.stderr)
        sys.exit(1)


def step_export_onnx(skip_validation: bool = False) -> None:
    """Export the trained PyTorch model to ONNX format."""
    log("EXPORT", f"Exporting {PT_CHECKPOINT} to ONNX…")
    cmd = [
        "uv", "run", "python", "-m", "audiovec.export_onnx",
        "--checkpoint", str(PT_CHECKPOINT),
        "--output", str(ONNX_DIR / "crnn-transformer.onnx"),
    ]
    if skip_validation:
        cmd.append("--skip-validation")

    t0 = time.time()
    run(cmd)
    elapsed = time.time() - t0

    # Report all expected ONNX artefacts
    for fname in ONNX_FILES:
        fp = ONNX_DIR / fname
        if fp.exists():
            size_mb = fp.stat().st_size / 1024 / 1024
            log("EXPORT", f"  {fname}  ({size_mb:.1f} MB)")
        else:
            log("EXPORT", f"  {fname}  (not created — model uses inline weights)")

    log("EXPORT", f"ONNX export done ({elapsed:.0f}s)")


def step_copy_to_web() -> None:
    """Copy ONNX files to the web app's model directory so they're
    ready to commit to GitHub."""
    log("COPY", f"Copying ONNX files to {WEB_MODELS_DIR} …")
    WEB_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    copied_any = False
    for fname in ONNX_FILES:
        src = ONNX_DIR / fname
        dst = WEB_MODELS_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            size_mb = dst.stat().st_size / 1024 / 1024
            log("COPY", f"  {fname}  → {dst}  ({size_mb:.1f} MB)")
            copied_any = True
        else:
            log("COPY", f"  {fname}  (skipped — file not found)")

    # ── Write version.json ────────────────────────────────────────────────
    import json
    version_file = WEB_MODELS_DIR / "version.json"
    version_data = {
        "version": MODEL_VERSION,
        "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    version_file.write_text(json.dumps(version_data, indent=2) + "\n")
    log("COPY", f"  version.json  → {version_file}  ({MODEL_VERSION})")

    if copied_any:
        log("COPY",
            "ONNX files copied to web app. "
            "Run the following to deploy them to GitHub:\n"
            "    git add web/app/api/predict/models/ && git commit -m \"Update ONNX model {MODEL_VERSION}\" && git push")
    else:
        log("COPY", "No ONNX files copied.")


def step_upload_hf() -> None:
    """Upload the PyTorch checkpoint to Hugging Face Hub."""
    log("UPLOAD", f"Uploading {PT_CHECKPOINT} to HF Hub ({HF_REPO_ID})…")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "  ⚠ huggingface_hub is not installed.\n"
            "    Install it with:  uv pip install huggingface_hub  or  pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "  ⚠ HF_TOKEN environment variable not set.\n"
            "    Create a token at https://huggingface.co/settings/tokens and set:\n"
            "      export HF_TOKEN=hf_…",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi(token=token)
    t0 = time.time()

    commit_message = f"Deploy audiovec model v{MODEL_VERSION}"
    api.upload_file(
        repo_id=HF_REPO_ID,
        path_or_fileobj=str(PT_CHECKPOINT),
        path_in_repo=HF_FILENAME,
        repo_type="space",
        commit_message=commit_message,
    )

    elapsed = time.time() - t0
    size_mb = PT_CHECKPOINT.stat().st_size / 1024 / 1024
    log("UPLOAD", f"Uploaded {size_mb:.1f} MB → hf.co/{HF_REPO_ID}/{HF_FILENAME} ({elapsed:.0f}s)")
    log("UPLOAD", f"  HF commit: {commit_message}")


def step_redeploy_vercel() -> None:
    """Trigger a Vercel deploy hook to redeploy the Next.js app."""
    log("REDEPLOY", "Triggering Vercel deploy hook…")

    if not VERCEL_DEPLOY_HOOK_URL:
        print(
            "  ⚠ VERCEL_DEPLOY_HOOK_URL not set.\n"
            "    Create a hook in Vercel (Settings → Git → Deploy Hooks) and set:\n"
            "      export VERCEL_DEPLOY_HOOK_URL=https://api.vercel.com/v1/integrations/deploy/…",
            file=sys.stderr,
        )
        sys.exit(1)

    import urllib.request
    t0 = time.time()

    try:
        req = urllib.request.Request(
            VERCEL_DEPLOY_HOOK_URL,
            method="POST",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
    except Exception as e:
        print(f"  ❌ Failed to trigger Vercel deploy: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - t0
    log("REDEPLOY", f"Vercel deploy triggered ({elapsed:.0f}s)")
    log("REDEPLOY", f"  Response: {body[:200]}")
    log("REDEPLOY",
        "  Monitor the deploy at:\n"
        "    https://vercel.com/<team>/audiovec/deployments")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified retrain → export → upload → redeploy pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment variables:\n"
            "  HF_TOKEN               Hugging Face token (required for upload)\n"
            "  VERCEL_DEPLOY_HOOK_URL  Vercel deploy hook URL (required for redeploy)\n"
            "\n"
            "Examples:\n"
            "  uv run python deploy_pipeline.py\n"
            "  uv run python deploy_pipeline.py --skip-train\n"
            "  uv run python deploy_pipeline.py --skip-train --skip-export\n"
        ),
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training; use existing models/audiovec_model.pt",
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Skip ONNX export and copy",
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Skip HF Hub upload",
    )
    parser.add_argument(
        "--skip-redeploy", action="store_true",
        help="Skip Vercel redeploy hook",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip ONNX Runtime validation during export",
    )
    parser.add_argument(
        "--model-version", type=str, default=None,
        help="Version string written to version.json and attached to traces. "
             "Auto-generated from git describe + date if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve MODEL_VERSION globally so all steps use the same label
    global MODEL_VERSION
    MODEL_VERSION = resolve_model_version(args.model_version)
    log("VERSION", f"Model version: {MODEL_VERSION}")

    print("╔══════════════════════════════════════════════════╗")
    print("║   audiovec  —  Deploy Pipeline                  ║")
    print("╚══════════════════════════════════════════════════╝")

    # ── 1. Train ────────────────────────────────────────────────────────
    if not args.skip_train:
        if not (PROJECT_ROOT / "train_crnn.py").exists():
            print("  ❌ train_crnn.py not found at project root", file=sys.stderr)
            sys.exit(1)
        step_train()
    else:
        if PT_CHECKPOINT.exists():
            log("SKIP", f"Training skipped (using existing {PT_CHECKPOINT})")
        else:
            print(f"  ❌ No checkpoint found at {PT_CHECKPOINT}", file=sys.stderr)
            sys.exit(1)

    # ── 2. Export ONNX ──────────────────────────────────────────────────
    if not args.skip_export:
        step_export_onnx(skip_validation=args.skip_validation)

    # ── 3. Copy ONNX to web app ─────────────────────────────────────────
    onnx_files_exist = any((ONNX_DIR / f).exists() for f in ONNX_FILES)
    if onnx_files_exist:
        step_copy_to_web()
    else:
        log("SKIP", "ONNX copy skipped (no ONNX files found in models/)")

    # ── 4. Upload to HF Hub ─────────────────────────────────────────────
    if not args.skip_upload:
        step_upload_hf()
    else:
        log("SKIP", "HF Hub upload skipped")

    # ── 5. Vercel redeploy ──────────────────────────────────────────────
    if not args.skip_redeploy:
        step_redeploy_vercel()
    else:
        log("SKIP", "Vercel redeploy skipped")

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   Pipeline complete ✓                           ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print("  Next steps:")
    print("    1. Commit & push the ONNX files to GitHub:")
    print("       git add web/app/api/predict/models/")
    print("       git commit -m \"Update ONNX model\"")
    print("       git push")
    print("    2. Monitor the Vercel deploy")
    print("    3. Check the new model in action!")
    print()


if __name__ == "__main__":
    main()
