"""Download and verify local zero-VRAM models in models/ for L.O.R.A.

Downloads and installs:
1. Kokoro TTS (kokoro-v1.0.onnx + voices-v1.0.bin) -> models/kokoro
2. Laya Decision Router (AgencyL1/Laya-Lora-4B) -> models/laya-lora-ft
3. Embedding Gemma 2 (google/embeddinggemma-2) -> models/embeddinggemma-2
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_models")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

KOKORO_ONNX_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    logger.info("Downloading %s -> %s", url, dest.name)
    try:
        def _reporthook(count, block_size, total_size):
            if total_size > 0:
                pct = min(100.0, (count * block_size / total_size) * 100.0)
                sys.stdout.write(f"\r  Downloading {dest.name}: {pct:.1f}% ({total_size // (1024*1024)} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, tmp, reporthook=_reporthook)
        sys.stdout.write("\n")
        sys.stdout.flush()
        if tmp.exists():
            shutil.move(str(tmp), str(dest))
    except Exception as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def setup_kokoro(force: bool = False) -> bool:
    """Setup Kokoro TTS ONNX weights in models/kokoro."""
    target_dir = MODELS_DIR / "kokoro"
    target_dir.mkdir(parents=True, exist_ok=True)
    onnx_file = target_dir / "kokoro-v1.0.onnx"
    voices_file = target_dir / "voices-v1.0.bin"

    # Check cache from C:\Models\kokoro if present
    c_kokoro = Path(r"C:\Models\kokoro")
    if c_kokoro.exists():
        if not onnx_file.exists() and (c_kokoro / "kokoro-v1.0.onnx").exists():
            logger.info("Copying kokoro-v1.0.onnx from C:\\Models\\kokoro...")
            shutil.copy2(c_kokoro / "kokoro-v1.0.onnx", onnx_file)
        if not voices_file.exists() and (c_kokoro / "voices-v1.0.bin").exists():
            logger.info("Copying voices-v1.0.bin from C:\\Models\\kokoro...")
            shutil.copy2(c_kokoro / "voices-v1.0.bin", voices_file)

    if force or not onnx_file.exists() or onnx_file.stat().st_size < 100_000_000:
        _download_file(KOKORO_ONNX_URL, onnx_file)
    if force or not voices_file.exists() or voices_file.stat().st_size < 10_000_000:
        _download_file(KOKORO_VOICES_URL, voices_file)

    ok = onnx_file.exists() and voices_file.exists()
    if ok:
        logger.info("[OK] Kokoro TTS ready at %s", target_dir)
    return ok


def setup_laya(force: bool = False) -> bool:
    """Setup Laya Decision Router weights in models/laya-lora-ft."""
    target_dir = MODELS_DIR / "laya-lora-ft"
    target_dir.mkdir(parents=True, exist_ok=True)
    safetensors = target_dir / "model.safetensors"

    if not force and safetensors.exists() and safetensors.stat().st_size > 500_000_000:
        logger.info("[OK] Laya Router already installed at %s", target_dir)
        return True

    try:
        from huggingface_hub import snapshot_download

        logger.info("Downloading Laya model from Hugging Face (AgencyL1/Laya-Lora-4B)...")
        snapshot_download(
            repo_id="AgencyL1/Laya-Lora-4B",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        ok = safetensors.exists()
        if ok:
            logger.info("[OK] Laya Router downloaded to %s", target_dir)
        return ok
    except Exception as exc:
        logger.warning("Could not download Laya from HF: %s. Calibrated heuristic router will be used.", exc)
        return False


def setup_embeddinggemma(force: bool = False) -> bool:
    """Setup Embedding Gemma 2 model in models/embeddinggemma-2."""
    target_dir = MODELS_DIR / "embeddinggemma-2"
    target_dir.mkdir(parents=True, exist_ok=True)
    safetensors = target_dir / "model.safetensors"

    # Check cache from C:\Models\embeddinggemma-2 if present
    c_embed = Path(r"C:\Models\embeddinggemma-2")
    if c_embed.exists() and not safetensors.exists():
        logger.info("Copying embeddinggemma-2 from C:\\Models\\embeddinggemma-2...")
        for item in c_embed.iterdir():
            dest = target_dir / item.name
            if not dest.exists():
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

    if not force and safetensors.exists() and safetensors.stat().st_size > 500_000_000:
        logger.info("[OK] Embedding Gemma 2 already installed at %s", target_dir)
        return True

    try:
        from huggingface_hub import snapshot_download

        logger.info("Downloading Embedding Gemma 2 from Hugging Face (google/embeddinggemma-2)...")
        snapshot_download(
            repo_id="google/embeddinggemma-2",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        ok = safetensors.exists()
        if ok:
            logger.info("[OK] Embedding Gemma 2 downloaded to %s", target_dir)
        return ok
    except Exception as exc:
        logger.warning("Could not download Embedding Gemma 2 from HF: %s. CPU fallback embedder will be used.", exc)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and setup models/ directory for L.O.R.A")
    parser.add_argument("--force", action="store_true", help="Force redownload existing models")
    parser.add_argument("--kokoro-only", action="store_true", help="Setup Kokoro only")
    parser.add_argument("--laya-only", action="store_true", help="Setup Laya only")
    parser.add_argument("--embedding-only", action="store_true", help="Setup Embedding Gemma 2 only")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Verifying and setting up models in %s ...", MODELS_DIR)

    results = {}
    if args.kokoro_only:
        results["kokoro"] = setup_kokoro(args.force)
    elif args.laya_only:
        results["laya"] = setup_laya(args.force)
    elif args.embedding_only:
        results["embeddinggemma"] = setup_embeddinggemma(args.force)
    else:
        results["kokoro"] = setup_kokoro(args.force)
        results["laya"] = setup_laya(args.force)
        results["embeddinggemma"] = setup_embeddinggemma(args.force)

    logger.info("--------------------------------------------------")
    logger.info("Model Setup Summary:")
    for k, v in results.items():
        status = "READY" if v else "SKIPPED/OPTIONAL"
        logger.info("  - %-18s: %s", k, status)
    logger.info("--------------------------------------------------")
    return 0


if __name__ == "__main__":
    sys.exit(main())
