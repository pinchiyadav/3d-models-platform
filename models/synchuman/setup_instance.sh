#!/usr/bin/env bash
set -euo pipefail

# One-shot setup for SyncHuman + API on a fresh GPU instance.
# Usage: HF_TOKEN=your_hf_token bash models/synchuman/setup_instance.sh

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required (your Hugging Face token)" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "[1/6] Installing system packages..."
if command -v apt-get >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git ffmpeg libgl1 libglib2.0-0 libssl-dev pkg-config python3-dev unzip build-essential
fi

echo "[2/6] Creating venv..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

echo "[3/6] Installing core torch stack (cu121)..."
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

echo "[4/6] Installing SyncHuman + TRELLIS deps..."
pip install \
  accelerate \
  safetensors==0.4.5 \
  diffusers==0.29.1 \
  transformers==4.36.0 \
  huggingface_hub \
  xformers==0.0.23 \
  flash-attn==2.5.8 \
  spconv-cu120 \
  rembg \
  onnxruntime \
  imageio[ffmpeg] \
  opencv-python \
  kaolin==0.17.0 \
  nvdiffrast
pip install git+https://github.com/EasternJournalist/utils3d@9a4eb15e
pip install --no-build-isolation "$ROOT/GaussianRenderer/diff-gaussian-rasterization"

echo "[5/6] Install SyncHuman editable..."
pip install -e "$ROOT"

echo "[6/6] Downloading weights..."
HF_TOKEN="$HF_TOKEN" python download.py

echo "Done."
echo "To run API:"
echo "  source venv/bin/activate"
echo "  export ATTN_BACKEND=flash_attn SPARSE_ATTN_BACKEND=flash_attn"
echo "  python api_server.py  # listens on 0.0.0.0:8000 by default"
