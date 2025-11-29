#!/usr/bin/env bash
set -euo pipefail

# One-shot setup for SyncHuman + API on a fresh GPU instance (no Docker).
# Usage: HF_TOKEN=your_hf_token [SYNC_ROOT=/workspace/SyncHuman] bash models/synchuman/setup_instance.sh

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required (your Hugging Face token)" >&2
  exit 1
fi

SYNC_ROOT="${SYNC_ROOT:-/workspace/SyncHuman}"
if [[ ! -d "${SYNC_ROOT}" ]]; then
  echo "SYNC_ROOT '${SYNC_ROOT}' not found. Clone https://github.com/IGL-HKUST/SyncHuman.git there first." >&2
  exit 1
fi
cd "$SYNC_ROOT"

echo "[1/7] Installing system packages..."
if command -v apt-get >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git ffmpeg libgl1 libglib2.0-0 libssl-dev pkg-config python3-dev unzip build-essential libglm-dev
fi

echo "[2/7] Creating venv..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel packaging ninja psutil "numpy<2"
if [[ -f "${SYNC_ROOT}/requirements.lock" ]]; then
  echo "[2b] Installing pinned lockfile..."
  pip install -r "${SYNC_ROOT}/requirements.lock"
fi

echo "[3/7] Installing core torch stack (cu121)..."
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

echo "[4/7] Installing SyncHuman + TRELLIS deps..."
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
  trimesh \
  easydict \
  xatlas \
  pyvista \
  pymeshfix \
  python-igraph \
  open3d==0.17.0
pip install git+https://github.com/NVlabs/nvdiffrast.git@v0.3.1 --no-build-isolation
pip install git+https://github.com/EasternJournalist/utils3d@9a4eb15e
pip install --no-build-isolation --no-cache-dir git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.17.0
git submodule update --init --recursive
pip install --no-build-isolation ./GaussianRenderer/diff-gaussian-rasterization

echo "[5/7] Install SyncHuman editable..."
pip install -e .

echo "[6/7] Downloading weights..."
HF_TOKEN="$HF_TOKEN" python download.py

echo "[7/7] Ready. Run API with:"
echo "  source venv/bin/activate"
echo "  export PYTHONPATH=${SYNC_ROOT}"
echo "  export ATTN_BACKEND=flash_attn SPARSE_ATTN_BACKEND=flash_attn"
echo "  python api_server.py  # listens on 0.0.0.0:8000 by default"

echo "Done."
