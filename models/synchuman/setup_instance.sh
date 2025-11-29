#!/usr/bin/env bash
set -euo pipefail

# One-shot setup for SyncHuman + API on a fresh GPU instance (no Docker).
# Usage: HF_TOKEN=your_hf_token [SYNC_ROOT=/workspace/SyncHuman] [USE_MAMBA=true] bash models/synchuman/setup_instance.sh

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

USE_MAMBA="${USE_MAMBA:-true}"

if [[ "${USE_MAMBA}" == "true" ]]; then
  echo "[2/7] Installing micromamba env (py3.10)..."
  MICROMAMBA_ROOT="${MICROMAMBA_ROOT:-/workspace/micromamba}"
  export MAMBA_ROOT_PREFIX="${MICROMAMBA_ROOT}"
  if [[ ! -x "${MICROMAMBA_ROOT}/bin/micromamba" ]]; then
    mkdir -p "${MICROMAMBA_ROOT}/bin"
    TMP_MAMBA="$(mktemp -d)"
    MAMBA_URL="https://micro.mamba.pm/api/micromamba/linux-64/latest"
    if curl -fLs "${MAMBA_URL}" -o "${TMP_MAMBA}/micromamba.tar.bz2"; then
      tar -xvjf "${TMP_MAMBA}/micromamba.tar.bz2" -C "${TMP_MAMBA}" "bin/micromamba"
      install -m 755 "${TMP_MAMBA}/bin/micromamba" "${MICROMAMBA_ROOT}/bin/micromamba"
    else
      echo "micromamba download failed; falling back to venv." >&2
      USE_MAMBA="false"
    fi
    rm -rf "${TMP_MAMBA}"
  fi
fi

if [[ "${USE_MAMBA}" == "true" ]]; then
  eval "$("${MICROMAMBA_ROOT}/bin/micromamba" shell hook -s bash)"
  micromamba create -y -n synchuman python=3.10
  micromamba activate synchuman
else
  echo "[2/7] Creating venv..."
  PYTHON_BIN="${PYTHON:-python3}"
  "${PYTHON_BIN}" -m venv venv
  source venv/bin/activate
fi

pip install --upgrade pip setuptools wheel packaging ninja psutil "numpy<2"
if [[ -f "${SYNC_ROOT}/requirements.lock" ]]; then
  echo "[2b] Installing pinned lockfile..."
  pip install -r "${SYNC_ROOT}/requirements.lock"
fi

echo "[3/7] Installing core torch stack (cu121)..."
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/cu121 \
  --retries 15 --timeout 3
# Try prebuilt flash-attn wheel matching torch 2.1.1/cu121; fallback to source
FLASH_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu121torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
if ! pip install --no-cache-dir "$FLASH_WHEEL"; then
  pip install flash-attn==2.5.8 --no-build-isolation --no-cache-dir
fi
pip install xformers==0.0.23

echo "[4/7] Installing SyncHuman + TRELLIS deps..."
pip install \
  accelerate \
  safetensors==0.4.5 \
  diffusers==0.29.1 \
  transformers==4.36.0 \
  huggingface_hub \
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
  open3d==0.17.0 \
  fastapi uvicorn python-multipart
pip install git+https://github.com/NVlabs/nvdiffrast.git@v0.3.1 --no-build-isolation
pip install git+https://github.com/EasternJournalist/utils3d@9a4eb15e
pip install --no-build-isolation --no-cache-dir git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.17.0
git submodule update --init --recursive
# Ensure rasterizer source is present
if [[ ! -d "./GaussianRenderer/diff-gaussian-rasterization" ]]; then
  mkdir -p ./GaussianRenderer
  git clone --depth 1 https://github.com/graphdeco-inria/diff-gaussian-rasterization.git ./GaussianRenderer/diff-gaussian-rasterization
fi
pip install --no-build-isolation ./GaussianRenderer/diff-gaussian-rasterization

echo "[5/7] Skipping editable install (no setup.py/pyproject); using PYTHONPATH at runtime..."

echo "[6/7] Downloading weights..."
# verify required files; if missing, download
REQUIRED_FILES=(
  "./ckpts/OneStage/SyncHuman_2D3DCrossSpaceDiffusion/model.safetensors"
  "./ckpts/OneStage/image_encoder/model.safetensors"
  "./ckpts/OneStage/text_encoder/model.safetensors"
  "./ckpts/OneStage/vae/diffusion_pytorch_model.safetensors"
  "./ckpts/OneStage/sparse_structure_decoder/model.safetensors"
  "./ckpts/SecondStage/ckpts/decoder_GS/model.ckpt"
  "./ckpts/SecondStage/ckpts/decoder_Mesh/model.ckpt"
  "./ckpts/SecondStage/ckpts/slat_flow/model.ckpt"
)
# remove incomplete ckpts if any required file is missing
NEED_DL=0
for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    NEED_DL=1
    break
  fi
done

if [[ "$NEED_DL" -eq 0 ]]; then
  echo "[6/7] ckpts present; skipping download."
else
  echo "[6/7] Missing ckpts; downloading..."
  export HF_HUB_ENABLE_HF_TRANSFER=1
  pip install -q hf_transfer || true

  ROBUST_DOWNLOADER_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/robust_downloader.py"
  
  echo "Starting robust downloader..."
  # Use timeout to prevent the script from hanging indefinitely. 2 hours.
  if ! timeout 7200 env HF_TOKEN="$HF_TOKEN" python "$ROBUST_DOWNLOADER_PATH"; then
    echo "Download script failed or timed out. Checking for completeness..."
  fi

  # Final check for required files
  HAVE_ALL=1
  for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
      HAVE_ALL=0
      echo "ERROR: Missing required file after download: $f" >&2
    fi
  done
  if [[ "$HAVE_ALL" -eq 1 ]]; then
    echo "[6/7] ckpts download complete."
  else
    echo "ERROR: Download failed, some required files are missing." >&2
    exit 1
  fi
fi

echo "[7/7] Ready. Run API with:"
echo "  # if micromamba was used (default):"
echo "  eval \\\"\\\$(/workspace/micromamba/bin/micromamba shell hook -s bash)\\\" && micromamba activate synchuman"
echo "  # else venv:"
echo "  source venv/bin/activate"
echo "  export PYTHONPATH=${SYNC_ROOT}"
echo "  export ATTN_BACKEND=flash_attn SPARSE_ATTN_BACKEND=flash_attn"
echo "  python api_server.py  # listens on 0.0.0.0:8000 by default"

echo "Done."
