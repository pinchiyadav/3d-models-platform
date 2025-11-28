#!/usr/bin/env bash
set -euo pipefail

# Build-time deps are already installed; install CUDA extensions at runtime (driver available here).
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
cd /workspace/SyncHuman

# Ensure rasterizer source exists (submodule or cloned)
if [ ! -d "GaussianRenderer/diff-gaussian-rasterization" ]; then
  git clone --depth 1 https://github.com/graphdeco-inria/diff-gaussian-rasterization.git GaussianRenderer/diff-gaussian-rasterization
fi

# Install Kaolin and rasterizer
python -m pip install --no-build-isolation --no-cache-dir git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.17.0
python -m pip install --no-build-isolation ./GaussianRenderer/diff-gaussian-rasterization

export PYTHONPATH=/workspace/SyncHuman
export ATTN_BACKEND=flash_attn
export SPARSE_ATTN_BACKEND=flash_attn

exec python api_server.py
