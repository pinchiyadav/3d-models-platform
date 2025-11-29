# SyncHuman (containerized setup for L40 target)

What this does
- Builds a CUDA 12.1 container with a `synchuman` conda env (PyTorch 2.1.1 + cu121) and required pip packages (accelerate, safetensors 0.4.5, diffusers 0.29.1, transformers 4.36.0, xformers 0.0.23.post1).
- Clones the official repo `IGL-HKUST/SyncHuman` and installs it editable.
- Prefetches checkpoints via the official `download.py` (HF repo `xishushu/SyncHuman`).
- Provides a wrapper `run_pipeline.py` to run OneStage and TwoStage with a user-specified input image.
- Optional TRELLIS install hook (set `--build-arg INSTALL_TRELLIS=1`) per SyncHuman docs.

Recommended Vast.ai instance
- L40 1/3 fraction (ID 26099273), ~46GB VRAM, ~0.189 $/h, Vietnam region, inet ~687/404 Mbps, reliability ~0.9967. Meets SyncHuman’s 40GB+ VRAM guidance at lowest cost.

Build
```bash
cd $(git rev-parse --show-toplevel)
docker build -t synchuman:cu121 \
  -f models/synchuman/Dockerfile \
  models/synchuman
# To install TRELLIS inside the image (may increase build time):
# docker build -t synchuman:cu121 --build-arg INSTALL_TRELLIS=1 -f models/synchuman/Dockerfile models/synchuman
```

Run inference (with your image mounted)
```bash
export INPUT=/Users/pinchiyadav/3d-models-platform/3d_img_input.jpeg
docker run --gpus all --rm \
  -v "$(dirname "$INPUT")":/data \
  -w /workspace/SyncHuman \
  synchuman:cu121 \
  micromamba run -n synchuman python run_pipeline.py --image /data/$(basename "$INPUT") --workdir /workspace/SyncHuman/outputs
```
Outputs:
- Stage1: `/workspace/SyncHuman/outputs/OneStage`
- Stage2 + final GLB: `/workspace/SyncHuman/outputs/SecondStage/output.glb`

Notes
- CUDA required; tested settings per official README: Python 3.10, torch 2.1.1 + cu121.
- `run_pipeline.py` keeps Stage1/Stage2 logic intact; use `--skip_stage1` to reuse existing Stage1 outputs.
- TRELLIS install is optional here; enable if their latest requirements are needed by upstream updates.

API option
- FastAPI wrapper: `api_server.py`; see `API_USAGE.md` for `/generate` usage.
- Quick setup on a new GPU (mamba by default, venv optional):  
  ```
  # ensure python3.10 is available if you disable mamba
  HF_TOKEN=<your_hf_token> SYNC_ROOT=/workspace/SyncHuman \
  bash models/synchuman/bootstrap.sh            # installs deps, downloads weights, starts API
  ```
  - If you prefer venv: `USE_MAMBA=false PYTHON=python3.10 ... bootstrap.sh`
  - API log: `/workspace/sync_api.log`; env exports handled inside bootstrap (flash-attn).
- One-command container: `docker build -t synchuman-api -f models/synchuman/Dockerfile models/synchuman && docker run --gpus all -p 8000:8000 synchuman-api`. Image includes weights and starts uvicorn on port 8000.
