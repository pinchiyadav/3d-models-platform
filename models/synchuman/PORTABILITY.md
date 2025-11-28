SyncHuman portability guide (new GPU instance)

Target instance template
- GPU: >=40GB VRAM (A40/L40), CUDA 12.1 capable (SM>=80 for flash-attn).
- Disk: ~200GB.
- Base image: `pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel` (matches official deps).

Bootstrap steps (venv)
```bash
# 0) System deps
apt-get update && apt-get install -y git ffmpeg libgl1 libglib2.0-0 libssl-dev pkg-config python3-dev unzip build-essential

# 1) Clone code
cd /workspace
git clone https://github.com/IGL-HKUST/SyncHuman.git
cd /workspace/SyncHuman

# 2) Python env (uses the same versions as our working A40)
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r <(cat <<'EOF'
accelerate
safetensors==0.4.5
diffusers==0.29.1
transformers==4.36.0
huggingface_hub
xformers==0.0.23.post1
flash-attn==2.5.8
spconv-cu120
rembg
onnxruntime
imageio[ffmpeg]
opencv-python
kaolin==0.17.0
nvdiffrast
EOF
)

# TRELLIS pieces required by SyncHuman
pip install git+https://github.com/EasternJournalist/utils3d@9a4eb15e
pip install --no-build-isolation /workspace/SyncHuman/GaussianRenderer/diff-gaussian-rasterization

# 3) Install SyncHuman editable
pip install -e .

# 4) Download weights (requires your HF token in env HF_TOKEN)
python download.py

# 5) Patch pipeline.json to local ckpt paths (already done in our repo copy)
#    decoder_GS -> ckpts/decoder_GS, decoder_Mesh -> ckpts/decoder_Mesh, slat_flow -> ckpts/slat_flow

# 6) (Optional) copy our helper scripts
#    api_server.py, run_pipeline.py, API_USAGE.md, TROUBLESHOOTING.md
```

Running inference/API
- Activate env: `source /workspace/SyncHuman/venv/bin/activate`.
- Set attention backend (fast + official): `export ATTN_BACKEND=flash_attn; export SPARSE_ATTN_BACKEND=flash_attn`.
- One-off run: `python run_pipeline.py --image /path/to/input.png`.
- API: `python api_server.py` (FastAPI on port 8000). See `API_USAGE.md` for usage.

Notes
- Keep inputs RGBA with solid foreground masks for best results (see TROUBLESHOOTING.md).
- Outputs live under `/workspace/SyncHuman/outputs`; each API run uses `outputs/api_<run_id>/`.
- For Docker-based portability, extend `models/synchuman/Dockerfile` to copy `api_server.py` and install `api_requirements.txt`, then run uvicorn in your container entrypoint.
