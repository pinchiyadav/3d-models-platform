SyncHuman API (FastAPI wrapper)

What this is
- A lightweight FastAPI server (`api_server.py`) that wraps the official SyncHuman OneStage/TwoStage pipelines.
- Defaults to the official high-quality settings (Stage1=50 steps, Stage2=25 steps) but lets you override steps per request.
- Uses the same preprocessing as our successful runs: rembg → crop to bbox → square pad → resize 768x768 RGBA.

Requirements
- GPU with >=40GB VRAM (A40/L40) and the SyncHuman repo + weights already in place.
- Python env with SyncHuman deps plus API deps: `pip install -r models/synchuman/api_requirements.txt` (fastapi, uvicorn, python-multipart).
- `SYNCHUMAN_ROOT` optional; defaults to current directory. Run from the repo root `/workspace/SyncHuman`.
- Inputs must already be RGBA (background-removed). The API will crop/pad/resize to 768x768 and expects a meaningful alpha channel.

Run the server
```bash
cd /workspace/SyncHuman
export ATTN_BACKEND=flash_attn
export SPARSE_ATTN_BACKEND=flash_attn
python api_server.py  # uses uvicorn internally on 0.0.0.0:8000
# or: uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Endpoints
- `GET /health` → `{"status":"ok"}`
- `POST /generate` (multipart/form-data)
  - `image` (file, optional): input photo.
  - `image_url` (str, optional): http/https URL to fetch the image.
  - `stage` (str, optional): `both` (default), `stage1`, or `stage2` (stage2 requires `reuse_stage1` path).
  - `stage1_steps` (int, optional): override Stage1 diffusion steps (default 50).
  - `stage2_steps` (int, optional): override Stage2 sampler steps (default 25).
  - `reuse_stage1` (str, optional): absolute path to an existing OneStage folder when `stage=stage2`.
  - `download` (bool, optional): if true and Stage2 ran, stream the GLB file.

Example requests
```bash
# Run both stages with defaults
curl -X POST http://localhost:8000/generate \
  -F "image=@/path/to/input.png"

# Run using an image URL
curl -X POST http://localhost:8000/generate \
  -F "image_url=https://example.com/your_rgba.png"

# Run both stages with higher quality
curl -X POST http://localhost:8000/generate \
  -F "image=@/path/to/input.png" \
  -F "stage1_steps=100" \
  -F "stage2_steps=50"

# Stage2 only using an existing Stage1 folder (no upload)
curl -X POST http://localhost:8000/generate \
  -F "image=@/tmp/dummy.png" \
  -F "stage=stage2" \
  -F "reuse_stage1=/workspace/SyncHuman/outputs/OneStage_official"
```

Outputs
- Each run gets its own folder under `/workspace/SyncHuman/outputs/api_<run_id>/`.
- Stage1: `OneStage` folder with `color_*.png`, `normal_*.png`, `voxel.ply`, `latent.npz`, `input.png`.
- Stage2: `SecondStage/ouput.glb` (note the upstream filename typo).
- Response JSON includes the paths; set `download=true` to stream the GLB directly.
