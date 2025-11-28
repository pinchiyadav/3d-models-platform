High-level rules
- Follow each model's official documentation exactly; pin all dependency versions as specified and prefer recommended/maximum-quality settings.
- Keep per-model assets isolated (own folder with Dockerfile/venv, preprocessing scripts, config, and README notes).
- Include preprocessing steps if a model requires specific input handling; keep them in that model's folder.
- Optimize for cost: pick one of the cheapest Vast.ai locations that offers >=45GB GPU VRAM, ~200GB+ storage, and strong bandwidth; prepare a reusable template for quick reprovisioning.
- Security relaxed per request: keys can be stored in code for this project.
- Prefetch model weights into images to speed cold starts (accepting larger image sizes).

Work plan (update as we go)
- [x] Clarify constraints: acceptable budget/locations on Vast.ai (cheap, no hard ceiling), backend stack (recommendation allowed), prefetch weights (yes), model list (SyncHuman, PShuman, Idol, Humanyun3d v2.1, SiTh).
- [ ] Research docs for SyncHuman, PShuman, Idol, Humanyun3d v2.1, SiTh; capture exact required versions, preprocessing, and recommended settings.
- [ ] Design architecture: Vast instance template (GPU, disk, network options), startup/bootstrap scripts, per-model container layout, and API flow (upload -> select model -> job submission -> fetch outputs).
- [ ] Prepare repository scaffolding: per-model directories with Dockerfile or venv specs, pinned requirements, README usage, and preprocessing scripts.
- [ ] Implement per-model build scripts: Docker build/run commands, env templates, and data paths; verify with sample input `3d_img_input.jpeg` locally where feasible.
- [ ] Implement backend API: endpoints for upload, model selection, job submission to Vast instance, polling status, retrieving artifacts, and managing instance power state.
- [ ] Integrate Firebase Storage for input/output persistence and history logging; design format for output metadata.
- [ ] Implement simple frontend (Firebase Hosting): upload UI, model selector, status of GPU instance with start control, history table, and download links.
- [ ] Testing and validation: run sample generation per model (as feasible), smoke-test API and frontend, document known gaps.

Open questions / decisions
- Backend stack defaulting to Python + FastAPI unless otherwise requested.
- Queue mechanism for jobs: choose simple in-process vs Redis based on expected throughput/scale.
- Prefer cheapest regions on Vast.ai that meet GPU/storage/bandwidth requirements.

Current SyncHuman run (instance 28302366, A40, pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel, 200GB disk)
- venv at `/workspace/SyncHuman/venv` with torch 2.1.1/cu121, xformers 0.0.23, flash-attn 2.5.8, spconv-cu120, diffusers 0.29.1, transformers 4.36.0, rembg, onnxruntime, kaolin 0.17.0 (cu121), nvdiffrast, utils3d (TRELLIS pinned), diff-gaussian-rasterization (mip-splatting submodule).
- Weights prefetched via `download.py`; SecondStage weights pulled from HF `xishushu/SyncHuman` into `ckpts/SecondStage/ckpts`. `pipeline.json` points to local decoder_GS/Mesh/slat_flow.
- Official samples 000 and 001 run cleanly with ATTN_BACKEND=flash_attn: GLBs at `outputs/SecondStage_official/ouput.glb` (~3.0MB) and `outputs/SecondStage_001/ouput.glb` (~3.1MB).
- User photo now succeeds after rembg + square RGBA (alpha ~64%): `outputs/SecondStage_userRG/ouput.glb` (~3.8MB) copied locally to `models/synchuman/outputs/synchuman_user_rgba.glb`.
- Copied official GLBs locally under `models/synchuman/outputs/`; added `models/synchuman/TROUBLESHOOTING.md` with fixes.

Research notes (in progress)
- SyncHuman: Python 3.10; conda install torch 2.1.1/torchvision 0.16.1/torchaudio 2.1.1 + CUDA 12.1; follow TRELLIS env setup; pip install accelerate, safetensors 0.4.5, diffusers 0.29.1, transformers 4.36.0; weights via `download.py` (HF repo `xishushu/SyncHuman`); inference scripts `inference_OneStage.py` and `inference_SecondStage.py` expect `image_path` override; outputs at `outputs/SecondStage/output.glb`.
- PShuman: Python 3.10; torch 2.1.0 + cu121; kaolin 0.17.0 (cu121 wheel); other deps pinned in requirements.txt; preprocessing: remove BG via `utils/remove_bg.py` (rembg) or Clipdrop; requires SMPLX models from ECON/SIFU bundle (OneDrive link in README); inference command uses `configs/inference-768-6view.yaml`, `with_smpl=false`, `num_views=7`, `crop_size~740`, seeds 42/600 for quality tweaks.
- IDOL: Python 3.10, CUDA 11.8, torch 2.3.1+cu118; pip packages in `scripts/pip_install.sh` (note stray comment line “Install pip packages in bulk” needs a `#`); installs pytorch3d v0.7.7, simple-knn, gaussian-splatting, facebook/sapiens (engine+pretrain) as editable; template files from SMPL-X/FLAME via `scripts/fetch_template.sh`; pretrained weights and caches via `scripts/download_files.sh` (HF ckpt + sapiens torchscript + cache_sub2.zip); run via `python run_demo.py --render_mode ...`.
- SiTh: GitHub repo `yawnyanor35/SITH-PLUS-PLUS` is currently placeholder-only (README has no setup/code); need official docs/code link to pin dependencies.
- Humanyun3d v2.1: No repository/docs found yet; need official source to follow required versions/settings.

SyncHuman-first checklist
- [x] Write Dockerfile for SyncHuman on CUDA 12.1 runtime, installing conda env with pinned torch 2.1.1 stack + Trellis deps + pip packages.
- [x] Add environment.yml and wrapper script to run OneStage/SecondStage with user-provided image path/output path.
- [x] Prefetch SyncHuman weights inside image via `download.py` (built into Dockerfile).
- [x] Document usage (build/run commands, expected outputs) and GPU requirements (>=40GB VRAM; targeting L40 1/3 instance).
