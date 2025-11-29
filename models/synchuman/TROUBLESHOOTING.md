SyncHuman troubleshooting notes (A40, CUDA 12.1, torch 2.1.1)

Common issues and fixes
- ImportError libGL.so.1 / cv2 fails: install system libs `apt-get install -y libgl1 libglib2.0-0` (also pulls mesa). Do this before starting the API so rembg/OpenCV can import.
- Missing nvdiffrast/xatlas/pyvista/pymeshfix/igraph/open3d/easydict: install via pip in the env:
  - `pip install git+https://github.com/NVlabs/nvdiffrast.git@v0.3.1 --no-build-isolation`
  - `pip install xatlas pyvista pymeshfix python-igraph open3d==0.17.0 easydict`
- Gaussian rasterizer build errors (diff_gaussian_rasterization): ensure `libglm-dev` and `build-essential` are installed, and pin numpy to `<2` (e.g. `pip install "numpy<2" --force-reinstall`). Then `git submodule update --init --recursive` and `pip install --no-build-isolation ./GaussianRenderer/diff-gaussian-rasterization`.
- Missing TRELLIS deps (`utils3d`, `diff_gaussian_rasterization`): install `utils3d` from commit `EasternJournalist/utils3d@9a4eb15e`; build Gaussian rasterizer from the mip-splatting submodule via `pip install . --no-build-isolation` inside `GaussianRenderer/diff-gaussian-rasterization`. The pip release alone is insufficient because SyncHuman needs the kernel_size branch.
- FlashAttention support: A40/L40 (SM 8.0/8.9) work with flash-attn 2.5.8 built from source; set `ATTN_BACKEND=flash_attn` and `SPARSE_ATTN_BACKEND=flash_attn`. Older GPUs without SM>=80 will fall back to memory-inefficient attention.
- SecondStage checkpoints: the upstream `pipeline.json` points to HF ids; patch it to local paths (`ckpts/SecondStage/ckpts/decoder_GS`, `decoder_Mesh`, `slat_flow`) so inference runs offline.
- Output filename typo: upstream writes `ouput.glb` (missing “t”). No action needed; just look for that name.

Input quality requirements (root cause of tiny GLBs)
- Use an RGBA image with a solid foreground mask (alpha close to 1.0 on the person, 0 on background). The official samples are 768x768 RGBA; matching that framing/aspect prevents over-cropping.
- If using JPEGs, convert to PNG and run background removal first (e.g., `rembg p --alpha-matting`). A nearly empty mask yields tiny meshes (~5 KB), as seen with the user sample.
- Avoid truncated/corrupted JPEGs; Pillow emits “truncated JPEG” warnings otherwise.

Known-good commands (from instance 28302366)
- Activate env: `source venv/bin/activate` then `export PYTHONPATH=/workspace/SyncHuman`.
- OneStage (official 000.png): `ATTN_BACKEND=flash_attn SPARSE_ATTN_BACKEND=flash_attn python inference_OneStage.py`.
- TwoStage: `ATTN_BACKEND=flash_attn SPARSE_ATTN_BACKEND=flash_attn python inference_SecondStage.py` after OneStage outputs exist.
- Official outputs: `outputs/SecondStage_official/ouput.glb` (~3.0 MB) and `outputs/SecondStage_001/ouput.glb` (~3.1 MB) verify the pipeline.

Signs something is off
- GLB << 1 MB: mask likely empty or Stage1 outputs mismatched. Recreate the RGBA mask and rerun Stage1+2.
- Import errors for `utils3d` or `diff_gaussian_rasterization`: rerun the TRELLIS install steps noted above.
