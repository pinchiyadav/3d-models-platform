"""
Minimal FastAPI wrapper around SyncHuman pipelines.

Features
- POST /generate: upload an image, optionally override Stage 1 and Stage 2 steps, choose stage ("both", "stage1", "stage2").
- Uses the official pipeline defaults if steps are not provided (Stage1=50, Stage2=25).
- Returns JSON with output paths; optionally streams the GLB when `download=true`.

Assumptions
- This script runs inside the SyncHuman repo (or set SYNCHUMAN_ROOT to that path).
- CUDA GPU (>=40GB) available; flash-attn/xformers installed.
"""
import os
import sys
import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

# Resolve repo root
REPO_ROOT = Path(os.environ.get("SYNCHUMAN_ROOT", Path(__file__).resolve().parent))
if (REPO_ROOT / "SyncHumanOneStagePipeline.py").exists():
    # Script placed directly inside SyncHuman package
    sys.path.append(str(REPO_ROOT.parent))
else:
    sys.path.append(str(REPO_ROOT))

try:
    from SyncHuman.pipelines.SyncHumanOneStagePipeline import SyncHumanOneStagePipeline
    from SyncHuman.pipelines.SyncHumanTwoStage import SyncHumanTwoStagePipeline
    from SyncHuman.utils.inference_utils import save_coords_to_npz, save_images, add_margin
    from SyncHuman.utils.voxel_utils import writeocc
    from SyncHuman.utils import postprocessing_utils
except Exception as exc:  # pragma: no cover - runtime check
    raise RuntimeError("Failed to import SyncHuman modules. Set SYNCHUMAN_ROOT to the repo path.") from exc

DEFAULT_STAGE1_STEPS = 50
DEFAULT_STAGE2_STEPS = 25

os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")

app = FastAPI(title="SyncHuman API", version="0.1.0")

_stage1_pipe: Optional[SyncHumanOneStagePipeline] = None
_stage2_pipe: Optional[SyncHumanTwoStagePipeline] = None


def _load_pipelines():
    global _stage1_pipe, _stage2_pipe
    if _stage1_pipe is None:
        _stage1_pipe = SyncHumanOneStagePipeline.from_pretrained("./ckpts/OneStage")
        _stage1_pipe.SyncHuman_2D3DCrossSpaceDiffusion.enable_xformers_memory_efficient_attention()
    if _stage2_pipe is None:
        _stage2_pipe = SyncHumanTwoStagePipeline.from_pretrained("./ckpts/SecondStage")
        _stage2_pipe.cuda()


def _prepare_rgba(input_path: Path, output_path: Path) -> float:
    """
    Expect a pre-masked RGBA input (caller provides alpha). Crop to bbox, square-pad, resize to 768x768.
    Raise if no alpha content.
    """
    rgba = Image.open(input_path).convert("RGBA")
    alpha = torch.from_numpy(np.array(rgba.split()[-1], dtype=np.uint8))
    covered = (alpha > 0).float().mean().item()
    coords = torch.nonzero(alpha > 10, as_tuple=False)
    if coords.numel() == 0:
        raise ValueError("Input must include an alpha mask; none detected.")
    ymin, xmin = coords.min(dim=0).values.tolist()
    ymax, xmax = coords.max(dim=0).values.tolist()
    cropped = rgba.crop((xmin, ymin, xmax + 1, ymax + 1))
    side = int(max(cropped.size) * 1.1)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    offset = ((side - cropped.width) // 2, (side - cropped.height) // 2)
    canvas.paste(cropped, offset)
    final = canvas.resize((768, 768), Image.LANCZOS)
    final.save(output_path)
    return covered


def _write_status(status_path: Path, data: dict) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(data))


def _run_stage1(image_path: Path, out_dir: Path, num_steps: Optional[int]) -> Path:
    _load_pipelines()
    assert _stage1_pipe is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: finer-grained progress is limited by the underlying pipeline; this marks start/end only.
    torch.manual_seed(43)
    image_raw = Image.open(image_path)
    imgs_in = torch.cat([_stage1_pipe.get_mv_input(str(image_path))] * 2, dim=0)
    with torch.autocast("cuda"):
        out, voxel, coords = _stage1_pipe.run_model(
            imgs_in,
            image_raw,
            None,
            prompt_embeds=_stage1_pipe.prompt_embeddings,
            guidance_scale=3.0,
            output_type="pt",
            num_images_per_prompt=1,
            num_inference_steps=num_steps or DEFAULT_STAGE1_STEPS,
        )
        out = out.images
        bsz = out.shape[0] // 2
        normals_pred = out[:bsz]
        images_pred = out[bsz:]
        images_pred[0] = imgs_in[0]
        normals_face = F.interpolate(normals_pred[-1].unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False).squeeze(0)
        normals_pred[0][:, :256, 256:512] = normals_face

        image_raw.save(out_dir / "input.png")
        save_images(images_pred, normals_pred, str(out_dir))
        save_coords_to_npz(coords, out_dir / "latent.npz")
        v = voxel.unsqueeze(0).cpu().numpy()
        writeocc(v, str(out_dir), "voxel.ply")

    torch.cuda.empty_cache()
    return out_dir


def _run_stage2(stage1_dir: Path, out_dir: Path, num_steps: Optional[int]) -> Path:
    _load_pipelines()
    assert _stage2_pipe is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sampling/postprocess are coarse-grained; we mark checkpoints here.
    mv_generate = _stage2_pipe._get_mv_generate(str(stage1_dir))
    coords = _stage2_pipe._get_coords_gen(str(stage1_dir))
    image = _stage2_pipe.preprocess_image(stage1_dir / "input.png")
    cond = _stage2_pipe.get_cond([image])
    sampler_params = {}
    if num_steps:
        sampler_params["steps"] = num_steps
    slat = _stage2_pipe.sample_slat(cond, coords.cuda(), sampler_params=sampler_params)
    mv_img = _stage2_pipe.get_cond_mv(mv_generate["mv_img"].unsqueeze(0).cuda())
    mv_normal = _stage2_pipe.get_cond_mv(mv_generate["mv_normal"].unsqueeze(0).cuda())
    gs = _stage2_pipe.models["slat_decoder_gs"](slat, mv_img, mv_normal)
    mesh = _stage2_pipe.models["slat_decoder_mesh"](slat, mv_img, mv_normal)
    # Avoid grad-required tensors in geometry to keep to_glb happy while allowing texture baking.
    try:
        if hasattr(mesh[0], "vertices") and isinstance(mesh[0].vertices, torch.Tensor):
            mesh[0].vertices = mesh[0].vertices.detach()
        if hasattr(gs[0], "means") and isinstance(gs[0].means, torch.Tensor):
            gs[0].means = gs[0].means.detach()
    except Exception:
        pass
    glb = postprocessing_utils.to_glb(
        gs[0],
        mesh[0],
        mv_generate["mv_img"],
        mv_generate["mv_normal"],
        mv_generate["faces_img"],
        mv_generate["faces_normal"],
        simplify=0.7,
        texture_size=1024,
    )
    glb_path = out_dir / "ouput.glb"
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    stage: str = Form("both"),  # "both", "stage1", "stage2"
    stage1_steps: Optional[int] = Form(None),
    stage2_steps: Optional[int] = Form(None),
    reuse_stage1: Optional[str] = Form(None),
    download: bool = Form(False),
):
    if not torch.cuda.is_available():
        raise HTTPException(status_code=500, detail="CUDA GPU required.")

    if stage not in {"both", "stage1", "stage2"}:
        raise HTTPException(status_code=400, detail="stage must be one of: both, stage1, stage2.")

    if stage == "stage2" and not reuse_stage1 and image is None and image_url is None:
        raise HTTPException(status_code=400, detail="stage2 requires reuse_stage1 or an image/image_url.")

    run_id = uuid4().hex[:8]
    workdir = Path("/workspace/SyncHuman").resolve()
    inputs_dir = workdir / "inputs" / f"api_{run_id}"
    outputs_dir = workdir / "outputs" / f"api_{run_id}"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    status_path = outputs_dir / "status.json"

    raw_path: Optional[Path] = None
    alpha_cov = None

    if image_url:
        try:
            import urllib.request
            url_suffix = Path(urllib.request.urlparse(image_url).path).suffix or ".png"
            raw_path = inputs_dir / f"upload{url_suffix}"
            with urllib.request.urlopen(image_url) as resp:
                raw_path.write_bytes(resp.read())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to download image_url: {exc}")
    elif image is not None:
        suffix = Path(image.filename or "upload").suffix or ".png"
        raw_path = inputs_dir / f"upload{suffix}"
        raw_bytes = await image.read()
        raw_path.write_bytes(raw_bytes)

    if stage != "stage2":
        if raw_path is None or not raw_path.exists():
            raise HTTPException(status_code=400, detail="Image or image_url is required for Stage1.")
        rgba_path = inputs_dir / "input_rgba_768.png"
        try:
            alpha_cov = _prepare_rgba(raw_path, rgba_path)
        except Exception as exc:
            _write_status(status_path, {"status": "error", "stage": "prepare", "message": str(exc)})
            raise HTTPException(status_code=400, detail=f"Failed to prepare input: {exc}")
        stage1_dir = outputs_dir / "OneStage"
        _write_status(status_path, {"status": "running", "stage": "stage1", "progress": "start"})
        try:
            _run_stage1(rgba_path, stage1_dir, stage1_steps)
        except Exception as exc:
            _write_status(status_path, {"status": "error", "stage": "stage1", "message": str(exc)})
            raise HTTPException(status_code=500, detail=f"Stage1 failed: {exc}")
        _write_status(status_path, {"status": "running", "stage": "stage1", "progress": "done"})
    else:
        stage1_dir = Path(reuse_stage1).expanduser().resolve()
        if not stage1_dir.exists():
            raise HTTPException(status_code=400, detail=f"reuse_stage1 not found: {stage1_dir}")

    glb_path = None
    if stage in {"both", "stage2"}:
        stage2_dir = outputs_dir / "SecondStage"
        _write_status(status_path, {"status": "running", "stage": "stage2", "progress": "sampling"})
        try:
            glb_path = _run_stage2(stage1_dir, stage2_dir, stage2_steps)
            _write_status(status_path, {"status": "running", "stage": "stage2", "progress": "postprocess"})
        except Exception as exc:
            _write_status(status_path, {"status": "error", "stage": "stage2", "message": str(exc)})
            raise HTTPException(status_code=500, detail=f"Stage2 failed: {exc}")

    result = {
        "run_id": run_id,
        "stage": stage,
        "input_raw": str(raw_path) if raw_path else None,
        "stage1_dir": str(stage1_dir),
        "stage1_steps": stage1_steps or DEFAULT_STAGE1_STEPS,
        "stage2_steps": stage2_steps or DEFAULT_STAGE2_STEPS,
        "alpha_coverage": alpha_cov,
    }
    if glb_path:
        result["glb"] = str(glb_path)
        _write_status(status_path, {"status": "completed", "stage": "done", "glb": str(glb_path)})

    if download and glb_path:
        return FileResponse(glb_path, media_type="model/gltf-binary", filename="synchuman.glb")

    return JSONResponse(result)


def main():
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)


if __name__ == "__main__":
    main()
