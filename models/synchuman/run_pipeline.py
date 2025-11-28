#!/usr/bin/env python3
"""
Wrapper to run SyncHuman two-stage inference with a user-provided input image.
Stage 1: OneStage pipeline writes intermediate outputs.
Stage 2: TwoStage pipeline produces the final GLB mesh.
"""
import argparse
import os
from pathlib import Path

import torch

from SyncHuman.pipelines.SyncHumanOneStagePipeline import SyncHumanOneStagePipeline
from SyncHuman.pipelines.SyncHumanTwoStage import SyncHumanTwoStagePipeline


def run_stage_one(image_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = SyncHumanOneStagePipeline.from_pretrained("./ckpts/OneStage")
    pipeline.SyncHuman_2D3DCrossSpaceDiffusion.enable_xformers_memory_efficient_attention()
    pipeline.run(image_path=str(image_path), save_path=str(out_dir))
    return out_dir


def run_stage_two(stage_one_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = SyncHumanTwoStagePipeline.from_pretrained("./ckpts/SecondStage")
    pipeline.cuda()
    pipeline.run(image_path=str(stage_one_dir), outpath=str(out_dir))
    return out_dir / "output.glb"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SyncHuman OneStage + TwoStage on a single input image.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--workdir",
        default="./outputs",
        help="Directory to store intermediate and final outputs (default: ./outputs).",
    )
    parser.add_argument(
        "--skip_stage1",
        action="store_true",
        help="Skip OneStage and reuse existing outputs (expects workdir/OneStage).",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    workdir = Path(args.workdir).expanduser().resolve()
    stage_one_dir = workdir / "OneStage"
    stage_two_dir = workdir / "SecondStage"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Please run on a GPU instance.")

    if args.skip_stage1:
        if not stage_one_dir.exists():
            raise FileNotFoundError(f"--skip_stage1 set but missing stage1 outputs at {stage_one_dir}")
    else:
        print(f"[Stage 1] Running OneStage on {image_path} -> {stage_one_dir}")
        run_stage_one(image_path, stage_one_dir)

    print(f"[Stage 2] Running TwoStage from {stage_one_dir} -> {stage_two_dir}")
    final_glb = run_stage_two(stage_one_dir, stage_two_dir)

    print(f"Done. Final GLB: {final_glb}")


if __name__ == "__main__":
    main()
