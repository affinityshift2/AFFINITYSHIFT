#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def read_images(file_path: Path) -> list[Path]:
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    paths = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line).expanduser()
        if not candidate.is_absolute():
            candidate = (file_path.parent / candidate).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Image not found: {candidate}")
        paths.append(candidate)
    if not paths:
        raise ValueError("Input list contains no images")
    return paths


def load_rgb_uint8(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(rgb, dtype=np.uint8)


def init_official_srec(repo: Path, checkpoint: Path, device: str, scale: int,
                       resblocks: int, n_feats: int, mixtures: int):
    """Load actual SReC `network.Compressor` with strict checkpoint validation."""
    import torch

    if not repo.is_dir() or not (repo / "src" / "network.py").is_file():
        raise FileNotFoundError("--srec-repo must point to the official SReC clone")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    sys.path.insert(0, str(repo.resolve()))
    from src import configs, network

    configs.scale = scale
    configs.resblocks = resblocks
    configs.n_feats = n_feats
    configs.K = mixtures
    configs.log_likelihood = True
    configs.collect_probs = False

    compressor = network.Compressor().to(device)
    try:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch
        saved = torch.load(checkpoint, map_location="cpu")
    if not isinstance(saved, dict) or "nets" not in saved:
        raise ValueError("Expected official SReC checkpoint with a 'nets' state dict")
    # FAIL if any weights do not match. Do not partially load unrelated CNN weights.
    compressor.nets.load_state_dict(saved["nets"], strict=True)
    compressor.eval()
    compressor.requires_grad_(False)
    return compressor


def compute_official_bpsp(compressor, image: np.ndarray, device: str) -> float:
    """Official SReC theoretical bpsp over all RGB subpixels."""
    import torch
    pixels = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
    pixels = pixels.to(device=device, dtype=torch.float32)  # [0,255], not ImageNet normalized
    with torch.inference_mode():
        bits = compressor(pixels)
        bpsp = float(bits.get_total_bpsp(pixels.numel()).item())
    if not math.isfinite(bpsp):
        raise ValueError("SReC produced a non-finite bpsp")
    return bpsp


def validate_probability_map(name: str, data, size: int) -> np.ndarray:
    probs = np.asarray(data, dtype=np.float64)
    if probs.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}), got {probs.shape}")
    if not np.isfinite(probs).all() or np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError(f"{name} must contain finite discrete pixel probabilities in [0,1]")
    return probs


def compute_paper_predictability(provider, compressor, image: np.ndarray,
                                 lambda_factor: float) -> float:
    """Equations (8)-(9): arithmetic mixture of local and global pixel probabilities.

    Provider contract:
      get_probability_maps(compressor, image_rgb_uint8) -> (local_hw, global_hw)
    Each map contains probability MASS of the observed RGB pixel at that
    spatial location under the correctly conditioned SReC distribution.
    The provider must ensure the observed target pixel is not exposed in
    its conditioning input and must preserve genuine SReC likelihoods.
    """
    local, global_ = provider.get_probability_maps(compressor, image)
    height, width = image.shape[:2]
    if height != width:
        raise ValueError("Expected square preprocessed image")
    local = validate_probability_map("local", local, height)
    global_ = validate_probability_map("global", global_, height)
    return float(np.mean(lambda_factor * global_ + (1.0 - lambda_factor) * local))


def main() -> None:
    parser = argparse.ArgumentParser(description="AffinityShift: official SReC or paper-specific image predictability")
    parser.add_argument("--file", type=Path, required=True, help="One image path per line")
    parser.add_argument("--load", type=Path, required=True, help="Official SReC .pth checkpoint")
    parser.add_argument("--srec-repo", type=Path, required=True, help="Path to official SReC repository")
    parser.add_argument("--output-csv", "--output_csv", type=Path, required=True)
    parser.add_argument("--mode", choices=["official-bpsp", "paper-dual-context"], default="official-bpsp")
    parser.add_argument("--provider", type=str, help="Importable Python module for genuine paper dual-context probabilities")
    parser.add_argument("--lambda-factor", "--lmbda", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--resblocks", type=int, default=3)
    parser.add_argument("--n-feats", type=int, default=64)
    parser.add_argument("--mixtures", type=int, default=10)
    args = parser.parse_args()

    if args.resolution <= 0 or args.resolution % (2 ** args.scale):
        parser.error("Resolution must be positive and divisible by 2**scale")
    if not 0 <= args.lambda_factor <= 1:
        parser.error("--lambda-factor must lie in [0,1]")
    if args.mode == "paper-dual-context" and not args.provider:
        parser.error("Paper dual-context mode requires --provider with genuine per-pixel SReC probabilities; "
                     "the stock SReC evaluation code does not expose this automatically")

    images = read_images(args.file)
    compressor = init_official_srec(args.srec_repo, args.load, args.device,
                                    args.scale, args.resblocks, args.n_feats, args.mixtures)
    provider = None
    if args.mode == "paper-dual-context":
        provider = importlib.import_module(args.provider)
        if not callable(getattr(provider, "get_probability_maps", None)):
            raise TypeError("Provider must export get_probability_maps(compressor, image_rgb_uint8)")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if args.mode == "official-bpsp":
            # Negative bpsp is an ordering proxy, NOT the paper's P_i.
            writer.writerow(["Image_Path", "SReC_BPSP", "Predictability_Proxy_NegBPSP"])
        else:
            writer.writerow(["Image_Path", "Image_Predictability"])
        for index, path in enumerate(images, 1):
            rgb = load_rgb_uint8(path, args.resolution)
            if provider is None:
                bpsp = compute_official_bpsp(compressor, rgb, args.device)
                writer.writerow([str(path), f"{bpsp:.10g}", f"{-bpsp:.10g}"])
                print(f"[{index}/{len(images)}] {path.name}: bpsp={bpsp:.6f}")
            else:
                score = compute_paper_predictability(provider, compressor, rgb, args.lambda_factor)
                writer.writerow([str(path), f"{score:.10g}"])
                print(f"[{index}/{len(images)}] {path.name}: P_i={score:.8f}")
            stream.flush()
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
