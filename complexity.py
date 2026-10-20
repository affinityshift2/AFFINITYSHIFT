from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

RESOLUTION = 512
JPEG_QUALITY = 50  # Fixed JPEG setting; must match the experimental protocol.
EPS = 1e-12
FEATURE_NAMES = (
    "edge_density", "gradient_magnitude", "laplacian_variance",
    "shannon_entropy", "colorfulness", "jpeg_psnr",
)
POSITIVE_NAMES = FEATURE_NAMES[:-1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    # Match the 512x512 resolution specified in the manuscript.
    return cv2.resize(image, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR)


def shannon_entropy(img_bgr: np.ndarray) -> float:
    entropies = []
    for channel in cv2.split(img_bgr):
        counts = np.bincount(channel.ravel(), minlength=256).astype(np.float64)
        p = counts[counts > 0] / counts.sum()
        entropies.append(float(-(p * np.log2(p)).sum()))
    return float(np.mean(entropies))


def edge_density(img_gray: np.ndarray) -> float:
    median = float(np.median(img_gray))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(img_gray, lower, upper, L2gradient=True)
    return float(np.mean(edges > 0))


def gradient_magnitude(img_gray: np.ndarray) -> float:
    gx = cv2.Scharr(img_gray, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(img_gray, cv2.CV_64F, 0, 1)
    return float(np.hypot(gx, gy).mean())


def laplacian_variance(img_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def colorfulness(img_bgr: np.ndarray) -> float:
    # Hasler-Suesstrunk colorfulness (signed opponent-color channels).
    b, g, r = cv2.split(img_bgr.astype(np.float64))
    rg = r - g
    yb = (r + g) / 2.0 - b
    std = math.sqrt(float(rg.var() + yb.var()))
    mean = math.sqrt(float(rg.mean() ** 2 + yb.mean() ** 2))
    return float(std + 0.3 * mean)


def jpeg_psnr(img_bgr: np.ndarray, quality: int = JPEG_QUALITY) -> float:
    success, encoded = cv2.imencode(
        ".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    mse = float(np.mean((img_bgr.astype(np.float64) - decoded.astype(np.float64)) ** 2))
    # Capped finite PSNR avoids infinite/non-serializable feature values.
    if mse <= EPS:
        return 100.0
    return float(10.0 * math.log10((255.0 ** 2) / mse))


def extract_features(image_path: str | Path) -> dict[str, float]:
    image = load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "edge_density": edge_density(gray),
        "gradient_magnitude": gradient_magnitude(gray),
        "laplacian_variance": laplacian_variance(gray),
        "shannon_entropy": shannon_entropy(image),
        "colorfulness": colorfulness(image),
        "jpeg_psnr": jpeg_psnr(image),
    }


def fit_reference_statistics(
    reference_features: list[Mapping[str, float]],
) -> dict:
    """Call only with verified non-member REFERENCE-FITTING samples."""
    if len(reference_features) < 2:
        raise ValueError("At least two reference-fitting non-members are required")
    matrix = np.array(
        [[sample[name] for name in FEATURE_NAMES] for sample in reference_features],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite features in reference fitting data")
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)  # Population standard deviation.
    if (stds <= EPS).any():
        constant = [FEATURE_NAMES[i] for i in np.where(stds <= EPS)[0]]
        raise ValueError(f"Reference feature has near-zero variance: {constant}")
    return {
        "feature_names": list(FEATURE_NAMES),
        "mean": dict(zip(FEATURE_NAMES, means.tolist())),
        "std": dict(zip(FEATURE_NAMES, stds.tolist())),
        "jpeg_quality": JPEG_QUALITY,
        "resolution": RESOLUTION,
        "normalization_split": "nonmember_reference_fitting_only",
        "reference_count": len(reference_features),
    }


def compute_complexity(features: Mapping[str, float], stats: Mapping) -> float:
    """Apply the same frozen reference z-scores to any sample."""
    if stats.get("normalization_split") != "nonmember_reference_fitting_only":
        raise ValueError("Expected statistics fitted on non-member reference subset")
    if stats.get("jpeg_quality") != JPEG_QUALITY or stats.get("resolution") != RESOLUTION:
        raise ValueError("Saved statistics preprocessing differs from current configuration")
    z = {}
    for name in FEATURE_NAMES:
        value = float(features[name])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {name}")
        mu = float(stats["mean"][name])
        sigma = float(stats["std"][name])
        if sigma <= EPS:
            raise ValueError(f"Invalid reference scale for {name}")
        z[name] = (value - mu) / sigma
    return float(sum(z[name] for name in POSITIVE_NAMES) - z["jpeg_psnr"])


def read_image_list(list_path: str | Path) -> list[Path]:
    base = Path(list_path).resolve().parent
    paths = []
    for line in Path(list_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            path = Path(line).expanduser()
            paths.append(path if path.is_absolute() else base / path)
    if not paths:
        raise ValueError("Image list is empty")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension: {path}")
    if len({str(p.resolve()) for p in paths}) != len(paths):
        raise ValueError("Image list contains duplicate paths")
    return paths


def save_csv(paths: list[Path], samples: list[dict], stats: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Image_Path", *FEATURE_NAMES, "Complexity"])
        writer.writeheader()
        for path, sample in zip(paths, samples):
            writer.writerow({
                "Image_Path": str(path),
                **sample,
                "Complexity": compute_complexity(sample, stats),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="AffinityShift intrinsic image complexity")
    commands = parser.add_subparsers(dest="mode", required=True)
    fit = commands.add_parser("fit", help="Fit z-score stats on reference-fitting NON-MEMBERS only")
    fit.add_argument("--images", required=True, help="Text file, one image path per line")
    fit.add_argument("--stats", required=True, help="Where to write reference statistics JSON")
    fit.add_argument("--output-csv", required=True)
    apply = commands.add_parser("transform", help="Apply saved stats to calibration/evaluation/members")
    apply.add_argument("--images", required=True)
    apply.add_argument("--stats", required=True)
    apply.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    paths = read_image_list(args.images)
    samples = [extract_features(path) for path in paths]
    stats_path = Path(args.stats)
    if args.mode == "fit":
        if stats_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing reference stats: {stats_path}")
        stats = fit_reference_statistics(samples)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    else:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    save_csv(paths, samples, stats, Path(args.output_csv))
    print(f"Saved {len(paths)} image complexity scores to {args.output_csv}")
    print(f"Reference statistics: {stats_path}")


if __name__ == "__main__":
    main()
