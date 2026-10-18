import cv2
import numpy as np
import math
from typing import Dict, List


# -------------------------
# Individual complexity features
# -------------------------
def shannon_entropy(img_bgr):
    ent = []
    for c in range(3):
        hist = cv2.calcHist([img_bgr], [c], None, [256], [0, 256]).ravel()
        p = hist / (hist.sum() + 1e-12)
        ent.append(-np.sum(p * np.log2(p + 1e-12)))
    return float(np.mean(ent))


def edge_density(img_gray):
    v = float(np.median(img_gray))
    low = int(max(0, 0.66 * v))
    high = int(min(255, 1.33 * v))
    edges = cv2.Canny(img_gray, low, high, L2gradient=True)
    return float((edges > 0).mean())


def gradient_magnitude(img_gray):
    gx = cv2.Scharr(img_gray, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(img_gray, cv2.CV_64F, 0, 1)
    return float(np.sqrt(gx**2 + gy**2).mean())


def laplacian_var(img_gray):
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def colorfulness(img_bgr):
    B, G, R = cv2.split(img_bgr.astype(np.float32))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    return float(np.sqrt(np.var(rg) + np.var(yb)) +
                 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))


def jpeg_psnr_q50(img_bgr):
    ok, enc = cv2.imencode(".jpg", img_bgr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    if not ok:
        return float("nan")
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    mse = np.mean((img_bgr.astype(np.float32) -
                   dec.astype(np.float32)) ** 2)
    if mse < 1e-12:
        return 100.0
    return 20 * math.log10(255.0) - 10 * math.log10(mse)


# -------------------------
# Normalization utilities
# -------------------------
def zscore(x: List[float]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-8)


def minmax(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


# -------------------------
# Final complexity computation
# -------------------------
def compute_complexity(features: Dict[str, List[float]]) -> np.ndarray:
    """
    Complexity_norm in [0,1]

    Complexity_raw = sum(z_k) - z_PSNR
    Complexity_norm = minmax(Complexity_raw)
    """

    positive_keys = [
        "entropy",
        "edge_density",
        "gradient",
        "laplacian",
        "colorfulness"
    ]

    Z = {k: zscore(features[k]) for k in features}

    complexity_raw = np.zeros_like(Z["entropy"])
    for k in positive_keys:
        complexity_raw += Z[k]

    complexity_raw -= Z["jpeg_psnr"]

    complexity_norm = minmax(complexity_raw)
    return complexity_norm
