#!/usr/bin/env python3
"""AffinityShift: conditional non-member reference fitting and membership scoring.

Inputs contain raw SSIM, LPIPS, Ensemble_Variance, Image_Predictability,
Complexity, and Source labels. Intrinsic-property scaling/statistics used upstream
must already have been fitted on the non-member reference-fitting subset only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
RIDGE_ALPHAS = np.logspace(-4, 4, 30)
FEATURES = ["Image_Predictability", "Complexity"]
AFFINITY = ["SSIM", "LPIPS", "Ensemble_Variance"]
REQUIRED = ["Source", *FEATURES, *AFFINITY]


def parse_args():
    p = argparse.ArgumentParser(description="AffinityShift: non-member-only ridge reference detector")
    p.add_argument("--input", required=True, help="Input CSV or XLSX file with features and Source label")
    p.add_argument("--output", required=True, help="Per-image evaluation predictions (.csv or .xlsx)")
    p.add_argument("--summary", help="Optional JSON summary file")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--alpha", type=float, default=0.05, help="Nominal calibration FPR (default 0.05)")
    p.add_argument("--member-label", default="Member")
    p.add_argument("--nonmember-label", default="Non Member")
    p.add_argument("--id-column", help="Optional unique sample identifier for checking duplicate records")
    return p.parse_args()


def load_table(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError("--input must be .csv, .xlsx, or .xls")


def save_table(frame, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        frame.to_csv(output, index=False)
    elif output.suffix.lower() == ".xlsx":
        frame.to_excel(output, index=False)
    else:
        raise ValueError("--output must end in .csv or .xlsx")


def ridge_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
    ])


def train_reference_models(reference, seed):
    if len(reference) < 5:
        raise ValueError("At least five non-member reference-fitting images are required")
    features = reference[FEATURES]
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    models, sigmas, selected_alphas = {}, {}, {}
    for metric in AFFINITY:
        y = reference[metric].to_numpy(dtype=np.float64)
        model = ridge_pipeline()
        oof = cross_val_predict(model, features, y, cv=cv)
        residuals = y - oof
        sigma = float(np.std(residuals, ddof=1))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Nonpositive out-of-fold residual standard deviation for {metric}")
        model.fit(features, y)
        models[metric] = model
        sigmas[metric] = sigma
        selected_alphas[metric] = float(model.named_steps["ridge"].alpha_)
    return models, sigmas, selected_alphas


def directional_scores(frame, models, sigmas):
    x = frame[FEATURES]
    expected_s = models["SSIM"].predict(x)
    expected_l = models["LPIPS"].predict(x)
    expected_v = models["Ensemble_Variance"].predict(x)
    z_s = np.maximum(0.0, (frame["SSIM"].to_numpy() - expected_s) / sigmas["SSIM"])
    z_l = np.maximum(0.0, (expected_l - frame["LPIPS"].to_numpy()) / sigmas["LPIPS"])
    z_v = np.maximum(0.0, (expected_v - frame["Ensemble_Variance"].to_numpy()) / sigmas["Ensemble_Variance"])
    return (z_s + z_l + z_v) / 3.0, z_s, z_l, z_v


def main():
    args = parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1")
    if args.member_label == args.nonmember_label:
        raise ValueError("Member and non-member labels must differ")
    data = load_table(args.input)
    # Accept original column names from prior scripts without silently changing their semantics.
    aliases = {"Naturalness": "Image_Predictability", "Image Predictability": "Image_Predictability",
               "Image_Predictibility": "Image_Predictability", "complexity": "Complexity"}
    for old, new in aliases.items():
        if old in data.columns and new not in data.columns:
            data = data.rename(columns={old: new})
    missing = [col for col in REQUIRED if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: {REQUIRED}")
    if data[REQUIRED].isnull().any().any():
        raise ValueError("Missing feature/label values; fix input rather than silently dropping images")
    labels = data["Source"].astype(str).str.strip()
    member = args.member_label.strip()
    nonmember = args.nonmember_label.strip()
    unknown = sorted(set(labels) - {member, nonmember})
    if unknown:
        raise ValueError(f"Unexpected Source labels: {unknown}")
    data = data.copy().reset_index(drop=True)
    data["Source"] = labels
    for col in FEATURES + AFFINITY:
        data[col] = pd.to_numeric(data[col], errors="raise")
    if not np.isfinite(data[FEATURES + AFFINITY].to_numpy(dtype=float)).all():
        raise ValueError("All features must be finite")
    if args.id_column:
        if args.id_column not in data:
            raise ValueError(f"Missing ID column: {args.id_column}")
        if data[args.id_column].isnull().any() or data[args.id_column].duplicated().any():
            raise ValueError("ID column must be unique and nonmissing")

    member_rows = data[data["Source"] == member]
    nonmember_rows = data[data["Source"] == nonmember]
    if len(member_rows) < 2 or len(nonmember_rows) < 12:
        raise ValueError("Insufficient examples for evaluation plus 5-fold reference fitting")

    # Exactly half of each class is held out for the balanced final evaluation.
    # For odd class counts, a perfect 50/50 split is impossible.
    if len(member_rows) % 2 or len(nonmember_rows) % 2:
        raise ValueError("For the paper's 50/50 protocol, each class must have an even number of images")
    member_eval, _unused_members = train_test_split(
        member_rows, test_size=0.5, random_state=args.seed, shuffle=True)
    nonmember_eval, nonmember_pool = train_test_split(
        nonmember_rows, test_size=0.5, random_state=args.seed, shuffle=True)
    # Both halves have equal class size if input is class-balanced, as in the paper.
    if len(member_eval) != len(nonmember_eval):
        raise ValueError("Paper uses equal original member and non-member counts for balanced evaluation")
    reference, calibration = train_test_split(
        nonmember_pool, test_size=0.25, random_state=args.seed, shuffle=True)
    evaluation = pd.concat([member_eval, nonmember_eval], ignore_index=True)
    evaluation = evaluation.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    models, sigmas, ridge_alphas = train_reference_models(reference, args.seed)
    calibration_scores, *_ = directional_scores(calibration, models, sigmas)
    threshold = float(np.quantile(calibration_scores, 1.0 - args.alpha))
    cal_fpr = float(np.mean(calibration_scores > threshold))
    scores, z_s, z_l, z_v = directional_scores(evaluation, models, sigmas)
    predictions = scores > threshold
    truth = (evaluation["Source"] == member).to_numpy()
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[False, True]).ravel()
    result = evaluation.copy()
    result["z_SSIM"] = z_s
    result["z_LPIPS"] = z_l
    result["z_Ensemble_Variance"] = z_v
    result["Membership_Score"] = scores
    result["Prediction"] = np.where(predictions, member, nonmember)
    result["Threshold"] = threshold
    save_table(result, args.output)

    summary = {
        "seed": args.seed, "nominal_calibration_fpr": args.alpha,
        "calibration_fpr": cal_fpr,
        "threshold": threshold,
        "counts": {"members_total": len(member_rows), "nonmembers_total": len(nonmember_rows),
                   "reference_nonmembers": len(reference), "calibration_nonmembers": len(calibration),
                   "evaluation_members": len(member_eval), "evaluation_nonmembers": len(nonmember_eval)},
        "ridge_alpha": ridge_alphas, "residual_sigma": sigmas,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "accuracy_percent": float(100 * accuracy_score(truth, predictions)),
        "TPR_percent": float(100 * tp / (tp + fn)),
        "FPR_percent": float(100 * fp / (fp + tn)),
        "AUC_percent": float(100 * roc_auc_score(truth, scores)),
    }
    print(json.dumps(summary, indent=2))
    if args.summary:
        target = Path(args.summary)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation predictions to: {args.output}")


if __name__ == "__main__":
    main()
