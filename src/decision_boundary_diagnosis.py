from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .metrics import expected_calibration_error, softmax
from .uncertainty import fit_temperature


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _load_split(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(run_dir / f"embeddings_{split}.npz")
    return data["embeddings"].astype(np.float32), data["logits"].astype(np.float32), data["y"].astype(np.int64)


def _centroid_distances(train_emb: np.ndarray, train_y: np.ndarray, emb: np.ndarray) -> np.ndarray:
    centroids = np.stack([train_emb[train_y == c].mean(axis=0) for c in range(3)])
    return np.linalg.norm(emb[:, None, :] - centroids[None, :, :], axis=2)


def _assign_mechanism(y: np.ndarray, pred: np.ndarray, nearest_proto: np.ndarray) -> np.ndarray:
    mechanism = np.full(len(y), "correct", dtype=object)
    error = y != pred
    proto_matches_truth = nearest_proto == y
    pred_matches_proto = pred == nearest_proto
    mechanism[error & proto_matches_truth] = "classifier_boundary_mismatch"
    mechanism[error & (~proto_matches_truth) & pred_matches_proto] = "representation_overlap"
    mechanism[error & (~proto_matches_truth) & (~pred_matches_proto)] = "mixed_or_outlying"
    return mechanism


def _fit_frozen_linear_head(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    test_emb: np.ndarray,
    test_y: np.ndarray,
    balanced: bool,
) -> dict[str, float]:
    head = LogisticRegression(
        max_iter=2000,
        class_weight="balanced" if balanced else None,
        solver="lbfgs",
    )
    head.fit(train_emb, train_y)
    pred = head.predict(test_emb)
    cm = pd.crosstab(pd.Series(test_y, name="true"), pd.Series(pred, name="pred"), dropna=False)
    vtvf = int(cm.reindex(index=[1, 2], columns=[1, 2], fill_value=0).loc[1, 2]) + int(
        cm.reindex(index=[1, 2], columns=[1, 2], fill_value=0).loc[2, 1]
    )
    return {
        "macro_f1": float(f1_score(test_y, pred, average="macro")),
        "accuracy": float((pred == test_y).mean()),
        "vtvf_cross_errors": vtvf,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    train_emb, train_logits, train_y = _load_split(run_dir, "train")
    val_emb, val_logits, val_y = _load_split(run_dir, "val")
    test_emb, test_logits, test_y = _load_split(run_dir, "test")

    probs = softmax(test_logits)
    pred = probs.argmax(axis=1)
    temp = fit_temperature(val_logits, val_y)
    probs_t = softmax(test_logits, temperature=temp)

    dist = _centroid_distances(train_emb, train_y, test_emb)
    nearest_proto = dist.argmin(axis=1)
    proto_margin_vtvf = dist[:, 2] - dist[:, 1]
    logit_margin_vtvf = test_logits[:, 1] - test_logits[:, 2]
    prob_margin_vtvf = probs[:, 1] - probs[:, 2]
    mechanism = _assign_mechanism(test_y, pred, nearest_proto)

    df = pd.DataFrame(
        {
            "y_true": test_y,
            "y_pred": pred,
            "nearest_prototype": nearest_proto,
            "mechanism": mechanism,
            "confidence": probs.max(axis=1),
            "temperature_confidence": probs_t.max(axis=1),
            "prob_sr": probs[:, 0],
            "prob_vt": probs[:, 1],
            "prob_vf": probs[:, 2],
            "d_sr": dist[:, 0],
            "d_vt": dist[:, 1],
            "d_vf": dist[:, 2],
            "prototype_margin_vtvf_dvf_minus_dvt": proto_margin_vtvf,
            "logit_margin_vtvf_vt_minus_vf": logit_margin_vtvf,
            "prob_margin_vtvf_vt_minus_vf": prob_margin_vtvf,
        }
    )
    df["is_error"] = df["y_true"] != df["y_pred"]
    df["is_vtvf"] = df["y_true"].isin([1, 2])
    df["is_vtvf_cross_error"] = ((df["y_true"] == 1) & (df["y_pred"] == 2)) | (
        (df["y_true"] == 2) & (df["y_pred"] == 1)
    )
    df["classifier_proto_disagree"] = df["y_pred"] != df["nearest_prototype"]
    df.to_csv(run_dir / "decision_boundary_diagnosis.csv", index=False)

    rows = []
    for group_name, mask in {
        "all": np.ones(len(df), dtype=bool),
        "vtvf_true": df["is_vtvf"].to_numpy(),
        "any_error": df["is_error"].to_numpy(),
        "vtvf_cross_error": df["is_vtvf_cross_error"].to_numpy(),
        "classifier_boundary_mismatch": df["mechanism"].eq("classifier_boundary_mismatch").to_numpy(),
        "representation_overlap": df["mechanism"].eq("representation_overlap").to_numpy(),
        "mixed_or_outlying": df["mechanism"].eq("mixed_or_outlying").to_numpy(),
    }.items():
        sub = df.loc[mask]
        if len(sub) == 0:
            continue
        rows.append(
            {
                "group": group_name,
                "n": int(len(sub)),
                "error_rate": float(sub["is_error"].mean()),
                "vtvf_cross_error_rate": float(sub["is_vtvf_cross_error"].mean()),
                "classifier_proto_disagreement_rate": float(sub["classifier_proto_disagree"].mean()),
                "mean_confidence": float(sub["confidence"].mean()),
                "mean_d_vt": float(sub["d_vt"].mean()),
                "mean_d_vf": float(sub["d_vf"].mean()),
                "mean_abs_proto_vtvf_margin": float(np.abs(sub["prototype_margin_vtvf_dvf_minus_dvt"]).mean()),
                "mean_abs_logit_vtvf_margin": float(np.abs(sub["logit_margin_vtvf_vt_minus_vf"]).mean()),
            }
        )

    mechanism_counts = df["mechanism"].value_counts().rename_axis("mechanism").reset_index(name="n")
    mechanism_counts["fraction"] = mechanism_counts["n"] / len(df)
    mechanism_counts.to_csv(run_dir / "decision_boundary_mechanism_counts.csv", index=False)

    frozen_plain = _fit_frozen_linear_head(train_emb, train_y, test_emb, test_y, balanced=False)
    frozen_balanced = _fit_frozen_linear_head(train_emb, train_y, test_emb, test_y, balanced=True)
    summary = pd.DataFrame(rows)
    for key, value in {
        "ece_before_temperature": expected_calibration_error(test_y, probs),
        "ece_after_temperature": expected_calibration_error(test_y, probs_t),
        "temperature": temp,
        "frozen_plain_macro_f1": frozen_plain["macro_f1"],
        "frozen_plain_vtvf_cross_errors": frozen_plain["vtvf_cross_errors"],
        "frozen_balanced_macro_f1": frozen_balanced["macro_f1"],
        "frozen_balanced_vtvf_cross_errors": frozen_balanced["vtvf_cross_errors"],
    }.items():
        summary[key] = value
    summary.to_csv(run_dir / "decision_boundary_summary.csv", index=False)

    vmask = df["is_vtvf"].to_numpy()
    plt.figure(figsize=(7, 5))
    correct = vmask & ~df["is_vtvf_cross_error"].to_numpy()
    cross = df["is_vtvf_cross_error"].to_numpy()
    plt.scatter(
        df.loc[correct, "prototype_margin_vtvf_dvf_minus_dvt"],
        df.loc[correct, "logit_margin_vtvf_vt_minus_vf"],
        s=10,
        alpha=0.45,
        label="correct VT/VF",
    )
    plt.scatter(
        df.loc[cross, "prototype_margin_vtvf_dvf_minus_dvt"],
        df.loc[cross, "logit_margin_vtvf_vt_minus_vf"],
        s=18,
        alpha=0.8,
        label="VT/VF cross-error",
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Prototype margin: d(VF center) - d(VT center)")
    plt.ylabel("Logit margin: logit(VT) - logit(VF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "decision_boundary_vtvf_margin_map.png", dpi=180)
    plt.close()

    report = {
        "run_dir": str(run_dir),
        "temperature": float(temp),
        "frozen_plain": frozen_plain,
        "frozen_balanced": frozen_balanced,
        "mechanism_counts": mechanism_counts.to_dict(orient="records"),
    }
    (run_dir / "decision_boundary_diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary)
    print(mechanism_counts)


if __name__ == "__main__":
    main()
