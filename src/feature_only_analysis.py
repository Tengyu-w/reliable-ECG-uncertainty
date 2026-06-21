from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import CLASS_NAMES, REGULARITY_FEATURE_NAMES, extract_regularity_features_batch, load_rhythm_windows, make_splits
from .metrics import classification_metrics, expected_calibration_error
from .uncertainty import entropy, msp


def _safe_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def _safe_aupr(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def _boundary_score(probs: np.ndarray) -> np.ndarray:
    ventricular_mass = probs[:, 1] + probs[:, 2]
    balance = 1.0 - np.abs(probs[:, 1] - probs[:, 2]) / np.maximum(ventricular_mass, 1e-8)
    return ventricular_mass * balance


def _selective_row(y: np.ndarray, probs: np.ndarray, risk_score: np.ndarray, coverage: float = 0.8) -> dict[str, float]:
    n_keep = max(1, int(round(len(y) * coverage)))
    keep = np.argsort(risk_score)[:n_keep]
    metrics = classification_metrics(y[keep], probs[keep])
    return {
        "coverage": coverage,
        "accepted": int(n_keep),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "risk": float(1.0 - metrics["accuracy"]),
    }


def _model_specs(seed: int) -> dict[str, object]:
    return {
        "logistic_l2": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=800, random_state=seed),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=100)
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path("results") / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_feature_only")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_rhythm_windows(args.mat)
    splits = make_splits(dataset.x, dataset.y, groups=dataset.record_ids, seed=args.seed)
    x_train = extract_regularity_features_batch(splits.x_train, sample_rate=args.sample_rate)
    x_val = extract_regularity_features_batch(splits.x_val, sample_rate=args.sample_rate)
    x_test = extract_regularity_features_batch(splits.x_test, sample_rate=args.sample_rate)

    np.savez(
        out_dir / "regularity_feature_splits.npz",
        x_train=x_train,
        y_train=splits.y_train,
        x_val=x_val,
        y_val=splits.y_val,
        x_test=x_test,
        y_test=splits.y_test,
        names=np.asarray(REGULARITY_FEATURE_NAMES),
    )

    rows = []
    selective_rows = []
    trained = {}
    for name, model in _model_specs(args.seed).items():
        model.fit(x_train, splits.y_train)
        val_probs = model.predict_proba(x_val)
        test_probs = model.predict_proba(x_test)
        trained[name] = model

        val_metrics = classification_metrics(splits.y_val, val_probs)
        test_metrics = classification_metrics(splits.y_test, test_probs)
        test_pred = test_probs.argmax(axis=1)
        error = (test_pred != splits.y_test).astype(int)
        boundary_error = (((splits.y_test == 1) & (test_pred == 2)) | ((splits.y_test == 2) & (test_pred == 1))).astype(int)
        ent = entropy(test_probs)
        msp_score = msp(test_probs)
        boundary = _boundary_score(test_probs)

        rows.append(
            {
                "model": name,
                "val_macro_f1": val_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_nll": test_metrics["nll"],
                "test_ece": expected_calibration_error(splits.y_test, test_probs),
                "error_auroc_entropy": _safe_auroc(error, ent),
                "error_aupr_entropy": _safe_aupr(error, ent),
                "error_auroc_msp": _safe_auroc(error, msp_score),
                "vtvf_boundary_auroc": _safe_auroc(boundary_error, boundary),
                "vtvf_boundary_aupr": _safe_aupr(boundary_error, boundary),
                "vt_as_vf": int(((splits.y_test == 1) & (test_pred == 2)).sum()),
                "vf_as_vt": int(((splits.y_test == 2) & (test_pred == 1)).sum()),
                "confusion_matrix": json.dumps(test_metrics["confusion_matrix"]),
            }
        )
        selective = _selective_row(splits.y_test, test_probs, ent, coverage=0.8)
        selective["model"] = name
        selective["score"] = "entropy"
        selective_rows.append(selective)

        pred_df = pd.DataFrame(test_probs, columns=[f"prob_{c}" for c in CLASS_NAMES])
        pred_df["y_true"] = splits.y_test
        pred_df["y_pred"] = test_pred
        pred_df["entropy"] = ent
        pred_df["vtvf_boundary_score"] = boundary
        pred_df.to_csv(out_dir / f"{name}_test_predictions.csv", index=False)

    summary = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    summary.to_csv(out_dir / "feature_only_summary.csv", index=False)
    pd.DataFrame(selective_rows).to_csv(out_dir / "feature_only_selective_80.csv", index=False)

    best_name = str(summary.iloc[0]["model"])
    best_model = trained[best_name]
    perm = permutation_importance(
        best_model,
        x_test,
        splits.y_test,
        scoring="f1_macro",
        n_repeats=30,
        random_state=args.seed,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": REGULARITY_FEATURE_NAMES,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(out_dir / "feature_permutation_importance.csv", index=False)

    plt.figure(figsize=(7, 4))
    plot_df = importance.sort_values("importance_mean")
    plt.barh(plot_df["feature"], plot_df["importance_mean"], xerr=plot_df["importance_std"], color="#4C78A8")
    plt.xlabel("Macro-F1 drop after permutation")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_permutation_importance.png", dpi=180)
    plt.close()

    print(summary)
    print("\nBest feature-only model:", best_name)
    print(importance)


if __name__ == "__main__":
    main()
