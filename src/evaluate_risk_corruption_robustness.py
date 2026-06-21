from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .data import extract_regularity_features_batch, load_rhythm_windows, make_splits
from .evaluate_corruption_severity import _corrupt
from .metrics import softmax
from .models import build_model
from .train import predict
from .train_embedding_risk_head import EmbeddingRiskHead


LABELS = {0: "SR", 1: "VT", 2: "VF"}


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _ecg_loader(x: np.ndarray, batch_size: int, features: np.ndarray | None = None) -> DataLoader:
    y = np.zeros(len(x), dtype=np.int64)
    if features is None:
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    else:
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(features), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size)


@torch.no_grad()
def _risk_scores(model: EmbeddingRiskHead, emb: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    ds = TensorDataset(torch.from_numpy(emb.astype(np.float32)))
    for (xb,) in DataLoader(ds, batch_size=batch_size):
        rows.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(rows, axis=0)


def _review_curve(y_true: np.ndarray, y_pred: np.ndarray, risk_score: np.ndarray) -> pd.DataFrame:
    any_error = y_true != y_pred
    vtvf_error = ((y_true == 1) & (y_pred == 2)) | ((y_true == 2) & (y_pred == 1))
    order = np.argsort(-risk_score)
    rows = []
    for burden in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
        n_review = max(1, int(round(len(y_true) * burden)))
        review_idx = order[:n_review]
        auto_idx = order[n_review:]
        rows.append(
            {
                "review_burden": burden,
                "reviewed": int(n_review),
                "auto_coverage": float(len(auto_idx) / len(y_true)),
                "all_error_captured": float(any_error[review_idx].sum() / max(any_error.sum(), 1)),
                "vtvf_error_captured": float(vtvf_error[review_idx].sum() / max(vtvf_error.sum(), 1)),
                "auto_error_rate": float(any_error[auto_idx].mean()) if len(auto_idx) else float("nan"),
                "auto_vtvf_error_rate": float(vtvf_error[auto_idx].mean()) if len(auto_idx) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _binary_auc(labels: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, score)), float(average_precision_score(labels, score))


def _feature_transformer(run_dir: Path, model_name: str):
    if model_name not in {"regularity_fusion", "reliability_gated_fusion"}:
        return lambda x: None
    scaler = np.load(run_dir / "feature_scaler.npz", allow_pickle=True)
    mean, std = scaler["mean"], scaler["std"]

    def features_for(x: np.ndarray) -> np.ndarray:
        return ((extract_regularity_features_batch(x) - mean) / std).astype(np.float32)

    return features_for


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether a trained embedding RISK head remains useful under progressive ECG signal degradation."
    )
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--classifier-run-dir", type=Path, required=True)
    parser.add_argument("--risk-head-run-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["resnet1d", "regularity_fusion", "reliability_gated_fusion"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=[
            "gaussian_noise",
            "baseline_wander",
            "random_masking",
            "spike",
            "clipping_saturation",
            "time_scaling",
            "mixed_noise_baseline",
        ],
    )
    args = parser.parse_args()

    classifier_dir = _resolve_run_dir(args.classifier_run_dir)
    risk_dir = _resolve_run_dir(args.risk_head_run_dir)
    out_dir = args.out or (risk_dir / "risk_corruption_robustness")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_rhythm_windows(args.mat)
    splits = make_splits(dataset.x, dataset.y, groups=dataset.record_ids, seed=args.seed)
    x_test, y_test = splits.x_test.astype(np.float32), splits.y_test.astype(np.int64)
    features_for = _feature_transformer(classifier_dir, args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = build_model(args.model).to(device)
    classifier.load_state_dict(torch.load(classifier_dir / "best_model.pt", map_location=device, weights_only=True)["model"])

    emb_dim = np.load(classifier_dir / "embeddings_train.npz")["embeddings"].shape[1]
    risk_head = EmbeddingRiskHead(emb_dim).to(device)
    risk_state = torch.load(risk_dir / "embedding_risk_head.pt", map_location=device, weights_only=True)
    risk_head.load_state_dict(risk_state["model"])

    rng = np.random.default_rng(args.seed)
    rows = []
    review_rows = []
    prediction_rows = []
    severity_values = [0, 1, 2, 3, 4]
    for corruption in ["clean"] + args.corruptions:
        for severity in severity_values:
            if corruption == "clean":
                if severity > 0:
                    continue
                x_eval = x_test
            else:
                if severity == 0:
                    x_eval = x_test
                else:
                    x_eval = _corrupt(x_test, corruption, severity, rng).astype(np.float32)

            logits, emb, _ = predict(classifier, _ecg_loader(x_eval, args.batch_size, features_for(x_eval)), device)
            probs = softmax(logits)
            y_pred = logits.argmax(axis=1)
            risk_score = _risk_scores(risk_head, emb, args.batch_size, device)
            any_error = (y_test != y_pred).astype(int)
            vtvf_error = (((y_test == 1) & (y_pred == 2)) | ((y_test == 2) & (y_pred == 1))).astype(int)
            error_auc, error_aupr = _binary_auc(any_error, risk_score)
            vtvf_auc, vtvf_aupr = _binary_auc(vtvf_error, risk_score)
            entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
            rows.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    "n": int(len(y_test)),
                    "accuracy": float(np.mean(y_test == y_pred)),
                    "all_errors": int(any_error.sum()),
                    "vtvf_errors": int(vtvf_error.sum()),
                    "risk_score_mean": float(np.mean(risk_score)),
                    "risk_score_median": float(np.median(risk_score)),
                    "risk_score_p90": float(np.quantile(risk_score, 0.9)),
                    "entropy_mean": float(np.mean(entropy)),
                    "error_auroc": error_auc,
                    "error_aupr": error_aupr,
                    "vtvf_error_auroc": vtvf_auc,
                    "vtvf_error_aupr": vtvf_aupr,
                }
            )
            curve = _review_curve(y_test, y_pred, risk_score)
            curve.insert(0, "severity", severity)
            curve.insert(0, "corruption", corruption)
            review_rows.append(curve)
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "corruption": corruption,
                        "severity": severity,
                        "sample_index": np.arange(len(y_test)),
                        "y_true": y_test,
                        "y_pred": y_pred,
                        "risk_score": risk_score,
                        "entropy": entropy,
                        "prob_SR": probs[:, 0],
                        "prob_VT": probs[:, 1],
                        "prob_VF": probs[:, 2],
                    }
                )
            )

    summary = pd.DataFrame(rows)
    reviews = pd.concat(review_rows, ignore_index=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    summary.to_csv(out_dir / "risk_corruption_summary.csv", index=False)
    reviews.to_csv(out_dir / "risk_corruption_review_curves.csv", index=False)
    predictions.to_csv(out_dir / "risk_corruption_predictions.csv", index=False)

    mono_rows = []
    for corruption, sub in summary[summary["corruption"] != "clean"].groupby("corruption"):
        ordered = sub.sort_values("severity")
        corr = spearmanr(ordered["severity"], ordered["risk_score_mean"]).correlation
        err_corr = spearmanr(ordered["severity"], ordered["all_errors"]).correlation
        mono_rows.append(
            {
                "corruption": corruption,
                "risk_mean_severity_spearman": float(corr) if not np.isnan(corr) else float("nan"),
                "error_count_severity_spearman": float(err_corr) if not np.isnan(err_corr) else float("nan"),
                "risk_mean_severity_4_minus_0": float(
                    ordered.loc[ordered["severity"].eq(4), "risk_score_mean"].mean()
                    - ordered.loc[ordered["severity"].eq(0), "risk_score_mean"].mean()
                ),
            }
        )
    monotonicity = pd.DataFrame(mono_rows)
    monotonicity.to_csv(out_dir / "risk_corruption_monotonicity.csv", index=False)

    selected = summary[summary["corruption"] != "clean"]
    plt.figure(figsize=(8, 5))
    for corruption, sub in selected.groupby("corruption"):
        sub = sub.sort_values("severity")
        plt.plot(sub["severity"], sub["risk_score_mean"], marker="o", linewidth=1.5, label=corruption)
    plt.xlabel("Corruption severity")
    plt.ylabel("Mean RISK score")
    plt.title("RISK score under progressive ECG signal degradation")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "risk_score_vs_severity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for corruption, sub in selected.groupby("corruption"):
        sub = sub.sort_values("severity")
        plt.plot(sub["severity"], sub["error_auroc"], marker="o", linewidth=1.5, label=corruption)
    plt.xlabel("Corruption severity")
    plt.ylabel("Error AUROC using RISK score")
    plt.ylim(0, 1.02)
    plt.title("RISK error-detection AUROC under signal degradation")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "risk_error_auroc_vs_severity.png", dpi=180)
    plt.close()

    review20 = reviews[reviews["review_burden"].eq(0.20) & reviews["corruption"].ne("clean")]
    plt.figure(figsize=(8, 5))
    for corruption, sub in review20.groupby("corruption"):
        sub = sub.sort_values("severity")
        plt.plot(sub["severity"], sub["all_error_captured"], marker="o", linewidth=1.5, label=corruption)
    plt.xlabel("Corruption severity")
    plt.ylabel("All-error capture at 20% review")
    plt.ylim(0, 1.02)
    plt.title("RISK review capture under signal degradation")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "risk_review20_capture_vs_severity.png", dpi=180)
    plt.close()

    manifest = {
        "classifier_run_dir": str(classifier_dir),
        "risk_head_run_dir": str(risk_dir),
        "model": args.model,
        "seed": args.seed,
        "out_dir": str(out_dir),
        "interpretation": (
            "This evaluates corruption robustness of the deployable RISK score. It should be reported as "
            "an additional validation, distinct from the existing PRO severity robustness."
        ),
    }
    (out_dir / "risk_corruption_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(summary[summary["severity"].isin([0, 2, 4])].head(30))
    print(monotonicity)


if __name__ == "__main__":
    main()
