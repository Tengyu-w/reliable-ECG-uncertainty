from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REP_COLUMNS = [
    "silhouette_test",
    "sr_vt_norm_dist_test",
    "sr_vf_norm_dist_test",
    "vt_vf_norm_dist_test",
    "local_purity_mean",
    "vtvf_mixing_ventricular",
    "prototype_vtvf_ambiguity_correct_vtvf",
    "prototype_vtvf_ambiguity_vtvf_error",
]

BOUNDARY_GROUPS = [
    "all",
    "vtvf_true",
    "any_error",
    "vtvf_cross_error",
    "classifier_boundary_mismatch",
    "representation_overlap",
    "mixed_or_outlying",
]


def _load_run(seed: str, model: str, run_dir: Path) -> dict[str, float | int | str]:
    metrics_path = run_dir / "metrics.json"
    rep_path = run_dir / "layerwise_representation_summary.csv"
    boundary_path = run_dir / "decision_boundary_summary.csv"
    mechanism_path = run_dir / "decision_boundary_mechanism_counts.csv"
    missing = [path for path in [metrics_path, rep_path, boundary_path, mechanism_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing diagnostic files for {run_dir}: {missing}")

    metrics = pd.read_json(metrics_path, typ="series")
    rep = pd.read_csv(rep_path)
    boundary = pd.read_csv(boundary_path)
    mechanisms = pd.read_csv(mechanism_path)

    final_rep = rep[rep["layer"].eq("final_embedding")].iloc[0]
    logits_rep = rep[rep["layer"].eq("classifier_logits")].iloc[0]
    all_boundary = boundary[boundary["group"].eq("all")].iloc[0]
    vtvf_boundary = boundary[boundary["group"].eq("vtvf_true")].iloc[0]
    cross_boundary = boundary[boundary["group"].eq("vtvf_cross_error")].iloc[0] if boundary["group"].eq("vtvf_cross_error").any() else None

    row: dict[str, float | int | str] = {
        "seed": seed,
        "model": model,
        "run_dir": str(run_dir),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "ece": float(metrics["ece"]),
        "total_errors": int(metrics["total_errors"]),
        "vtvf_cross_errors": int(metrics["vtvf_cross_errors"]),
        "frozen_plain_macro_f1": float(all_boundary["frozen_plain_macro_f1"]),
        "frozen_plain_vtvf_cross_errors": int(all_boundary["frozen_plain_vtvf_cross_errors"]),
        "frozen_balanced_macro_f1": float(all_boundary["frozen_balanced_macro_f1"]),
        "frozen_balanced_vtvf_cross_errors": int(all_boundary["frozen_balanced_vtvf_cross_errors"]),
        "vtvf_classifier_proto_disagreement_rate": float(vtvf_boundary["classifier_proto_disagreement_rate"]),
        "vtvf_cross_mean_confidence": float(cross_boundary["mean_confidence"]) if cross_boundary is not None else float("nan"),
        "vtvf_cross_mean_abs_proto_margin": float(cross_boundary["mean_abs_proto_vtvf_margin"]) if cross_boundary is not None else float("nan"),
        "vtvf_cross_mean_abs_logit_margin": float(cross_boundary["mean_abs_logit_vtvf_margin"]) if cross_boundary is not None else float("nan"),
    }
    for col in REP_COLUMNS:
        row[f"final_{col}"] = float(final_rep[col])
        row[f"logits_{col}"] = float(logits_rep[col])

    for _, mech in mechanisms.iterrows():
        key = str(mech["mechanism"])
        row[f"mechanism_{key}_n"] = int(mech["n"])
        row[f"mechanism_{key}_fraction"] = float(mech["fraction"])
    for key in ["correct", "representation_overlap", "classifier_boundary_mismatch", "mixed_or_outlying"]:
        row.setdefault(f"mechanism_{key}_n", 0)
        row.setdefault(f"mechanism_{key}_fraction", 0.0)
    return row


def _paired_deltas(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, group in runs.groupby("seed"):
        if set(group["model"]) != {"CNN", "CNN-LSTM"}:
            continue
        base = group[group["model"].eq("CNN")].iloc[0]
        comp = group[group["model"].eq("CNN-LSTM")].iloc[0]
        row = {"seed": seed, "baseline": "CNN", "comparator": "CNN-LSTM"}
        for col in [
            "accuracy",
            "macro_f1",
            "ece",
            "total_errors",
            "vtvf_cross_errors",
            "final_vt_vf_norm_dist_test",
            "final_vtvf_mixing_ventricular",
            "final_prototype_vtvf_ambiguity_vtvf_error",
            "mechanism_representation_overlap_fraction",
            "mechanism_classifier_boundary_mismatch_fraction",
            "frozen_plain_macro_f1",
            "frozen_plain_vtvf_cross_errors",
            "frozen_balanced_macro_f1",
            "frozen_balanced_vtvf_cross_errors",
        ]:
            row[f"{col}_delta"] = comp[col] - base[col]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate representation and decision-boundary diagnostics.")
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("SEED", "MODEL", "RUN_DIR"),
        required=True,
        help="Example: --run 42 CNN results/.../20260626_003641_cnn",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame([_load_run(seed, model, Path(run_dir)) for seed, model, run_dir in args.run])
    runs = runs.sort_values(["seed", "model"])
    runs.to_csv(args.out / "representation_boundary_run_summary.csv", index=False)

    numeric_cols = runs.select_dtypes(include="number").columns.tolist()
    aggregate = runs.groupby("model")[numeric_cols].agg(["mean", "std", "median"])
    aggregate.columns = ["_".join(col).strip("_") for col in aggregate.columns.to_flat_index()]
    aggregate.reset_index().to_csv(args.out / "representation_boundary_model_aggregate.csv", index=False)

    deltas = _paired_deltas(runs)
    deltas.to_csv(args.out / "representation_boundary_paired_deltas.csv", index=False)
    delta_aggregate = deltas.drop(columns=["seed", "baseline", "comparator"]).agg(["mean", "median", "min", "max"]).T
    delta_aggregate.reset_index(names="metric").to_csv(args.out / "representation_boundary_delta_aggregate.csv", index=False)

    report = {
        "runs": runs.to_dict(orient="records"),
        "paired_deltas": deltas.to_dict(orient="records"),
        "delta_aggregate": delta_aggregate.reset_index(names="metric").to_dict(orient="records"),
        "notes": {
            "representation_overlap": "Prediction error where nearest embedding prototype also disagrees with the true label and matches the classifier prediction.",
            "classifier_boundary_mismatch": "Prediction error where nearest embedding prototype matches the true label but classifier prediction disagrees.",
            "vtvf_mixing_ventricular": "Among VT/VF samples, fraction of ventricular KNN neighbors from the opposite ventricular class.",
        },
    }
    (args.out / "representation_boundary_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(runs[["seed", "model", "accuracy", "macro_f1", "vtvf_cross_errors", "final_vt_vf_norm_dist_test", "final_vtvf_mixing_ventricular", "mechanism_representation_overlap_fraction", "mechanism_classifier_boundary_mismatch_fraction"]])
    print(delta_aggregate)


if __name__ == "__main__":
    main()
