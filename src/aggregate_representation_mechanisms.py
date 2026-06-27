from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _parse_run(values: list[str]) -> dict:
    if len(values) != 5:
        raise argparse.ArgumentTypeError("--run expects: FAMILY VARIANT SEED MODEL RUN_DIR")
    family, variant, seed, model, run_dir = values
    return {"family": family, "variant": variant, "seed": int(seed), "model": model, "run_dir": Path(run_dir)}


def _load(run: dict, filename: str) -> pd.DataFrame:
    path = run["run_dir"] / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "run_dir", str(run["run_dir"]))
    df.insert(0, "model", run["model"])
    df.insert(0, "variant", run["variant"])
    df.insert(0, "family", run["family"])
    df.insert(0, "seed", run["seed"])
    return df


def _mean_table(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    agg = {metric: ["mean", "std", "median"] for metric in metrics if metric in df.columns}
    out = df.groupby(group_cols).agg(agg)
    out.columns = ["_".join(col).strip("_") for col in out.columns]
    return out.reset_index()


def _variant_delta(table: pd.DataFrame, family: str, baseline: str, comparator: str, key_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    sub = table[table["family"].eq(family)]
    base = sub[sub["variant"].eq(baseline)]
    comp = sub[sub["variant"].eq(comparator)]
    rows = []
    merge_cols = ["seed", *key_cols]
    merged = base.merge(comp, on=merge_cols, suffixes=("_baseline", "_comparator"))
    for _, row in merged.iterrows():
        out = {"family": family, "baseline": baseline, "comparator": comparator}
        for col in merge_cols:
            out[col] = row[col]
        for metric in metrics:
            if metric in row:
                continue
            left = f"{metric}_baseline"
            right = f"{metric}_comparator"
            if left in row and right in row:
                out[f"{metric}_delta"] = row[right] - row[left]
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate representation mechanism analyses.")
    parser.add_argument("--run", nargs=5, action="append", required=True, metavar=("FAMILY", "VARIANT", "SEED", "MODEL", "RUN_DIR"))
    parser.add_argument("--out", type=Path, default=Path("results/representation_mechanisms"))
    args = parser.parse_args()

    runs = [_parse_run(item) for item in args.run]
    args.out.mkdir(parents=True, exist_ok=True)

    readability = pd.concat([_load(run, "mechanism_layer_readability.csv") for run in runs], ignore_index=True)
    stability = pd.concat([_load(run, "mechanism_layer_perturbation_stability.csv") for run in runs], ignore_index=True)
    tradeoff = pd.concat([_load(run, "mechanism_stability_readability_tradeoff.csv") for run in runs], ignore_index=True)
    stable_errors = pd.concat([_load(run, "mechanism_stable_confident_errors.csv") for run in runs], ignore_index=True)

    readability.to_csv(args.out / "mechanism_readability_all_runs.csv", index=False)
    stability.to_csv(args.out / "mechanism_layer_stability_all_runs.csv", index=False)
    tradeoff.to_csv(args.out / "mechanism_stability_readability_tradeoff_all_runs.csv", index=False)
    stable_errors.to_csv(args.out / "mechanism_stable_errors_all_runs.csv", index=False)

    readability_summary = _mean_table(
        readability,
        ["family", "variant", "model", "layer", "probe"],
        ["accuracy", "macro_f1", "auroc", "vtvf_cross_errors"],
    )
    stability_summary = _mean_table(
        stability,
        ["family", "variant", "model", "layer"],
        [
            "embedding_shift_mean",
            "embedding_shift_error_mean",
            "embedding_shift_vtvf_cross_error_mean",
            "cosine_preservation_mean",
            "prediction_flip_rate",
            "prediction_flip_error_rate",
            "prediction_flip_vtvf_cross_error_rate",
        ],
    )
    tradeoff_summary = _mean_table(
        tradeoff,
        ["family", "variant", "model", "layer"],
        [
            "embedding_shift_mean",
            "prediction_flip_rate",
            "cosine_preservation_mean",
            "vtvf_probe_auroc",
            "multiclass_probe_macro_f1",
            "probe_vtvf_cross_errors",
        ],
    )
    stable_error_summary = _mean_table(
        stable_errors,
        ["family", "variant", "model", "group"],
        ["n", "fraction", "mean_confidence", "mean_flip_rate", "mean_final_embedding_shift"],
    )

    readability_summary.to_csv(args.out / "mechanism_readability_summary.csv", index=False)
    stability_summary.to_csv(args.out / "mechanism_layer_stability_summary.csv", index=False)
    tradeoff_summary.to_csv(args.out / "mechanism_tradeoff_summary.csv", index=False)
    stable_error_summary.to_csv(args.out / "mechanism_stable_error_summary.csv", index=False)

    delta_frames = []
    if set(["CNN", "CNN-LSTM"]).issubset(set(tradeoff["variant"])):
        delta_frames.append(
            _variant_delta(
                tradeoff,
                "cnn_lstm_10seed",
                "CNN",
                "CNN-LSTM",
                ["layer"],
                ["embedding_shift_mean", "prediction_flip_rate", "vtvf_probe_auroc", "multiclass_probe_macro_f1"],
            )
        )
    if set(["baseline", "pro"]).issubset(set(tradeoff["variant"])):
        delta_frames.append(
            _variant_delta(
                tradeoff,
                "pro_3seed",
                "baseline",
                "pro",
                ["layer"],
                ["embedding_shift_mean", "prediction_flip_rate", "vtvf_probe_auroc", "multiclass_probe_macro_f1"],
            )
        )
    deltas = pd.concat([df for df in delta_frames if not df.empty], ignore_index=True) if delta_frames else pd.DataFrame()
    if not deltas.empty:
        deltas.to_csv(args.out / "mechanism_paired_tradeoff_deltas.csv", index=False)
        delta_summary = deltas.groupby(["family", "baseline", "comparator", "layer"], as_index=False).agg(
            embedding_shift_mean_delta_mean=("embedding_shift_mean_delta", "mean"),
            prediction_flip_rate_delta_mean=("prediction_flip_rate_delta", "mean"),
            vtvf_probe_auroc_delta_mean=("vtvf_probe_auroc_delta", "mean"),
            multiclass_probe_macro_f1_delta_mean=("multiclass_probe_macro_f1_delta", "mean"),
        )
        delta_summary.to_csv(args.out / "mechanism_paired_tradeoff_delta_summary.csv", index=False)

    report = {
        "n_runs": len(runs),
        "outputs": {
            "readability_summary": str(args.out / "mechanism_readability_summary.csv"),
            "stability_summary": str(args.out / "mechanism_layer_stability_summary.csv"),
            "tradeoff_summary": str(args.out / "mechanism_tradeoff_summary.csv"),
            "stable_error_summary": str(args.out / "mechanism_stable_error_summary.csv"),
        },
    }
    (args.out / "representation_mechanism_aggregate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
