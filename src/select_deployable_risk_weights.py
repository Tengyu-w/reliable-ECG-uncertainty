from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


COMPONENTS = [
    "entropy",
    "local_instability",
    "vtvf_mixing",
    "knn",
    "softmax_vtvf_ambiguity",
]


def _capture(df: pd.DataFrame, score: np.ndarray, burden: float) -> tuple[float, float, float]:
    y = df["y_true"].to_numpy(int)
    pred = df["y_pred"].to_numpy(int)
    any_error = y != pred
    vtvf = ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))
    order = np.argsort(-score)
    reviewed = order[: max(1, int(round(len(df) * burden)))]
    auto = order[max(1, int(round(len(df) * burden))) :]
    return (
        float(any_error[reviewed].sum() / max(any_error.sum(), 1)),
        float(vtvf[reviewed].sum() / max(vtvf.sum(), 1)),
        float(any_error[auto].mean()) if len(auto) else np.nan,
    )


def _weight_grid(units: int):
    for values in itertools.product(range(units + 1), repeat=len(COMPONENTS)):
        if sum(values) == units:
            yield np.asarray(values, dtype=float) / units


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--grid-units", type=int, default=10)
    parser.add_argument(
        "--out", type=Path, default=Path("results/risk_validation_selected_20260620")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    selection_rows = []
    for seed in args.seeds:
        component_dir = args.components_root / f"seed{seed}" / "full_components"
        split_frames = {
            split: pd.read_csv(component_dir / f"risk_target_components_{split}.csv")
            for split in ["train", "val", "test"]
        }
        val = split_frames["val"]
        matrix = val[COMPONENTS].to_numpy(float)
        candidates = []
        for weights in _weight_grid(args.grid_units):
            score = matrix @ weights
            all_10, vtvf_10, auto_10 = _capture(val, score, 0.10)
            all_20, vtvf_20, auto_20 = _capture(val, score, 0.20)
            objective = 0.35 * vtvf_10 + 0.45 * vtvf_20 + 0.20 * all_20
            candidates.append(
                {
                    **{name: float(weight) for name, weight in zip(COMPONENTS, weights)},
                    "val_objective": objective,
                    "val_all_capture_10": all_10,
                    "val_vtvf_capture_10": vtvf_10,
                    "val_auto_error_rate_10": auto_10,
                    "val_all_capture_20": all_20,
                    "val_vtvf_capture_20": vtvf_20,
                    "val_auto_error_rate_20": auto_20,
                }
            )
        candidates_df = pd.DataFrame(candidates).sort_values(
            [
                "val_objective",
                "val_vtvf_capture_10",
                "val_vtvf_capture_20",
                "val_all_capture_20",
                "val_auto_error_rate_20",
            ],
            ascending=[False, False, False, False, True],
        )
        seed_dir = args.out / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        candidates_df.head(50).to_csv(seed_dir / "top_validation_weight_candidates.csv", index=False)
        selected = candidates_df.iloc[0]
        weights = selected[COMPONENTS].to_numpy(float)
        targets = {}
        for split, frame in split_frames.items():
            risk = np.clip(frame[COMPONENTS].to_numpy(float) @ weights, 0.0, 1.0).astype(
                np.float32
            )
            targets[split] = risk
            out_frame = frame.copy()
            out_frame["risk_target"] = risk
            out_frame.to_csv(seed_dir / f"selected_components_{split}.csv", index=False)
        np.savez(seed_dir / "selected_targets.npz", **targets)
        metadata = {
            "seed": seed,
            "selection_split": "validation",
            "test_labels_used_for_selection": False,
            "objective": "0.35*VT/VF capture@10 + 0.45*VT/VF capture@20 + 0.20*all-error capture@20",
            "grid_units": args.grid_units,
            "weights": {name: float(value) for name, value in zip(COMPONENTS, weights)},
            "validation_metrics": {
                key: float(selected[key])
                for key in selected.index
                if key.startswith("val_")
            },
        }
        (seed_dir / "selection.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        selection_rows.append(
            {
                "seed": seed,
                **metadata["weights"],
                **metadata["validation_metrics"],
                "target_path": str(seed_dir / "selected_targets.npz"),
            }
        )
    pd.DataFrame(selection_rows).to_csv(args.out / "selected_weights.csv", index=False)
    print(args.out)


if __name__ == "__main__":
    main()
