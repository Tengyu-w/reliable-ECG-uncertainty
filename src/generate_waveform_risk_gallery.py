from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .data import CLASS_NAMES, load_rhythm_windows


def _test_indices(y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    indices = np.arange(len(y))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, test_idx = next(splitter.split(indices, y, groups=groups))
    return test_idx


def _take_unique(
    frame: pd.DataFrame,
    mask: pd.Series,
    category: str,
    n: int,
    ascending: bool,
    used: set[int],
) -> list[pd.Series]:
    selected = []
    sub = frame[mask & ~frame["sample_index"].isin(used)].sort_values(
        "risk_score", ascending=ascending
    )
    for _, row in sub.head(n).iterrows():
        selected.append(row)
        used.add(int(row["sample_index"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate internal waveform evidence cases for RISK review routing."
    )
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--classifier-run-dir", type=Path, required=True)
    parser.add_argument("--risk-run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=100)
    parser.add_argument(
        "--out", type=Path, default=Path("results/waveform_risk_gallery_20260620/seed42")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dataset = load_rhythm_windows(args.mat)
    test_idx = _test_indices(dataset.y, dataset.record_ids, args.seed)
    pred = pd.read_csv(args.classifier_run_dir / "test_predictions.csv")
    components = pd.read_csv(args.classifier_run_dir / "risk_target_components_test.csv")
    risk = pd.read_csv(args.risk_run_dir / "risk_scores_test.csv")
    if not (len(test_idx) == len(pred) == len(components) == len(risk)):
        raise ValueError("Test split and saved prediction lengths do not match.")

    frame = pred[["y_true", "y_pred", "prob_SR", "prob_VT", "prob_VF"]].copy()
    frame.insert(0, "sample_index", np.arange(len(frame)))
    frame["global_index"] = test_idx
    frame["window_id"] = dataset.window_ids[test_idx]
    frame["record_id"] = dataset.record_ids[test_idx]
    for col in [
        "entropy",
        "knn",
        "local_instability",
        "vtvf_mixing",
        "softmax_vtvf_ambiguity",
        "risk_target",
    ]:
        frame[col] = components[col].to_numpy()
    frame["risk_score"] = risk["risk_score"].to_numpy()
    frame["is_error"] = frame["y_true"] != frame["y_pred"]
    frame["is_vtvf_cross_error"] = (
        ((frame["y_true"] == 1) & (frame["y_pred"] == 2))
        | ((frame["y_true"] == 2) & (frame["y_pred"] == 1))
    )
    frame["reviewed_at_20pct"] = (
        frame["risk_score"] >= frame["risk_score"].quantile(0.80)
    )

    used: set[int] = set()
    selected: list[tuple[str, pd.Series]] = []
    categories = [
        (
            "captured_high-risk VT/VF cross-error",
            frame["is_vtvf_cross_error"] & frame["reviewed_at_20pct"],
            False,
        ),
        (
            "missed_low-risk VT/VF cross-error",
            frame["is_vtvf_cross_error"] & ~frame["reviewed_at_20pct"],
            True,
        ),
        (
            "high-risk correct ventricular case",
            ~frame["is_error"] & frame["y_true"].isin([1, 2]),
            False,
        ),
        (
            "captured non-boundary error",
            frame["is_error"] & ~frame["is_vtvf_cross_error"] & frame["reviewed_at_20pct"],
            False,
        ),
    ]
    for category, mask, ascending in categories:
        for row in _take_unique(frame, mask, category, 3, ascending, used):
            selected.append((category, row))

    selected_rows = []
    for category, row in selected:
        item = row.to_dict()
        item["category"] = category
        item["true_label"] = CLASS_NAMES[int(row["y_true"])]
        item["predicted_label"] = CLASS_NAMES[int(row["y_pred"])]
        selected_rows.append(item)
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(args.out / "waveform_risk_cases.csv", index=False)

    fig, axes = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
    axes = axes.ravel()
    time = np.arange(dataset.x.shape[-1]) / args.sample_rate
    for ax, (category, row) in zip(axes, selected):
        signal = dataset.x[int(row["global_index"]), 0]
        color = "#C0392B" if bool(row["is_error"]) else "#167D73"
        ax.plot(time, signal, color=color, linewidth=0.9)
        ax.axhline(0, color="#bbbbbb", linewidth=0.5)
        true_label = CLASS_NAMES[int(row["y_true"])]
        pred_label = CLASS_NAMES[int(row["y_pred"])]
        ax.set_title(
            f"{category}\n{true_label}->{pred_label} | risk={row['risk_score']:.3f} | "
            f"H={row['entropy']:.2f}, KNN={row['knn']:.2f}",
            fontsize=9,
        )
        ax.set_ylabel("Normalized amplitude")
        ax.grid(True, color="#eeeeee", linewidth=0.5)
    for ax in axes[len(selected) :]:
        ax.axis("off")
    for ax in axes[-3:]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(
        "Internal RISK evidence gallery: successful review routing and failure cases",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out / "waveform_risk_gallery_12_cases.png", dpi=200)
    plt.close(fig)

    notes = [
        "# Waveform RISK evidence gallery",
        "",
        "- Cases are selected algorithmically from seed 42, not manually chosen for appearance.",
        "- The gallery includes successful VT/VF error capture, missed VT/VF errors, high-risk correct ventricular cases, and captured non-boundary errors.",
        "- Waveforms are normalized five-second research windows. They are not diagnostic strips and have not received clinician adjudication.",
        "- The missed-error row is especially important for discussing residual risk and why forced expert review cannot be replaced by the model.",
        "- This artifact is for internal dissertation analysis and should not be redistributed with private dataset content.",
    ]
    (args.out / "gallery_notes.md").write_text("\n".join(notes), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
