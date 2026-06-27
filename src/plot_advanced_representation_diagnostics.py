from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_COLORS = {"CNN": "#4C78A8", "CNN-LSTM": "#F58518"}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _layer_order(model: str) -> list[str]:
    if model == "CNN":
        return [
            "conv1",
            "pool1",
            "conv2",
            "pool2",
            "conv3",
            "pre_embedding_pool",
            "final_embedding",
            "classifier_logits",
        ]
    return [
        "cnn_conv1",
        "cnn_pool1",
        "cnn_conv2",
        "cnn_pool2",
        "cnn_conv3",
        "cnn_sequence",
        "lstm_last_state",
        "final_embedding",
        "classifier_logits",
    ]


def _plot_layerwise_probe(root: Path, out: Path) -> list[Path]:
    df = pd.read_csv(root / "layerwise_linear_probe_summary_6seed.csv")
    outputs: list[Path] = []

    for probe, metric, ylabel, filename in [
        ("vt_vs_vf_binary", "auroc_mean", "VT vs VF probe AUROC", "01_layerwise_vtvf_probe_auroc.png"),
        ("sr_vt_vf_multiclass", "macro_f1_mean", "SR/VT/VF probe Macro-F1", "02_layerwise_multiclass_probe_macro_f1.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9.8, 4.8))
        for model in ["CNN", "CNN-LSTM"]:
            sub = df[(df["model"] == model) & (df["probe"] == probe)].copy()
            order = _layer_order(model)
            sub["layer"] = pd.Categorical(sub["layer"], categories=order, ordered=True)
            sub = sub.sort_values("layer")
            x = np.arange(len(sub))
            ax.plot(x, sub[metric], marker="o", linewidth=2.0, color=MODEL_COLORS[model], label=model)
            err_col = metric.replace("_mean", "_std")
            if err_col in sub:
                ax.fill_between(
                    x,
                    sub[metric] - sub[err_col].fillna(0),
                    sub[metric] + sub[err_col].fillna(0),
                    color=MODEL_COLORS[model],
                    alpha=0.14,
                )
            for xi, label in zip(x, sub["layer"].astype(str)):
                ax.text(xi, sub[metric].iloc[xi] - 0.035, label, rotation=35, ha="right", va="top", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.set_ylim(0.2 if "AUROC" in ylabel else 0.45, 0.98 if "AUROC" in ylabel else 0.78)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        path = out / filename
        _save(fig, path)
        outputs.append(path)

    delta = pd.read_csv(root / "layerwise_linear_probe_delta_summary.csv")
    sub = delta[delta["probe"] == "vt_vs_vf_binary"].copy()
    sub["pair"] = sub["cnn_layer"] + " -> " + sub["cnn_lstm_layer"]
    fig, ax = plt.subplots(figsize=(9.8, 4.2))
    colors = np.where(sub["mean_delta"] >= 0, "#59A14F", "#E15759")
    ax.bar(np.arange(len(sub)), sub["mean_delta"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(sub["pair"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("CNN-LSTM minus CNN AUROC")
    ax.set_title("Layer-wise VT/VF Readability Delta")
    ax.grid(axis="y", alpha=0.25)
    path = out / "03_layerwise_vtvf_probe_delta.png"
    _save(fig, path)
    outputs.append(path)
    return outputs


def _plot_cka(root: Path, out: Path) -> list[Path]:
    df = pd.read_csv(root / "advanced_representation_cka.csv")
    outputs: list[Path] = []

    sub = df[df["comparison_type"] == "same_model_cross_seed"]
    grouped = (
        sub.groupby(["left_model", "representation"], as_index=False)
        .agg(linear_cka=("linear_cka", "mean"), svcca_mean_corr=("svcca_mean_corr", "mean"))
        .sort_values(["representation", "left_model"])
    )
    reps = ["final_embedding", "classifier_logits", "regularity_features"]
    x = np.arange(len(reps))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for offset, model, metric, hatch in [
        (-1.5 * width, "CNN", "linear_cka", ""),
        (-0.5 * width, "CNN-LSTM", "linear_cka", ""),
        (0.5 * width, "CNN", "svcca_mean_corr", "//"),
        (1.5 * width, "CNN-LSTM", "svcca_mean_corr", "//"),
    ]:
        vals = [
            grouped[(grouped["left_model"] == model) & (grouped["representation"] == rep)][metric].mean()
            for rep in reps
        ]
        label = f"{model} {'CKA' if metric == 'linear_cka' else 'SVCCA'}"
        ax.bar(x + offset, vals, width=width, label=label, color=MODEL_COLORS[model], alpha=0.82, hatch=hatch)
    ax.set_xticks(x)
    ax.set_xticklabels(["final embedding", "classifier logits", "regularity features"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Cross-seed similarity")
    ax.set_title("Cross-seed Representation Consistency")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2, fontsize=8)
    path = out / "04_cross_seed_cka_svcca.png"
    _save(fig, path)
    outputs.append(path)

    same_seed = df[(df["comparison_type"] == "same_seed_model_pair") & (df["representation"] != "regularity_features")]
    pivot = same_seed.pivot_table(index="left_seed", columns="representation", values="linear_cka", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Representation")
    ax.set_ylabel("Seed")
    ax.set_title("Same-seed CNN vs CNN-LSTM CKA")
    fig.colorbar(im, ax=ax, label="linear CKA")
    path = out / "05_same_seed_model_pair_cka_heatmap.png"
    _save(fig, path)
    outputs.append(path)
    return outputs


def _plot_distribution_and_perturbation(root: Path, out: Path) -> list[Path]:
    outputs: list[Path] = []
    dist = pd.read_csv(root / "advanced_distribution_geometry_all_runs.csv")
    vtvf = dist[(dist["representation"] == "final_embedding") & (dist["scope"] == "VT_vs_VF")]
    metrics = [
        ("mahalanobis_centroid_distance", "VT/VF Mahalanobis distance"),
        ("fisher_ratio", "VT/VF Fisher ratio"),
        ("euclidean_centroid_distance", "VT/VF Euclidean distance"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.0))
    for ax, (metric, title) in zip(axes, metrics):
        grouped = vtvf.groupby("model")[metric].agg(["mean", "std"]).reindex(["CNN", "CNN-LSTM"])
        ax.bar(grouped.index, grouped["mean"], yerr=grouped["std"], color=[MODEL_COLORS[m] for m in grouped.index], capsize=4)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    path = out / "06_vtvf_distribution_geometry.png"
    _save(fig, path)
    outputs.append(path)

    pert = pd.read_csv(root / "advanced_perturbation_stability_all_runs.csv")
    grouped = pert.groupby("model").agg(
        embedding_shift=("embedding_shift_mean", "mean"),
        prediction_flip=("prediction_flip_rate", "mean"),
        prototype_flip=("prototype_flip_rate", "mean"),
        neighbor_jaccard=("neighbor_jaccard_mean", "mean"),
    ).reindex(["CNN", "CNN-LSTM"])
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.8))
    labels = [
        ("embedding_shift", "Embedding shift"),
        ("prediction_flip", "Prediction flip"),
        ("prototype_flip", "Prototype flip"),
        ("neighbor_jaccard", "Neighbour Jaccard"),
    ]
    for ax, (metric, title) in zip(axes, labels):
        ax.bar(grouped.index, grouped[metric], color=[MODEL_COLORS[m] for m in grouped.index])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    path = out / "07_perturbation_stability_summary.png"
    _save(fig, path)
    outputs.append(path)
    return outputs


def _plot_concept_alignment(root: Path, out: Path) -> list[Path]:
    concept = pd.read_csv(root / "advanced_concept_alignment_all_runs.csv")
    sub = concept[concept["representation"] == "final_embedding"]
    grouped = sub.groupby(["model", "regularity_feature"], as_index=False).agg(
        corr=("max_abs_dim_correlation", "mean"),
        r2=("ridge_r2_from_representation", "mean"),
    )
    features = list(grouped["regularity_feature"].drop_duplicates())
    models = ["CNN", "CNN-LSTM"]
    outputs: list[Path] = []
    for metric, title, filename in [
        ("corr", "Regularity Concept Alignment: Max Dimension Correlation", "08_concept_alignment_correlation_heatmap.png"),
        ("r2", "Regularity Concept Alignment: Ridge R2 from Embedding", "09_concept_alignment_r2_heatmap.png"),
    ]:
        mat = np.zeros((len(models), len(features)))
        for i, model in enumerate(models):
            for j, feature in enumerate(features):
                val = grouped[(grouped["model"] == model) & (grouped["regularity_feature"] == feature)][metric]
                mat[i, j] = val.mean() if len(val) else np.nan
        fig, ax = plt.subplots(figsize=(10.5, 3.6))
        im = ax.imshow(mat, aspect="auto", cmap="magma" if metric == "corr" else "coolwarm")
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels(models)
        ax.set_xticks(np.arange(len(features)))
        ax.set_xticklabels(features, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=metric)
        path = out / filename
        _save(fig, path)
        outputs.append(path)
    return outputs


def _make_contact_sheet(paths: list[Path], out: Path) -> Path:
    import matplotlib.image as mpimg

    n = len(paths)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4.2))
    axes = np.asarray(axes).reshape(-1)
    for ax, path in zip(axes, paths):
        ax.imshow(mpimg.imread(path))
        ax.set_title(path.stem, fontsize=9)
        ax.axis("off")
    for ax in axes[len(paths) :]:
        ax.axis("off")
    sheet = out / "contact_sheet.png"
    _save(fig, sheet)
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot advanced ECG representation diagnostics.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/cnn_lstm_baseline_20260626/advanced_representation_diagnostics_6seed"),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or (args.root / "figures")
    paths: list[Path] = []
    paths += _plot_layerwise_probe(args.root, out)
    paths += _plot_cka(args.root, out)
    paths += _plot_distribution_and_perturbation(args.root, out)
    paths += _plot_concept_alignment(args.root, out)
    sheet = _make_contact_sheet(paths, out)
    print("Generated figures:")
    for path in paths:
        print(path)
    print(sheet)


if __name__ == "__main__":
    main()
