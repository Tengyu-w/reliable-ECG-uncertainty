from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {0: "SR", 1: "VT", 2: "VF"}


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _normalise(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if hi <= lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def _level(x: float) -> str:
    if x >= 0.75:
        return "high"
    if x >= 0.45:
        return "medium"
    return "low"


def _routing(row: pd.Series, threshold: float, force_threshold: float) -> str:
    if row["risk_score"] >= force_threshold:
        return "forced_expert_review"
    if row["is_vtvf_pair"] and (row["risk_score"] >= threshold or row["vtvf_ambiguity_norm"] >= 0.5):
        return "boundary_review"
    if row["knn_norm"] >= 0.75 or row["entropy_norm"] >= 0.75:
        return "signal_quality_or_uncertainty_review"
    return "automatic_accept_with_low_risk_flag"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate decision-support risk evidence cards for high-risk ECG cases from an existing RISK run."
    )
    parser.add_argument("--classifier-run-dir", type=Path, required=True)
    parser.add_argument("--risk-head-run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    classifier_dir = _resolve_run_dir(args.classifier_run_dir)
    risk_dir = _resolve_run_dir(args.risk_head_run_dir)
    out_dir = args.out or (risk_dir / "risk_evidence_cards")
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = classifier_dir / f"{args.split}_predictions.csv"
    comp_path = classifier_dir / f"risk_target_components_{args.split}.csv"
    risk_path = risk_dir / f"risk_scores_{args.split}.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not comp_path.exists():
        raise FileNotFoundError(comp_path)
    if not risk_path.exists():
        raise FileNotFoundError(risk_path)

    preds = pd.read_csv(pred_path)
    comps = pd.read_csv(comp_path)
    risk = pd.read_csv(risk_path)
    n = min(len(preds), len(comps), len(risk))
    df = pd.DataFrame({"sample_index": np.arange(n)})
    for col in ["prob_SR", "prob_VT", "prob_VF", "y_true", "y_pred"]:
        df[col] = preds[col].iloc[:n].to_numpy()
    for col in ["entropy", "knn", "local_instability", "vtvf_mixing", "softmax_vtvf_ambiguity", "risk_target"]:
        df[col] = comps[col].iloc[:n].to_numpy()
    df["risk_score"] = risk["risk_score"].iloc[:n].to_numpy()

    probs = df[["prob_SR", "prob_VT", "prob_VF"]].to_numpy(float)
    sorted_idx = np.argsort(-probs, axis=1)
    df["predicted_label"] = df["y_pred"].map(LABELS)
    df["true_label"] = df["y_true"].map(LABELS)
    df["alternative_label"] = [LABELS[int(i)] for i in sorted_idx[:, 1]]
    df["alternative_probability"] = probs[np.arange(n), sorted_idx[:, 1]]
    df["predicted_probability"] = probs[np.arange(n), sorted_idx[:, 0]]
    df["vtvf_ambiguity"] = np.minimum(df["prob_VT"], df["prob_VF"]) / np.maximum(df["prob_VT"] + df["prob_VF"], 1e-8)
    df["is_error"] = df["y_true"] != df["y_pred"]
    df["is_vtvf_pair"] = df["y_pred"].isin([1, 2]) | df["y_true"].isin([1, 2])

    for col in ["entropy", "knn", "local_instability", "vtvf_mixing", "softmax_vtvf_ambiguity", "vtvf_ambiguity"]:
        df[f"{col}_norm"] = _normalise(df[col].astype(float))
        df[f"{col}_level"] = df[f"{col}_norm"].map(_level)

    review_threshold = float(df["risk_score"].quantile(0.80))
    force_threshold = float(df["risk_score"].quantile(0.95))
    df["routing_decision"] = df.apply(lambda row: _routing(row, review_threshold, force_threshold), axis=1)
    df["evidence_summary"] = df.apply(
        lambda row: (
            f"pred={row['predicted_label']}({row['predicted_probability']:.3f}); "
            f"alt={row['alternative_label']}({row['alternative_probability']:.3f}); "
            f"risk={row['risk_score']:.3f}; entropy={row['entropy_level']}; "
            f"VT/VF ambiguity={row['vtvf_ambiguity_level']}; local mixing={row['vtvf_mixing_level']}; "
            f"KNN atypicality={row['knn_level']}; route={row['routing_decision']}"
        ),
        axis=1,
    )

    ranked = df.sort_values(["risk_score", "vtvf_ambiguity_norm"], ascending=False).head(args.top_n).copy()
    card_cols = [
        "sample_index",
        "true_label",
        "predicted_label",
        "predicted_probability",
        "alternative_label",
        "alternative_probability",
        "risk_score",
        "risk_target",
        "entropy_level",
        "vtvf_ambiguity_level",
        "local_instability_level",
        "vtvf_mixing_level",
        "knn_level",
        "routing_decision",
        "is_error",
        "evidence_summary",
    ]
    ranked[card_cols].to_csv(out_dir / f"risk_evidence_cards_top{args.top_n}_{args.split}.csv", index=False)
    df.to_csv(out_dir / f"risk_evidence_cards_all_{args.split}.csv", index=False)

    markdown = out_dir / f"risk_evidence_cards_top{args.top_n}_{args.split}.md"
    lines = [
        "# Risk Evidence Cards",
        "",
        "These cards are decision-support explanations, not autonomous diagnosis. They summarize why a short-window ECG case is routed to review.",
        "",
    ]
    for _, row in ranked.iterrows():
        lines.extend(
            [
                f"## Case {int(row['sample_index'])}",
                "",
                f"- Predicted label: **{row['predicted_label']}** ({row['predicted_probability']:.3f})",
                f"- Alternative label: **{row['alternative_label']}** ({row['alternative_probability']:.3f})",
                f"- True label: {row['true_label']} | error: {bool(row['is_error'])}",
                f"- RISK score: {row['risk_score']:.3f}",
                f"- Entropy: {row['entropy_level']}; VT/VF ambiguity: {row['vtvf_ambiguity_level']}; local mixing: {row['vtvf_mixing_level']}; KNN atypicality: {row['knn_level']}",
                f"- Routing decision: **{row['routing_decision']}**",
                "",
            ]
        )
    markdown.write_text("\n".join(lines), encoding="utf-8")

    plot_df = ranked.head(min(12, len(ranked))).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.42 * len(plot_df))))
    y = np.arange(len(plot_df))
    plt.barh(y, plot_df["risk_score"], color="#0B7A75")
    plt.yticks(y, [f"#{int(i)} {p}/{a}" for i, p, a in zip(plot_df["sample_index"], plot_df["predicted_label"], plot_df["alternative_label"])])
    plt.xlabel("RISK score")
    plt.title("Top risk evidence cards")
    plt.tight_layout()
    plt.savefig(out_dir / f"risk_evidence_cards_top{min(12, len(ranked))}_{args.split}.png", dpi=180)
    plt.close()

    print(ranked[card_cols].head(args.top_n))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
