from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "mechanism_derived_model_search_3seed_20260701"
ASSET_DIR = ROOT / "docs" / "final_model_selection_assets"
REPORT_PATH = ROOT / "docs" / "FINAL_MODEL_SELECTION_REPORT_CN.md"

CLASS_NAMES = {0: "SR", 1: "VT", 2: "VF"}
CLASS_COLORS = {0: "#4C78A8", 1: "#F58518", 2: "#54A24B"}

KEY_CANDIDATES = [
    "baseline",
    "boundary075",
    "boundary075_prototype",
    "proto_center_only",
    "proto_center_margin",
    "boundary075_center",
    "boundary075_margin",
    "boundary100_center",
    "boundary075_contrastive",
    "boundary075_center_calibrated",
]

MECHANISM_ROWS = [
    {
        "analysis_finding": "VT/VF boundary samples concentrate dangerous errors.",
        "mechanism": "Boundary-risk weighting",
        "model_terms": "boundary_ce_weight",
        "tested_configs": "boundary075; boundary075_center; boundary100_center; boundary075_prototype",
        "outcome_interpretation": "Boundary weighting improves total errors, but boundary075 alone only weakly reduces VT/VF cross-errors; it is not sufficient as the final mechanism.",
    },
    {
        "analysis_finding": "Embedding neighborhoods are mixed and same-class samples are dispersed.",
        "mechanism": "Prototype-center compactness",
        "model_terms": "prototype_center_weight",
        "tested_configs": "proto_center_only; proto_center_margin; boundary075_center; boundary075_prototype",
        "outcome_interpretation": "This is the strongest minimal mechanism: proto_center_only improves all six outcomes in 3/3 paired seeds.",
    },
    {
        "analysis_finding": "VT/VF prototype ambiguity suggests insufficient inter-class separation.",
        "mechanism": "VT/VF prototype margin",
        "model_terms": "prototype_margin_weight + prototype_vtvf_margin",
        "tested_configs": "proto_margin_only; proto_center_margin; boundary075_margin; boundary075_prototype",
        "outcome_interpretation": "Margin alone is weak; it can help in combination, but should not be treated as a standalone core explanation.",
    },
    {
        "analysis_finding": "Local purity and VT/VF neighborhood mixing are associated with outcome changes.",
        "mechanism": "Contrastive local-purity control",
        "model_terms": "contrastive_weight + contrastive boundary/negative anchors",
        "tested_configs": "boundary075_contrastive",
        "outcome_interpretation": "Fails the guard in this search because VT/VF cross-errors increase despite some overall improvement.",
    },
    {
        "analysis_finding": "Calibration and confident-risk mismatch remain visible in the uncertainty analysis.",
        "mechanism": "Entropy / anti-confident risk calibration",
        "model_terms": "risk_entropy_weight + anti_confident_risk_weight",
        "tested_configs": "boundary075_center_calibrated",
        "outcome_interpretation": "Does not pass the guard: calibration add-ons do not justify inclusion when VT/VF safety outcomes degrade.",
    },
    {
        "analysis_finding": "Waveform regularity and signal morphology are useful diagnostic evidence.",
        "mechanism": "Regularity / waveform evidence",
        "model_terms": "regularity auxiliary losses or routing evidence",
        "tested_configs": "not retained in this 36-run final candidate set",
        "outcome_interpretation": "Kept as diagnostic/recover evidence, not as the selected model-layer training constraint.",
    },
]


def read_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(RESULT_DIR / "model_layer_causal_pareto_search_summary.csv")
    paired = pd.read_csv(RESULT_DIR / "model_layer_causal_pareto_search_paired_effects.csv")
    run = pd.read_csv(RESULT_DIR / "model_layer_causal_pareto_search_run_level.csv")
    return summary, paired, run


def build_absolute_table(run: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "macro_f1", "ece", "vtvf_cross_errors", "total_errors", "error_migration_penalty"]
    grouped = run.groupby("candidate", as_index=False)[metric_cols].agg(["mean", "std"])
    grouped.columns = ["candidate"] + [f"{metric}_{stat}" for metric, stat in grouped.columns[1:]]
    grouped = grouped.reset_index(drop=True)
    order = {candidate: i for i, candidate in enumerate(KEY_CANDIDATES)}
    grouped["order"] = grouped["candidate"].map(order).fillna(999)
    return grouped.sort_values(["order", "candidate"]).drop(columns=["order"])


def build_mechanism_bridge() -> pd.DataFrame:
    return pd.DataFrame(MECHANISM_ROWS)


def fmt(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def save_outcome_bar(summary: pd.DataFrame) -> Path:
    plot_df = summary.copy()
    plot_df["label"] = plot_df["candidate"].str.replace("_", "\n")
    plot_df = plot_df.sort_values("total_errors_delta_mean")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    specs = [
        ("total_errors_delta_mean", "Total errors delta\n(lower is better)", "#4C78A8"),
        ("vtvf_cross_errors_delta_mean", "VT/VF cross-errors delta\n(lower is better)", "#F58518"),
        ("macro_f1_delta_mean", "Macro-F1 delta\n(higher is better)", "#54A24B"),
    ]
    for ax, (col, title, color) in zip(axes, specs):
        ax.barh(plot_df["label"], plot_df[col], color=color, alpha=0.85)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=8)
    out = ASSET_DIR / "01_outcome_delta_ranking.png"
    fig.suptitle("Mechanism-derived model search: paired outcome deltas vs baseline", fontsize=13)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def save_pareto_plot(summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    colors = summary["is_pareto"].map({True: "#D62728", False: "#7F7F7F"})
    sizes = summary["selected_for_full_validation"].map({True: 120, False: 55})
    ax.scatter(
        summary["total_errors_delta_mean"],
        summary["macro_f1_delta_mean"],
        c=colors,
        s=sizes,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )
    for _, row in summary.iterrows():
        ax.text(
            row["total_errors_delta_mean"] + 2,
            row["macro_f1_delta_mean"],
            row["candidate"].replace("_", " "),
            fontsize=8,
        )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Total errors delta vs baseline (lower is better)")
    ax.set_ylabel("Macro-F1 delta vs baseline (higher is better)")
    ax.set_title("Pareto view: performance gain should not hide safety trade-offs")
    out = ASSET_DIR / "02_pareto_tradeoff.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def save_seed_heatmap(summary: pd.DataFrame) -> Path:
    cols = [
        "accuracy_good_direction_n",
        "macro_f1_good_direction_n",
        "ece_good_direction_n",
        "vtvf_cross_errors_good_direction_n",
        "total_errors_good_direction_n",
        "error_migration_penalty_good_direction_n",
    ]
    labels = ["Acc", "Macro-F1", "ECE", "VT/VF", "Total", "Migration"]
    plot_df = summary.sort_values(["mean_good_objective_count", "total_errors_delta_mean"], ascending=[False, True])
    values = plot_df[cols].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    im = ax.imshow(values, cmap="YlGn", vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(cols)), labels=labels)
    ax.set_yticks(range(len(plot_df)), labels=plot_df["candidate"].str.replace("_", " "))
    ax.set_title("Seed-level guard consistency: number of paired seeds improving per objective")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{int(values[i, j])}/3", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="good-direction seeds")
    out = ASSET_DIR / "03_seed_guard_consistency.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def find_run_dir(run: pd.DataFrame, candidate: str, seed: int) -> Path | None:
    sub = run[(run["candidate"] == candidate) & (run["seed"] == seed)]
    if sub.empty:
        return None
    path = ROOT / str(sub.iloc[0]["run_dir"])
    return path if path.exists() else None


def load_embedding_view(run: pd.DataFrame, candidates: list[str], seed: int = 42):
    chunks = []
    for candidate in candidates:
        run_dir = find_run_dir(run, candidate, seed)
        if not run_dir:
            continue
        emb_path = run_dir / "embeddings_test.npz"
        pred_path = run_dir / "test_predictions.csv"
        if not emb_path.exists() or not pred_path.exists():
            continue
        emb_npz = np.load(emb_path)
        preds = pd.read_csv(pred_path)
        emb = emb_npz["embeddings"].astype(np.float32)
        y = emb_npz["y"].astype(int)
        y_pred = preds["y_pred"].to_numpy(int)
        chunks.append((candidate, emb, y, y_pred))
    return chunks


def save_embedding_contact_sheet(run: pd.DataFrame) -> Path | None:
    candidates = ["baseline", "boundary075", "boundary075_prototype", "proto_center_only", "proto_center_margin"]
    chunks = load_embedding_view(run, candidates, seed=42)
    if not chunks:
        return None
    all_emb = np.vstack([chunk[1] for chunk in chunks])
    coords_all = PCA(n_components=2, random_state=0).fit_transform(all_emb)
    offset = 0
    fig, axes = plt.subplots(1, len(chunks), figsize=(4.4 * len(chunks), 4), sharex=True, sharey=True, constrained_layout=True)
    if len(chunks) == 1:
        axes = [axes]
    for ax, (candidate, emb, y, y_pred) in zip(axes, chunks):
        coords = coords_all[offset : offset + len(emb)]
        offset += len(emb)
        for cls in [0, 1, 2]:
            mask = y == cls
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=7,
                alpha=0.35,
                color=CLASS_COLORS[cls],
                label=CLASS_NAMES[cls],
                linewidth=0,
            )
        cross = ((y == 1) & (y_pred == 2)) | ((y == 2) & (y_pred == 1))
        ax.scatter(coords[cross, 0], coords[cross, 1], s=18, marker="x", color="#D62728", linewidth=0.8)
        ax.set_title(candidate.replace("_", "\n"), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Seed 42 test embeddings: class geometry and VT/VF cross-errors", fontsize=13)
    out = ASSET_DIR / "04_embedding_pca_contact_sheet_seed42.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def to_markdown_table(df: pd.DataFrame, cols: list[str], rename: dict[str, str] | None = None, max_rows: int | None = None) -> str:
    sub = df[cols].copy()
    if max_rows:
        sub = sub.head(max_rows)
    if rename:
        sub = sub.rename(columns=rename)
    return sub.to_markdown(index=False)


def write_report(summary: pd.DataFrame, absolute: pd.DataFrame, mechanism: pd.DataFrame, figures: list[Path]) -> None:
    selected = summary[summary["selected_for_full_validation"] == True]["candidate"].tolist()
    pareto = summary[summary["is_pareto"] == True]["candidate"].tolist()
    display = summary.copy()
    display = display.sort_values(["selected_for_full_validation", "total_errors_delta_mean"], ascending=[False, True])
    for col in [
        "accuracy_delta_mean",
        "macro_f1_delta_mean",
        "ece_delta_mean",
        "vtvf_cross_errors_delta_mean",
        "total_errors_delta_mean",
        "error_migration_penalty_delta_mean",
    ]:
        display[col] = display[col].map(lambda x: fmt(float(x), 4 if "delta" in col and "errors" not in col and "penalty" not in col else 1))

    abs_display = absolute[absolute["candidate"].isin(KEY_CANDIDATES)].copy()
    abs_display["candidate"] = pd.Categorical(abs_display["candidate"], categories=KEY_CANDIDATES, ordered=True)
    abs_display = abs_display.sort_values("candidate")
    for col in abs_display.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            abs_display[col] = abs_display[col].map(lambda x: fmt(float(x), 4))

    rel_figs = [path.relative_to(REPORT_PATH.parent).as_posix() for path in figures if path is not None]
    abs_metric_table = to_markdown_table(
        abs_display,
        [
            "candidate",
            "accuracy_mean",
            "macro_f1_mean",
            "ece_mean",
            "vtvf_cross_errors_mean",
            "total_errors_mean",
            "error_migration_penalty_mean",
        ],
        {
            "candidate": "模型",
            "accuracy_mean": "Acc mean",
            "macro_f1_mean": "Macro-F1 mean",
            "ece_mean": "ECE mean",
            "vtvf_cross_errors_mean": "VT/VF互错 mean",
            "total_errors_mean": "总错误 mean",
            "error_migration_penalty_mean": "迁移惩罚 mean",
        },
    )
    summary_table = to_markdown_table(
        display,
        [
            "candidate",
            "accuracy_delta_mean",
            "macro_f1_delta_mean",
            "ece_delta_mean",
            "vtvf_cross_errors_delta_mean",
            "total_errors_delta_mean",
            "error_migration_penalty_delta_mean",
            "is_pareto",
            "passes_basic_guard",
            "selected_for_full_validation",
        ],
        {
            "candidate": "候选模型",
            "accuracy_delta_mean": "Acc Δ",
            "macro_f1_delta_mean": "Macro-F1 Δ",
            "ece_delta_mean": "ECE Δ",
            "vtvf_cross_errors_delta_mean": "VT/VF互错 Δ",
            "total_errors_delta_mean": "总错误 Δ",
            "error_migration_penalty_delta_mean": "迁移惩罚 Δ",
            "is_pareto": "Pareto",
            "passes_basic_guard": "Guard",
            "selected_for_full_validation": "Full validation",
        },
    )
    mechanism_table = mechanism.to_markdown(index=False)
    traditional_table = ""
    traditional_path = ROOT / "results" / "cnn_tcn_validity_20260626" / "summary" / "combined_model_metric_summary.csv"
    if traditional_path.exists():
        traditional = pd.read_csv(traditional_path)
        metrics = ["accuracy", "macro_f1", "ece", "vtvf_cross_errors", "total_errors", "embedding_silhouette"]
        sub = traditional[
            traditional["model"].isin(["CNN", "CNN-LSTM"]) & traditional["metric"].isin(metrics)
        ][["model", "metric", "n", "mean", "std"]]
        pivot = sub.pivot_table(index="model", columns="metric", values="mean", aggfunc="first").reset_index()
        for col in pivot.columns:
            if col != "model":
                pivot[col] = pivot[col].map(lambda x: fmt(float(x), 4))
        traditional_table = pivot.to_markdown(index=False)

    text = f"""# Final Model Selection Report CN

## 一句话结论

本轮 `mechanism-derived` 搜索完成了 12 个候选配置、3 个 seed、共 36 个训练运行。结果显示：最终模型不应再理解为“把所有分析指标都塞进网络”，而应理解为“用大量机制分析生成候选干预，再用 outcome guard 筛出最小充分约束”。目前最有论文主模型价值的是 `proto_center_only`，因为它用最少的约束项在 3/3 paired seeds 上同时改善 accuracy、macro-F1、ECE、VT/VF cross-errors、total errors 和 error migration penalty。`proto_center_margin` 可作为第二候选或机制组合对照；原四项 `boundary075_prototype` 仍是重要历史模型和桥梁模型，但本轮结果提示它不是最小充分配置。

## 证据范围

- 结果目录：`results/mechanism_derived_model_search_3seed_20260701`
- 运行规模：12 candidates x 3 seeds = 36 runs
- 配对方式：每个候选与同 seed baseline 配对比较
- baseline：`reliability_gated_fusion` without extra model-layer constraint
- 核心 outcomes：accuracy、macro-F1、ECE、VT/VF cross-errors、total errors、error migration penalty

## 防止“拿自己的钥匙开自己的锁”

本项目确实做过这一层防护，但需要在论文里明确写出来。核心原则是：机制分析用于生成候选约束，最终模型选择不能只靠同一套机制指标自证成功。

具体做法有四点：

1. 数据划分层面：使用 record-level / duplicate-family split audit，避免同一来源记录或近重复窗口跨 train/test 造成泄漏。公开证据见 `results_public/tables/duplicate_family_*` 和 `dataset_split_statistics.csv`。
2. 机制发现层面：embedding、KNN local purity、prototype ambiguity、softmax entropy、waveform regularity 等只作为候选机制来源，不单独作为最终成功标准。
3. 干预验证层面：每个候选模型与同 seed baseline 做 paired comparison，只改变训练约束权重，观察 outcome delta。
4. 选择层面：最终选择依据是 outcome guard，包括 accuracy、macro-F1、ECE、VT/VF cross-errors、total errors 和 error migration penalty，而不是某一张 embedding 图或某一个机制变量变好。

因此，`proto_center_only` 被推荐为主候选，并不是因为它让 prototype/embedding 指标看起来更好，而是因为它在 matched-seed outcome guard 上同时改善六个目标。表征可视化仍然有用，但角色是解释机制，不是替代 outcome validation。

## 传统模型对照在论文中的位置

这份报告的核心是最终机制约束模型选择；传统模型负责论文前段的问题定义。CNN/CNN-LSTM 的 10-seed 证据说明：CNN-LSTM 能降低部分 VT/VF 互错并改善整体 embedding silhouette，但同时 accuracy、ECE 和 total errors 并不全面占优。因此传统模型不是最终答案，而是引出“VT/VF boundary confusion 需要机制解释和 outcome guard”的起点。

{traditional_table}

## 因果式定量分析如何表述

这里可以说“因果式机制验证”或“model-layer causal-style intervention analysis”，但不建议写成严格临床因果推断。更准确的表述是：

> We treated model constraints as intervenable variables and evaluated paired outcome changes under do(weight = value)-style interventions across matched seeds.

中文论文里可写为：

> 本文将训练约束权重视为可干预变量，在相同 seed / split 下比较不同权重组合对模型 outcome 的配对影响。该设计不能替代外部临床因果验证，但可以定量回答每个机制约束是否真正改善模型可靠性。

这正好回应“四个权重”的问题：`boundary075_prototype` 的四个项不是凭经验保留，而是可以被拆解为 boundary weighting、prototype center compactness、prototype margin 和 VT/VF margin target 等干预单元，再通过 outcome guard 判断哪些单元真正必要。

## 候选模型结果

{summary_table}

## 候选模型绝对指标审计

{abs_metric_table}

## 机制到权重的桥梁

{mechanism_table}

## 主模型建议

### 推荐主模型：`proto_center_only`

选择理由：

1. 它是最小充分机制：只引入 `prototype_center_weight=0.02`，但六个 outcome 全部 3/3 seeds 同向改善。
2. 它直接对应前期表征分析发现的核心问题：VT/VF 混淆不仅是边界样本难，而是 embedding 空间中同类样本不够紧凑、局部邻域混杂。
3. 它比 `boundary075` 更能解决 VT/VF 互错：`boundary075` 的 VT/VF cross-errors 平均只减少 2.7，而 `proto_center_only` 减少 20.7。
4. 它比原四项 `boundary075_prototype` 更适合作为博士申请叙事中的“机制筛选结果”：复杂模型有效，但经过拆解后发现 center compactness 是更核心、更小、更可解释的干预。

### 第二候选：`proto_center_margin`

`proto_center_margin` 同样通过 Pareto 和 guard，并且 ECE 改善更强。它适合作为“表征紧凑 + VT/VF 原型间隔”的组合模型，对照 `proto_center_only` 说明 margin 是否提供额外价值。

### 不建议作为主模型：`boundary075_prototype`

原四项模型仍然有价值：它证明 boundary + prototype 方向整体可行，并构成从旧版本到新版本的桥梁。但本轮 36-run 搜索中它没有进入 Pareto selected set，说明它不是最小充分模型。论文里应把它定位为“机制组合历史模型/桥梁模型”，而不是最终主张。

### 明确反例

`boundary075_contrastive` 和 `boundary075_center_calibrated` 很重要，因为它们证明“看起来合理的机制不一定可加入最终模型”。前者在整体指标上有一定改善，但 VT/VF cross-errors 平均增加 14.3，未通过 safety guard。这是论文中解释为什么需要 outcome-guarded multi-objective selection 的关键反例。

## 视觉证据

"""
    for fig in rel_figs:
        text += f"![{Path(fig).stem}]({fig})\n\n"

    text += """## 下一轮 full validation 建议

建议进入下一轮重点验证的不是所有模型，而是以下分层组合：

1. `proto_center_only`：主模型，验证最小充分机制是否稳定。
2. `proto_center_margin`：机制组合对照，验证 margin 是否提供额外收益。
3. `boundary075_prototype`：历史四项桥梁模型，保留作为旧模型对照。
4. `boundary075`：简单 boundary-risk 对照，说明单纯边界加权不够。
5. CNN / CNN-LSTM / GatedFusion baseline：放在论文主线前段，说明传统结构和无约束融合模型的局限。

`boundary100_center` 和 `boundary075_margin` 可以作为 supplementary sensitivity，不建议放在主线中心，因为它们虽然被 Pareto 选中，但解释性不如 `proto_center_only`，且 seed-level 稳定性或机制简洁性较弱。

## 限制

- 当前是 3-seed 内部验证，不是外部队列临床验证。
- 所有结论应表述为模型可靠性研究证据，不应表述为临床诊断性能声明。
- Pareto 选择依赖当前 outcome 定义；若未来加入外部数据、病人级 split 或更严格 review budget，最终模型仍需复核。
- 表征可视化是机制解释证据，不是单独成功标准；最终判断仍以 paired outcome guard 为准。
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    summary, paired, run = read_tables()
    absolute = build_absolute_table(run)
    mechanism = build_mechanism_bridge()

    absolute.to_csv(ASSET_DIR / "model_selection_absolute_metrics.csv", index=False)
    summary.to_csv(ASSET_DIR / "model_selection_delta_summary.csv", index=False)
    paired.to_csv(ASSET_DIR / "model_selection_paired_effects.csv", index=False)
    mechanism.to_csv(ASSET_DIR / "mechanism_weight_bridge.csv", index=False)

    figures = [
        save_outcome_bar(summary),
        save_pareto_plot(summary),
        save_seed_heatmap(summary),
        save_embedding_contact_sheet(run),
    ]
    write_report(summary, absolute, mechanism, figures)

    manifest = {
        "result_dir": str(RESULT_DIR.relative_to(ROOT)),
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "assets": [str(path.relative_to(ROOT)) for path in ASSET_DIR.iterdir() if path.is_file()],
    }
    (ASSET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
