from __future__ import annotations

import json
import math
import shutil
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PUBLIC_FIGURES = Path("results_public/figures")
ROUTING_DIR = Path("results/evidence_informed_mechanism_routing_10seed_v4_20260627")
V5D_DIR = ROUTING_DIR / "hierarchical_router_v5d_reserved"
EXPLANATION_DIR = ROUTING_DIR / "explanation_reliability_audit"
SSL_DIR = Path("results/frozen_ssl_encoder_comparison_20260627")


GROUPS = {
    "12_v5d_hierarchical_router": {
        "title": "V5d Hierarchical Router",
        "purpose": "Final mechanism-separated hierarchical routing policy with reserved residual budget.",
    },
    "13_frozen_ssl_encoder": {
        "title": "Frozen Self-Supervised Encoder",
        "purpose": "Lightweight frozen self-supervised encoder baseline for foundation-model readiness.",
    },
    "14_explanation_reliability": {
        "title": "Explanation Reliability",
        "purpose": "Quantitative evidence that explanation families align with intended error mechanisms.",
    },
}


def _ensure_group(group: str) -> Path:
    group_dir = PUBLIC_FIGURES / group
    individual = group_dir / "individual_figures"
    if group_dir.exists():
        shutil.rmtree(group_dir)
    individual.mkdir(parents=True, exist_ok=True)
    return group_dir


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _percent_axis(ax) -> None:
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def _line_with_std(ax, df: pd.DataFrame, x: str, y: str, label: str, color: str, marker: str = "o") -> None:
    std = f"{y}_std"
    mean = f"{y}_mean"
    xs = df[x].to_numpy(float)
    ys = df[mean].to_numpy(float)
    ax.plot(xs, ys, marker=marker, linewidth=2.2, label=label, color=color)
    if std in df.columns:
        err = df[std].fillna(0).to_numpy(float)
        ax.fill_between(xs, ys - err, ys + err, color=color, alpha=0.12, linewidth=0)


def _find_rows(df: pd.DataFrame, method_contains: str | None = None, reserve: float | None = None) -> pd.DataFrame:
    out = df.copy()
    if method_contains is not None:
        out = out[out["method"].astype(str).str.contains(method_contains, case=False, regex=False)]
    if reserve is not None and "reserve_fraction" in out.columns:
        out = out[np.isclose(out["reserve_fraction"].astype(float), reserve)]
    return out.sort_values("budget")


def _v5d_public_summary() -> pd.DataFrame:
    summary = pd.read_csv(V5D_DIR / "v5d_reserved_policy_mean_std.csv")
    seed_level = pd.read_csv(V5D_DIR / "all_seed_v5d_reserved_policy_summary.csv")
    baseline = seed_level[
        seed_level["method"].astype(str).isin(["optimized_mechanism_router_v4", "boundary_first_v5b"])
    ].copy()
    if baseline.empty:
        return summary
    metric_cols = [
        "action_rate",
        "stage1_boundary_rate",
        "stage2_residual_rate",
        "all_error_addressed",
        "vtvf_cross_error_addressed",
        "automatic_unresolved_error_rate",
        "automatic_unresolved_vtvf_cross_error_rate",
        "single_label_error_rate_after_routing",
        "single_label_vtvf_cross_error_rate_after_routing",
    ]
    rows = []
    for (method, policy_family, budget), sub in baseline.groupby(["method", "policy_family", "budget"], sort=True):
        row = {
            "method": method,
            "policy_family": policy_family,
            "reserve_fraction": np.nan,
            "budget": float(budget),
            "n_seeds": int(sub["seed"].nunique()),
        }
        for col in metric_cols:
            if col in sub.columns:
                row[f"{col}_mean"] = float(sub[col].mean())
                row[f"{col}_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else np.nan
        rows.append(row)
    return pd.concat([summary, pd.DataFrame(rows)], ignore_index=True, sort=False)


def _draw_method_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    boxes = {
        "ECG window": (0.08, 0.80, 0.18, 0.10),
        "Base classifier\nSR / VT / VF": (0.36, 0.80, 0.22, 0.10),
        "Evidence layer\nRISK, softmax, validity,\nwavelet, representation": (0.70, 0.78, 0.24, 0.14),
        "Stage 1\nVT/VF boundary-first": (0.18, 0.48, 0.26, 0.13),
        "Stage 2\nResidual mechanism router": (0.58, 0.48, 0.28, 0.13),
        "{VT,VF} set\nor boundary review": (0.12, 0.18, 0.28, 0.12),
        "Mechanism-specific review\nor automatic label": (0.58, 0.18, 0.30, 0.12),
    }
    colors = {
        "ECG window": "#eef3f7",
        "Base classifier\nSR / VT / VF": "#e7f0fb",
        "Evidence layer\nRISK, softmax, validity,\nwavelet, representation": "#edf7ef",
        "Stage 1\nVT/VF boundary-first": "#fff0df",
        "Stage 2\nResidual mechanism router": "#f2ebfa",
        "{VT,VF} set\nor boundary review": "#ffe9e5",
        "Mechanism-specific review\nor automatic label": "#eef3f7",
    }
    for text, (x, y, w, h) in boxes.items():
        rect = plt.Rectangle((x, y), w, h, facecolor=colors[text], edgecolor="#333333", linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, weight="bold")

    arrows = [
        ("ECG window", "Base classifier\nSR / VT / VF"),
        ("Base classifier\nSR / VT / VF", "Evidence layer\nRISK, softmax, validity,\nwavelet, representation"),
        ("Evidence layer\nRISK, softmax, validity,\nwavelet, representation", "Stage 1\nVT/VF boundary-first"),
        ("Evidence layer\nRISK, softmax, validity,\nwavelet, representation", "Stage 2\nResidual mechanism router"),
        ("Stage 1\nVT/VF boundary-first", "{VT,VF} set\nor boundary review"),
        ("Stage 2\nResidual mechanism router", "Mechanism-specific review\nor automatic label"),
        ("Stage 1\nVT/VF boundary-first", "Stage 2\nResidual mechanism router"),
    ]

    def center(name: str) -> tuple[float, float]:
        x, y, w, h = boxes[name]
        return x + w / 2, y + h / 2

    for src, dst in arrows:
        sx, sy = center(src)
        dx, dy = center(dst)
        ax.annotate(
            "",
            xy=(dx, dy),
            xytext=(sx, sy),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5, shrinkA=28, shrinkB=28),
        )
    ax.text(
        0.5,
        0.04,
        "Final framing: RISK and related scores are evidence; v5d is the decision policy.",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
    )
    _savefig(path)


def make_v5d_figures(group_dir: Path) -> list[Path]:
    out = group_dir / "individual_figures"
    paths: list[Path] = []
    df = _v5d_public_summary()
    v4 = _find_rows(df, method_contains="optimized")
    if v4.empty:
        v4 = _find_rows(df, method_contains="v4")
    r10 = _find_rows(df, method_contains="v5d", reserve=0.10)
    r20 = _find_rows(df, method_contains="v5d", reserve=0.20)
    r30 = _find_rows(df, method_contains="v5d", reserve=0.30)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, metric, title in [
        (axes[0], "all_error_addressed", "All-error capture"),
        (axes[1], "vtvf_cross_error_addressed", "VT/VF cross-error capture"),
    ]:
        if not v4.empty:
            _line_with_std(ax, v4, "budget", metric, "v4 optimized router", "#4b5563", "s")
        for data, label, color in [
            (r10, "v5d reserve 10%", "#0f766e"),
            (r20, "v5d reserve 20%", "#2563eb"),
            (r30, "v5d reserve 30%", "#9333ea"),
        ]:
            if not data.empty:
                _line_with_std(ax, data, "budget", metric, label, color)
        ax.set_title(title)
        ax.set_xlabel("Action budget")
        ax.set_ylabel("Captured errors")
        ax.grid(True, alpha=0.25)
        ax.set_xticks([0.05, 0.10, 0.20, 0.30])
        ax.set_xticklabels(["5%", "10%", "20%", "30%"])
        _percent_axis(ax)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("v5d reserved residual budget improves VT/VF capture while preserving all-error capture", fontsize=13)
    path = out / "fig_085_v5d_capture_vs_budget.png"
    _savefig(path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for data, label, color in [(v4, "v4 optimized router", "#4b5563"), (r20, "v5d reserve 20%", "#2563eb")]:
        if not data.empty:
            _line_with_std(
                ax,
                data,
                "budget",
                "automatic_unresolved_vtvf_cross_error_rate",
                label,
                color,
            )
    ax.set_title("Automatic-route residual VT/VF cross-error rate")
    ax.set_xlabel("Action budget")
    ax.set_ylabel("Residual VT/VF error rate")
    ax.set_xticks([0.05, 0.10, 0.20, 0.30])
    ax.set_xticklabels(["5%", "10%", "20%", "30%"])
    ax.grid(True, alpha=0.25)
    _percent_axis(ax)
    ax.legend()
    path = out / "fig_086_v5d_unresolved_vtvf_rate.png"
    _savefig(path)
    paths.append(path)

    subset = df[df["method"].astype(str).str.contains("v5d", case=False, regex=False)]
    subset = subset[np.isclose(subset["budget"].astype(float), 0.20)]
    subset = subset.sort_values("reserve_fraction")
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{int(r * 100)}%" for r in subset["reserve_fraction"].to_numpy(float)]
    x = np.arange(len(labels))
    stage1 = subset["stage1_boundary_rate_mean"].to_numpy(float)
    stage2 = subset["stage2_residual_rate_mean"].to_numpy(float)
    ax.bar(x, stage1, label="Stage 1 boundary-first", color="#f59e0b")
    ax.bar(x, stage2, bottom=stage1, label="Stage 2 residual", color="#7c3aed")
    ax.set_title("v5d action allocation at 20% action budget")
    ax.set_xlabel("Reserved residual budget fraction")
    ax.set_ylabel("Action rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    _percent_axis(ax)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = out / "fig_087_v5d_stage_allocation.png"
    _savefig(path)
    paths.append(path)

    path = out / "fig_088_v5d_method_diagram.png"
    _draw_method_diagram(path)
    paths.append(path)
    return paths


def make_ssl_figures(group_dir: Path) -> list[Path]:
    out = group_dir / "individual_figures"
    paths: list[Path] = []
    metrics = pd.read_csv(SSL_DIR / "frozen_ssl_metrics_mean_std.csv").iloc[0]
    metric_names = ["accuracy", "macro_f1", "ece", "any_error_auroc", "vtvf_error_auroc"]
    labels = ["Accuracy", "Macro-F1", "ECE", "Any-error\nAUROC", "VT/VF-error\nAUROC"]
    vals = [metrics[f"{name}_mean"] for name in metric_names]
    errs = [metrics.get(f"{name}_std", 0.0) for name in metric_names]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, yerr=errs, capsize=4, color=["#60a5fa", "#60a5fa", "#fbbf24", "#34d399", "#34d399"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Frozen self-supervised encoder: classifier vs risk-ranking behavior")
    ax.set_ylabel("Metric value")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.025, f"{val:.3f}", ha="center", fontsize=9)
    path = out / "fig_089_ssl_metrics_summary.png"
    _savefig(path)
    paths.append(path)

    policy = pd.read_csv(SSL_DIR / "frozen_ssl_review_policy_mean_std.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    colors = {"ssl_any_error_risk": "#0891b2", "ssl_vtvf_boundary_risk": "#dc2626"}
    for method, sub in policy.groupby("method"):
        sub = sub.sort_values("budget")
        _line_with_std(axes[0], sub, "budget", "all_error_addressed", method, colors.get(method, "#333333"))
        _line_with_std(axes[1], sub, "budget", "vtvf_cross_error_addressed", method, colors.get(method, "#333333"))
    for ax, title in zip(axes, ["All-error capture", "VT/VF cross-error capture"]):
        ax.set_title(title)
        ax.set_xlabel("Action budget")
        ax.set_ylabel("Captured errors")
        ax.set_xticks([0.10, 0.20, 0.30])
        ax.set_xticklabels(["10%", "20%", "30%"])
        ax.grid(True, alpha=0.25)
        _percent_axis(ax)
    axes[0].legend(fontsize=8)
    fig.suptitle("Frozen SSL risk heads: strong ranking despite weaker classifier", fontsize=13)
    path = out / "fig_090_ssl_review_capture_vs_budget.png"
    _savefig(path)
    paths.append(path)

    v5d = _v5d_public_summary()
    v5d20 = _find_rows(v5d, method_contains="v5d", reserve=0.20)
    v5d20 = v5d20[np.isclose(v5d20["budget"].astype(float), 0.20)]
    ssl20 = policy[np.isclose(policy["budget"].astype(float), 0.20)]
    rows = []
    if not v5d20.empty:
        rows.append(
            {
                "method": "v5d reserve 20%",
                "All-error": float(v5d20["all_error_addressed_mean"].iloc[0]),
                "VT/VF": float(v5d20["vtvf_cross_error_addressed_mean"].iloc[0]),
            }
        )
    for _, row in ssl20.iterrows():
        rows.append(
            {
                "method": row["method"].replace("ssl_", "SSL ").replace("_", " "),
                "All-error": row["all_error_addressed_mean"],
                "VT/VF": row["vtvf_cross_error_addressed_mean"],
            }
        )
    comp = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(comp))
    width = 0.36
    ax.bar(x - width / 2, comp["All-error"], width, label="All-error capture", color="#2563eb")
    ax.bar(x + width / 2, comp["VT/VF"], width, label="VT/VF capture", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(comp["method"], rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_title("20% action budget: v5d policy vs frozen SSL risk heads")
    ax.set_ylabel("Captured errors")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = out / "fig_091_ssl_vs_v5d_20pct.png"
    _savefig(path)
    paths.append(path)
    return paths


def make_explanation_figures(group_dir: Path) -> list[Path]:
    out = group_dir / "individual_figures"
    paths: list[Path] = []
    df = pd.read_csv(EXPLANATION_DIR / "explanation_alignment_mean_std.csv")
    evidence_order = [
        "boundary_explanation",
        "representation_explanation",
        "second_opinion_explanation",
        "sr_ventricular_explanation",
        "regularity_atypicality_explanation",
        "hidden_confidence_explanation",
    ]
    target_order = [
        "vtvf_cross_error",
        "representation_conflict_error",
        "any_error",
        "sr_ventricular_error",
        "atypical_signal_error",
        "hidden_confident_error",
    ]
    matrix = df.pivot(index="evidence_family", columns="target_error_type", values="auroc_mean")
    matrix = matrix.reindex(index=evidence_order, columns=target_order)
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(matrix.to_numpy(float), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(target_order)))
    ax.set_xticklabels([t.replace("_", "\n") for t in target_order], fontsize=8)
    ax.set_yticks(np.arange(len(evidence_order)))
    ax.set_yticklabels([e.replace("_", " ") for e in evidence_order], fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.65 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="AUROC")
    ax.set_title("Explanation reliability: evidence family vs target error type")
    path = out / "fig_092_explanation_alignment_heatmap.png"
    _savefig(path)
    paths.append(path)

    preferred = [
        ("boundary_explanation", "vtvf_cross_error", "Boundary -> VT/VF"),
        ("representation_explanation", "representation_conflict_error", "Representation -> conflict"),
        ("second_opinion_explanation", "any_error", "Second opinion -> any error"),
        ("sr_ventricular_explanation", "sr_ventricular_error", "SR-ventricular"),
        ("regularity_atypicality_explanation", "atypical_signal_error", "Regularity -> atypical"),
        ("hidden_confidence_explanation", "hidden_confident_error", "Hidden confidence"),
    ]
    rows = []
    for evidence, target, label in preferred:
        row = df[(df["evidence_family"] == evidence) & (df["target_error_type"] == target)]
        if row.empty:
            continue
        row = row.iloc[0]
        rows.append({"label": label, "Top 10%": row["top10_capture_mean"], "Top 20%": row["top20_capture_mean"]})
    pref = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(pref))
    width = 0.36
    ax.bar(x - width / 2, pref["Top 10%"], width, label="Top 10% capture", color="#0f766e")
    ax.bar(x + width / 2, pref["Top 20%"], width, label="Top 20% capture", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(pref["label"], rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_title("Preferred explanation-to-mechanism alignment")
    ax.set_ylabel("Target error capture")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = out / "fig_093_preferred_explanation_capture.png"
    _savefig(path)
    paths.append(path)

    route = pd.read_csv(EXPLANATION_DIR / "v5d_route_alignment_mean_std.csv")
    focus = route[(np.isclose(route["budget"].astype(float), 0.20)) & (np.isclose(route["reserve_fraction"].astype(float), 0.20))]
    focus = focus.sort_values("mechanism_route")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(focus))
    width = 0.32
    ax.bar(x - width / 2, focus["all_error_precision_mean"], width, label="All-error precision", color="#64748b")
    ax.bar(x + width / 2, focus["route_target_precision_mean"], width, label="Route-target precision", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in focus["mechanism_route"]], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Precision among selected samples")
    ax.set_title("v5d route-level alignment at 20% budget / 20% residual reserve")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = out / "fig_094_v5d_route_alignment_precision.png"
    _savefig(path)
    paths.append(path)
    return paths


def make_contact_sheet(group_dir: Path, paths: list[Path]) -> None:
    thumbs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((640, 420))
        thumbs.append((path, img.copy()))
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    cell_w, cell_h = 700, 500
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    for idx, (path, img) in enumerate(thumbs):
        r, c = divmod(idx, cols)
        x = c * cell_w + (cell_w - img.width) // 2
        y = r * cell_h + 35
        sheet.paste(img, (x, y))
        caption = path.name
        draw.text((c * cell_w + 20, r * cell_h + 10), caption, fill="#111111", font=font)
        wrapped = textwrap.wrap(GROUPS[group_dir.name]["purpose"], width=72)
        if idx == 0:
            draw.text((c * cell_w + 20, r * cell_h + 455), "\n".join(wrapped[:2]), fill="#555555", font=small)
    sheet.save(group_dir / "contact_sheet.png")


def write_group_readme(group_dir: Path, paths: list[Path]) -> None:
    meta = GROUPS[group_dir.name]
    lines = [
        f"# {meta['title']}",
        "",
        meta["purpose"],
        "",
        "## Contact Sheet",
        "",
        "![Contact sheet](contact_sheet.png)",
        "",
        "## Included Figures",
        "",
    ]
    for idx, path in enumerate(paths, start=1):
        rel = Path("individual_figures") / path.name
        lines.append(f"{idx}. [`{path.name}`]({rel.as_posix()})")
    (group_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_public_indexes(new_entries: list[dict[str, str]]) -> None:
    readme = PUBLIC_FIGURES / "README.md"
    text = readme.read_text(encoding="utf-8")
    additions = "\n".join(
        [
        "| V5d Hierarchical Router | `12_v5d_hierarchical_router/` | 4 | Final mechanism-separated hierarchical routing policy and budget behavior. |",
        "| Frozen Self-Supervised Encoder | `13_frozen_ssl_encoder/` | 3 | Frozen encoder foundation-readiness baseline and review-routing curves. |",
        "| Explanation Reliability | `14_explanation_reliability/` | 3 | Mechanism-alignment checks for explanation evidence families. |",
        ]
    )
    if "| V5d Hierarchical Router | `12_v5d_hierarchical_router/`" not in text:
        marker = "| V6 RISK Distillation | `11_v6_risk_distillation/` | 2 | V6 evidence for deployable RISK distillation and review-routing validation. |"
        text = text.replace(marker, marker + "\n" + additions)
    readme.write_text(text, encoding="utf-8")

    atlas = Path("docs/FIGURE_ATLAS.md")
    text = atlas.read_text(encoding="utf-8")
    additions = "\n".join(
        [
        "| V5d hierarchical router | `results_public/figures/12_v5d_hierarchical_router/` | Final decision-policy evidence: v5d budget curves, residual VT/VF rate, stage allocation, and method diagram. |",
        "| Frozen self-supervised encoder | `results_public/figures/13_frozen_ssl_encoder/` | Lightweight frozen encoder baseline showing classification limits but strong risk-ranking behavior. |",
        "| Explanation reliability | `results_public/figures/14_explanation_reliability/` | Quantitative alignment between explanation evidence families and intended error mechanisms. |",
        ]
    )
    if "| V5d hierarchical router | `results_public/figures/12_v5d_hierarchical_router/`" not in text:
        marker = "| V6 RISK distillation | `results_public/figures/11_v6_risk_distillation/` | The final deployable RISK framing and review-budget evidence. |"
        text = text.replace(marker, marker + "\n" + additions)
    if "Finish with `12_v5d_hierarchical_router/`" not in text:
        text = text.replace(
            "and `11_v6_risk_distillation/`, which support the final RISK and\n   review-routing story.",
            "and `11_v6_risk_distillation/`, which support the RISK evidence layer.\n6. Finish with `12_v5d_hierarchical_router/`, then check\n   `13_frozen_ssl_encoder/` and `14_explanation_reliability/` for the new\n   final-method, frozen-encoder, and explanation-reliability evidence.",
        )
    atlas.write_text(text, encoding="utf-8")

    manifest_path = PUBLIC_FIGURES / "figure_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = []
    existing = {entry.get("file") for entry in manifest}
    for entry in new_entries:
        if entry["file"] not in existing:
            manifest.append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    PUBLIC_FIGURES.mkdir(parents=True, exist_ok=True)
    makers = {
        "12_v5d_hierarchical_router": make_v5d_figures,
        "13_frozen_ssl_encoder": make_ssl_figures,
        "14_explanation_reliability": make_explanation_figures,
    }
    manifest_entries = []
    for group, maker in makers.items():
        group_dir = _ensure_group(group)
        paths = maker(group_dir)
        make_contact_sheet(group_dir, paths)
        write_group_readme(group_dir, paths)
        for idx, path in enumerate(paths, start=1):
            manifest_entries.append(
                {
                    "file": path.as_posix(),
                    "section": GROUPS[group]["title"],
                    "subsection": "GitHub visual pack",
                    "source_image_index": idx,
                    "location": "generated_from_aggregate_csv",
                    "reason": "public-safe aggregate figure generated for GitHub release",
                    "group": group,
                    "group_title": GROUPS[group]["title"],
                }
            )
    update_public_indexes(manifest_entries)
    print("Generated GitHub visual pack:")
    for entry in manifest_entries:
        print(" -", entry["file"])


if __name__ == "__main__":
    main()
