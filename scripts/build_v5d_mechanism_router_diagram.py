from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(
    "results_public/figures/12_v5d_hierarchical_router/individual_figures/"
    "fig_089_v5d_mechanism_evidence_router.png"
)


def _box(
    ax,
    xy,
    wh,
    title,
    body="",
    fc="#eef7f6",
    ec="#0f766e",
    lw=1.5,
    title_size=10,
    body_size=7.8,
    wrap_width=30,
):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.02,
        y + h - 0.027,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color="#0f172a",
    )
    if body:
        wrapped = "\n".join(textwrap.wrap(body, width=wrap_width))
        ax.text(
            x + 0.02,
            y + h - 0.065,
            wrapped,
            ha="left",
            va="top",
            fontsize=body_size,
            color="#334155",
            linespacing=1.15,
        )
    return patch


def _arrow(ax, start, end, color="#0f766e", lw=1.45, rad=0.0, alpha=0.9):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    ax.text(
        0.03,
        0.96,
        "Mechanism Evidence Router: from multi-source analysis to Stage 1 / Stage 2 recovery",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.03,
        0.915,
        "The analysis layer is not a decorative representation result; each evidence family feeds a specific failure mechanism and is judged by held-out routing outcomes.",
        ha="left",
        va="top",
        fontsize=10,
        color="#475569",
    )

    evidence = [
        (
            "Classifier uncertainty",
            "entropy, MSP, margin, VT/VF softmax ambiguity, logit margin",
        ),
        (
            "Representation geometry",
            "embedding neighborhood, KNN mixing, local purity, prototype distance and disagreement",
        ),
        (
            "Validity / boundary evidence",
            "validity gate, boundary score, gate x boundary, low-validity confidence",
        ),
        (
            "ECG signal structure",
            "regularity, autocorrelation, spectral entropy, wavelet VT/VF boundary risk",
        ),
        (
            "Consistency / hidden error",
            "model disagreement, high-confidence errors, corruption response, cluster stress",
        ),
    ]

    y0 = 0.765
    for i, (title, body) in enumerate(evidence):
        _box(
            ax,
            (0.04, y0 - i * 0.13),
            (0.24, 0.105),
            title,
            body,
            fc="#f0fdfa",
            ec="#0f766e",
            body_size=7.2,
            wrap_width=31,
        )

    _box(
        ax,
        (0.35, 0.36),
        (0.17, 0.22),
        "Mechanism graph\n(CGR-style)",
        "evidence becomes mechanism scores, not a single opaque risk number",
        fc="#ecfeff",
        ec="#0891b2",
        title_size=11,
        body_size=8.4,
        wrap_width=25,
    )

    mechanisms = [
        ("VT/VF boundary", ""),
        ("Validity boundary", ""),
        ("SR-ventricular", ""),
        ("Representation conflict", ""),
        ("Atypical signal", ""),
        ("Hidden confident error", ""),
    ]
    mech_pos = []
    for i, (title, body) in enumerate(mechanisms):
        y = 0.78 - i * 0.102
        mech_pos.append((0.58, y + 0.04))
        _box(
            ax,
            (0.56, y),
            (0.21, 0.072),
            title,
            body,
            fc="#eef7f6",
            ec="#0f766e",
            title_size=8.8,
            body_size=7.1,
            wrap_width=30,
        )

    _box(
        ax,
        (0.83, 0.64),
        (0.13, 0.16),
        "Stage 1",
        "boundary-first gate: VT/VF review or {VT,VF} prediction set",
        fc="#f0fdfa",
        ec="#0f766e",
        title_size=11,
        body_size=7.8,
        wrap_width=25,
    )
    _box(
        ax,
        (0.83, 0.38),
        (0.13, 0.20),
        "Stage 2",
        "reserved residual budget: mechanism-specific review before automatic output",
        fc="#ecfeff",
        ec="#0891b2",
        title_size=11,
        body_size=7.6,
        wrap_width=25,
    )
    _box(
        ax,
        (0.83, 0.16),
        (0.13, 0.14),
        "Outcome guard",
        "fixed-budget capture, residual VT/VF risk, calibration, migration",
        fc="#f8fafc",
        ec="#64748b",
        title_size=10,
        body_size=7.2,
        wrap_width=24,
    )

    for i in range(len(evidence)):
        _arrow(ax, (0.28, y0 - i * 0.13 + 0.052), (0.35, 0.47), color="#14b8a6", lw=1.2, alpha=0.75)

    for _, y in mech_pos:
        _arrow(ax, (0.52, 0.47), (0.56, y), color="#0891b2", lw=1.2, alpha=0.8)

    _arrow(ax, (0.77, 0.818), (0.83, 0.72), color="#0f766e", lw=1.7)
    _arrow(ax, (0.77, 0.716), (0.83, 0.72), color="#0f766e", lw=1.7)
    for y in [0.614, 0.512, 0.410, 0.308]:
        _arrow(ax, (0.77, y), (0.83, 0.48), color="#0891b2", lw=1.4, rad=0.02)
    _arrow(ax, (0.89, 0.64), (0.89, 0.58), color="#64748b", lw=1.2)
    _arrow(ax, (0.89, 0.38), (0.89, 0.30), color="#64748b", lw=1.2)

    ax.text(
        0.50,
        0.085,
        "Research prototype only: review-prioritization and mechanism evidence; not clinical validation or a medical-device claim.",
        ha="center",
        va="center",
        fontsize=9,
        color="#64748b",
    )
    ax.text(
        0.50,
        0.045,
        "Key distinction: model-layer constraints are selected by outcome guards; routing-layer mechanisms use analysis evidence to decide which errors require review.",
        ha="center",
        va="center",
        fontsize=9,
        color="#475569",
    )

    plt.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
