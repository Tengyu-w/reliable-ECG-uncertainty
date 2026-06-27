# Documentation Guide

This folder contains the supervisor-facing documentation for the project. The
repository is intentionally organized as a research portfolio, not as a raw
experiment dump.

## Recommended Reading Order

| Time | File | Purpose |
| --- | --- | --- |
| 2 min | [Project README](../README.md) | Main contribution, result snapshot, and limitations. |
| 5 min | [PhD application brief](phd_application_project_brief.md) | Short supervisor-facing project pitch. |
| 10 min | [Experiment evidence summary](EXPERIMENT_EVIDENCE_SUMMARY.md) | The full research logic in one compact document. |
| Reuse | [Transferable method framework](VT_VF_TRANSFERABLE_METHOD_FRAMEWORK_CN.md) | Method template for other manuscripts or projects. |
| 15 min | [Evidence index](evidence_index.md) | Claim-by-claim pointers to public figures and tables. |
| 30 min | [Research report](RESEARCH_REPORT.md) | Detailed stage-ordered report. |

## Core Documents

- [APPLICATION_INDEX.md](APPLICATION_INDEX.md): advisor-facing entry point.
- [EXPERIMENT_EVIDENCE_SUMMARY.md](EXPERIMENT_EVIDENCE_SUMMARY.md): compact
  story of what was tested, why, what worked, and what failed.
- [VT_VF_TRANSFERABLE_METHOD_FRAMEWORK_CN.md](VT_VF_TRANSFERABLE_METHOD_FRAMEWORK_CN.md):
  reusable method framework distilled from the VT/VF project.
- [RESEARCH_REPORT.md](RESEARCH_REPORT.md): full research narrative.
- [METHOD_OVERVIEW.md](METHOD_OVERVIEW.md): method diagram and reliability
  signal families.
- [FIGURE_ATLAS.md](FIGURE_ATLAS.md): public figure inventory.
- [DATA_STATEMENT.md](DATA_STATEMENT.md): what data is excluded from GitHub.
- [EXPERIMENT_PIPELINE.md](EXPERIMENT_PIPELINE.md): reproducible code entry
  points.

## What Was Removed From The Public Main Path

Earlier drafts, group-meeting scripts, and stage-by-stage scratch notes were
removed from the main documentation path. Their useful conclusions were
merged into the evidence summary and research report. This keeps the repository
readable for a PhD supervisor while preserving the experiment code and public
aggregate evidence.

## Scope Boundary

This repository does not distribute raw ECG data, model checkpoints,
embeddings, window-level prediction files, or private review examples. It is a
research prototype for reliability analysis, not a clinical validation package.
