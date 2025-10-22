"""
Plot cost reduction (%) of OctoSelector over single-LLM baselines.

Inputs
- compare_single_model_cost_saving.csv   # flat CSV with a 'benchmark' column (Spider/BIRD/IMDb)
- Model_comparison_batch_{benchmark}_{family}_ns{n_s}.csv  # per-family summary files

Output
- cost_saving_percentage_single_llm_feasible_region.png
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# Config (relative paths)
# -----------------------------
FILE_SINGLE_COMPARE = Path("compare_single_model_cost_saving.csv")
OUT_FIG = Path("cost_saving_percentage_single_llm_feasible_region.png")

# Per-benchmark default n_s used in file naming
NS_BY_BENCH = {"spider": 8, "BIRD": 6, "IMDb": None}  # None → filename without _ns suffix

# Models to exclude from the figure
EXCLUDE_MODELS = {"gpt-4o-mini", "gemini-1.5-flash-8b", "claude-3-5-haiku-20241022"}

# Desired order left→right (keep only those present)
DESIRED_ORDER = [
    "gpt-3.5-turbo-0125",
    "gpt-4o",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "claude-3-5-sonnet-20241022",
]


# -----------------------------
# Helpers
# -----------------------------
def shorten(name: str) -> str:
    """Compact display names for x-tick labels."""
    name = re.sub(r"^gpt-3\.5-turbo.*$", "gpt-3.5", name)
    name = re.sub(r"^gpt-4o.*$", "gpt-4o", name)
    name = re.sub(r"^gemini-1\.5-flash.*$", "gemini-flash", name)
    name = re.sub(r"^gemini-1\.5-pro.*$", "gemini-pro", name)
    name = re.sub(r"^claude-.*-sonnet.*$", "claude-sonnet", name)
    return name


def _family_of(model_name: str) -> str:
    """Infer model family token used in summary filenames."""
    return model_name.split("-", 1)[0]


def _summary_csv_name(benchmark: str, family: str) -> Path:
    """Build per-family summary CSV path."""
    ns = NS_BY_BENCH.get(benchmark, None)
    if ns is None:
        fname = f"Model_comparison_batch_{benchmark}_{family}.csv"
    else:
        fname = f"Model_comparison_batch_{benchmark}_{family}_ns{ns}.csv"
    return Path(fname)


def get_price_per_model(benchmark: str, model_name: str) -> tuple[float, float]:
    """Read baseline & OctoSelector price for a given benchmark+model from its family summary CSV."""
    family = _family_of(model_name)
    csv_file = _summary_csv_name(benchmark, family)
    df = pd.read_csv(csv_file, index_col=0)

    base = pd.to_numeric(df.loc["Baseline price (USD)", model_name], errors="coerce")
    octo = pd.to_numeric(df.loc["OctoSelector price (USD)", model_name], errors="coerce")
    return float(base), float(octo)


# -----------------------------
# Main plotting routine
# -----------------------------
def main():
    df = pd.read_csv(FILE_SINGLE_COMPARE)

    # Drop excluded models’ columns (both *_baseline and *_octoSel)
    for m in EXCLUDE_MODELS:
        cols = [c for c in df.columns if c.startswith(m + "_")]
        df = df.drop(columns=cols, errors="ignore")

    # Keep models that have both baseline and octo columns
    model_order: List[str] = [
        m for m in DESIRED_ORDER if (m + "_baseline" in df.columns and m + "_octoSel" in df.columns)
    ]
    labels_short = [shorten(m) for m in model_order]

    # Compute reductions per benchmark (override with precise per-family summary CSVs)
    reductions_per_bench: Dict[str, List[float | None]] = {}
    for _, row in df.iterrows():
        bench = str(row["benchmark"])
        vals: List[float | None] = []
        for m in model_order:
            base, octo = get_price_per_model(bench, m)
            vals.append(None if base == 0 or np.isnan(base) else (base - octo) / base * 100.0)
        reductions_per_bench[bench] = vals

    # Plot: one row per benchmark, shared x
    nrows = len(reductions_per_bench)
    fig, axes = plt.subplots(nrows, 1, figsize=(5, 6), sharex=True)
    if nrows == 1:
        axes = [axes]

    bar_width = 0.55

    for ax, (bench, vals) in zip(axes, reductions_per_bench.items()):
        x = np.arange(len(model_order))
        ax.bar(x, vals, width=bar_width, edgecolor="gray", alpha=0.85)
        title = "Spider" if bench.lower() == "spider" else bench
        ax.set_title(title, fontsize=10, loc="center")

        valid = [v for v in vals if v is not None]
        ymax = max(valid) if valid else 1.0
        ax.set_ylim(0, ymax * 1.25)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for xi, v in enumerate(vals):
            if v is not None:
                ax.text(xi, v + ymax * 0.03, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    # Bottom x labels
    axes[-1].set_xticks(list(range(len(model_order))))
    axes[-1].set_xticklabels(labels_short, rotation=20, ha="right")

    # Y label on top subplot
    axes[0].set_ylabel("Cost Reduction (%)")

    fig.suptitle("Cost Reduction (%) on Each Feasible Region of Single-LLM Baselines", fontsize=10)
    plt.tight_layout()

    fig.savefig(
        OUT_FIG,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )


if __name__ == "__main__":
    main()