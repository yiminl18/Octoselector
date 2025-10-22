"""
Visualize feasible regions for accuracy/latency constraints and OctoSelector.

- Parses ILP results CSV into a grid.
- Builds a feasibility mask using monotonic closure (left-bottom fill from feasible points).
- Overlays per-model feasible rectangles and the OctoSelector feasible region.
- Saves a single PNG under ./figures/.

All comments are minimal and in English. No personal info or absolute paths.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, to_rgba


# -----------------------------
# Parsing and small utilities
# -----------------------------

def parse_results(filepath: str):
    """Read results CSV and coerce key columns to numeric."""
    df = pd.read_csv(filepath)
    df["acc_constraint"] = pd.to_numeric(df["acc_constraint"], errors="coerce")
    df["lat_constraint"] = pd.to_numeric(df["lat_constraint"], errors="coerce")
    df["objective_value"] = pd.to_numeric(df["objective_value"], errors="coerce")
    df["solver_status"] = df["solver_status"].astype(str)
    return df.to_dict(orient="records")


def find_closest_index(arr: np.ndarray, value: float) -> int:
    """Index of closest element in arr to value."""
    return int(np.argmin(np.abs(arr - value)))


def left_bottom_mask(point_acc, point_lat, acc_range, lat_range):
    """Mask for cells with acc <= point_acc and lat >= point_lat."""
    n_lat = len(lat_range)
    n_acc = len(acc_range)
    mask = np.zeros((n_lat, n_acc), dtype=bool)
    for j in range(n_lat):
        for i in range(n_acc):
            if acc_range[i] <= point_acc and lat_range[j] >= point_lat:
                mask[j, i] = True
    return mask


def avg_price_lower_left(point_acc, point_lat, acc_range, lat_range, heatmap_data):
    """Average price in the left-bottom region under (acc, lat)."""
    valid_acc_idx = np.where(acc_range <= point_acc)[0]
    valid_lat_idx = np.where(lat_range >= point_lat)[0]
    if valid_acc_idx.size == 0 or valid_lat_idx.size == 0:
        return np.nan
    sub = heatmap_data[np.ix_(valid_lat_idx, valid_acc_idx)]
    vals = sub[~np.isnan(sub)]
    return np.nan if vals.size == 0 else np.mean(vals)


def union_rectangles_min_price(
    acc_points, lat_points, baseline_price_points,
    heatmap_data, acc_range, lat_range
):
    """Union of model rectangles; per cell take min(heatmap, baseline) where feasible."""
    masks = [left_bottom_mask(acc_points[i], lat_points[i], acc_range, lat_range)
             for i in range(len(acc_points))]

    union_mask = np.zeros(heatmap_data.shape, dtype=bool)
    for m in masks:
        union_mask |= m

    n_lat, n_acc = heatmap_data.shape
    combined = np.full((n_lat, n_acc), np.nan)
    baseline_combined = np.full((n_lat, n_acc), np.nan)

    for j in range(n_lat):
        for i in range(n_acc):
            cand_prices, cand_bases = [], []
            for idx, m in enumerate(masks):
                if m[j, i]:
                    price_val = heatmap_data[j, i]
                    if not np.isnan(price_val):
                        cand_prices.append(price_val)
                        cand_bases.append(baseline_price_points[idx])
            if cand_prices:
                combined[j, i] = min(cand_prices)
                baseline_combined[j, i] = min(cand_bases)

    outside_mask = ~union_mask
    feasible_outside = outside_mask & ~np.isnan(heatmap_data)
    avg_price_outside = np.nanmean(heatmap_data[feasible_outside]) if np.any(feasible_outside) else np.nan
    print("Average price outside the union (feasible cells):", avg_price_outside)

    return combined, baseline_combined, avg_price_outside


# -----------------------------
# Grid construction
# -----------------------------

def create_heatmap_data(results, acc_range, lat_range):
    """
    Build a dense grid from sparse results:
      1) snap points to grid
      2) fill left-bottom from feasible points
      3) optionally prune right-upper from infeasible points (feasible has priority)
    """
    acc_range = np.asarray(acc_range)
    lat_range = np.asarray(lat_range)

    n_acc = len(acc_range)
    n_lat = len(lat_range)
    heatmap = np.full((n_lat, n_acc), np.nan, dtype=float)

    feasible_idx = []
    infeasible_idx = []

    # Snap and collect feasibility
    for res in results:
        if "acc_constraint" not in res or "lat_constraint" not in res:
            continue
        a = res.get("acc_constraint", np.nan)
        l = res.get("lat_constraint", np.nan)
        if np.isnan(a) or np.isnan(l):
            continue

        i = find_closest_index(acc_range, a)
        j = find_closest_index(lat_range, l)

        status = str(res.get("solver_status", "")).lower()
        if status in ("infeasible", "undefined"):
            infeasible_idx.append((i, j))
        else:
            val = res.get("objective_value", np.nan)
            heatmap[j, i] = val if not np.isnan(val) else 1.0
            feasible_idx.append((i, j))

    # Left-bottom fill from feasible points
    for i_f, j_f in feasible_idx:
        block = heatmap[j_f:, :i_f + 1]
        block = np.where(np.isnan(block), 1.0, block)
        heatmap[j_f:, :i_f + 1] = block

    # Right-upper prune from infeasible points (no override of known feasible)
    for i_i, j_i in infeasible_idx:
        region = heatmap[:j_i + 1, i_i:]
        # keep existing feasible values; ensure unknowns remain NaN
        region[:] = region  # explicit no-op to emphasize region selection
        heatmap[:j_i + 1, i_i:] = region

    return heatmap


# -----------------------------
# Constraint presets
# -----------------------------

def return_constraint_range(benchmark: str, model_family: str):
    if benchmark == "spider":
        if model_family == "gpt":
            acc_constraint_range = np.linspace(0.72, 0.77, 10)
            lat_constraint_range = np.linspace(0.32, 0.58, 10)
            acc_points = [0.74951, 0.74130, 0.76519]
            lat_points = [0.54864, 0.3894205, 0.48524]
            price_points = [0.02379, 0.325455, 0.4342775]
            model_list = ["gpt-4o-mini", "gpt-3.5-turbo-0125", "gpt-4o"]
        elif model_family == "gemini":
            acc_constraint_range = np.linspace(0.75, 0.82, 10)
            lat_constraint_range = np.linspace(0.18, 0.5, 10)
            acc_points = [0.77817, 0.78433, 0.81479]
            lat_points = [0.24809, 0.26169, 0.481259]
            price_points = [0.006573, 0.011709, 0.195791]
            model_list = ["gemini-1.5-flash-8b", "gemini-1.5-flash", "gemini-1.5-pro"]
        elif model_family == "claude":
            acc_constraint_range = np.linspace(0.55, 0.74, 10)
            lat_constraint_range = np.linspace(0.5, 0.7, 10)
            acc_points = [0.6609, 0.70021]
            lat_points = [0.58283, 0.54633]
            price_points = [0.15614, 0.597462]
            model_list = ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"]
        # union of feasibility bounds
        acc_constraint_range = np.linspace(0.6609, 0.815, 20)
        lat_constraint_range = np.linspace(0.24809, 0.58283, 20)

    elif benchmark == "BIRD":
        if model_family == "gpt":
            acc_constraint_range = np.linspace(0.35, 0.49, 10)
            lat_constraint_range = np.linspace(0.41, 0.89, 10)
            acc_points = [0.42335, 0.39854, 0.47631]
            lat_points = [0.87199, 0.47505, 0.74221]
            price_points = [0.06238, 0.79425, 1.04768]
            model_list = ["gpt-4o-mini", "gpt-3.5-turbo-0125", "gpt-4o"]
        elif model_family == "gemini":
            acc_constraint_range = np.linspace(0.35, 0.45, 10)
            lat_constraint_range = np.linspace(0.30, 0.81, 10)
            acc_points = [0.396, 0.41672, 0.43896]
            lat_points = [0.45733, 0.50185, 0.75791]
            price_points = [0.01744, 0.03032, 0.50509]
            model_list = ["gemini-1.5-flash-8b", "gemini-1.5-flash", "gemini-1.5-pro"]
        elif model_family == "claude":
            acc_constraint_range = np.linspace(0.40, 0.485, 10)
            lat_constraint_range = np.linspace(0.83, 1.5, 10)
            acc_points = [0.43467, 0.46831]
            lat_points = [0.97358, 0.90164]
            price_points = [0.41542, 1.40628]
            model_list = ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"]
        acc_constraint_range = np.linspace(0.39854, 0.485, 20)
        lat_constraint_range = np.linspace(0.45733, 0.97358, 20)

    elif benchmark == "IMDb":
        if model_family == "gpt":
            acc_constraint_range = np.linspace(1.80, 2.1, 10)
            lat_constraint_range = np.linspace(0.42, 0.58, 10)
            acc_points = [2.0193, 2.0448, 1.8631]
            lat_points = [0.5260, 0.4637, 0.5728]
            price_points = [0.0876, 1.7456, 1.4597]
            model_list = ["gpt-4o-mini", "gpt-3.5-turbo-0125", "gpt-4o"]
        elif model_family == "gemini":
            acc_constraint_range = np.linspace(2.15, 3.6, 10)
            lat_constraint_range = np.linspace(0.34, 0.9, 10)
            acc_points = [3.5070, 2.8149, 2.1766]
            lat_points = [0.4596, 0.4068, 0.8851]
            price_points = [0.0220, 0.0441, 0.7349]
            model_list = ["gemini-1.5-flash-8b", "gemini-1.5-flash", "gemini-1.5-pro"]
        elif model_family == "claude":
            acc_constraint_range = np.linspace(1.95, 2.1, 10)
            lat_constraint_range = np.linspace(0.95, 1.90, 10)
            acc_points = [2.0407, 1.9689]
            lat_points = [1.1646, 1.8735]
            price_points = [0.4686, 1.7546]
            model_list = ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"]
    else:
        raise ValueError("Unknown benchmark")

    return acc_constraint_range, lat_constraint_range, acc_points, lat_points, price_points, model_list


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    benchmark = "spider"
    model_family = "gpt"
    n_s = 8

    # Input and grid
    results = parse_results(f"eval_new_batching_{benchmark}_{model_family}.csv")
    acc_range, lat_range, acc_pts, lat_pts, price_pts, model_list = return_constraint_range(
        benchmark, model_family
    )

    total_cells = len(acc_range) * len(lat_range)

    # Baseline rectangle coverage
    for acc, lat, base_price, model in zip(acc_pts, lat_pts, price_pts, model_list):
        count_x = np.sum(acc_range <= acc)
        count_y = np.sum(lat_range >= lat)
        cover = count_x * count_y
        fail_rate = round((total_cells - cover) / total_cells, 3)
        print(model, 1 - fail_rate)

    heatmap = create_heatmap_data(results, acc_range, lat_range)
    masked = np.ma.masked_invalid(heatmap)
    infeasible = np.count_nonzero(np.isnan(heatmap))
    print("feasible region rate of OctoSelector: ", 1 - round(infeasible / total_cells, 3))

    # Build bin edges for pcolormesh
    def to_edges(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        d = np.diff(arr)
        left = arr[0] - d[0] / 2
        right = arr[-1] + d[-1] / 2
        mids = (arr[:-1] + arr[1:]) / 2
        return np.concatenate(([left], mids, [right]))

    acc_edges = to_edges(acc_range)
    lat_edges = to_edges(lat_range)

    # Colors
    octo_color, octo_alpha = "powderblue", 0.4
    color_list = ["mediumorchid", "dodgerblue", "steelblue"]
    model_colors = {name: color_list[i % len(color_list)] for i, name in enumerate(model_list)}
    model_alpha = 0.4

    # Figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # 1) cell grid (white cells + light edges)
    dummy_Z = np.zeros_like(masked, dtype=float)
    ax.pcolormesh(
        acc_edges, lat_edges, dummy_Z,
        cmap=ListedColormap(["#FFFFFF"]),
        shading="auto", edgecolors="lightgray", linewidth=0.2, zorder=0
    )

    # 2) OctoSelector feasible region overlay
    octo_mask = ~np.isnan(masked)
    octo_Z = np.where(octo_mask, 1.0, np.nan)
    octo_cmap_single = ListedColormap([octo_color])
    octo_artist = ax.pcolormesh(
        acc_edges, lat_edges, octo_Z,
        cmap=octo_cmap_single, shading="auto",
        alpha=octo_alpha, vmin=0, vmax=1, zorder=1
    )
    octo_artist.set_edgecolor("none")

    # 3) Per-model rectangles
    ax.set_xlim(acc_edges[0], acc_edges[-1])
    ax.set_ylim(lat_edges[0], lat_edges[-1])
    ax.invert_yaxis()

    x_left, x_right = acc_edges[0], acc_edges[-1]
    y_top, y_bottom = lat_edges[0], lat_edges[-1]
    for acc, lat, name in zip(acc_pts, lat_pts, model_list):
        col = model_colors.get(name, "#999999")
        ax.fill_between([x_left, acc], y_bottom, lat, color=col, alpha=model_alpha, zorder=2)

    # 4) Model points + guides
    ax.scatter(acc_pts, lat_pts, s=150, marker="o", edgecolor="darkgray", facecolors="none", zorder=3)
    for acc, lat, name in zip(acc_pts, lat_pts, model_list):
        ax.plot([x_left, acc], [lat, lat], color="darkgray", linestyle="--", linewidth=0.9, alpha=0.85, zorder=4)
        ax.plot([acc, acc], [lat, y_bottom], color="darkgray", linestyle="--", linewidth=0.9, alpha=0.85, zorder=4)
        ax.annotate(name, xy=(acc, lat), xycoords="data", xytext=(6, 6), textcoords="offset points",
                    fontsize=10, color="black", zorder=5)

    # 5) Legend
    legend_handles = [Patch(facecolor=to_rgba(octo_color, octo_alpha), edgecolor="none", label="OctoSelector feasible")]
    for name in model_list:
        col = model_colors.get(name, "#999999")
        legend_handles.append(Patch(facecolor=to_rgba(col, model_alpha), edgecolor="none", label=name))
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=1.0, fontsize=9)

    ax.set_title(f"{model_family.upper()}: Feasible Regions\n(Colored = feasible; White = infeasible)")
    ax.set_xlabel("Accuracy Constraint")
    ax.set_ylabel("Latency Constraint (Lower is Better)")
    plt.tight_layout()

    # Save to ./figures/
    import os
    os.makedirs("figures", exist_ok=True)
    out_path = f"figures/{benchmark}_{model_family}_feasible_region.png"
    plt.savefig(out_path, dpi=400, bbox_inches="tight", facecolor="white", transparent=False)