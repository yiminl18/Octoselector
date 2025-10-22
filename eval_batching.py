"""
Visualize ILP results as a heatmap and compare against baseline models.

- Reads per-(accuracy, latency) objective values from CSVs.
- Builds a price heatmap (NaN = infeasible).
- Computes per-model averages in the lower-left rectangles
  (acc <= point_acc and lat >= point_lat; lower latency is better so y is inverted).
- Combines rectangles by taking the minimum feasible price per cell.
- Saves a compact CSV summary per K.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from read_select_json import load_json
import config


# ----------------------------
# Parsing & grid preparation
# ----------------------------

def parse_results(filepath: str):
    """Load and normalize result rows from a CSV file."""
    df = pd.read_csv(filepath)
    df["acc_constraint"] = pd.to_numeric(df["acc_constraint"], errors="coerce")
    df["lat_constraint"] = pd.to_numeric(df["lat_constraint"], errors="coerce")
    df["objective_value"] = pd.to_numeric(df["objective_value"], errors="coerce")
    df["solver_status"] = df["solver_status"].astype(str)
    return df.to_dict(orient="records")


def find_closest_index(arr: np.ndarray, value: float) -> int:
    """Index of the element in arr closest to value."""
    return int(np.argmin(np.abs(arr - value)))


def create_heatmap_data(results, acc_range: np.ndarray, lat_range: np.ndarray) -> np.ndarray:
    """
    Map (acc_constraint, lat_constraint) -> objective_value on a dense grid.
    Infeasible cells remain NaN.
    """
    n_acc, n_lat = len(acc_range), len(lat_range)
    heatmap = np.full((n_lat, n_acc), np.nan, dtype=float)

    for res in results:
        if "acc_constraint" not in res or "lat_constraint" not in res:
            continue
        i = find_closest_index(acc_range, res["acc_constraint"])   # column (acc)
        j = find_closest_index(lat_range, res["lat_constraint"])   # row (lat)
        status = (res.get("solver_status", "") or "").lower()
        if status in {"infeasible", "undefined"}:
            heatmap[j, i] = np.nan
        else:
            heatmap[j, i] = res.get("objective_value", np.nan)
    return heatmap


# ----------------------------
# Region & aggregation helpers
# ----------------------------

def left_bottom_mask(point_acc, point_lat, acc_range, lat_range):
    """
    Boolean mask for cells with acc <= point_acc and lat >= point_lat.
    (We invert y later so lower latency is visually higher.)
    """
    n_lat, n_acc = len(lat_range), len(acc_range)
    mask = np.zeros((n_lat, n_acc), dtype=bool)
    for j in range(n_lat):
        for i in range(n_acc):
            if acc_range[i] <= point_acc and lat_range[j] >= point_lat:
                mask[j, i] = True
    return mask


def avg_price_lower_left(point_acc, point_lat, acc_range, lat_range, heatmap_data):
    """Average price over the left-bottom region; NaN if empty."""
    acc_idx = np.where(acc_range <= point_acc)[0]
    lat_idx = np.where(lat_range >= point_lat)[0]
    if acc_idx.size == 0 or lat_idx.size == 0:
        return np.nan
    sub = heatmap_data[np.ix_(lat_idx, acc_idx)]
    vals = sub[~np.isnan(sub)]
    return np.mean(vals) if vals.size else np.nan


def union_rectangles_min_price(
    acc_points, lat_points, baseline_price_points,
    heatmap_data, acc_range, lat_range
):
    """
    For each cell, if it belongs to ≥1 rectangle, take the min feasible price across rectangles.
    Also compute average price of feasible cells outside the union.
    """
    masks = [left_bottom_mask(a, l, acc_range, lat_range) for a, l in zip(acc_points, lat_points)]
    union_mask = np.any(np.stack(masks, axis=0), axis=0) if masks else np.zeros_like(heatmap_data, dtype=bool)

    n_lat, n_acc = heatmap_data.shape
    combined = np.full((n_lat, n_acc), np.nan)
    baseline_combined = np.full((n_lat, n_acc), np.nan)

    for j in range(n_lat):
        for i in range(n_acc):
            prices = []
            bases = []
            for idx, m in enumerate(masks):
                if m[j, i]:
                    v = heatmap_data[j, i]
                    if not np.isnan(v):
                        prices.append(v)
                        bases.append(baseline_price_points[idx])
            if prices:
                combined[j, i] = min(prices)
                baseline_combined[j, i] = min(bases)

    outside_mask = (~union_mask) & ~np.isnan(heatmap_data)
    avg_price_outside = float(np.nanmean(heatmap_data[outside_mask])) if np.any(outside_mask) else np.nan

    return combined, baseline_combined, avg_price_outside


# ----------------------------
# Constraint grids (per dataset/model family)
# ----------------------------

def return_constraint_range(benchmark, model_family):
    """
    Return (acc_range, lat_range, acc_points, lat_points, price_points, model_list)
    for plotting and baseline comparison.
    """
    if benchmark == 'spider':
        if model_family == 'gpt':
            acc_constraint_range = np.linspace(0.72, 0.77, 10)
            lat_constraint_range = np.linspace(0.32, 0.58, 10)
            acc_points = [0.74951, 0.74130, 0.76419]
            lat_points = [0.54864, 0.3894205, 0.48524]
            price_points = [0.02379, 0.325455, 0.4342775]
            model_list = ['gpt-4o-mini', 'gpt-3.5-turbo-0125', 'gpt-4o']
        elif model_family == 'gemini':
            acc_constraint_range = np.linspace(0.75, 0.82, 10)
            lat_constraint_range = np.linspace(0.18, 0.5, 10)
            acc_points = [0.77817, 0.78433, 0.8121]
            lat_points = [0.24809, 0.26169, 0.481259]
            price_points = [0.006573, 0.011709, 0.195791]
            model_list = ['gemini-1.5-flash-8b', 'gemini-1.5-flash', 'gemini-1.5-pro']
        elif model_family == 'claude':
            acc_constraint_range = np.linspace(0.55, 0.74, 10)
            lat_constraint_range = np.linspace(0.5, 0.7, 10)
            acc_points = [0.6609, 0.70021]
            lat_points = [0.58283, 0.54633]
            price_points = [0.15614, 0.597462]
            model_list = ['claude-3-5-haiku-20241022', 'claude-3-5-sonnet-20241022']
        # Optional override to a shared feasible box:
        acc_constraint_range = np.linspace(0.6609, 0.815, 20)
        lat_constraint_range = np.linspace(0.24809, 0.58283, 20)

    elif benchmark == 'BIRD':
        if model_family == 'gpt':
            acc_constraint_range = np.linspace(0.35, 0.47, 10)
            lat_constraint_range = np.linspace(0.47, 0.89, 10)
            acc_points = [0.42335, 0.39854, 0.47031]
            lat_points = [0.87199, 0.47505, 0.76221]
            price_points = [0.06238, 0.79425, 1.04768]
            model_list = ['gpt-4o-mini', 'gpt-3.5-turbo-0125', 'gpt-4o']
        elif model_family == 'gemini':
            acc_constraint_range = np.linspace(0.35, 0.45, 10)
            lat_constraint_range = np.linspace(0.30, 0.81, 10)
            acc_points = [0.396, 0.41672, 0.43896]
            lat_points = [0.45733, 0.50185, 0.75791]
            price_points = [0.01744, 0.03032, 0.50509]
            model_list = ['gemini-1.5-flash-8b', 'gemini-1.5-flash', 'gemini-1.5-pro']
        elif model_family == 'claude':
            acc_constraint_range = np.linspace(0.40, 0.485, 10)
            lat_constraint_range = np.linspace(0.83, 1.5, 10)
            acc_points = [0.43467, 0.46531]
            lat_points = [0.97358, 0.90164]
            price_points = [0.41542, 1.40628]
            model_list = ['claude-3-5-haiku-20241022', 'claude-3-5-sonnet-20241022']
        # Optional override to a shared feasible box:
        acc_constraint_range = np.linspace(0.39854, 0.485, 20)
        lat_constraint_range = np.linspace(0.45733, 0.97358, 20)

    elif benchmark == 'IMDb':
        if model_family == 'gpt':
            acc_constraint_range = np.linspace(1.80, 2.1, 10)
            lat_constraint_range = np.linspace(0.42, 0.58, 10)
            acc_points = [2.0193, 2.0448, 1.8631]
            lat_points = [0.5260, 0.4637, 0.5728]
            price_points = [0.0876, 1.7456, 1.4597]
            model_list = ['gpt-4o-mini', 'gpt-3.5-turbo-0125', 'gpt-4o']
        elif model_family == 'gemini':
            acc_constraint_range = np.linspace(2.15, 3.6, 10)
            lat_constraint_range = np.linspace(0.34, 0.9, 10)
            acc_points = [3.5070, 2.8149, 2.1766]
            lat_points = [0.4596, 0.4068, 0.8851]
            price_points = [0.0220, 0.0441, 0.7349]
            model_list = ['gemini-1.5-flash-8b', 'gemini-1.5-flash', 'gemini-1.5-pro']
        elif model_family == 'claude':
            acc_constraint_range = np.linspace(1.95, 2.1, 10)
            lat_constraint_range = np.linspace(0.95, 1.90, 10)
            acc_points = [2.0407, 1.9689]
            lat_points = [1.1646, 1.8735]
            price_points = [0.4686, 1.7546]
            model_list = ['claude-3-5-haiku-20241022', 'claude-3-5-sonnet-20241022']
    else:
        raise ValueError("Unknown benchmark/model_family")

    return acc_constraint_range, lat_constraint_range, acc_points, lat_points, price_points, model_list


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    benchmark = 'spider'
    model_family = 'gpt'
    if benchmark == 'spider':
        n_s = 8
    elif benchmark == 'BIRD':
        n_s = 6
    cluster_number_K_list = [15]
    # cluster_number_K_list = [5, 10, 15, 20, 25, 30, 40]

    for K in cluster_number_K_list:
        results = parse_results(f'eval_batching_K{K}_{benchmark}_{model_family}.csv')

        # Grid and baseline points
        acc_range, lat_range, acc_points, lat_points, price_points, model_list = return_constraint_range(
            benchmark, model_family
        )
        total_cells = len(acc_range) * len(lat_range)

        # Heatmap data
        heatmap = create_heatmap_data(results, acc_range, lat_range)
        masked = np.ma.masked_invalid(heatmap)
        infeasible = int(np.count_nonzero(np.isnan(heatmap)))
        fail_rate_overall = round(infeasible / total_cells, 3)
        print('OctoSelector fail rate:', fail_rate_overall)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        cmap = plt.get_cmap('coolwarm').copy()
        cmap.set_bad(color='white')

        def to_edges(arr):
            arr = np.asarray(arr)
            d = np.diff(arr)
            left = arr[0] - d[0] / 2
            right = arr[-1] + d[-1] / 2
            mids = (arr[:-1] + arr[1:]) / 2
            return np.concatenate(([left], mids, [right]))

        acc_edges = to_edges(acc_range)
        lat_edges = to_edges(lat_range)

        im = ax.pcolormesh(
            acc_edges, lat_edges, masked,
            cmap=cmap, shading='auto',
            edgecolors='lightgray', linewidth=0.3
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Objective Function Value (price)")

        ax.set_xlabel("Accuracy Constraint")
        ax.set_ylabel("Latency Constraint (Lower is Better)")
        ax.invert_yaxis()
        ax.set_title("Heatmap of Objective Value\n(White = Infeasible; Lower latency is better)")

        # Draw baseline rectangles (left/down from each point)
        ax.scatter(
            acc_points, lat_points, s=150, marker='o',
            c=price_points, cmap=cmap, norm=im.norm,
            edgecolor='black', facecolors='none', label='Baselines', zorder=6
        )

        adjust_x = (acc_range[1] - acc_range[0]) / 10
        adjust_y = (lat_range[-1] - lat_range[0]) / 10

        for acc, lat, price, name in zip(acc_points, lat_points, price_points, model_list):
            ax.plot([acc_range[0] - adjust_x, acc], [lat, lat],
                    color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=4)
            ax.plot([acc, acc], [lat, lat_range[-1] + adjust_y],
                    color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=4)
            ax.annotate(f"{name}: {price:.3f}", xy=(acc, lat), xycoords='data',
                        xytext=(6, 6), textcoords='offset points',
                        fontsize=10, color='black',
                        arrowprops=dict(arrowstyle="->", color='gray', lw=0.6, alpha=0.8))

        plt.show()

        # Compute per-model and combined averages
        fail_rates = [fail_rate_overall]  # first col = OctoSelector
        baseline_prices = [np.nan]        # first col not applicable for baseline
        octo_prices = [np.nan]            # placeholder for OctoSelector col

        for acc, lat, base_price, name in zip(acc_points, lat_points, price_points, model_list):
            cnt_x = int(np.sum(acc_range <= acc))
            cnt_y = int(np.sum(lat_range >= lat))
            covered = cnt_x * cnt_y
            fr = round((total_cells - covered) / total_cells, 3)
            print(name, 1 - fr)

            avg_price = avg_price_lower_left(acc, lat, acc_range, lat_range, heatmap)
            print(f"{name}: fail_rate={fr}, baseline={base_price}, OctoSelector={avg_price}")

            fail_rates.append(fr)
            baseline_prices.append(base_price)
            octo_prices.append(avg_price)

        combined, baseline_combined, avg_price_outside = union_rectangles_min_price(
            acc_points, lat_points, price_points,
            heatmap, acc_range, lat_range
        )
        overall_avg_price = float(np.nanmean(combined))
        baseline_avg_price = float(np.nanmean(baseline_combined))

        print("Union (min-per-cell) OctoSelector avg:", overall_avg_price)
        print("Union baseline avg:", baseline_avg_price)
        print("Avg price outside union (feasible cells):", avg_price_outside)

        # Append "overall opt" column
        baseline_prices.append(baseline_avg_price)
        octo_prices.append(overall_avg_price)
        fail_rates.append(np.nan)  # no fail rate for the 'overall opt' col

        # Build summary table
        row_labels = [
            "Overall Fail rate",
            "Baseline price (USD)",
            "OctoSelector price (USD)",
            "Outside baseline feasible price (USD)"
        ]
        columns = ['OctoSelector'] + model_list + ['overall opt']

        # Place the outside-feasible price only in the last column; NaN elsewhere
        outside_row = [np.nan] * (len(columns) - 1) + [avg_price_outside]

        data = [
            fail_rates,
            baseline_prices,
            octo_prices,
            outside_row
        ]
        df = pd.DataFrame(data, index=row_labels, columns=columns)
        df.index.name = "Metric"

        out_name = f"Model_comparison_batch_K{K}_{benchmark}_{model_family}_ns{n_s}.csv"
        os.makedirs(os.path.dirname(out_name) or ".", exist_ok=True)
        df.to_csv(out_name)