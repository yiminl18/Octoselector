"""
ILP-based LLM assignment for batched NL2SQL queries.

Given:
- precomputed batch metadata (pure full batches vs residual mixed batches),
- per-cluster accuracy/latency/token-price summaries for each model,
- and per-query predicted cluster IDs,

this module builds and solves an integer linear program that minimizes total
token cost subject to global average accuracy/latency constraints. Pure full
batches use integer counts (x_ij); residual batches use per-batch binary
assignments (y_bj).
"""

from __future__ import annotations

import os
import csv
import time
from typing import Dict, Tuple, List, Callable, Optional
from collections import defaultdict

import joblib
import pandas as pd
import pulp

# Keep explicit imports to avoid wildcards.
from batch_metric_models import (
    load_batches_index,
    load_cluster_metric_df,
    load_llm_models_config,
    cost_full_pure_per_batch_cfg,
    lat_full_pure_per_batch_cfg,
    get_acc_cluster,
    cost_on_demand_residual_cfg,
    lat_on_demand_residual_cfg,
    acc_on_demand,
    default_batch_prompt,
    tokenizer_fn,
    predict_feature_rf,
)
from nl2sql_dataprepare import db_queries_clusterID_dict
from eval_graph_batch import return_constraint_range
from read_select_json import load_json
import config


# -------------------------
# Batch utilities
# -------------------------

def summarize_batches_for_ilp(batches_idx: Dict[str, dict], n_s: int):
    """
    Return:
      - full_count_by_cluster: {cluster_id -> # full pure batches}
      - residual_batches: [batch_name, ...]
      - total_full, total_res
    """
    full_count_by_cluster = defaultdict(int)
    residual_batches: List[str] = []

    for bname, rec in batches_idx.items():
        is_full = rec.get("is_full", False)
        btype = rec.get("type", "mixed")
        size = int(rec.get("size", 0))
        cids = rec.get("cluster_ids", [])
        if btype == "pure" and is_full and size == n_s and len(cids) == 1:
            full_count_by_cluster[int(cids[0])] += 1
        else:
            residual_batches.append(bname)

    total_full = sum(full_count_by_cluster.values())
    total_res = len(residual_batches)
    return dict(full_count_by_cluster), residual_batches, total_full, total_res


# -------------------------
# ILP solver
# -------------------------

def solve_llm_assignment_ilp(
    *,
    batches_csv_path: str,
    cluster_summary_csv: str,
    model_config_json: str,
    queries_cluster_dict: Dict[Tuple[str, str], int],
    n_s: int,
    model_list: Optional[List[str]] = None,
    sigma_A: float = 0.44,
    sigma_T: float = 1.0,
    accuracy_is_lower_bound: bool = True,
    latency_is_upper_bound: bool = True,
    prompt_tok_counter: Optional[Callable[[str, List[str], str], int]] = None,
    solver: Optional[pulp.LpSolver] = None,
    msg: bool = True,
    benchmark: str = "spider",
):
    """
    Build and solve the ILP:
      minimize total token cost subject to global average accuracy/latency constraints.
    Returns:
      {
        status, x_assign, y_assign, avg_accuracy, avg_latency, total_cost, totals
      }
    """
    batches_idx = load_batches_index(batches_csv_path)
    cluster_metric_df = load_cluster_metric_df(cluster_summary_csv)
    model_configs = load_llm_models_config(model_config_json)

    if model_list is None:
        model_list = list(model_configs.keys())

    full_cnt_by_cluster, residual_batches, TOTAL_FULL, TOTAL_RES = summarize_batches_for_ilp(
        batches_idx, n_s
    )

    I = sorted(full_cnt_by_cluster.keys())  # clusters with full-pure batches
    J = list(range(len(model_list)))        # model indices
    MNAME = {j: m for j, m in enumerate(model_list)}
    Bres = residual_batches                 # residual batch names

    # Precompute coefficients: full batches
    C_full = {
        (i, j): cost_full_pure_per_batch_cfg(
            model_configs=model_configs,
            model_name=MNAME[j],
            cluster_id=i,
            n_s=n_s,
            cluster_metric_df=cluster_metric_df,
        )
        for i in I for j in J
    }
    T_full = {
        (i, j): lat_full_pure_per_batch_cfg(
            model_configs=model_configs,
            model_name=MNAME[j],
            cluster_id=i,
            n_s=n_s,
            cluster_metric_df=cluster_metric_df,
        )
        for i in I for j in J
    }
    A_full = {(i, j): get_acc_cluster(cluster_metric_df, MNAME[j], i) for i in I for j in J}

    # Residual batches
    C_res, T_res, A_res = {}, {}, {}
    for b in Bres:
        brec = batches_idx[b]
        for j in J:
            m = MNAME[j]
            C_res[(b, j)] = cost_on_demand_residual_cfg(
                brec,
                m,
                model_configs=model_configs,
                cluster_metric_df=cluster_metric_df,
                queries_cluster_dict=queries_cluster_dict,
                prompt_tok_counter=prompt_tok_counter,
                benchmark=benchmark,
            )
            T_res[(b, j)] = lat_on_demand_residual_cfg(
                brec,
                m,
                model_configs=model_configs,
                cluster_metric_df=cluster_metric_df,
                queries_cluster_dict=queries_cluster_dict,
                prompt_tok_counter=prompt_tok_counter,
                benchmark=benchmark,
            )
            A_res[(b, j)] = acc_on_demand(
                brec,
                m,
                cluster_metric_df=cluster_metric_df,
                queries_cluster_dict=queries_cluster_dict,
                n_s=n_s,
            )

    # Model
    prob = pulp.LpProblem("LLM_Assignment", pulp.LpMinimize)

    # Vars
    x = pulp.LpVariable.dicts("x", ((i, j) for i in I for j in J), lowBound=0, cat="Integer")
    y = pulp.LpVariable.dicts("y", ((b, j) for b in Bres for j in J), lowBound=0, upBound=1, cat="Binary")

    # Coverage of full-pure batches
    for i in I:
        prob += pulp.lpSum(x[(i, j)] for j in J) == full_cnt_by_cluster[i], f"cover_full_cluster_{i}"

    # Assign each residual batch to exactly one model
    for b in Bres:
        prob += pulp.lpSum(y[(b, j)] for j in J) == 1, f"assign_residual_{b}"

    # Objective
    cost_full_expr = pulp.lpSum(C_full[(i, j)] * x[(i, j)] for i in I for j in J) if I else 0
    cost_res_expr = pulp.lpSum(C_res[(b, j)] * y[(b, j)] for b in Bres for j in J) if Bres else 0
    prob += cost_full_expr + cost_res_expr, "MinCost"

    # Global averages
    acc_full_sum = pulp.lpSum(A_full[(i, j)] * x[(i, j)] for i in I for j in J) if I else 0
    acc_res_sum = pulp.lpSum(A_res[(b, j)] * y[(b, j)] for b in Bres for j in J) if Bres else 0
    lat_full_sum = pulp.lpSum(T_full[(i, j)] * x[(i, j)] for i in I for j in J) if I else 0
    lat_res_sum = pulp.lpSum(T_res[(b, j)] * y[(b, j)] for b in Bres for j in J) if Bres else 0

    TOTAL_BATCHES = TOTAL_FULL + TOTAL_RES
    denom = float(TOTAL_BATCHES) if TOTAL_BATCHES > 0 else 1.0

    avg_acc_expr = (acc_full_sum + acc_res_sum) / denom
    avg_lat_expr = (lat_full_sum + lat_res_sum) / denom

    # Constraints
    if accuracy_is_lower_bound:
        prob += avg_acc_expr >= sigma_A, "AccuracyMin"
    else:
        prob += avg_acc_expr <= sigma_A, "AccuracyMax"

    if latency_is_upper_bound:
        prob += avg_lat_expr <= sigma_T, "LatencyMax"
    else:
        prob += avg_lat_expr >= sigma_T, "LatencyMin"

    # Solve
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=msg)
    prob.solve(solver)

    status = pulp.LpStatus.get(prob.status, "Unknown")

    # If infeasible, return zeros to avoid noisy upstream handling.
    if status == "Infeasible":
        return {
            "status": status,
            "x_assign": {},
            "y_assign": {},
            "avg_accuracy": 0.0,
            "avg_latency": 0.0,
            "total_cost": 0.0,
            "totals": {"TOTAL_FULL": 0, "TOTAL_RES": 0},
        }

    # Extract solution
    x_assign: Dict[int, Dict[str, int]] = defaultdict(dict)
    for i in I:
        for j in J:
            val = int(round(pulp.value(x[(i, j)]) or 0))
            if val > 0:
                x_assign[i][MNAME[j]] = val

    y_assign: Dict[str, str] = {}
    for b in Bres:
        best_m, best_v = None, -1.0
        for j in J:
            v = float(pulp.value(y[(b, j)]) or 0.0)
            if v > best_v:
                best_v, best_m = v, MNAME[j]
        if best_m is not None:
            y_assign[b] = best_m

    total_cost = pulp.value(cost_full_expr + cost_res_expr)
    avg_acc = pulp.value(avg_acc_expr)
    avg_lat = pulp.value(avg_lat_expr)

    return {
        "status": status,
        "x_assign": dict(x_assign),
        "y_assign": y_assign,
        "avg_accuracy": round(float(avg_acc), 5) if avg_acc is not None else None,
        "avg_latency": round(float(avg_lat), 5) if avg_lat is not None else None,
        "total_cost": round(float(total_cost) / 1e6, 5) if total_cost is not None else None,
        "totals": {"TOTAL_FULL": TOTAL_FULL, "TOTAL_RES": TOTAL_RES},
    }


# -------------------------
# Reporting helpers
# -------------------------

def summarize_batches_per_model(
    x_assign: Dict[int, Dict[str, int]],
    y_assign: Dict[str, str],
    model_list: List[str],
) -> Dict[str, Dict[str, int]]:
    """Return {model: {'full': #full_pure, 'res': #residual, 'total': sum}}."""
    out = {m: {'full': 0, 'res': 0, 'total': 0} for m in model_list}

    for _cid, per_model in x_assign.items():
        for m, cnt in per_model.items():
            out.setdefault(m, {'full': 0, 'res': 0, 'total': 0})
            out[m]['full'] += int(cnt)

    for _b, m in y_assign.items():
        out.setdefault(m, {'full': 0, 'res': 0, 'total': 0})
        out[m]['res'] += 1

    for m in out:
        out[m]['total'] = out[m]['full'] + out[m]['res']

    return out


def result_batches_per_model(
    per_model: Dict[str, Dict[str, int]],
    order: str = "total_desc",
) -> Dict[str, List[int]]:
    """Return {model: [#full, #res]} sorted by desired order."""
    if order == "total_desc":
        items = sorted(per_model.items(), key=lambda kv: (-kv[1]['total'], kv[0]))
    elif order == "name":
        items = sorted(per_model.items(), key=lambda kv: kv[0])
    else:
        items = per_model.items()

    out: Dict[str, List[int]] = {}
    for m, stats in items:
        out[m] = [stats['full'], stats['res']]
    return out


def prompt_tok_counter(db_name: str, nlqs: List[str], benchmark: str) -> int:
    """Default prompt token counter; replace with a real tokenizer if needed."""
    prompt = default_batch_prompt(db_name, nlqs, benchmark)
    return tokenizer_fn(prompt)


# -------------------------
# Main
# -------------------------

def main():
    n_s = 8
    model_family = 'gpt'
    benchmark = "spider"

    if benchmark == 'spider':
        dev_json_file = getattr(config, "SPIDER_DEV_FILE", "spider/dev.json")
    elif benchmark == 'BIRD':
        dev_json_file = getattr(config, "BIRD_DEV_FILE", "BIRD/dev/dev.json")
    else:
        raise ValueError('Invalid benchmark')

    dev_json_data = load_json(dev_json_file)

    cluster_number_K_list = [5, 10, 15, 20, 25, 30, 40]
    avg_runtime = 0.0
    optimal_counter = 0

    for K in cluster_number_K_list:
        kmeans_loaded = joblib.load(os.path.join(config.KMEANS_MODEL_DIR, f'kmeans_model_{benchmark}_K{K}.pkl'))
        rf_model = joblib.load(os.path.join(config.KMEANS_MODEL_DIR, f'{config.RF_MODEL_BASENAME}_{benchmark}.pkl'))

        feature_vectors = predict_feature_rf(rf_model, dev_json_data)
        predicted_clusters = kmeans_loaded.predict(feature_vectors)

        batches_csv_path = f"batches_{benchmark}_dev_ns{n_s}_K{K}.csv"
        cluster_summary_csv = f"cluster_summary_{benchmark}_K{K}.csv"
        model_config_json = getattr(config, "MODEL_CONFIG_FILE", "model_file/LLM_models_config.json")

        queries_cluster_dict: Dict[Tuple[str, str], int] = db_queries_clusterID_dict(
            predicted_clusters, dev_json_data
        )

        acc_range, lat_range, acc_points, lat_points, price_points, model_list = return_constraint_range(
            benchmark, model_family
        )

        filename = f'eval_batching_K{K}_{benchmark}_{model_family}.csv'
        base_cols = ['acc_constraint', 'lat_constraint', 'solver_status', 'objective_value', 'final_acc', 'final_latency']
        model_cols = list(model_list)
        fieldnames = base_cols + model_cols

        file_exists = os.path.isfile(filename)
        with open(filename, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for acc_min in acc_range:
                for lat_max in lat_range:
                    start_time = time.time()
                    res = solve_llm_assignment_ilp(
                        batches_csv_path=batches_csv_path,
                        cluster_summary_csv=cluster_summary_csv,
                        model_config_json=model_config_json,
                        queries_cluster_dict=queries_cluster_dict,
                        n_s=n_s,
                        model_list=model_list,
                        sigma_A=acc_min,
                        sigma_T=lat_max,
                        accuracy_is_lower_bound=True,
                        latency_is_upper_bound=True,
                        prompt_tok_counter=prompt_tok_counter,
                        msg=False,
                        benchmark=benchmark,
                    )
                    if res['status'] == 'Optimal':
                        runtime = time.time() - start_time
                        avg_runtime += runtime
                        optimal_counter += 1

                    per_model = summarize_batches_per_model(res["x_assign"], res["y_assign"], model_list=model_list)

                    row = {
                        'acc_constraint': acc_min,
                        'lat_constraint': lat_max,
                        'solver_status': res['status'],
                        'objective_value': res['total_cost'],
                        'final_acc': res['avg_accuracy'],
                        'final_latency': res['avg_latency'],
                    }
                    for m in model_cols:
                        stats = per_model.get(m, {'full': 0, 'res': 0, 'total': 0})
                        row[m] = f"[{stats['full']},{stats['res']}]"

                    writer.writerow(row)

    # Optionally, print average optimal runtime.
    # if optimal_counter:
    #     print("Average optimal runtime (s):", round(avg_runtime / optimal_counter, 5))


if __name__ == "__main__":
    main()