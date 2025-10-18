# llm_assignment_ilp.py
# Implements the ILP in your formulation with interchangeable full-pure batches (x_ij)
# and per-batch binary assignment for residual batches (y_bj).

from __future__ import annotations
import json
import csv
from pathlib import Path
from typing import Dict, Tuple, List, Mapping, Any, Callable, Optional
from collections import Counter, defaultdict

import pandas as pd
import pulp
from batch_metric_models import *
from nl2sql_dataprepare import db_queries_clusterID_dict
from eval_graph_batch import return_constraint_range
import time



# =========================
# ---- Batch utilities ----
# =========================


def summarize_batches_for_ilp(batches_idx: Dict[str, dict], n_s: int):
    """
    Returns:
      full_count_by_cluster: {cluster_id -> #full_pure_batches}
      residual_batches     : [batch_name, ...]
      TOTAL_FULL, TOTAL_RES
    """
    full_count_by_cluster = defaultdict(int)
    residual_batches: List[str] = []

    for bname, rec in batches_idx.items():
        is_full = rec["is_full"]
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


# =========================
# ------ ILP Solver -------
# =========================

def solve_llm_assignment_ilp(
    *,
    batches_csv_path: str,
    cluster_summary_csv: str,
    model_config_json: str,
    queries_cluster_dict: Dict[Tuple[str, str], int],
    n_s: int,
    model_list: Optional[List[str]] = None,     # if None, use all models in config
    sigma_A: float = 0.44,                      # accuracy threshold
    sigma_T: float = 1.0,                       # latency threshold
    accuracy_is_lower_bound: bool = True,       # True -> >= sigma_A, False -> <= sigma_A
    latency_is_upper_bound: bool = True,        # True -> <= sigma_T, False -> >= sigma_T
    prompt_tok_counter: Optional[Callable[[str, List[str],str], int]] = None,
    solver: Optional[pulp.LpSolver] = None,
    msg: bool = True,
    benchmark: str
):
    """
    Build and solve the ILP:
        minimize total cost subject to accuracy & latency average constraints.

    Returns a dict with:
        - status
        - x_assign (cluster_id -> {model -> count})
        - y_assign (batch -> model)
        - avg_accuracy, avg_latency
        - total_cost
    """
    # ---- Load data ----
    batches_idx = load_batches_index(batches_csv_path)
    cluster_metric_df = load_cluster_metric_df(cluster_summary_csv)
    model_configs = load_llm_models_config(model_config_json)

    if model_list is None:
        model_list = list(model_configs.keys())

    # default prompt token counter (very rough). Replace with your tokenizer-based function.
    # if prompt_tok_counter is None:
    #     def prompt_tok_counter(db_name: str, nlqs: List[str]) -> int:
    #         # naive: word count as token proxy
    #         return sum(len(q.split()) for q in nlqs)

    # ---- Summarize batches ----
    full_cnt_by_cluster, residual_batches, TOTAL_FULL, TOTAL_RES = summarize_batches_for_ilp(batches_idx, n_s)

    I = sorted(full_cnt_by_cluster.keys())          # clusters with full-pure batches
    J = list(range(len(model_list)))                # model indices
    MNAME = {j: m for j, m in enumerate(model_list)}
    Bres = residual_batches                         # residual batch names

    # ---- Precompute coefficients ----
    # Full: per-batch cost/latency; accuracy per batch equals A(G_i, L_j)
    C_full = {(i, j): cost_full_pure_per_batch_cfg(
        model_configs=model_configs, model_name=MNAME[j], cluster_id=i, n_s=n_s,
        cluster_metric_df=cluster_metric_df
    ) for i in I for j in J}

    T_full = {(i, j): lat_full_pure_per_batch_cfg(
        model_configs=model_configs, model_name=MNAME[j], cluster_id=i, n_s=n_s,
        cluster_metric_df=cluster_metric_df
    ) for i in I for j in J}

    A_full = {(i, j): get_acc_cluster(cluster_metric_df, MNAME[j], i) for i in I for j in J}

    # Residual: per-batch per-model coefficients
    C_res = {}
    T_res = {}
    A_res = {}
    for b in Bres:
        brec = batches_idx[b]
        for j in J:
            m = MNAME[j]
            C_res[(b, j)] = cost_on_demand_residual_cfg(
                brec, m, model_configs=model_configs, cluster_metric_df=cluster_metric_df,
                queries_cluster_dict=queries_cluster_dict, prompt_tok_counter=prompt_tok_counter,benchmark=benchmark
            )
            T_res[(b, j)] = lat_on_demand_residual_cfg(
                brec, m, model_configs=model_configs, cluster_metric_df=cluster_metric_df,
                queries_cluster_dict=queries_cluster_dict, prompt_tok_counter=prompt_tok_counter, benchmark=benchmark
            )
            A_res[(b, j)] = acc_on_demand(
                brec, m, cluster_metric_df=cluster_metric_df,
                queries_cluster_dict=queries_cluster_dict, n_s=n_s
            )

    # ---- Build ILP ----
    prob = pulp.LpProblem("LLM_Assignment", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts("x", ((i, j) for i in I for j in J), lowBound=0, cat="Integer")
    y = pulp.LpVariable.dicts("y", ((b, j) for b in Bres for j in J), lowBound=0, upBound=1, cat="Binary")

    # Coverage constraints for full-pure clusters
    for i in I:
        prob += pulp.lpSum(x[(i, j)] for j in J) == full_cnt_by_cluster[i], f"cover_full_cluster_{i}"

    # Assignment constraints for residual batches
    for b in Bres:
        prob += pulp.lpSum(y[(b, j)] for j in J) == 1, f"assign_residual_{b}"

    # Objective: minimize cost
    cost_full_expr = pulp.lpSum(C_full[(i, j)] * x[(i, j)] for i in I for j in J) if I else 0
    cost_res_expr  = pulp.lpSum(C_res[(b, j)]  * y[(b, j)] for b in Bres for j in J) if Bres else 0
    prob += cost_full_expr + cost_res_expr, "MinCost"

    # Accuracy average constraint
    # ----- 先构造“分子”的和（不用先除以类别内批数）-----
    acc_full_sum = pulp.lpSum(A_full[(i, j)] * x[(i, j)] for i in I for j in J) if I else 0
    acc_res_sum  = pulp.lpSum(A_res[(b, j)]  * y[(b, j)] for b in Bres for j in J) if Bres else 0

    lat_full_sum = pulp.lpSum(T_full[(i, j)] * x[(i, j)] for i in I for j in J) if I else 0
    lat_res_sum  = pulp.lpSum(T_res[(b, j)]  * y[(b, j)] for b in Bres for j in J) if Bres else 0

    # ----- 总批数是常数（由 CSV + 覆盖约束决定）-----
    TOTAL_BATCHES = TOTAL_FULL + TOTAL_RES
    # 防御：极端情况下没有任何批
    denom = float(TOTAL_BATCHES) if TOTAL_BATCHES > 0 else 1.0

    # ----- 组合后的全局加权平均 -----
    avg_acc_expr = (acc_full_sum + acc_res_sum) / denom
    avg_lat_expr = (lat_full_sum + lat_res_sum) / denom

    # ----- 约束（方向按你的设定）-----
    if accuracy_is_lower_bound:
        prob += avg_acc_expr >= sigma_A, "AccuracyMin"
    else:
        prob += avg_acc_expr <= sigma_A, "AccuracyMax"

    if latency_is_upper_bound:
        prob += avg_lat_expr <= sigma_T, "LatencyMax"
    else:
        prob += avg_lat_expr >= sigma_T, "LatencyMin"
    # ---- Solve ----
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=msg)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]


    # ---- 如果 infeasible（或其它非最优），统一返回 0 值，避免上层写出“不可行”的结果 ----
    if status == "Infeasible":
        zero_res = {
            "status": status,
            "x_assign": {},          # 空指派
            "y_assign": {},          # 空指派
            "avg_accuracy": 0.0,     # 统一置 0
            "avg_latency": 0.0,      # 统一置 0
            "total_cost": 0.0,       # 统一置 0
            "totals": {"TOTAL_FULL": 0, "TOTAL_RES": 0},
        }
        return zero_res

    # Extract results
    x_assign: Dict[int, Dict[str, int]] = defaultdict(dict)
    for i in I:
        for j in J:
            val = int(round(pulp.value(x[(i, j)]) or 0))
            if val > 0:
                x_assign[i][MNAME[j]] = val

    y_assign: Dict[str, str] = {}
    for b in Bres:
        best_m, best_v = None, -1
        for j in J:
            v = pulp.value(y[(b, j)]) or 0.0
            if v > best_v:
                best_v, best_m = v, MNAME[j]
        if best_m is not None:
            y_assign[b] = best_m

    # Compute realized averages and cost
    total_cost = pulp.value(cost_full_expr + cost_res_expr)
    avg_acc    = pulp.value(avg_acc_expr)
    avg_lat    = pulp.value(avg_lat_expr)   

    return {
        "status": status,
        "x_assign": dict(x_assign),
        "y_assign": y_assign,
        "avg_accuracy": round(float(avg_acc),5) if avg_acc is not None else None,
        "avg_latency": round(float(avg_lat),5) if avg_lat is not None else None,
        "total_cost": round(float(total_cost)/1e6,5) if total_cost is not None else None,
        "totals": {"TOTAL_FULL": TOTAL_FULL, "TOTAL_RES": TOTAL_RES},
    }

from typing import Dict, List

def summarize_batches_per_model(
    x_assign: Dict[int, Dict[str, int]],   # {cluster_id: {model_name: count}}
    y_assign: Dict[str, str],              # {batch_name: model_name}
    model_list: List[str],
) -> Dict[str, Dict[str, int]]:
    """
    返回 {model: {'full': #full_pure, 'res': #residual, 'total': sum}}
    """
    out = {m: {'full': 0, 'res': 0, 'total': 0} for m in model_list}

    # full-pure：按簇聚合的批次数
    for _cid, per_model in x_assign.items():
        for m, cnt in per_model.items():
            out.setdefault(m, {'full': 0, 'res': 0, 'total': 0})
            out[m]['full'] += int(cnt)

    # residual：逐批指派
    for _b, m in y_assign.items():
        out.setdefault(m, {'full': 0, 'res': 0, 'total': 0})
        out[m]['res'] += 1

    # 总数
    for m in out:
        out[m]['total'] = out[m]['full'] + out[m]['res']

    return out

def result_batches_per_model(
    per_model: Dict[str, Dict[str, int]],
    order: str = "total_desc"
):
    llm_count_dict = {}
    if order == "total_desc":
        items = sorted(per_model.items(), key=lambda kv: (-kv[1]['total'], kv[0]))
    elif order == "name":
        items = sorted(per_model.items(), key=lambda kv: kv[0])
    else:
        items = per_model.items()

    for m, stats in items:
        # 用 list：
        llm_count_dict[m] = [stats['full'], stats['res']]
        # 若想是一列字符串，用下面这行替换：
        # llm_count_dict[m] = f"[{stats['full']},{stats['res']}]"
    return llm_count_dict


def prompt_tok_counter(db_name: str, nlqs: List[str], benchmark: str) -> int:
    prompt = default_batch_prompt(db_name, nlqs, benchmark)
    return tokenizer_fn(prompt)
# =========================
# -------- main -----------
# =========================

def main():
    # --- Inputs (edit these) ---
    n_s = 8
    # LLM_list = ["gpt-4o-mini","gpt-3.5-turbo-0125","gpt-4o"]
    avg_runtime =0
    optimal_counter = 0
    model_family = 'gpt'
    benchmark = "spider"
    if benchmark == 'spider':
        dev_json_file = '/Users/delilah/Documents/Research/LLM/spider/dev.json'
    elif benchmark == 'BIRD':
        dev_json_file = '/Users/delilah/Documents/Research/LLM/BIRD/dev/dev.json'
    else:
        raise ValueError('Check your benchmark name !')
    
    dev_json_data = load_json(dev_json_file)
     # Load models
    cluster_number_K_list = [5,10,15,20,25,30,40]
    for cluster_number_K in cluster_number_K_list:
        kmeans_loaded = joblib.load(f'model_file/kmeans_model_{benchmark}_K{cluster_number_K}.pkl')
        rf_model = joblib.load('model_file/random_forest_model_'+ benchmark +'.pkl')
        # predicted cluster ID
        feature_vectors = predict_feature_rf(rf_model, dev_json_data)
        predicted_clusters = kmeans_loaded.predict(feature_vectors)
        # model_family = "gpt"

        batches_csv_path   = f"batches_{benchmark}_dev_ns{n_s}_K{cluster_number_K}.csv"
        cluster_summary_csv = f"cluster_summary_{benchmark}_K{cluster_number_K}.csv"
        model_config_json   = "model_file/LLM_models_config.json"

        queries_cluster_dict: Dict[Tuple[str, str], int] = db_queries_clusterID_dict(predicted_clusters, dev_json_data) # TODO: provide real mapping

        acc_constraint_range, lat_constraint_range, acc_points, lat_points, price_points, model_list = return_constraint_range(benchmark, model_family)
        # acc_constraint_range = acc_points
        # lat_constraint_range = lat_points
        print(acc_constraint_range, lat_constraint_range, acc_points, lat_points, price_points, model_list)

        # filtered_model_configs = {key: model_configs[key] for key in model_list if key in model_configs}
     
        filename = f'eval_batching_K{cluster_number_K}_{benchmark}_{model_family}.csv'

        # 1) 固定的基础列
        base_cols = [
            'acc_constraint', 'lat_constraint',
            'solver_status', 'objective_value',
            'final_acc', 'final_latency'
        ]

        # 2) 固定的模型列顺序（建议直接用 return_constraint_range 返回的 model_list；
        #    若想按字母序固定，也可以：model_list = sorted(model_list)）
        model_cols = list(model_list)  # 保持稳定的既定顺序

        # 3) 最终固定表头
        fieldnames = base_cols + model_cols

        # 4) 只在最外层打开一次文件 & 创建 writer；首次写入 header
        file_exists = os.path.isfile(filename)


        with open(filename, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            ### ILP solver
            for constraint_acc_min in acc_constraint_range:
                for constraint_lat_max in lat_constraint_range: 
            # for idx in range(len(acc_constraint_range)):
            #         constraint_acc_min = acc_constraint_range[idx]
            #         constraint_lat_max = lat_constraint_range[idx]
                    start_time = time.time()
                    res = solve_llm_assignment_ilp(
                        batches_csv_path=batches_csv_path,
                        cluster_summary_csv=cluster_summary_csv,
                        model_config_json=model_config_json,
                        queries_cluster_dict=queries_cluster_dict,
                        n_s=n_s,
                        model_list=model_list,                 # use all models in config
                        sigma_A=constraint_acc_min,
                        sigma_T=constraint_lat_max,
                        accuracy_is_lower_bound=True,    # AvgAcc >= sigma_A
                        latency_is_upper_bound=True,     # AvgLat <= sigma_T
                        prompt_tok_counter=prompt_tok_counter,
                        msg=False,
                        benchmark=benchmark
                    )
                    if res['status'] == 'Optimal':
                        end_time = time.time()
                        runtime = end_time-start_time
                        avg_runtime += runtime
                        optimal_counter += 1

                    
                    # print(res['status'],res['avg_accuracy'],res['avg_latency'],res['total_cost'],res['totals'])
                    per_model = summarize_batches_per_model(res["x_assign"], res["y_assign"], model_list=model_list)
                    llm_batch_count = result_batches_per_model(per_model, order="total_desc")
                    # print(llm_batch_count)
                                #save data
                    row_data = {
                        'acc_constraint': constraint_acc_min,
                        'lat_constraint': constraint_lat_max,
                        'solver_status': res['status'],
                        'objective_value': res['total_cost'],
                        'final_acc': res['avg_accuracy'],
                        'final_latency': res['avg_latency']
                    }
                    # print(row_data)
                    # Add model counts.

                    # 6) 按固定顺序写模型列；不存在的一律填 0
                    #    为了 CSV 可读性，写成 "[full,res]" 的字符串；也可以写两个/三个独立列（见下方可选方案）
                    for m in model_cols:
                        stats = per_model.get(m, {'full': 0, 'res': 0, 'total': 0})
                        row_data[m] = f"[{stats['full']},{stats['res']}]"

                    # 7) 写入这一行
                    writer.writerow(row_data)

            # filename = 'eval_new_batching_'+benchmark+'_'+model_family+'.csv'
            # # If the file doesn't exist yet, write the header.
            # file_exists = os.path.isfile(filename)
            # with open(filename, mode='a', newline='') as csvfile:
            #     fieldnames = list(row_data.keys())
            #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #     if not file_exists:
            #         writer.writeheader()
            #     writer.writerow(row_data)

        # print('avg_runtime: ', round(avg_runtime/optimal_counter,5))

if __name__ == "__main__":
    main()  # Uncomment and fill in real data/mapping to run a quick test
    pass
