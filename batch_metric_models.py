"""
Batch-level metrics and cost/latency estimators for NL2SQL routing.

Provides:
- CSV loaders for batches and per-cluster metrics
- Token, latency, and price estimators for full-pure and residual batches
- Utility prompt/schema helpers
Note: function names are preserved.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from sqlite_utils import concat_schema
from read_select_json import load_json
from nl2sql_dataprepare import norm_key
import config


# -----------------------------
# Tokenizer / prompt / schema
# -----------------------------

def tokenizer_fn(prompt: str, tokenizer_model: str = "gpt-4o-mini") -> int:
    """Rough token count using tiktoken for the given model."""
    import tiktoken
    enc = tiktoken.encoding_for_model(tokenizer_model)
    return len(enc.encode(prompt))


def get_schema_fn(benchmark: str, db_name: str) -> str:
    """Read sqlite schema for a DB (relative paths via config)."""
    if benchmark == "spider":
        base = getattr(config, "SPIDER_DB_PATH", "spider/database/")
    elif benchmark == "BIRD":
        base = getattr(config, "BIRD_DEV_DB_PATH", "BIRD/dev/dev_databases/")
    else:
        raise ValueError("Unknown benchmark")
    db_sqlite_path = f"{base}{db_name}/{db_name}.sqlite"
    return concat_schema(db_sqlite_path)


def default_batch_prompt(db_name: str, queries: List[str], benchmark: str) -> str:
    """Simple batch prompt: schema + numbered questions, no extra prose."""
    table_schema = get_schema_fn(benchmark, db_name)
    questions = [f"{i+1}. Q: {q}" for i, q in enumerate(queries)]
    return (
        f"Schema: {table_schema}\n"
        + "\n".join(questions)
        + "\nProvide one single SQL query for each question, numbered like '1. SELECT ...;'. Do not include any additional text."
    )


# -----------------------------
# CSV / metrics loaders
# -----------------------------

def load_batches_index(batches_csv_path: str) -> Dict[str, dict]:
    """
    {batch_name -> batch_record}
    Expected columns: batch, DB, size, is_full, type, cluster_ids, NLqueries
    """
    idx: Dict[str, dict] = {}
    with open(batches_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rec = dict(r)
            rec["size"] = int(rec.get("size", 0))
            rec["is_full"] = str(rec.get("is_full", "0")).strip() in {"1", "True", "true"}

            cid_str = (rec.get("cluster_ids") or "").strip()
            rec["cluster_ids"] = [int(x) for x in cid_str.split("|") if x.strip().isdigit()]

            qs = (rec.get("NLqueries") or "").strip()
            parts = [p.strip() for p in qs.replace(" || ", "||").split("||")] if qs else []
            rec["NLqueries"] = [p for p in parts if p]

            idx[rec["batch"]] = rec
    return idx


def load_cluster_metric_df(cluster_res_file: str) -> pd.DataFrame:
    """Per-cluster summary (accuracy/latency/tokens/prices)."""
    df = pd.read_csv(cluster_res_file)
    if "cluster" in df.columns:
        df = df.set_index("cluster", drop=False)
    return df


def load_llm_models_config(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Internal lookups
# -----------------------------

def _lookup_col_any(df: pd.DataFrame, cluster_id: int, *colnames: str) -> float:
    for col in colnames:
        if col in df.columns:
            try:
                return float(df.loc[cluster_id, col])
            except Exception:
                return float(df[col].values[int(cluster_id)])
    raise KeyError(f"None of columns {colnames} found in cluster_metric_df.")


def _lookup_metric(cluster_metric_df: pd.DataFrame, prefix: str, model_name: str, cluster_id: int) -> float:
    col = f"{prefix}_{model_name}"
    return _lookup_col_any(cluster_metric_df, cluster_id, col)


def _count_clusters_in_batch(
    db_name: str,
    nlqueries: List[str],
    queries_cluster_dict: Dict[Tuple[str, str], int],
) -> Counter:
    ctr = Counter()
    for q in nlqueries:
        cid = int(queries_cluster_dict[norm_key(db_name, q)])
        ctr[cid] += 1
    return ctr


# -----------------------------
# Accuracy lookups
# -----------------------------

def get_acc_cluster(cluster_metric_df: pd.DataFrame, model_name: str, cluster_id: int) -> float:
    """Expected column: acc_<model>."""
    return _lookup_col_any(cluster_metric_df, cluster_id, f"acc_{model_name}")


def acc_on_demand(
    batch_rec: Mapping[str, Any],
    model_name: str,
    *,
    cluster_metric_df: pd.DataFrame,
    queries_cluster_dict: Dict[Tuple[str, str], int],
    n_s: int,
) -> float:
    """
    Average accuracy for a batch:
    - pure full (size==n_s and single cluster): acc(cluster, model)
    - otherwise: cluster-weighted average over NLqueries
    """
    btype = batch_rec.get("type", "mixed")
    is_full = bool(batch_rec.get("is_full", False))
    size = int(batch_rec.get("size", 0))
    cids = batch_rec.get("cluster_ids", [])
    if btype == "pure" and is_full and size == n_s and len(cids) == 1:
        return _lookup_metric(cluster_metric_df, "acc", model_name, int(cids[0]))

    db = batch_rec["DB"]
    nlqs: List[str] = batch_rec.get("NLqueries", [])
    ctr = _count_clusters_in_batch(db, nlqs, queries_cluster_dict)
    total = sum(ctr.values())
    if total == 0:
        if cids:
            w = 1.0 / len(cids)
            return sum(w * _lookup_metric(cluster_metric_df, "acc", model_name, cid) for cid in cids)
        return 0.0

    val = 0.0
    for cid, cnt in ctr.items():
        val += (cnt / total) * _lookup_metric(cluster_metric_df, "acc", model_name, cid)
    return val


# -----------------------------
# Latency estimators
# -----------------------------

def get_latency_params(cfg: Dict[str, dict], model_name: str) -> Tuple[float, float, float]:
    """Return (theta_in, theta_out, t_network) from config."""
    if model_name not in cfg:
        raise KeyError(f"model '{model_name}' not found in config")
    lp = cfg[model_name].get("latency_params", {})
    return float(lp["theta_in"]), float(lp["theta_out"]), float(lp["t_network"])


def get_out_tokens_per_query(cluster_metric_df: pd.DataFrame, model_name: str, cluster_id: int) -> float:
    """Expected column: outputTok_<model>."""
    return _lookup_col_any(cluster_metric_df, cluster_id, f"outputTok_{model_name}")


def get_tok_batch_prompt_avg(cluster_metric_df: pd.DataFrame, cluster_id: int) -> float:
    """Expected column: tok_batch_prompt_avg (or TokBatchPromptAvg)."""
    return _lookup_col_any(cluster_metric_df, cluster_id, "tok_batch_prompt_avg", "TokBatchPromptAvg")


def lat_full_pure_per_batch_cfg(
    *,
    model_configs: Dict[str, dict],
    model_name: str,
    cluster_id: int,
    n_s: int,
    cluster_metric_df: pd.DataFrame,
) -> float:
    """Per-query latency for a full pure batch (size n_s)."""
    theta_in, theta_out, t_net = get_latency_params(model_configs, model_name)
    tok_batch_avg = get_tok_batch_prompt_avg(cluster_metric_df, cluster_id)   # per-query avg input tokens
    tok_out_per_q = get_out_tokens_per_query(cluster_metric_df, model_name, cluster_id)
    tok_in = tok_batch_avg
    tok_out = n_s * tok_out_per_q
    return float((theta_in * tok_in + theta_out * tok_out + t_net) / n_s)


def lat_on_demand_residual_cfg(
    batch_rec: Mapping[str, Any],
    model_name: str,
    *,
    model_configs: Dict[str, dict],
    cluster_metric_df: pd.DataFrame,
    queries_cluster_dict: Dict[Tuple[str, str], int],
    prompt_tok_counter: Callable[[str, List[str], str], int],
    benchmark: str,
) -> float:
    """Per-query latency for a residual/mixed batch."""
    theta_in, theta_out, t_net = get_latency_params(model_configs, model_name)
    db = batch_rec["DB"]
    nlqs: List[str] = batch_rec.get("NLqueries", [])
    tok_in = int(prompt_tok_counter(db, nlqs, benchmark))

    ctr = _count_clusters_in_batch(db, nlqs, queries_cluster_dict)
    tok_out = 0.0
    for cid, n_i in ctr.items():
        tok_out += n_i * get_out_tokens_per_query(cluster_metric_df, model_name, cid)

    return float((theta_in * tok_in + theta_out * tok_out + t_net) / max(1, len(nlqs)))


# -----------------------------
# Cost estimators
# -----------------------------

def get_price_params(cfg: Dict[str, dict], model_name: str) -> Tuple[float, float]:
    """Return (IC, OC) input/output $ per token (usually per 1e6)."""
    if model_name not in cfg:
        raise KeyError(f"model '{model_name}' not found in config")
    m = cfg[model_name]
    if "price" in m and isinstance(m["price"], (list, tuple)) and len(m["price"]) >= 2:
        return float(m["price"][0]), float(m["price"][1])
    # Fallbacks
    for ic_key in ["ic", "IC", "input_price", "price_in"]:
        for oc_key in ["oc", "OC", "output_price", "price_out"]:
            if ic_key in m and oc_key in m:
                return float(m[ic_key]), float(m[oc_key])
    raise KeyError(f"price params for '{model_name}' not found")


def cost_full_pure_per_batch_cfg(
    *,
    model_configs: Dict[str, dict],
    model_name: str,
    cluster_id: int,
    n_s: int,
    cluster_metric_df: pd.DataFrame,
) -> float:
    """Total batch cost (input + output tokens)."""
    IC, OC = get_price_params(model_configs, model_name)
    tok_prompt_avg = get_tok_batch_prompt_avg(cluster_metric_df, cluster_id)   # per-query avg input tokens
    tok_out_per_q = get_out_tokens_per_query(cluster_metric_df, model_name, cluster_id)
    tok_in = tok_prompt_avg
    tok_out = n_s * tok_out_per_q
    return IC * tok_in + OC * tok_out


def cost_on_demand_residual_cfg(
    batch_rec: Mapping[str, Any],
    model_name: str,
    *,
    model_configs: Dict[str, dict],
    cluster_metric_df: pd.DataFrame,
    queries_cluster_dict: Dict[Tuple[str, str], int],
    prompt_tok_counter: Callable[[str, List[str], str], int],
    benchmark: str,
) -> float:
    """Total batch cost for a residual/mixed batch."""
    IC, OC = get_price_params(model_configs, model_name)
    db = batch_rec["DB"]
    nlqs: List[str] = batch_rec.get("NLqueries", [])
    tok_in = int(prompt_tok_counter(db, nlqs, benchmark))

    ctr = _count_clusters_in_batch(db, nlqs, queries_cluster_dict)
    tok_out = 0.0
    for cid, n_i in ctr.items():
        tok_out += n_i * get_out_tokens_per_query(cluster_metric_df, model_name, cid)

    return IC * tok_in + OC * tok_out


def cost_on_demand(
    batch_rec: Mapping[str, Any],
    model_name: str,
    *,
    n_s: int,
    cluster_metric_df: pd.DataFrame,
    queries_cluster_dict: Dict[Tuple[str, str], int],
    price_profile: Dict[str, Dict[str, float]],  # kept for compatibility; not used
    prompt_tok_counter: Callable[[str, List[str], str], int],
    benchmark: str,
) -> float:
    """
    Full-pure (size==n_s & single cluster) → fast path; otherwise residual path.
    Keeps the signature; loads model config internally for price/latency.
    """
    model_cfg_path = getattr(config, "MODEL_CONFIG_FILE", "model_file/LLM_models_config.json")
    model_configs = load_llm_models_config(model_cfg_path)

    btype = batch_rec.get("type", "mixed")
    is_full = bool(batch_rec.get("is_full", False))
    size = int(batch_rec.get("size", 0))
    cids = batch_rec.get("cluster_ids", [])

    if btype == "pure" and is_full and size == n_s and len(cids) == 1:
        return cost_full_pure_per_batch_cfg(
            model_configs=model_configs,
            model_name=model_name,
            cluster_id=int(cids[0]),
            n_s=n_s,
            cluster_metric_df=cluster_metric_df,
        )

    return cost_on_demand_residual_cfg(
        batch_rec,
        model_name,
        model_configs=model_configs,
        cluster_metric_df=cluster_metric_df,
        queries_cluster_dict=queries_cluster_dict,
        prompt_tok_counter=prompt_tok_counter,
        benchmark=benchmark,
    )