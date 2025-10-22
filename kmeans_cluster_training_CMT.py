# kmeans_cluster_training CMT.py
# training kmeans models and generate CMT (cluster metric table, as summary of kmean model result)

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from read_select_json import load_json
import config

warnings.filterwarnings("ignore")


def load_ground_truth_data(model_configs, benchmark):
    """Load per-model ground truth CSVs according to config."""
    model_ground_truths = {}
    for model_name, cfg in model_configs.items():
        folder = cfg["groundtruth_path"]
        fname = f"{model_name}_train_{benchmark}.csv"
        file_path = os.path.join(folder, fname)
        model_ground_truths[model_name] = pd.read_csv(file_path)
    return model_ground_truths


def add_model_data(store, model_ground_truths, idx, model_name, model_cfg):
    """Append accuracy/latency/token stats for a single model."""
    gt = model_ground_truths[model_name]
    acc_key = model_cfg["accuracy_key"]
    lat_key = model_cfg["latency_key"]
    in_tok_key = model_cfg["input_token_key"]
    out_tok_key = model_cfg["output_token_key"]

    store[acc_key].append(gt.loc[idx, "Accuracy"])

    # Convert ms to s only when value looks like ms (heuristic threshold 150).
    t = gt.loc[idx, "TimeCost"]
    store[lat_key].append(round(t / 1000, 3) if t > 150 else round(t, 3))

    store[in_tok_key].append(gt.loc[idx, "InputTokenSize"])
    store[out_tok_key].append(gt.loc[idx, "OutputTokenSize"])


def combine_input_data(store, input_json, model_ground_truths, model_configs, benchmark="spider"):
    """Build features (via RF) and attach per-model stats."""
    rf_path = os.path.join(config.KMEANS_MODEL_DIR, f"{config.RF_MODEL_BASENAME}_{benchmark}.pkl")
    rf_model = joblib.load(rf_path)

    for idx, item in enumerate(input_json):
        q = item.get("question")
        store["query"].append(q)

        # Use RF-predicted feature vector for clustering.
        predicted = rf_model.predict([q])[0]
        store["feature"].append(predicted)

        for model_name, model_cfg in model_configs.items():
            add_model_data(store, model_ground_truths, idx, model_name, model_cfg)

    return store


def create_dataframe(store, model_configs):
    """Pack features and per-model stats into a DataFrame."""
    data = {"feature": store["feature"]}
    for cfg in model_configs.values():
        data[cfg["accuracy_key"]] = store[cfg["accuracy_key"]]
        data[cfg["latency_key"]] = store[cfg["latency_key"]]
        data[cfg["input_token_key"]] = store[cfg["input_token_key"]]
        data[cfg["output_token_key"]] = store[cfg["output_token_key"]]
    return pd.DataFrame(data)


def train_kmeans_model(benchmark, store, model_configs, n_cluster=20, save_model=True, save_path="kmeans_model.pkl"):
    """Train KMeans on feature vectors."""
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    df = create_dataframe(store, model_configs)
    df["cluster"] = kmeans.fit_predict(np.vstack(df["feature"].values))

    if save_model:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(kmeans, save_path)

    return kmeans, df


def calculate_cluster_prices(cluster_summary, model_configs):
    """Compute price columns from average token usage."""
    for model_name, cfg in model_configs.items():
        in_tok_key = cfg["input_token_key"]
        out_tok_key = cfg["output_token_key"]
        price_key = f"price_{model_name}"
        in_cost, out_cost = cfg["price"]
        cluster_summary[price_key] = in_cost * cluster_summary[in_tok_key] + out_cost * cluster_summary[out_tok_key]
    return cluster_summary


def generate_cluster_summary(df, model_configs):
    """Aggregate cluster stats for each model."""
    # Start with a 'size' aggregation using the first accuracy key available.
    some_acc_key = next(iter(model_configs.values()))["accuracy_key"]
    agg_spec = {"cluster_ele_num": (some_acc_key, "size")}

    for cfg in model_configs.values():
        agg_spec[cfg["accuracy_key"]] = (cfg["accuracy_key"], "mean")
        agg_spec[cfg["latency_key"]] = (cfg["latency_key"], "mean")
        agg_spec[cfg["input_token_key"]] = (cfg["input_token_key"], "mean")
        agg_spec[cfg["output_token_key"]] = (cfg["output_token_key"], "mean")

    cluster_summary = df.groupby("cluster").agg(**agg_spec)
    cluster_summary = calculate_cluster_prices(cluster_summary, model_configs)

    # Column order: accuracies → latencies → prices → input tokens → output tokens → size
    ordered = []
    for cfg in model_configs.values():
        ordered.append(cfg["accuracy_key"])
    for cfg in model_configs.values():
        ordered.append(cfg["latency_key"])
    for model_name in model_configs.keys():
        ordered.append(f"price_{model_name}")
    for cfg in model_configs.values():
        ordered.append(cfg["input_token_key"])
    for cfg in model_configs.values():
        ordered.append(cfg["output_token_key"])
    ordered.append("cluster_ele_num")

    cluster_summary = cluster_summary[ordered]
    return cluster_summary


def save_rounded_cluster_summary(cluster_summary, file_path):
    """Save cluster summary with float columns rounded to 3 decimals."""
    out = cluster_summary.round(3)
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    out.to_csv(file_path, index=True)


def process_dataset(benchmark, json_file, model_configs, cluster_number_K, save_path):
    """Full pipeline for one dataset → (kmeans, df, summary)."""
    store = {"query": [], "feature": []}
    for cfg in model_configs.values():
        store[cfg["accuracy_key"]] = []
        store[cfg["latency_key"]] = []
        store[cfg["input_token_key"]] = []
        store[cfg["output_token_key"]] = []

    input_json = load_json(json_file)
    model_ground_truths = load_ground_truth_data(model_configs, benchmark)

    store = combine_input_data(store, input_json, model_ground_truths, model_configs, benchmark=benchmark)

    kmeans, df = train_kmeans_model(
        benchmark,
        store,
        model_configs,
        n_cluster=cluster_number_K,
        save_path=save_path,
    )

    from sklearn.metrics import silhouette_score, davies_bouldin_score

    feats = np.vstack(store["feature"])
    labels = kmeans.labels_
    print("Silhouette Score:", silhouette_score(feats, labels))
    print("Davies-Bouldin Index:", davies_bouldin_score(feats, labels))

    summary = generate_cluster_summary(df, model_configs)
    return kmeans, df, summary


def main():
    """Main runner for Spider (and optional BIRD)."""
    model_configs = load_json(config.MODEL_CONFIG_FILE)

    spider_file = config.SPIDER_TRAIN_FILE
    # bird_file = config.BIRD_TRAIN_FILE

    cluster_K_list = [15] 
    ## for finding the best K, uncomment the below 
    #cluster_K_list = [5, 10, 15, 20, 25, 30, 40]

    for K in cluster_K_list:
        km_path = os.path.join(config.KMEANS_MODEL_DIR, f"kmeans_model_spider_K{K}.pkl")
        _, _, spider_summary = process_dataset(
            benchmark="spider",
            json_file=spider_file,
            model_configs=model_configs,
            cluster_number_K=K,
            save_path=km_path,
        )
        save_rounded_cluster_summary(spider_summary, f"cluster_summary_spider_K{K}.csv")

    # To run BIRD later, uncomment and point to config.BIRD_TRAIN_FILE
    # km_path = os.path.join(config.KMEANS_MODEL_DIR, "kmeans_model_BIRD.pkl")
    # _, _, bird_summary = process_dataset(
    #     benchmark="BIRD",
    #     json_file=bird_file,
    #     model_configs=model_configs,
    #     cluster_number_K=20,
    #     save_path=km_path,
    # )
    # save_rounded_cluster_summary(bird_summary, "cluster_summary_BIRD.csv")


if __name__ == "__main__":
    main()