# config.py
# Relative default paths (edit to match your project structure)

## SPIDER benchmark config
SPIDER_DB_PATH = "spider/database/"
SPIDER_TRAIN_FILE = "spider/train_spider.json"
SPIDER_DEV_FILE = "spider/dev.json"

SPIDER_RF_MODEL_BASENAME = "random_forest_model_spider"
SPIDER_KMEAN_MODEL_BASENAME = f"kmeans_model_spider_K{15}"
SPIDER_CMT_BASENAME = f"cluster_summary_spider_K{15}"

## BIRD benchmark config
BIRD_TRAIN_DB_PATH = "BIRD/train/train_databases/"
BIRD_TRAIN_FILE = "BIRD/train/train.json"
BIRD_DEV_DB_PATH = "BIRD/dev/dev_databases/"
BIRD_DEV_FILE = "BIRD/dev/dev.json"

BIRD_RF_MODEL_BASENAME = "random_forest_model_BIRD"
BIRD_KMEAN_MODEL_BASENAME = f"kmeans_model_BIRD_K{15}"
BIRD_CMT_BASENAME = f"cluster_summary_BIRD_K{15}"

## trained random forest model dir
RF_MODEL_DIR = "model_file/"
KMEAN_MODEL_DIR = "model_file/"

## Replaced by your API key 
OPENAI_API_KEY = "sk-xxx"
GOOGLE_API_KEY =  "xxx"
ANTHROPIC_API_KEY = "xxx"
