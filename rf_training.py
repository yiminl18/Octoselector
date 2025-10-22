## rf_trianing.py 
## training random forest model for nl2sql dataset 

import re
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from extract_features_sql import extract_features  # keep your current extractor
from read_select_json import load_json
import config


def custom_preprocessor(text):
    """Basic text cleanup for TF-IDF."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)


def train_RF_model_old(
    json_data,
    custom_preprocessor,
    suffix='',
    RF_model_name=os.path.join("model_file", "random_forest_model")
):
    """Old version: builds features from SQL only."""
    data = {'query': [], 'feature': []}

    for item in json_data:
        question = item.get("question")
        SQL_query = item.get("query") or item.get("SQL")
        feature = extract_features(SQL_query)
        data['query'].append(question)
        data['feature'].append(feature)

    feature_name = [
        'join_table_num', 'groupBy_num', 'orderBy_bool', 'select_att_num',
        'select_distinct_bool', 'select_agg_bool', 'limit_bool',
        'where_num', 'nested_query_bool'
    ]
    y = pd.DataFrame(data['feature'], columns=feature_name)
    X = pd.Series(data['query'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    tfidf = TfidfVectorizer(
        stop_words=stopwords.words('english'),
        preprocessor=custom_preprocessor
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline = make_pipeline(tfidf, model)
    pipeline.fit(X_train, y_train)

    out_path = RF_model_name + suffix + '.pkl'
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(pipeline, f)

    return pipeline, X_test, y_test


def train_RF_model(
    json_data,
    custom_preprocessor,
    suffix='',
    RF_model_name=os.path.join("model_file", "random_forest_model")
):
    """New version: uses (SQL, NL question) for feature extractor."""
    data = {'query': [], 'feature': []}

    for item in json_data:
        question = item.get("question")
        SQL_query = item.get("query") or item.get("SQL")
        feature_vector = extract_features(SQL_query, question)
        data['query'].append(question)
        data['feature'].append(feature_vector)

    feature_names = [
        'num_tables', 'num_predicates', 'num_nested', 'num_sql_concepts',
        'num_groupby', 'bool_orderby', 'bool_limit', 'bool_distinct',
        'num_agg', 'num_projected_attributes',
        'num_joins', 'NLquery_len_nomalized', 'num_logical_operators',
        'num_having_conditions', 'bool_uses_like', 'num_involved_columns'
    ]

    y = pd.DataFrame(data['feature'], columns=feature_names)
    X = pd.Series(data['query'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer(
        stop_words=stopwords.words('english'),
        preprocessor=custom_preprocessor
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline = make_pipeline(tfidf, model)
    pipeline.fit(X_train, y_train)

    out_path = RF_model_name + suffix + '.pkl'
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(pipeline, f)

    return pipeline, X_test, y_test


if __name__ == "__main__":

    # Select dataset
    benchmark = 'spider'  # 'BIRD' or 'spider'

    if benchmark == 'BIRD':
        json_file = config.BIRD_TRAIN_FILE
    elif benchmark == 'spider':
        json_file = config.SPIDER_TRAIN_FILE
    else:
        raise ValueError("Invalid benchmark name")

    json_data = load_json(json_file)
    pipeline, X_test, y_test = train_RF_model(
        json_data,
        custom_preprocessor=custom_preprocessor,
        suffix='_' + benchmark,
        RF_model_name=os.path.join(config.RF_MODEL_DIR, "random_forest_model")
    )

    # Evaluation
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    y_pred = pipeline.predict(X_test)

    print(benchmark)
    print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))