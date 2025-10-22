# nl2sql_llm_groundtruth.py

from connect_llm import GPT_prompt_api, Gemini_api, Claude_api
from sqlite_utils import execute_sqlite_query, concat_schema
from read_select_json import load_json
import time
import re, sqlparse
import csv
import tiktoken
from datetime import date
from collections import defaultdict
import os

import OctoSelector.config as config


def evaluate_SQL_answer(SQLquery_GroundTruth, SQLquery_gpt, db_path):
    """Execute and compare LLM SQL vs. ground truth on SQLite."""
    dfsql_res_gpt = execute_sqlite_query(db_path, SQLquery_gpt)
    dfsql_res_GroundTruth = execute_sqlite_query(db_path=db_path, query=SQLquery_GroundTruth)
    correct_answer = False
    if compare_dataframes(dfsql_res_gpt, dfsql_res_GroundTruth):
        correct_answer = True
    return correct_answer


def generate_nl2sql_prompt_batch(batch, table_schemas, include_schema=True):
    """Build a batched NL2SQL prompt."""
    schema = table_schemas if include_schema else ""
    questions = [f"{i+1}. Q: {q['question']}" for i, q in enumerate(batch)]
    prompt = (
        f"Schema: {schema}\n"
        + "\n".join(questions)
        + "\nProvide one single SQL query for each question, numbered like '1. SELECT ...;'. Do not include any additional text."
    )
    return prompt


def cluster_questions_results(json_data, cluster_num=-1):
    """Group questions by db_id; optionally filter by cluster index file."""
    db_question_groups = defaultdict(list)
    if cluster_num != -1:
        cluster_json_idx_file = "json_data_cluster_" + str(cluster_num) + ".json"
        cluster_json_idx = load_json(cluster_json_idx_file)
        for item_idx in cluster_json_idx:
            item = json_data[item_idx]
            db_id = item.get("db_id")
            db_question_groups[db_id].append(item)
    else:
        for item in json_data:
            db_id = item.get("db_id")
            db_question_groups[db_id].append(item)
    return db_question_groups


def compare_dataframes(df1, df2):
    """Compare two DataFrames by sorting on their first column."""
    try:
        first_column_namedf1 = df1.columns[0]
        first_column_namedf2 = df2.columns[0]
        sorted_df1 = df1.sort_values(by=first_column_namedf1)
        sorted_df2 = df2.sort_values(by=first_column_namedf2)
        are_values_equal_1_2 = sorted_df1.values.tolist() == sorted_df2.values.tolist()
    except Exception:
        are_values_equal_1_2 = False
    return are_values_equal_1_2


def save_csv(save_path, save_tuple: list):
    """Append a row to CSV (creates parent folder if needed)."""
    folder = os.path.dirname(save_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(save_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows([save_tuple])


def generate_nl2sql_prompt(table_schemas, query_NL, evidence=""):
    """Build a single NL2SQL prompt."""
    prompt = "Given the table schema as follows:  "
    prompt += table_schemas
    if len(evidence) > 1:
        prompt += " Given the evidence as additional knowledge: "
        prompt += evidence
    prompt += " Task: convert the question to SQL, only return the SQL query:  "
    prompt = prompt + "question: " + query_NL + ";"
    return prompt


def stop_execution():
    """Raise a timeout error."""
    raise TimeoutError("The operation timed out!")


def get_SQLquery_gpt(prompt, model_name):
    """Query the selected LLM and return (response, latency_seconds)."""
    start_time_gpt = time.time()

    if "gemini" in model_name:
        llm_response = Gemini_api(prompt, model_name=model_name)
    elif "claude" in model_name:
        llm_response = Claude_api(prompt, model_name=model_name)
    else:
        llm_response = GPT_prompt_api(prompt, model_name=model_name)

    end_time_gpt = time.time()
    llm_response_time = end_time_gpt - start_time_gpt
    return llm_response, llm_response_time


def parse_gpt_response(response):
    """Extract SQL queries from an LLM response."""
    if "```sql" in response or "```" in response:
        clean_response = response.replace("```sql", "").replace("```", "").strip()
    else:
        clean_response = response.strip()
    clean_response = re.sub(r"\d+\.\s*", "", clean_response)
    parsed_queries = sqlparse.split(clean_response)
    formatted_queries = [sqlparse.format(query, reindent=True).strip() for query in parsed_queries]
    return formatted_queries


def eval_gpt_nl2sql(prompt, SQLquery_GroundTruth, model_name, tokenizer, db_path, iter_query_num=1):
    """Evaluate model on a single NL2SQL prompt (possibly multiple trials)."""
    avg_gpt_response_time = 0
    iter_correct_counter = 0
    output_tokensize = 0

    for _ in range(iter_query_num):
        gpt_response, gpt_response_time = get_SQLquery_gpt(prompt, model_name)
        if model_name == "gpt-4o":
            time.sleep(6)
        if gpt_response is None:
            gpt_response = ""
        avg_gpt_response_time += gpt_response_time
        output_tokensize += len(tokenizer.encode(gpt_response))
        try:
            SQLquery_gpt = parse_gpt_response(gpt_response)[0]
        except Exception:
            SQLquery_gpt = ""
        if evaluate_SQL_answer(SQLquery_GroundTruth=SQLquery_GroundTruth, SQLquery_gpt=SQLquery_gpt, db_path=db_path):
            iter_correct_counter += 1

    input_tokensize_counter = len(tokenizer.encode(prompt)) * iter_query_num
    output_tokensize_counter = int(output_tokensize)

    avg_gpt_accuracy = iter_correct_counter / iter_query_num
    avg_gpt_response_time /= iter_query_num

    eval_gpt_tuple = [
        round(avg_gpt_accuracy, 3),
        iter_correct_counter,
        round(avg_gpt_response_time, 3),
        output_tokensize_counter,
        input_tokensize_counter,
        gpt_response,
    ]
    return eval_gpt_tuple


def process_nl2sql_queries(json_data, model_name, tokenizer, file_name, iter_query_num=1, db_path=None):
    """Run evaluation for each NL question individually and append results to CSV."""
    if db_path is None:
        db_path = config.SPIDER_DB_PATH
    for item in json_data:
        db_id = item.get("db_id")
        NLquery = item.get("question")
        db_sqlite_path = os.path.join(db_path, db_id, f"{db_id}.sqlite")
        table_schemas = concat_schema(db_sqlite_path)
        evidence = item.get("evidence") or ""
        prompt = generate_nl2sql_prompt(table_schemas=table_schemas, query_NL=NLquery, evidence=evidence)
        SQLquery_GroundTruth = item.get("query") or item.get("SQL")
        eval_gpt_tuple = eval_gpt_nl2sql(
            prompt,
            SQLquery_GroundTruth,
            model_name,
            tokenizer,
            db_sqlite_path,
            iter_query_num=iter_query_num,
        )
        eval_gpt_tuple.append(NLquery)
        save_csv(file_name, eval_gpt_tuple)


def selected_clustered_data(json_data, cluster_num):
    """Subset json_data by cluster index file if cluster_num is specified."""
    if cluster_num != -1:
        cluster_json_idx_file = "json_data_cluster_" + str(cluster_num) + ".json"
        cluster_json_idx = load_json(cluster_json_idx_file)
        selected_data = [json_data[idx] for idx in cluster_json_idx]
    else:
        selected_data = json_data
    return selected_data


def naming_file(
    model_name, cluster_num=-1, batch_flag=False, version_name=str(date.today()), suffix=""
):
    """Build output filename pattern."""
    version_name = ""
    if batch_flag is False:
        if cluster_num != -1:
            file_name = model_name + "_cluster" + str(cluster_num) + "_" + version_name + suffix + ".csv"
        else:
            file_name = model_name + "_" + version_name + suffix + ".csv"
    else:
        if cluster_num != -1:
            file_name = model_name + "_batch_cluster" + str(cluster_num) + "_" + version_name + suffix + ".csv"
        else:
            file_name = model_name + "_batch_" + version_name + suffix + ".csv"
    return file_name


def main(
    data_file,
    model_name="gpt-4o-mini",
    cluster_num=-1,
    batch_flag=False,
    maxnum_batch_questions=50,
    suffix="",
    iter_query_num=1,
    db_path=None,
):
    """Main runner. Keeps original behavior with cleaner defaults."""
    if iter_query_num < 1:
        iter_query_num = 1

    original_data = load_json(data_file)[:2400]
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

    if batch_flag is False:
        csv_title = ["Accuracy", "", "TimeCost", "OutputTokenSize", "InputTokenSize", "OutputQuery", "InputQuery"]
        file_name = naming_file(model_name, cluster_num, batch_flag=batch_flag, suffix=suffix)
        save_csv(save_path=file_name, save_tuple=csv_title)
        selected_data = selected_clustered_data(original_data, cluster_num)
        process_nl2sql_queries(
            selected_data,
            model_name,
            tokenizer,
            file_name,
            iter_query_num=iter_query_num,
            db_path=db_path,
        )
    else:
        csv_title = ["AccuracyBatch", "LengthBatch", "TimeCostBatch", "OutputTokenSize", "InputTokenSize"]
        file_name = naming_file(model_name, cluster_num, batch_flag=batch_flag, suffix=suffix)
        save_csv(save_path=file_name, save_tuple=csv_title)
        db_question_groups = cluster_questions_results(original_data, cluster_num)

        for db_id, questions in db_question_groups.items():
            db_sqlite_path = os.path.join(db_path or "", db_id, f"{db_id}.sqlite")
            table_schemas = concat_schema(db_sqlite_path)

            for i in range(0, len(questions), maxnum_batch_questions):
                batch_true_counter = 0
                batch = questions[i : i + maxnum_batch_questions]
                input_tokensize_counter = 0
                output_tokensize_counter = 0
                avg_gpt_response_time = 0
                try:
                    for _ in range(iter_query_num):
                        prompt = generate_nl2sql_prompt_batch(batch, table_schemas)
                        input_tokensize_counter += len(tokenizer.encode(prompt))
                        gpt_response, gpt_response_time = get_SQLquery_gpt(prompt, model_name)
                        avg_gpt_response_time += gpt_response_time
                        output_tokensize_counter += len(tokenizer.encode(gpt_response))
                        gpt_queries = parse_gpt_response(gpt_response)

                        if benchmark == "spider":
                            gt_queries = [q["query"] for _, q in enumerate(batch)]
                        elif benchmark == "BIRD":
                            gt_queries = [q["SQL"] for _, q in enumerate(batch)]
                        else:
                            raise ValueError("check benchmark name!!")

                        for question_idx in range(len(batch)):
                            query_GPT = gpt_queries[question_idx]
                            query_GroundTruth = gt_queries[question_idx]
                            if evaluate_SQL_answer(
                                SQLquery_GroundTruth=query_GroundTruth,
                                SQLquery_gpt=query_GPT,
                                db_path=db_sqlite_path,
                            ):
                                batch_true_counter += 1
                except Exception:
                    continue

                input_tokensize_counter /= iter_query_num
                output_tokensize_counter /= iter_query_num
                batch_true_counter /= iter_query_num
                avg_gpt_response_time /= iter_query_num

                target_tuple = [
                    round(batch_true_counter, 3),
                    len(batch),
                    round(avg_gpt_response_time, 3),
                    output_tokensize_counter,
                    input_tokensize_counter,
                ]
                save_csv(file_name, target_tuple)


def print_num_question_inDB(spider_data_file=None):
    """Print number of questions per DB for Spider dataset."""
    if spider_data_file is None:
        spider_data_file = config.SPIDER_TRAIN_FILE
    spider_data = load_json(spider_data_file)
    dict_db = cluster_questions_results(spider_data)
    for k in dict_db:
        print(len(dict_db[k]))


if __name__ == "__main__":
    # Choose dataset/benchmark here
    benchmark = "spider"  # "spider" or "BIRD"
    db_part = "train"     # "train" or "dev" dataset

    if benchmark == "spider" and db_part == "train":
        db_path = config.SPIDER_DB_PATH
        json_data_file = config.SPIDER_TRAIN_FILE
    elif benchmark == "spider" and db_part == "dev":
        db_path = config.SPIDER_DB_PATH
        json_data_file = config.SPIDER_DEV_FILE
    elif benchmark == "BIRD" and db_part == "train":
        db_path = config.BIRD_TRAIN_DB_PATH
        json_data_file = config.BIRD_TRAIN_FILE
    elif benchmark == "BIRD" and db_part == "dev":
        db_path = config.BIRD_DEV_DB_PATH
        json_data_file = config.BIRD_DEV_FILE
    else:
        raise ValueError("Invalid benchmark/db_part combination")


    batch_size_list = [8,10]
    model_name = "gpt-4o-mini"
    ## batch_flag == False ,max_batch_questions = 1: running ground truth LLM invocation with single query 
    ## batch_flag == True: running ground truth LLM invocation with batched queries 
    for max_batch_questions in batch_size_list:
        print("max_batch_questions: ", max_batch_questions)
        main(
            json_data_file,
            model_name,
            cluster_num=-1, ## when no cluster_num specified , cluster_num = -1
            batch_flag=True,
            maxnum_batch_questions=max_batch_questions,
            suffix=f"{db_part}_{benchmark}_ns{max_batch_questions}", ## saving filename suffix 
            iter_query_num=1,
            db_path=db_path,
        )