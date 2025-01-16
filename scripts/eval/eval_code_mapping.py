import logging
import os
from typing import Union
from pathlib import Path
from statistics import mean, stdev
import pandas as pd

from scripts.utils.general import init_logger

logger = init_logger(__name__, logging.INFO)


def read_gold(gold_path: Union[str, Path]):
    df_gold = pd.read_excel(gold_path, sheet_name=0, index_col=0, dtype=str)
    df_gold.sort_values(by="mapsFromCode", inplace=True)
    return df_gold


def extract_code_from_result(llm_response: str, llm_model_name: str):
    if "flan" in llm_model_name:
        response_text = llm_response.replace("<pad>", "").replace("</s>", "").replace("ICD10CM:", "").strip()
        code_list = [i.strip() for i in response_text.split(",")]
    elif "llama" in llm_model_name:
        code_list = []
        start = llm_response.find("{")
        end = llm_response.find("}")
        if start != -1 and end != -1:
            try:
                code_list = eval(llm_response[start:end+1])["ICD10CM"].split(", ")
            except:
                pass
    else:
        try:
            result_dict = eval(llm_response)
            code_list = list(result_dict.values())
            code_list = [i.strip() for i in code_list]
        except SyntaxError:
            code_list = []
    return code_list


def read_prediction(prediction_path: Union[str, Path], model_name: str):
    if "flan" in model_name:
        df_prediction = pd.read_excel(prediction_path, dtype=str)
    else:
        df_prediction = pd.read_excel(prediction_path, sheet_name=0, index_col=0, skiprows=[0, 1, 2], dtype=str)
    df_prediction.columns = ["mapsFromCode", "raw_prediction"]
    df_prediction["prediction"] = df_prediction.apply(lambda x: extract_code_from_result(x["raw_prediction"],
                                                                                         model_name),
                                                      axis=1)
    df_prediction.sort_values(by="mapsFromCode", inplace=True)
    return df_prediction


def calculate_single_code_accuracy(row: pd.Series):
    if row["prediction"]:
        gold = [i.strip() for i in row["mapsToCode"].split(",")]
        pred = row["prediction"]

        # remove the . specifying sub-category level in prediction to align with gold
        pred = [item.replace(".", "") for item in pred]

        len_gold = len(gold)
        true_pred = 0
        for code in gold:
            if code in pred:
                true_pred += 1
        accuracy = true_pred / len_gold * 100
    else:
        accuracy = 0
    return accuracy


def calculate_overall_accuracy(df_gold: pd.DataFrame, df_pred: pd.DataFrame):
    df_merge = pd.merge(df_gold, df_pred, on="mapsFromCode", how="outer")
    df_merge = df_merge.where(pd.notnull(df_merge), None)
    df_merge["accuracy"] = df_merge.apply(lambda x: calculate_single_code_accuracy(x), axis=1)
    logger.debug(df_merge.head())
    overall_accuracy = df_merge["accuracy"].mean()
    return overall_accuracy


def evaluate(gold_path: Union[Path, str], pred_path: Union[Path, str], model_name: str):
    logger.info(f"Gold file path: {gold_path}")
    logger.info(f"Prediction file or directory path: {pred_path}")
    assert os.path.isfile(gold_path), "Expecting gold dataset to be a file, please check the path!"
    df_gold = read_gold(gold_path)
    if os.path.isfile(pred_path):
        df_pred = read_prediction(pred_path, model_name)
        result = calculate_overall_accuracy(df_gold, df_pred)
        return {"average accuracy": result}
    elif os.path.isdir(pred_path):
        result_list = []
        for i, pred in enumerate(os.listdir(pred_path)):
            if pred[0] != ".":
                pred_path_of_file = Path(pred_path, pred)
                df_pred = read_prediction(pred_path_of_file, model_name)
                result = calculate_overall_accuracy(df_gold, df_pred)
                result_list.append(result)
                logger.debug(f"Average accuracy for {pred}: {result}")
            else:
                continue
        mean_average_accuracy = round(mean(result_list), 2)
        std = round(stdev(result_list), 2)

        return {"mean average accuracy": mean_average_accuracy, "standard deviation": std}
    else:
        raise ValueError("Expecting prediction results to be either a file or a directory containing only "
                         "prediction result files. Please check the path!")

