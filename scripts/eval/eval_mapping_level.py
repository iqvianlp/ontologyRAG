import logging
import os, copy
from typing import Union
from pathlib import Path
from sklearn.metrics import confusion_matrix
from statistics import mean, stdev
import numpy as np
import pandas as pd

from scripts.utils.general import init_logger

logger = init_logger(__name__, logging.INFO)


def read_gold(gold_path: Union[str, Path]):
    df_gold = pd.read_excel(gold_path, sheet_name=0, dtype=str)
    return df_gold


def extract_level_from_result(llm_response: str):
    text_for_search = copy.copy(llm_response)
    while True:
        start = text_for_search.find("{")
        end = text_for_search.find("}")
        if start != -1 and end != -1:
            try:
                if "mapping_level" in eval(text_for_search[start:end+1]).keys():
                    level = eval(text_for_search[start:end+1])["mapping_level"]
                    return level
                else:
                    text_for_search = text_for_search[end+1:]
            except:
                break
        else:
            break
    text_for_search = copy.copy(llm_response).lower()
    start = text_for_search.find("mapping level is")
    if start != -1: 
        level = text_for_search[start+17:start+20].strip(":").strip("\n").strip(".").upper()
        return level

    
def read_prediction(prediction_path: Union[str, Path], processed_output_path: Union[str, Path]):
    filename = Path(prediction_path).stem
    df = pd.read_excel(prediction_path, dtype=str)
    level_dict = {}
    for idx, row in df.iloc[2:, 2:].iterrows():
        text = row[1]
        level = extract_level_from_result(text)
        level_dict[idx] = level
    df["level"] = pd.Series(level_dict)
    df.to_excel(f"{processed_output_path}/{filename}_processed.xlsx")
    return df


def calculate_accuracy(row: pd.Series):
    accuracy = 0
    if row["gold"] and row["pred"] and row["gold"]==row["pred"]:
        accuracy = 1
    return accuracy


def calculate_performance(df_gold: pd.DataFrame, df_pred: pd.DataFrame):
    df_merge = pd.merge(df_gold, df_pred, left_on="Label", right_on="Doc ID")
    df_merge.rename(columns={'output':'gold', 'level':'pred'}, inplace=True)
    
    # Calculate accuracy
    df_merge["accuracy"] = df_merge.apply(lambda x: calculate_accuracy(x), axis=1)
    accuracy = float(df_merge["accuracy"].mean()*100)
    logger.debug(df_merge.head())
    
    # Calculate precision
    cm = confusion_matrix(df_merge["gold"], df_merge["pred"], labels=["A", "B", "C"])
    precision_a = round(float(cm[0, 0]/np.sum(cm, axis=0)[0]*100), 2)
    precision_b = round(float(cm[1, 1]/np.sum(cm, axis=0)[1]*100), 2)
    precision_c = round(float(cm[2, 2]/np.sum(cm, axis=0)[2]*100), 2)

    return {"accuracy": accuracy, "precision_a": precision_a, "precision_b": precision_b, "precision_c": precision_c}


def evaluate(gold_path: Union[Path, str], raw_pred_path: Union[Path, str], processed_pred_output_dir: Union[Path, str]):
    logger.info(f"Gold file path: {gold_path}")
    logger.info(f"Prediction file or directory path: {raw_pred_path}")
    df_gold = read_gold(gold_path)
    if os.path.isfile(raw_pred_path):
        df_pred = read_prediction(raw_pred_path, processed_pred_output_dir)
        result = calculate_performance(df_gold, df_pred)
        return(result)

    elif os.path.isdir(raw_pred_path):
        result_accuracy_t02 = []
        result_accuracy_t06 = []
        result_accuracy_t10 = []
        result_precision_a = []
        result_precision_b = []
        result_precision_c = []
        for i, pred in enumerate(os.listdir(raw_pred_path)):
            if pred[0] != ".":
                pred_path_of_file = Path(raw_pred_path, pred)
                df_pred = read_prediction(pred_path_of_file, processed_pred_output_dir)
                result = calculate_performance(df_gold, df_pred)
                if "0.2" in pred:
                    result_accuracy_t02.append(result["accuracy"])
                elif "0.6" in pred:
                    result_accuracy_t06.append(result["accuracy"])
                elif "1.0" in pred:
                    result_accuracy_t10.append(result["accuracy"])
                result_precision_a.append(result["precision_a"])
                result_precision_b.append(result["precision_b"])
                result_precision_c.append(result["precision_c"])
                logger.debug(f"{pred}: {result}")
            else:
                continue 
        
        # Calculate average accuracy for each temperatures
        average_accuracy_t02 = round(mean(result_accuracy_t02), 2)
        average_accuracy_t06 = round(mean(result_accuracy_t06), 2)
        average_accuracy_t10 = round(mean(result_accuracy_t10), 2)
        std_accuracy_t02 = round(stdev(result_accuracy_t02), 2)
        std_accuracy_t06 = round(stdev(result_accuracy_t06), 2)
        std_accuracy_t10 = round(stdev(result_accuracy_t10), 2)

        # Calculate average precision for each level
        average_precision_a = round(mean(result_precision_a), 2)
        average_precision_b = round(mean(result_precision_b), 2)
        average_precision_c = round(mean(result_precision_c), 2)
        std_precision_a = round(stdev(result_precision_a), 2)
        std_precision_b = round(stdev(result_precision_b), 2)
        std_precision_c = round(stdev(result_precision_c), 2)

        return {"average accuracy (t0.2)": average_accuracy_t02, "standard deviation of accuracy (t0.2)": std_accuracy_t02, 
                "average accuracy (t0.6)": average_accuracy_t06, "standard deviation of accuracy (t0.6)": std_accuracy_t06, 
                "average accuracy (t1.0)": average_accuracy_t10, "standard deviation of accuracy (t1.0)": std_accuracy_t10,
                "average precision a": average_precision_a, "standard deviation of precision a": std_precision_a, 
                "average precision b": average_precision_b, "standard deviation of precision b": std_precision_b,
                "average precision c": average_precision_c, "standard deviation of precision c": std_precision_c}
    
    else:
        raise ValueError("Expecting prediction results to be either a file or a directory containing only "
                         "prediction result files. Please check the path!")
