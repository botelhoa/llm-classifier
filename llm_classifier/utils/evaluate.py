
import pandas as pd
import os

from sklearn.metrics import precision_recall_fscore_support
from llm_classifier.utils.data_handling import task_maps
from llm_classifier.utils.model import SUPPORTED_MODELS
from llm_classifier.utils.tasks import task_registry

data_dir = os.path.join(os.getcwd().split("llm_classifier")[0], "llm_classifier/data")

def clean_prediction(pred, task: str, missing_val: int):

    if pd.isna(pred) == False:
        return task_maps[task].get(pred.split(": ")[-1], missing_val)
    else:
        return missing_val

def evaluate(tasks: list=task_registry.keys(), models: list=SUPPORTED_MODELS, average: str="weighted"):

    models.remove("mosaicml/mpt-7b-instruct") # remove because of poor performance

    results = []

    for task in tasks:
        
        df = pd.read_parquet(f"{data_dir}/{task}/{task}_sample.parquet")

        id_col = "id" if task=="topic" else "account_id"

        df = df.rename(columns={"bias": "labels", "trustworthy": "labels", "topic": "labels", "final_text": "text"})
        df["labels_numeric"] = df["labels"].map(task_maps[task])

        for model in models:

            model = model.replace("/", "_")

            path = f"{data_dir}/{task}/{model}"
            
            try:
                assert os.path.isdir(path) == True
            except:
                continue
            
            for file in os.listdir(path):

                if "labels" in file:
                    continue
                
                temp_df = pd.read_csv(os.path.join(path, file))
                temp_df["id"] = temp_df["id"].astype(str)

                
                if "distil" in model or "ernie" in model:
                    temp_df["prediction_numeric"] = temp_df["prediction"]

                else:
                    missing_val = task_maps[task]["miscellaneous"] if task == "topic" else task_maps[task]["moderate"] if task == "partisanship" else 0
                    temp_df["prediction_numeric"] = temp_df["prediction"].map(lambda x: clean_prediction(x, task, missing_val))


                temp_df = df.merge(temp_df, left_on=id_col, right_on="id")

                prec, rec, f1, _ = precision_recall_fscore_support(temp_df["labels_numeric"], temp_df["prediction_numeric"], average=average, zero_division=0)

                results.append({
                    "model": model,
                    "examples": file.split(f"{task}_")[1].split("_predictions.csv")[0],
                    "task": task,
                    #f"precision ({average})": prec,
                    #f"recall ({average})": rec,
                    f"f1 ({average})": f1,
                })

    results = pd.DataFrame(results).pivot(index=["model", "examples"], columns='task')['f1 (weighted)']
    results["all"] = (results["partisanship"] + results["topic"]) / 2
    results = results.sort_values(by="all", ascending=False).reset_index().round(2)
    results.to_csv(f"{data_dir}/results.csv")
            