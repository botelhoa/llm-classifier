
import click
import os
import csv
import time
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download

from llm_classifier.utils.model import LLMCLassifier, CONFIG, SmallClassifier, SUPPORTED_MODELS, ConfidentLearner
from llm_classifier.utils.data_handling import PostDataset, create_text_field
from llm_classifier.utils.prompt import Prompter
from llm_classifier.utils.tasks import task_registry
from llm_classifier.utils.data_handling import task_maps
from llm_classifier.utils.evaluate import clean_prediction




@click.command()
@click.option("-model", "-m", type=str, required=True, help='Name of model in HuggingFace Hub.')
def download(model):
    save_path = f"{os.getcwd()}/models/{model.replace('/', '_')}"
    snapshot_download(repo_id=model, local_dir=save_path)



@click.command()
@click.option("-file", "-f", type=str, help='Local path to file') 
@click.option("-task", "-t", type=str, help='Name of task family') 
@click.option("-examples", "-e", type=click.Choice(['zero', 'few']), help='Whether to use zero or few shot learning') 
@click.option("-new_tokens", "-n", type=int, default=None, help='Number of tokens for model to generate') 
@click.option("-model_name", "-m", type=str, default=None, help='Name of model being used')
@click.option("-openai_key", "-k", type=str, default=None, help='OpenAI API key') 
@click.option("-save_name", "-s", type=str, default=None, help='Name of file to save predictions to') 
def run(file, task, examples, new_tokens, model_name, openai_key, save_name):

    print(f"Starting {model_name} on {task} with {examples} examples...")

    start = time.time()

    supported_models = SUPPORTED_MODELS
    supported_models = [model.replace("/", "_") for model in supported_models]+supported_models
    assert model_name.split("models/")[-1] in supported_models, f"Please use one of the following supported models: {supported_models}"

    assert os.path.isfile(os.path.join(os.getcwd(), file)) == True, "Please use a valid file path"

    if new_tokens:
        CONFIG["max_new_tokens"] = new_tokens

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    dataset = PostDataset(
            file_path=file, 
            task = task,
            )
    
    dataloader = DataLoader(
            dataset=dataset, 
            batch_size=1, 
            shuffle=True,
            )
    
    prompter = Prompter(
        model=model_name,
        task=task, 
        examples=examples, 
        max_length=CONFIG["max_seq_len"],
        )

    model = LLMCLassifier(
                model_path= model_name, 
                config=CONFIG, 
                candidate_labels= task_registry[task].categories,
                )

    model_name = model_name.split("models/")[-1]
    save_name = save_name if save_name else f"{task}_{examples}_predictions.csv"
    os.makedirs(f"data/{task.split('_')[0]}/{model_name}", exist_ok=True)
    with open(f'data/{task.split("_")[0]}/{model_name}/{save_name}','w+') as file:
        
        dict_writer = csv.DictWriter(file, fieldnames=["id", "prediction"])
        dict_writer.writeheader() 

        for row_dict in tqdm(dataloader):
            text = row_dict["text"][0]
            input_prompt = prompter.truncate(prompter.make_prompt(text), text)
            output = model.run(input_prompt)
            prompter.conv.clear_messages()
            dict_writer.writerow({"id": row_dict["id"][0], "prediction": output})

    print(f"Finished running {model_name} on {task} with {examples} examples after {time.time()-start} seconds")


@click.command()
@click.option("-file", "-f", type=str, help='Local path to file containing training examples with or without synthetic labels') 
@click.option("-task", "-t", type=str, help='Name of task family') 
@click.option("-confident", "-c", is_flag=True, help='Whether to use Confident Learning') 
@click.option("-openai_key", "-k", type=str, default=None, help='OpenAI API key') 
def train(file, task, confident, openai_key):

    training_file = file
    
    # Select best model
    assert os.path.isfile("data/results.csv") == True, "Please compute the metrics first in order to chose best model"
    results = pd.read_csv("data/results.csv", index_col=0).sort_values(by=task, ascending=False).head(1)
    labeler_name = results["model"].values[0]
    label_file = f'data/{task.split("_")[0]}/{labeler_name}/topic_training_labels.csv'

    # Generate labels
    if os.path.isfile(label_file) == False:
        run(["-f", file, "-t", task, "-e", results["examples"].values[0], "-n", 10, "-m", labeler_name, "-k", openai_key, "-s", "topic_training_labels.csv"]) 


    # Load data
    df = pd.read_parquet(training_file)
    df["final_text"] = df.apply(lambda row: create_text_field(row), axis=1)

    # Add labels
    missing_val = task_maps[task]["miscellaneous"] if task == "topic" else task_maps[task]["moderate"] if task == "partisanship" else 0
    df = df.merge(pd.read_csv(label_file), on="id").rename(columns = {"prediction": "label"})
    df["label"] = df["label"].map(lambda x: clean_prediction(x, task, missing_val))
    
    if confident == True:

        cl = ConfidentLearner()
        df = cl.run(df)
        del cl

    start = time.time()

    model = SmallClassifier(config=CONFIG, num_labels=len(df["label"].unique()))
    model.train(df) 

    # Evaluate
    model = SmallClassifier(
        config=CONFIG, 
        num_labels=len(df["label"].unique()),
        trained_path=model.trainer.state.best_model_checkpoint,
        )
    eval_df = pd.read_parquet(f"data/{task}/{task}_sample.parquet")
    eval_df["prediction"] = model.run(eval_df["final_text"].values.tolist())
    model.save(eval_df, task=task, confident=confident)
    

    runtime = int((time.time() - start) / 60)
    print(f"Training complete after {runtime} minutes")


