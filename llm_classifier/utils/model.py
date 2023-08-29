
import time
import torch
import math
import openai
import os
import backoff
import numpy as np
import pandas as pd

from typing import List
from openai.error import RateLimitError
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForSeq2SeqLM, LlamaTokenizer, pipeline,
    TrainingArguments, Trainer, EarlyStoppingCallback, ProgressCallback, DataCollatorWithPadding, pipeline, get_linear_schedule_with_warmup,
    StoppingCriteria, StoppingCriteriaList)
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from datasets import Dataset
from InstructorEmbedding import INSTRUCTOR
from cleanlab.classification import CleanLearning
from sklearn.linear_model import LogisticRegression

CONFIG = {
        'max_new_tokens': 30,
        'do_sample': True,
        'temperature': 0.01,
        'top_p': 0.01,
        'typical_p': 1,
        'repetition_penalty': 1.18,
        'top_k': 40,
        'min_length': 1,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'seed': 22,
        'add_bos_token': True,
        'max_seq_len': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': [],
        "model_type": "distilbert-base-uncased", #"nghuyong/ernie-2.0-base-en",
        "learning_rate": 2e-5,
        #"batch_size": ,
        "num_epochs": 100,
        "warmup_ratio": 0.1,
        "gradient_accumulation_steps": 1,
    }


SUPPORTED_MODELS = ["NousResearch/Nous-Hermes-13b", "mosaicml/mpt-7b-instruct", "TheBloke/stable-vicuna-13B-HF", "TheBloke/Wizard-Vicuna-13B-Uncensored-HF", 
"gpt-3.5-turbo", "nomic-ai/gpt4all-13b-snoozy", "distilbert-base-uncased", "nghuyong_ernie-2.0-base-en"]



class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer) -> None:
      self.stop_token_ids = [tokenizer.convert_tokens_to_ids(tokenizer.eos_token)]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
    

class OpenAIModel:

    def __init__(self, model_path: str, config: dict) -> None:
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.config = config
        self.model_path = model_path


    @backoff.on_exception(backoff.expo, RateLimitError)
    def run(self, prompt: List[dict]):  

        completion = openai.ChatCompletion.create(
            model=self.model_path,
            messages=prompt,
            max_tokens=self.config["max_new_tokens"],
            temperature=self.config["temperature"],
            top_p=1,
            n=1,
            stream=False,
            presence_penalty=0,
            frequency_penalty=0,
            stop=["\n",],
        )
      
        return completion["choices"][0]["message"]["content"].strip()



class HFModel:
    def __init__(self, model_path: str, config: dict, candidate_labels: list=None) -> None:
        
        self.config = config
        self.model_path = model_path

        if "mpt" in self.model_path:
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", model_input_names=['input_ids', 'attention_mask'])
        else:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path, model_input_names=['input_ids', 'attention_mask'])

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})


        try:
            self.pipe_type = "zero-shot-classification"
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map='auto',
                load_in_8bit=True,
                num_labels = len(candidate_labels),
            )

            self.pipe = pipeline(
                    task=self.pipe_type, 
                    model=self.base_model, 
                    tokenizer=self.tokenizer, 
                    candidate_labels=candidate_labels,
                    multi_label=False,
                )

        except:
            try:
                self.pipe_type = "text-generation"

                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    device_map='auto',
                    load_in_8bit=True,
                    max_seq_len=self.config["max_seq_len"],
                )

                self.pipe = pipeline(
                            task=self.pipe_type,
                            model=self.base_model, 
                            tokenizer=self.tokenizer, 
                            max_new_tokens=self.config["max_new_tokens"],
                            temperature=self.config["temperature"],
                            top_p=self.config["top_p"],
                            repetition_penalty=self.config["repetition_penalty"],
                            return_full_text=False,
                            stopping_criteria=StoppingCriteriaList([StopOnTokens(self.tokenizer)]),
                        )

            except:
                self.pipe_type = "text2text-generation"
                self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    device_map='auto',
                    load_in_8bit=True,
                    max_seq_len=self.config["max_seq_len"],
                )

                self.pipe = pipeline(
                        task=self.pipe_type,
                        model=self.base_model, 
                        tokenizer=self.tokenizer, 
                        max_new_tokens=self.config["max_new_tokens"],
                        temperature=self.config["temperature"],
                        top_p=self.config["top_p"],
                        repetition_penalty=self.config["repetition_penalty"],
                        return_full_text=False,
                        stopping_criteria=StoppingCriteriaList([StopOnTokens(self.tokenizer)]),
                    )

        print("Device: ", self.base_model.device)
        print("Pipe type: ", self.pipe_type)


    def run(self, prompt: str): 

        out = self.pipe(prompt)

        if self.pipe_type == "text-generation" or self.pipe_type == "text2text-generation":
            pred = out[0]["generated_text"]#.split("\n")[0].strip() # 'Topic name: ', '### Response: '
            return pred

            return pred

        elif self.pipe_type == "zero-shot-classification":
            pred = out["labels"][np.argmax(out["scores"])]
            return pred


class LLMCLassifier:

    def __init__(self, model_path: str, config: dict, candidate_labels: list=None) -> None:

        self.config = config
        self.model_path = model_path

        if "gpt" in self.model_path and "4all" not in self.model_path:
            self.pipe = OpenAIModel(model_path=self.model_path, config=self.config)

        else:
            self.pipe = HFModel(model_path=self.model_path, config=self.config, candidate_labels=candidate_labels)

    
    def run(self, text) -> str:
        return self.pipe.run(text)



class ConfidentLearner:
    def __init__(self) -> None:

        self.embedder = INSTRUCTOR('hkunlp/instructor-large')
        self.instruction = "Represent the Facebook post:"
        self.cl = CleanLearning(clf=LogisticRegression(max_iter=400), cv_n_folds=5)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        os.environ["TOKENIZERS_PARALLELISM"] = "True"

        embeddings = self.embedder.encode([[self.instruction, i] for i in df["final_text"]])
        issue_idx = self.cl.find_label_issues(X=embeddings, labels=df["label"]).query("is_label_issue == True").index
        df = df.drop(issue_idx).reset_index(drop=True)
        return df



class SmallClassifier:
    def __init__(self, config: dict, num_labels: int, label_col: str ="label", text_col: str="final_text", id_col: str="id", trained_path: str=None) -> None:
        super(SmallClassifier, self).__init__()

        self.config = config
        self.label_col = label_col
        self.text_col = text_col
        self.id_col = id_col


        model_path = trained_path if trained_path else self.config["model_type"]
    
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_type"], fast=True)
        self.pipe = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, truncation=True, device=self.model.device, max_length = 512)


    def train(self, df: pd.DataFrame) -> None:

        train, eval = train_test_split(df, test_size=0.2, stratify=df[self.label_col], random_state=22)
        train_dataset = Dataset.from_pandas(train[[self.id_col, self.text_col, self.label_col]]).map(self._tokenize, batched=True)
        eval_dataset = Dataset.from_pandas(eval[[self.id_col, self.text_col, self.label_col]]).map(self._tokenize, batched=True)

        training_args = TrainingArguments(
            output_dir="models",
            logging_dir="models/training_logs",
            learning_rate=float(self.config["learning_rate"]),
            auto_find_batch_size = True, 
            num_train_epochs=int(self.config["num_epochs"]),
            lr_scheduler_type="linear",
            warmup_ratio=float(self.config["warmup_ratio"]),
            weight_decay=0.01,
            evaluation_strategy="steps",
            save_strategy="steps",
            gradient_accumulation_steps = int(self.config["gradient_accumulation_steps"]),
            eval_steps = 100,
            save_steps = 100,
            save_total_limit = 1,
            logging_steps = 100,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            seed=22,
            fp16= False,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self._compute_metrics,
            callbacks = [
                EarlyStoppingCallback(
                                    early_stopping_patience=2, 
                                    early_stopping_threshold=0.0,
                                    ),
                ProgressCallback,
                ],
            optimizers = self._load_optimizer(train_len=len(train_dataset)),
        )

        self.trainer.train()


    def run(self, data):
        preds = self.pipe(data)
        preds = [int(pred["label"].split("_")[-1]) for pred in preds]
        return preds


    def save(self, df: pd.DataFrame, task: str, confident: bool):

        save_path = f"data/{task}/{self.config['model_type'].replace('/', '_')}"
        save_name = f"{task}_many_confident_predictions.csv" if confident else f"{task}_many_predictions.csv"
        os.makedirs(save_path, exist_ok=True)
        df[[self.id_col, "prediction"]].to_csv(os.path.join(save_path, save_name))


    def _compute_metrics(self, outs,): 
        preds = np.argmax(outs.predictions, axis=-1)
        prec, rec, f1, _ = precision_recall_fscore_support(outs.label_ids, preds, beta=1.0, average="weighted")
        return {"f1": f1, "precision": prec, "recall": rec,}


    def _tokenize(self, input):
        return self.tokenizer(
                    input[self.text_col], 
                    add_special_tokens=True,
                    return_tensors='pt',
                    return_attention_mask = False,
                    max_length = 512,
                    truncation = True,
                    padding = "max_length",
                    )


    def _load_optimizer(self, train_len: int, no_decay= ["bias", "LayerNorm.weight"], lr_decay: float=0.95,):
    
        optimizer_grouped_parameters = []
    
        try:

            for n, p in self.model.named_parameters():
                param_dict = {
                    "params": p,
                    "weight_decay": 0.01,
                    "learning_rate": self.config["learning_rate"],
                }

                if "layer" in n:
                    layer_n = int(n.split(".")[3])
                    param_dict["learning_rate"] = lr_decay**layer_n * self.config["learning_rate"]

                for nd in no_decay:
                    if nd in n:
                        param_dict["weight_decay"] = 0.0
                        
                optimizer_grouped_parameters.append(param_dict)


        except: 
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        
        total_steps = math.ceil(int(self.config["num_epochs"]) * max(train_len // int(self.config["gradient_accumulation_steps"]), 1))
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=float(self.config["learning_rate"]), eps=1e-8, amsgrad=True)
        scheduler = get_linear_schedule_with_warmup(
                                                    optimizer=optimizer, 
                                                    num_warmup_steps=float(self.config["warmup_ratio"])*total_steps, 
                                                    num_training_steps=total_steps,
                                                    )
        
        return (optimizer, scheduler)