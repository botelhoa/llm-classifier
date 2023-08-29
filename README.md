# Self-Hosting LLMs for Unsupervised Data Labeling

<!---
[![lint](https://github.com/CybersecurityForDemocracy/llm-classifier/actions/workflows/lint.yml/badge.svg)](https://github.com/CybersecurityForDemocracy/llm-classifier/actions/workflows/lint.yml)
[![test](https://github.com/CybersecurityForDemocracy/llm-classifier/actions/workflows/test.yml/badge.svg)](https://github.com/CybersecurityForDemocracy/llm-classifier/actions/workflows/test.yml)
[![Repo Size](https://img.shields.io/github/repo-size/CybersecurityForDemocracy/llm-classifie/badge.svg)](https://img.shields.io/github/repo-size/CybersecurityForDemocracy/llm-classifier)
-->
[![Website](https://img.shields.io/badge/Website-Cybersecurity%20For%20Democracy-blueviolet)](https://cybersecurityfordemocracy.org/)
[![Medium Blog](https://img.shields.io/badge/Medium-Cybersecurity%20For%20Democracy-blueviolet)](https://medium.com/p/9b5614b4f46d)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

The accompanying code to produce the results in [](https://medium.com/p/9b5614b4f46d). Compares various LLMs in their ability to do zero- and few-shot text classification.

## Table of Contents


- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)


## Installation

After cloning this repo and installing [Poetry](https://python-poetry.org/docs/#installation), run `poetry install`, then, `poe accelerate` (because `poetry` struggles with `torch`)

To download a model run: `python utils/download.py hugging-face-model`

## Usage

Reproduce the experiment results by running the two bash files with `bash pretrain_comparison.sh` and `bash two_stage`. 
This will produce a consolidated file of the perforamnce metrics by run in `data/results.csv`.

To do an individual pass of data through an LLM, use the following command:

`run -f path/to/data -t task -m hugging-face-model`

Optionally, you can pass the flags:

- `examples` / `-e`: Whether to use zero or few shot learning.
- `new_tokens` \ `-n`: The number of tokens for the model to generate.


New tasks can be added using the format in `utils/prompt.py`. Model config is housed in `utils/model.py` and can be customized there.


To train a self-supervised model on many examples, run:

`train -f path/to/data -t task`

Optionally, you can pass the flags:

- `-confident` / `-c`: Enables Confident Learning which filters out likely mislabeled data



## Results

### Accuracy

`evaluate`

| model | examples | partisanship |	topic | all	|
| --- | --- | --- |	--- | --- |
| gpt-3.5-turbo | few |	**0.46** | **0.62** | **0.54** |
| gpt-3.5-turbo| zero |	0.31 | 0.64 | 0.47 |
| TheBloke/Wizard-Vicuna-13B-Uncensored-HF | |zero | 0.34 | 0.20 | 0.27 |
| nomic-ai/gpt4all-13b-snoozy |	zero | 0.35 | 0.12 | 0.23 |
| nomic-ai/gpt4all-13b-snoozy | few | 0.13 | 0.33 | 0.23 |
| NousResearch/Nous-Hermes-13b | zero | 0.32 | 0.09 | 0.20 |
| TheBloke/Wizard-Vicuna-13B-Uncensored-HF | few | 0.20 | 0.15 | 0.17 |
| TheBloke/stable-vicuna-13B-HF | zero | 0.13 | 0.16 | 0.14 |
| NousResearch/Nous-Hermes-13b | few | 0.17 | 0.08 | 0.13 |
| TheBloke/stable-vicuna-13B-HF | few | 0.19 | 0.06 | 0.13 |
| nghuyong_ernie_2.0-base-en | many | - | 0.63 | - |
| nghuyong_ernie_2.0-base-en  | many_confident | - | 0.61 | - |
| distilbert-base-uncased | many_confident | - | 0.44 | - |
| distilbert-base-uncased | many | - | 0.39 | - |

### Speed
All models run on a single Nvidia A100

| Model | Runtime (per) |
| --- | ---| 
| distilbert-base-uncased | **0.0153** |
| nghuyong_ernie_2.0-base-en  | 0.0155 |
| nomic-ai/gpt4all-13b-snoozy | 0.7451 |
| TheBloke/Wizard-Vicuna-13B-Uncensored-HF | 0.7485 |
| TheBloke/stable-vicuna-13B-HF | 0.7572 |
| NousResearch/Nous-Hermes-13b | 0.9320 |
