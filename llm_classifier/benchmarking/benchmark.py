import os
import time
import json
import torch
import click
from statistics import mean

from llm_classifier.utils.model import LLMCLassifier, CONFIG, SmallClassifier

CONFIG["max_new_tokens"] = 1


def benchmark(prompt, model_name, is_small = bool, num_iter: int=100):
    start = time.monotonic()
    start_cpu = time.process_time()

    
    if is_small:
        model = SmallClassifier(
            config=CONFIG, 
            num_labels=2,
            )

    else:
        model = LLMCLassifier(
                    model_path= model_name, 
                    config=CONFIG, 
                    candidate_labels=["a", "b"],
                    )

    marginal_runtimes = []
    for _ in range(num_iter):
        marginal_start = time.time()
        _ = model.run(prompt)
        marginal_runtimes.append(time.time()-marginal_start)
    
    stop = time.monotonic()
    stop_cpu = time.process_time()

    return {
        "num_iter": num_iter,
        "wall_time": time.strftime("%H:%M:%S", time.gmtime(stop - start)),
        #"wall_time_per": (stop-start) / num_iter,
        "processor_time": time.strftime("%H:%M:%S", time.gmtime(stop_cpu - start_cpu)),
        #"processor_time_per": (stop_cpu-start_cpu) / num_iter,
        "seconds_per_token": mean(marginal_runtimes) / CONFIG["max_new_tokens"],
        "using_gpu": torch.cuda.is_available(),
        }


@click.command()
@click.option("-model_name", "-m", type=str, default=None, help='Name of model being used') 
@click.option("-small", "-s", is_flag=True, help='Whether to use SmallClassifier') 
def cli(model_name, small):

    """
    fil-profile run benchmark.py --no-browser
    """

    # Use arbitrary prompt length of 500 for tests

    prompt = "Hello, what is your name?"
    prompt = prompt * int(500 / len(prompt))
    times = benchmark(prompt, model_name, small)

    os.makedirs("benchmarking/results", exist_ok=True)
    file_name = model_name.split("models/")[-1].replace("/", "_")
    with open(f'benchmarking/results/{file_name}_runtime.json', 'w+') as file:
        file.write(json.dumps(times))