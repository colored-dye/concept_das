import os, argparse, yaml, json, glob, pickle, time, itertools, datetime
import shutil
import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

from axbench.utils.constants import * 
from axbench.utils.model_utils import get_prefix_length, get_suffix_length
from axbench.scripts.args.dataset_args import DatasetArgs
from axbench.scripts.args.training_args import TrainingArgs
from transformers import set_seed

# all supported methods
import axbench
from .benchmark_utils import (
    truthfulqa_eval,
)

import logging
import torch.distributed as dist
import sys

# Initialize the logger
logger = logging.getLogger(__name__)


BENCHMARK_FUNCTION_MAPPING = {
    "truthfulqa": truthfulqa_eval,
}


def benchmark_entry(
    inference_args: DatasetArgs,
    training_args: TrainingArgs,
    rank: int,
    device: torch.device,
    logger: logging.Logger,
):
    benchmark_name = inference_args.benchmark
    logger.warning(f"Benchmark: {benchmark_name}.")

    func = BENCHMARK_FUNCTION_MAPPING[benchmark_name]
    func(
        args=inference_args,
        training_args=training_args,
        device=device,
        logger=logger,
        rank=rank,
    )



def main():
    custom_args = [
        {
            'args': ['--benchmark'],
            'kwargs': {
                'type': str,
                'choices': ['truthfulqa'],
                'help': 'Benchmark name.'
            }
        },
        {
            'args': ['--hf_dataset_path'],
            'kwargs': {
                'type': str,
                'help': 'HF benchmark dataset path.'
            }
        },
    ]
    training_args = TrainingArgs(custom_args=custom_args, section="train", ignore_unknown=True)
    inference_args = DatasetArgs(custom_args=custom_args, section="inference", ignore_unknown=True)

    set_seed(inference_args.seed)

    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', 
                          timeout=datetime.timedelta(seconds=60000))

    # Get the rank and world_size from environment variables
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Set the device for this process
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # Configure the logger per rank
    logger.setLevel(logging.WARNING)  # Set the logging level as desired

    # Create a logging formatter that includes the rank
    formatter = logging.Formatter(
        fmt=f'%(asctime)s,%(msecs)03d %(levelname)-8s [Rank {rank}] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S'
    )

    # Create a console handler and set its formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(console_handler)

    benchmark_entry(
        inference_args=inference_args,
        training_args=training_args,
        rank=rank,
        device=device,
        logger=logger,
    )

    # Finalize the process group
    dist.destroy_process_group()

    # Remove handlers to prevent duplication if the script is run multiple times
    logger.removeHandler(console_handler)


if __name__ == "__main__":
    main()
