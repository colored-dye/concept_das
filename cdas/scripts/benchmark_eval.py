"""Multiple-choice benchmark evaluation."""


import os, datetime
import shutil
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.distributed as dist
from transformers import set_seed
from pathlib import Path

from cdas.args import DatasetArgs, TrainingArgs
from cdas.constants import CONFIG_FILE
from cdas.utils.model_utils import load_hf_model_tokenizer
from cdas.utils.logger_utils import logger_setup

from cdas.scripts.utils import load_config
from cdas.scripts.benchmark_utils import (
    truthfulqa_eval,
    mmlu_eval,
    tiny_arc_eval,
    tiny_mmlu_eval,
)

import logging


BENCHMARK_FUNCTION_MAPPING = {
    "truthfulqa": truthfulqa_eval,
    "mmlu": mmlu_eval,
    "tiny_arc": tiny_arc_eval,
    "tiny_mmlu": tiny_mmlu_eval,
}


def main():
    custom_args = [
        {
            "args": ["--benchmark"],
            "kwargs": {
                "type": str,
                "choices": ["truthfulqa", "mmlu", "tiny_arc", "tiny_mmlu"],
                "help": "Benchmark name.",
                "required": True,
            },
        },
        {
            "args": ["--hf_dataset_path"],
            "kwargs": {
                "type": str,
                "help": "HF benchmark dataset path.",
                "required": True,
            },
        },
        {
            "args": ["--num_shots"],
            "kwargs": {
                "type": int,
                "required": True,
            },
        },
    ]
    training_args = TrainingArgs(custom_args=custom_args, section="train", ignore_unknown=True)
    inference_args = DatasetArgs(custom_args=custom_args, section="inference", ignore_unknown=True)

    # Initialize the process group
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=60000)
    )

    # Get the rank and world_size from environment variables
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set the device for this process
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    inference_args.seed = int(inference_args.seed)
    set_seed(inference_args.seed)

    logger = logging.getLogger(__name__)
    logger_handlers = logger_setup(logger=logger, level=logging.INFO, rank=rank)

    # Paths
    dump_dir = Path(inference_args.dump_dir).resolve()
    train_dir = dump_dir / "train"
    overwrite_inference_dump_dir = (
        Path(inference_args.overwrite_inference_dump_dir)
        if inference_args.overwrite_inference_dump_dir is not None
        else dump_dir / "evaluate"
    )
    logger.warning(f"Dump dir: {overwrite_inference_dump_dir}.")

    if inference_args.overwrite_data_dir and Path(inference_args.overwrite_data_dir).exists():
        data_dir = Path(inference_args.overwrite_data_dir)
    else:
        data_dir = dump_dir / "generate"
    data_dir = data_dir.resolve()
    logger.warning(f"Data dir: {data_dir}.")

    config = load_config(train_dir)
    if config is None:
        raise ValueError(f"Config file `{CONFIG_FILE}` not found in {train_dir}.")
    layer = config["layer"]

    logger.warning(f"Benchmark: {inference_args.benchmark}.")
    logger.warning(f"Seed: {inference_args.seed}")
    logger.warning(f"Num of shots: {inference_args.num_shots}")

    hf_model, tokenizer = load_hf_model_tokenizer(
        model_name_or_path=inference_args.model_name,
        device=device,
        padding_side="left",
    )

    func = BENCHMARK_FUNCTION_MAPPING[inference_args.benchmark]
    func(
        args=inference_args,
        training_args=training_args,
        device=device,
        logger=logger,
        rank=rank,
        hf_model=hf_model,
        tokenizer=tokenizer,
        layer=layer,
        train_dir=train_dir,
        dump_dir=overwrite_inference_dump_dir,
    )

    # Finalize the process group
    dist.destroy_process_group()

    for hdlr in logger_handlers:
        logger.removeHandler(hdlr)


if __name__ == "__main__":
    main()
