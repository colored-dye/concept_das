from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch

import logging

from axbench.scripts.args.dataset_args import DatasetArgs
from axbench.scripts.args.training_args import TrainingArgs
from axbench.utils.model_utils import get_prefix_length, get_suffix_length


def truthfulqa_eval(
    args: DatasetArgs,
    training_args: TrainingArgs,
    rank: int,
    device: torch.device,
    logger: logging.Logger,
):
    logger.warning(f"Loading TruthfulQA from {args.hf_dataset_path}.")
    dataset = load_dataset(args.hf_dataset_path, "multiple_choice", split="validation")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, model_max_length=1024, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        print("adding a special padding token...")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        need_resize = True
    else:
        need_resize = False

    prefix_length = get_prefix_length(tokenizer)
    logger.warning(f"Chat model prefix length: {prefix_length}")

    model_instance = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.use_bf16 else None,
        trust_remote_code=True,
    ).eval()
    model_instance.to(device)

    if need_resize:
        model_instance.resize_token_embeddings(len(tokenizer))

