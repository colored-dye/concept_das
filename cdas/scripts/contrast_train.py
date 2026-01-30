"""
Interchange intervention training on contrastive data.
"""

import os
import datetime
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
import json
import pickle

import torch
import torch.distributed as dist
from transformers import (
    set_seed,
    PreTrainedTokenizer, PreTrainedModel,
)

from cdas.args.training_args import TrainingArgs
from cdas.constants import (
    CONFIG_FILE,
    METADATA_FILE,
    TRAIN_STATE_FILE,
    FACTOR_FILE,
)
from cdas.models import Model
import cdas.models as models_module
from cdas.utils.model_utils import load_hf_model_tokenizer
from cdas.utils.logger_utils import logger_setup
from cdas.utils.model_utils import get_prefix_length

from cdas.scripts.utils import (
    train_data_generator,
    load_metadata,
    partition_list,
    prepare_df_prompt_only,
)

import logging

# logger = logging.getLogger(__name__)


def load_state(dump_dir, rank):
    """
    Load the state from a file if it exists.
    """
    state_path = os.path.join(f"{dump_dir}", f"{TRAIN_STATE_FILE}_rank_{rank}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def save_state(dump_dir: Path, state, concept_metadata, rank):
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Save state
    state_path = os.path.join(dump_dir, f"{TRAIN_STATE_FILE}_rank_{rank}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)

    # Save metadata again
    metadata_path = os.path.join(dump_dir, f"rank_{rank}_{METADATA_FILE}")
    with open(metadata_path, "a") as f:
        f.write(json.dumps(concept_metadata) + "\n")


def train_and_save_model(
    args: TrainingArgs,
    hf_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    concept_id: int,
    concept_raw_df: DataFrame,
    model_name: str,
    device: torch.device,
    dump_dir: Path,
    rank: int,
    logger: logging.Logger,
):
    """
    Train and save a single model.

    Args:
        concept_raw_df: Concept-specific data.
    """
    model_args = args.models[model_name]
    model_class = getattr(models_module, model_name)
    benchmark_model: Model = model_class(
        model=hf_model,
        tokenizer=tokenizer,
        layer=args.layer,
        training_args=model_args,
        lm_model_name=args.model_name,
        device=device,
        seed=args.seed,
    )
    low_rank_dimension = model_args.low_rank_dimension

    benchmark_model.make_model(
        mode="train",
        embed_dim=hf_model.config.hidden_size,
        low_rank_dimension=low_rank_dimension,
        dtype=torch.bfloat16 if args.use_bf16 else None,
        concept_id=concept_id,
        intervention_positions=model_args.intervention_positions,
    )

    df = prepare_df_prompt_only(df=concept_raw_df, tokenizer=tokenizer)
    if args.output_length is not None:
        logger.warning(f"Max output length: {args.output_length}")
        for k, v in df.items():
            if k.endswith("output"):
                tokens = [tokenizer.encode(x)[:int(args.output_length)] for x in v]
                df[k] = [tokenizer.decode(x) for x in tokens]

    prefix_length = get_prefix_length(tokenizer)

    kwargs = dict(
        prefix_length=prefix_length,
        positions=model_args.intervention_positions,
        exclude_bos=model_args.exclude_bos,
        source_tokens=args.source_tokens,
        do_overwrite_vector=args.do_overwrite_vector,
    )
    benchmark_model.train(df, **kwargs)

    benchmark_model.save(dump_dir=dump_dir, model_name=f"rank_{rank}_{model_name}", **kwargs)
    logger.info(f"[Rank {rank}]: {model_name} saved to {dump_dir}")

    del benchmark_model
    torch.cuda.empty_cache()


def main():
    args = TrainingArgs(section="train", ignore_unknown=True)

    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=60000),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    logger = logging.getLogger(__name__)
    logger_handlers = logger_setup(logger=logger, level=logging.INFO, rank=rank)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    set_seed(args.seed + rank)

    hf_model, tokenizer = load_hf_model_tokenizer(
        model_name_or_path=args.model_name,
        device=device,
        dtype=torch.bfloat16,
        padding_side="right",
    )

    dump_dir = Path(args.dump_dir).resolve()
    # Train results go to dir
    dump_dir = dump_dir / "train"
    dump_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite_data_dir and Path(args.overwrite_data_dir).exists():
        data_dir = Path(args.overwrite_data_dir)
    else:
        data_dir = dump_dir / "generate"
    data_dir = data_dir.resolve()
    logger.warning(f"Data directory: {data_dir}")

    # Load metadata
    metadata_path = data_dir / METADATA_FILE
    metadata = load_metadata(metadata_path)

    # Load data
    df_generator = train_data_generator(data_dir)
    df_list = list(df_generator)
    logger.warning(f"Total number of concept df loaded: {len(df_list)}")
    if args.max_concepts:
        logger.warning(f"All ranks only processing {args.max_concepts} concepts")
        df_list = df_list[:args.max_concepts]

    df_list_per_rank = partition_list(df_list, world_size)
    my_df_list = df_list_per_rank[rank]

    # Load state
    state = load_state(dump_dir, rank)
    last_concept_id = state.get("last_concept_id", None) if state else None
    logger.warning(f"Rank {rank} last concept_id processed: {last_concept_id}")

    # Training loop
    for concept_id, concept_df in my_df_list:
        if last_concept_id is not None and concept_id <= last_concept_id:
            logger.warning(f"Rank {rank} skipping concept id {concept_id}.")
            continue

        logger.info(f"[Concept {concept_id}]: {metadata[concept_id]['concept']}")
        logger.info(str(concept_df.iloc[0].to_dict()))

        for model_name in sorted(args.models.keys()):
            train_and_save_model(
                args=args,
                hf_model=hf_model,
                tokenizer=tokenizer,
                concept_id=concept_id,
                concept_raw_df=concept_df,
                model_name=model_name,
                device=device,
                dump_dir=dump_dir,
                rank=rank,
                logger=logger,
            )

        # Finished concept; save state
        current_state = {'last_concept_id': concept_id}
        save_state(dump_dir, current_state, metadata[concept_id], rank)

    dist.barrier(device_ids=[torch.cuda.current_device()])

    # merge results
    if rank == 0:
        logger.warning("Rank 0 is merging results.")

        # Merging metadata
        metadata_entries = []
        metadata_files_existing = []
        for r in range(world_size):
            metadata_path = dump_dir / f"rank_{r}_{METADATA_FILE}"
            metadata_files_existing.append(metadata_path)
            try:
                with open(metadata_path, "r") as f:
                    for line in f:
                        metadata_entry = json.loads(line)
                        metadata_entries.append(metadata_entry)
            except Exception as e:
                logger.warning(f"Error reading file: {e}")
        metadata_path = os.path.join(dump_dir, METADATA_FILE)
        # delete per-rank files
        with open(metadata_path, "a") as f:
            for metadata_entry in metadata_entries:
                f.write(json.dumps(metadata_entry) + "\n")
        for f in metadata_files_existing:
            try:
                f.unlink()
                logger.warning(f"Deleted file {f.name}")
            except Exception as e:
                logger.error(f"Error deleting file {f.name}: {e}")

        # save config
        config = {
            "model_name": args.model_name,
            "layer": args.layer,
            "component": args.component,
        }
        config_path = dump_dir / CONFIG_FILE
        with open(config_path, 'w') as f:
            json.dump(config, f)
        logger.info(f"Config saved to {config_path}.")

        for model_name in sorted(args.models.keys()):
            weight_files = [dump_dir / f"rank_{r}_{model_name}_weight.pt" for r in range(world_size)]
            bias_files = [dump_dir / f"rank_{r}_{model_name}_bias.pt" for r in range(world_size)]

            # Check if files exist
            weight_files_existing = [f for f in weight_files if f.exists()]
            bias_files_existing = [f for f in bias_files if f.exists()]

            if not weight_files_existing or not bias_files_existing:
                logger.warning(f"No weight or bias files found for model {model_name}. Skipping.")
                continue

            # Load weights and biases
            weights = [torch.load(f, weights_only=True) for f in weight_files_existing]
            biases = [torch.load(f, weights_only=True) for f in bias_files_existing]

            # Concatenate weights and biases
            if isinstance(weights[0], dict):
                merged_weight = {}
                for key in weights[0].keys():
                    weight_tensors = [w[key] for w in weights]
                    merged_weight[key] = torch.cat(weight_tensors, dim=0)
            else:
                merged_weight = torch.cat(weights, dim=0)

            # Handle dictionary biases
            if isinstance(biases[0], dict):
                merged_bias = {}
                for key in biases[0].keys():
                    bias_tensors = [b[key] for b in biases]
                    merged_bias[key] = torch.cat(bias_tensors, dim=0)
            else:
                merged_bias = torch.cat(biases, dim=0)

            # Save merged weight and bias files
            weight_file = dump_dir / f"{model_name}_weight.pt"
            bias_file = dump_dir / f"{model_name}_bias.pt"

            # Also merge previous weights
            if Path(weight_file).exists():
                logger.warning("Merging previous weight.")
                previous_weight = torch.load(weight_file, weights_only=True)
                merged_weight = torch.cat([previous_weight, merged_weight])
            if Path(bias_file).exists():
                logger.warning("Merging previous bias.")
                previous_bias = torch.load(bias_file, weights_only=True)
                merged_bias = torch.cat([previous_bias, merged_bias])

            torch.save(merged_weight, weight_file)
            torch.save(merged_bias, bias_file)
            logger.warning(f"Saved merged weights and biases for model {model_name}")

            # Optionally delete per-rank files
            for f in weight_files_existing + bias_files_existing:
                try:
                    f.unlink()
                    logger.warning(f"Deleted file {f.name}")
                except Exception as e:
                    logger.error(f"Error deleting file {f.name}: {e}")

    logger.info("All finished")

    dist.destroy_process_group()
    for hdlr in logger_handlers:
        logger.removeHandler(hdlr)


if __name__ == "__main__":
    main()
