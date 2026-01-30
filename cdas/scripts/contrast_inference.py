"""
Predict latents, gather steering factors, or generate with steering.
"""

import os
import copy
import datetime
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import pickle
from tqdm import tqdm, trange
import numpy as np
import json
from typing import Literal

import torch
import torch.distributed as dist
from transformers import (
    set_seed,
    PreTrainedTokenizer, PreTrainedModel,
)

from cdas.args import TrainingArgs, DatasetArgs
from cdas.constants import (
    CONFIG_FILE,
    METADATA_FILE,
    CONTRAST_TRAIN_DATA_FILE,
    FACTOR_FILE,
    LATENT_FILE,
    STEERING_FILE,
    INFERENCE_STATE_FILE,
)
import cdas.models as models_module
from cdas.models import Model, CDASVector
from cdas.utils.model_utils import load_hf_model_tokenizer
from cdas.utils.logger_utils import logger_setup
from cdas.utils.model_utils import get_prefix_length

from cdas.scripts.utils import (
    load_metadata,
    partition_concept_ids,
    prepare_df_prompt_only,
    prepare_df_pair,
    save_df,
    separate_contrast_df,
)

import logging

# logger = logging.getLogger(__name__)


def load_config(config_path):
    """
    Load metadata from a JSON lines file.
    """
    if not os.path.exists(Path(config_path) / CONFIG_FILE):
        return None
    with open(Path(config_path) / CONFIG_FILE) as f:
        d = json.load(f)
    return d


def load_state(dump_dir: Path, mode, rank):
    """
    Load the state from a file if it exists.
    """
    state_path = dump_dir / f"{mode}_{INFERENCE_STATE_FILE}_rank_{rank}"
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def save_state(dump_dir, state, mode, rank):
    if not isinstance(dump_dir, Path):
        dump_dir = Path(dump_dir)
        
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save state
    state_path = os.path.join(dump_dir, f"{mode}_{INFERENCE_STATE_FILE}_rank_{rank}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)


def infer_latent(
    hf_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    args: DatasetArgs,
    training_args: TrainingArgs,
    rank: int,
    world_size: int,
    device: torch.device | str,
    logger: logging.Logger,
    train_dir: Path,
    dump_dir: Path,
):
    """Predict latents for concept detection."""
    config = load_config(train_dir)
    if config is None:
        raise ValueError(f"Config file `{CONFIG_FILE}` not found in {train_dir}.")
    layer = config["layer"]

    prefix_length = get_prefix_length(tokenizer)

    # Load metadata from trained dir
    metadata_path = train_dir / METADATA_FILE
    metadata = load_metadata(metadata_path)
    concept_ids = list(metadata.keys())
    logger.warning(f"Total number of concepts loaded: {len(concept_ids)}")

    # Partition concept_ids among ranks sequentially
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    all_df = pd.read_parquet(args.contrast_test_data_path)
    num_of_examples = args.latent_num_of_examples
    if num_of_examples is not None:
        logger.warning(f"Only select first {num_of_examples} examples.")
        num_of_examples = int(num_of_examples)

    for concept_id in my_concept_ids:
        logger.warning(f"[Concept {concept_id}]: {metadata[concept_id]['concept']}")
        concept_df = all_df[all_df['concept_id'] == concept_id]
        if num_of_examples is not None:
            concept_df = concept_df.iloc[:num_of_examples]
        positive_df, negative_df = separate_contrast_df(concept_df)
        df = pd.concat([positive_df, negative_df])
        df = prepare_df_pair(df=df, tokenizer=tokenizer, is_chat_model=True)

        for model_name in args.models:
            model_class = getattr(models_module, model_name)
            logger.info(f"Loading {model_class} on {device}.")
            benchmark_model: Model = model_class(
                model=hf_model,
                tokenizer=tokenizer,
                layer=layer,
                low_rank_dimension=1,
                device=device
            )
            benchmark_model.load(dump_dir=train_dir, mode="latent")
            benchmark_model.to(device=device)
            if hasattr(benchmark_model, 'ax'):
                benchmark_model.ax.eval()

            def do_latents(df):
                results = benchmark_model.predict_latent(
                    examples=df,
                    prefix_length=prefix_length,
                    batch_size=args.latent_batch_size,
                )
                for k, v in results.items():
                    if k == "tokens":
                        if "tokens" not in df:
                            df["tokens"] = v  # for tokens, they are global
                        else:
                            continue
                    else:
                        df[f"{model_name}_{k}"] = v
                return df
            df = do_latents(df)

            del benchmark_model
            torch.cuda.empty_cache()
        save_df(dump_dir=dump_dir, partition='latent', current_df=df, rank=rank)

    dist.barrier()

    # merge results
    if rank == 0:
        logger.warning("Rank 0 is merging results.")
        # Merge per-rank results
        all_parquet_files = list(dump_dir.glob(f"rank_*_{LATENT_FILE}"))
        # Parse filenames to extract rank
        import re
        pattern = re.compile(r'rank_(\d+)_latent_data\.parquet')

        file_info_list = []
        for parquet_file in all_parquet_files:
            match = pattern.match(parquet_file.name)
            if match:
                rank_str = match.group(1)
                rank_int = int(rank_str)
                file_info_list.append({
                    'rank': rank_int,
                    'file': parquet_file
                })
            else:
                logger.warning(f"Filename {parquet_file.name} does not match the expected pattern.")

        # Sort the file_info_list by rank
        file_info_list.sort(key=lambda x: x['rank'])

        # Read and concatenate dataframes
        dfs = []

        for info in file_info_list:
            df = pd.read_parquet(info['file'])
            dfs.append(df)
        if len(dfs) > 0:
            combined_df = pd.concat(dfs, ignore_index=True)

            # Also merge previous results
            previous_file = Path(dump_dir) / LATENT_FILE
            if previous_file.exists():
                old_df = pd.read_parquet(previous_file)
                combined_df = pd.concat(
                    [old_df, combined_df], ignore_index=True, axis=0
                )

            combined_df.to_parquet(dump_dir / LATENT_FILE, engine='pyarrow')
            logger.warning(f"Saved combined latent inference results to {dump_dir / LATENT_FILE}")
        else:
            logger.warning("No results to merge.")

        # Optionally, delete per-rank files
        for info in file_info_list:
            os.remove(info['file'])
            logger.warning(f"Deleted {info['file']}")


def infer_steering(
    steering_direction: Literal["positive", "negative", "two_way"],
    hf_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    args: DatasetArgs,
    training_args: TrainingArgs,
    rank: int,
    world_size: int,
    device: torch.device | str,
    logger: logging.Logger,
    train_dir: Path,
    overwrite_inference_dump_dir: Path,
    layer: int,
):
    """Generate with model steering."""
    use_saved_factors = True
    # Load steering factors
    factor_path = (train_dir / FACTOR_FILE).resolve()
    if not factor_path.exists():
        use_saved_factors = False
        logger.warning(f"Factor file `{factor_path}` does not exist.")
        # raise FileNotFoundError(f"Factor file `{factor_path}` does not exist.")
    else:
        factor_df = pd.read_parquet(factor_path)

    # Load metadata from trained dir
    metadata_path = train_dir / METADATA_FILE
    metadata = load_metadata(metadata_path)
    concept_ids = list(metadata.keys())
    logger.warning(f"Total number of concepts loaded: {len(concept_ids)}")

    # Partition concept_ids among ranks sequentially
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    # Load steering data
    logger.warning(f"Using contrastive test data: {args.contrast_test_data_path}.")
    all_df = pd.read_parquet(args.contrast_test_data_path)
    num_of_examples = args.steering_num_of_examples
    if num_of_examples is not None:
        logger.warning(f"Only select first {num_of_examples} examples.")
        num_of_examples = int(num_of_examples)

    prefix_length = get_prefix_length(tokenizer)

    for concept_id in my_concept_ids:
        logger.warning(f"[Concept {concept_id}]: {metadata[concept_id]['concept']}")
        concept_df = all_df[all_df['concept_id'] == concept_id]
        if num_of_examples is not None:
            concept_df = concept_df.iloc[:num_of_examples]
        positive_df, negative_df = separate_contrast_df(concept_df)
        positive_df['original_prompt'] = positive_df['input'].copy()
        negative_df['original_prompt'] = negative_df['input'].copy()
        # Format prompts
        positive_df = prepare_df_prompt_only(positive_df, tokenizer)
        negative_df = prepare_df_prompt_only(negative_df, tokenizer)

        if use_saved_factors:
            # Get factors
            concept_factor_df = factor_df[factor_df['concept_id'] == concept_id]
            concept_factor_labels = (concept_factor_df['category']=='positive').to_numpy()

        for model_name in args.models:
            low_rank_dimension = training_args.models[model_name].low_rank_dimension
            if low_rank_dimension is None:
                low_rank_dimension = 1
            for multiplier in args.steering_factors:
                if use_saved_factors:
                    # Prepare steering factors
                    def get_acts_end(row):
                        return row[f'{model_name}_acts'][-1]
                    acts = concept_factor_df.apply(get_acts_end, axis=1).to_numpy()
                    pos_acts = acts[concept_factor_labels == 1]
                    neg_acts = acts[concept_factor_labels == 0]
                    positive_factor = pos_acts.mean()
                    negative_factor = neg_acts.mean()
                    logger.info(f"[Model {model_name}]: multiplier={multiplier:.2f}, positive factor={positive_factor}, negative factor={negative_factor}.")
                    positive_factor *= multiplier
                    negative_factor *= multiplier
                    if low_rank_dimension > 1:
                        positive_factor = positive_factor.tolist()
                        negative_factor = negative_factor.tolist()
                else:
                    positive_factor = multiplier
                    negative_factor = multiplier

                model_class = getattr(models_module, model_name)
                logger.info(f"Loading {model_class} on {device}.")
                benchmark_model: Model = model_class(
                    model=hf_model,
                    tokenizer=tokenizer,
                    layer=layer,
                    low_rank_dimension=low_rank_dimension,
                    device=device,
                    training_args=training_args.models[model_name],
                    steering_layers=args.steering_layers,
                )
                benchmark_model.load(
                    dump_dir=train_dir,
                    mode="steering",
                    priority_mode="compute_priority",
                    low_rank_dimension=low_rank_dimension,
                    concept_id=concept_id,
                )
                benchmark_model.to(device=device)
                if isinstance(benchmark_model.ax, list):
                    for ax in benchmark_model.ax:
                        ax.eval()
                else:
                    benchmark_model.ax.eval()

                # Incorporate factors into data
                if steering_direction == "negative":
                    logger.warning("Steering direction: negative->positive")

                    positive_df['factor'] = [negative_factor] * len(positive_df)
                    df = positive_df
                elif steering_direction == "positive":
                    logger.warning("Steering direction: positive->negative")

                    negative_df['factor'] = [positive_factor] * len(negative_df)
                    df = negative_df
                elif steering_direction == "two_way":
                    logger.warning("Steering direction: bi-directional")

                    positive_df['factor'] = [negative_factor] * len(positive_df)
                    negative_df['factor'] = [positive_factor] * len(negative_df)
                    df = pd.concat([positive_df, negative_df])
                else:
                    raise ValueError(f"Unknown steering direction: `{steering_direction}`.")

                def do_steer(cur_df):
                    results = benchmark_model.predict_steer(
                        examples=cur_df,
                        batch_size=int(args.steering_batch_size),
                        eval_output_length=int(args.steering_output_length),
                        temperature=float(args.temperature),
                        prefix_length=prefix_length,
                        use_synergy=False,
                    )
                    for k, v in results.items():
                        cur_df[f"{model_name}_{k}"] = v
                    return cur_df

                df = do_steer(df)

                del benchmark_model
                torch.cuda.empty_cache()

                save_df(overwrite_inference_dump_dir, 'steering', df, rank)
                logger.warning(f"Saved inference results to rank_{rank}_{STEERING_FILE}")

    # Synchronize all processes
    dist.barrier()

    # Rank 0 merges results
    if rank == 0:
        logger.warning("Rank 0 is merging results.")
        # Merge per-rank results
        all_parquet_files = list(
            Path(overwrite_inference_dump_dir).glob(f"rank_*_{STEERING_FILE}")
        )

        # Parse filenames to extract rank
        import re
        pattern = re.compile(r'rank_(\d+)_steering_data\.parquet')

        file_info_list = []
        for parquet_file in all_parquet_files:
            match = pattern.match(parquet_file.name)
            if match:
                rank_str = match.group(1)
                rank_int = int(rank_str)
                file_info_list.append({
                    'rank': rank_int,
                    'file': parquet_file
                })
            else:
                logger.warning(f"Filename {parquet_file.name} does not match the expected pattern.")

        # Sort the file_info_list by rank
        file_info_list.sort(key=lambda x: x['rank'])

        # Read and concatenate dataframes
        dfs = []

        for info in file_info_list:
            df = pd.read_parquet(info['file'])
            dfs.append(df)

        if len(dfs) > 0:
            combined_df = pd.concat(dfs, ignore_index=True)

            # Also merge previous results
            previous_file = Path(overwrite_inference_dump_dir) / STEERING_FILE
            if previous_file.exists():
                logger.warning(f"Appending current results to old results: `{previous_file}`")
                old_df = pd.read_parquet(previous_file)
                combined_df = pd.concat(
                    [old_df, combined_df], ignore_index=True, axis=0
                )

            combined_df.to_parquet(
                Path(overwrite_inference_dump_dir)
                / STEERING_FILE,
                engine="pyarrow",
            )
            logger.warning(
                f"Saved combined steering inference results to {Path(overwrite_inference_dump_dir) / STEERING_FILE}"
            )
        else:
            logger.warning("No results to merge.")

        # Optionally, delete per-rank files
        for info in file_info_list:
            os.remove(info['file'])
            logger.warning(f"Deleted {info['file']}")

    torch.cuda.empty_cache()
    dist.barrier()


def infer_steering_factor(
    hf_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    args: DatasetArgs,
    training_args: TrainingArgs,
    rank: int,
    world_size: int,
    device: torch.device,
    logger: logging.Logger,
    train_dir: Path,
    data_dir: Path,
):
    """Compute and store steering factor."""
    mode = "factor"

    dump_dir = train_dir
    config = load_config(train_dir)
    if config is None:
        raise ValueError(f"Config file `{CONFIG_FILE}` not found in {train_dir}.")
    layer = config["layer"]

    prefix_length = get_prefix_length(tokenizer)

    # Load saved states
    state = load_state(dump_dir, mode, rank)
    last_concept_id_processed = state.get('last_concept_id', None) if state else None

    # Load training data
    all_df = pd.read_parquet(data_dir / CONTRAST_TRAIN_DATA_FILE)

    # Load metadata from trained dir
    metadata_path = train_dir / METADATA_FILE
    metadata = load_metadata(metadata_path)
    concept_ids = list(metadata.keys())
    logger.warning(f"Total number of concepts loaded: {len(concept_ids)}")

    # Partition concept_ids among ranks sequentially
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    for concept_id in my_concept_ids:
        if last_concept_id_processed is not None and concept_id <= last_concept_id_processed:
            logger.warning(f"Rank {rank} skipping concept id {concept_id}.")
            continue

        logger.warning(f"[Concept {concept_id}]: {metadata[concept_id]['concept']}")
        concept_df = all_df[all_df['concept_id'] == concept_id]
        df = prepare_df_prompt_only(concept_df, tokenizer)
        positive_df, negative_df = separate_contrast_df(df)
        df = pd.concat([positive_df, negative_df])

        # Only keep inputs till source tokens
        source_tokens = training_args.source_tokens
        source_tokens = tokenizer.decode(tokenizer.encode(source_tokens, add_special_tokens=False))
        def rstrip_source_tokens(row):
            x = row['input']
            idx = x.rfind(source_tokens)
            if idx == -1:
                raise ValueError(f"Substring `{source_tokens}` not found in `{x}`.")
            return x[:idx+len(source_tokens)]
        df['input'] = df.apply(rstrip_source_tokens, axis=1)
        logger.info(f"Keep first few tokens: {repr(df['input'].iloc[0])}")

        for model_name in args.models:
            model_class = getattr(models_module, model_name)
            logger.info(f"Loading {model_class} on {device}.")
            benchmark_model: Model = model_class(
                model=hf_model,
                tokenizer=tokenizer,
                layer=layer,
                low_rank_dimension=training_args.models[model_name].low_rank_dimension,
                device=device
            )
            benchmark_model.load(dump_dir=train_dir, mode=mode)
            benchmark_model.to(device=device)
            if hasattr(benchmark_model, 'ax'):
                benchmark_model.ax.eval()

            def do_latents(df):
                results = benchmark_model.predict_latent(
                    examples=df,
                    prefix_length=prefix_length,
                    batch_size=args.latent_batch_size,
                    overwrite_concept_id=None,
                    return_max_act_only=False,
                )
                for k, v in results.items():
                    if k == "tokens":
                        if "tokens" not in df:
                            df["tokens"] = v  # for tokens, they are global
                        else:
                            continue
                    else:
                        df[f"{model_name}_{k}"] = v
                return df
            df = do_latents(df)

            del benchmark_model
            torch.cuda.empty_cache()
        save_df(dump_dir=dump_dir, partition=mode, current_df=df, rank=rank)
        current_state = {'last_concept_id': concept_id}
        save_state(dump_dir, current_state, mode, rank)

    dist.barrier()

    # merge results
    if rank == 0:
        logger.warning("Rank 0 is merging results.")
        # Merge per-rank results
        all_parquet_files = list(dump_dir.glob(f"rank_*_{FACTOR_FILE}"))
        # Parse filenames to extract rank
        import re
        pattern = re.compile(r'rank_(\d+)_factor_data\.parquet')

        file_info_list = []
        for parquet_file in all_parquet_files:
            match = pattern.match(parquet_file.name)
            if match:
                rank_str = match.group(1)
                rank_int = int(rank_str)
                file_info_list.append({
                    'rank': rank_int,
                    'file': parquet_file
                })
            else:
                logger.warning(f"Filename {parquet_file.name} does not match the expected pattern.")

        # Sort the file_info_list by rank
        file_info_list.sort(key=lambda x: x['rank'])

        # Read and concatenate dataframes
        dfs = []

        for info in file_info_list:
            df = pd.read_parquet(info["file"])
            dfs.append(df)
        if len(dfs) > 0:
            combined_df = pd.concat(dfs, ignore_index=True)

            # Also merge previous results
            previous_file = Path(dump_dir) / FACTOR_FILE
            if previous_file.exists():
                old_df = pd.read_parquet(previous_file)
                combined_df = pd.concat(
                    [old_df, combined_df], ignore_index=True, axis=0
                )

            combined_df.to_parquet(dump_dir / FACTOR_FILE, engine="pyarrow")
            logger.warning(f"Saved combined factor inference results to {dump_dir / FACTOR_FILE}")
        else:
            logger.warning("No results to merge.")

        # Optionally, delete per-rank files
        for info in file_info_list:
            os.remove(info['file'])
            logger.warning(f"Deleted {info['file']}")


def main():
    custom_args = [
        {
            "args": ["--mode"],
            "kwargs": {
                "type": str,
                "choices": [
                    "latent",
                    "factor",
                    "positive_steering",
                    "negative_steering",
                    "two_way_steering",
                    "all",
                ],
                "nargs": "+",
                "required": True,
                "help": "The inference mode.",
            },
        },
        {
            "args": ["--contrast_test_data_path"],
            "kwargs": {"type": str, "default": None, "help": "Contrastive test data."},
        },
    ]
    training_args = TrainingArgs(custom_args=custom_args, section="train", ignore_unknown=True)
    inference_args = DatasetArgs(custom_args=custom_args, section="inference", ignore_unknown=True)

    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=60000)
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    logger = logging.getLogger(__name__)
    logger_handlers = logger_setup(logger=logger, level=logging.INFO, rank=rank)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    set_seed(int(inference_args.seed) + rank)

    # Paths
    dump_dir = Path(inference_args.dump_dir).resolve()
    train_dir = dump_dir / "train"
    overwrite_inference_dump_dir = (
        Path(inference_args.overwrite_inference_dump_dir)
        if inference_args.overwrite_inference_dump_dir is not None
        else dump_dir / "inference"
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

    hf_model, tokenizer = load_hf_model_tokenizer(
        model_name_or_path=inference_args.model_name,
        device=device,
        dtype=torch.bfloat16,
        padding_side="right",
    )

    modes = inference_args.mode
    for mode in modes:
        logger.warning(f"Current mode: {mode}.")

        if mode == "positive_steering" or mode == "negative_steering" or mode == "two_way_steering":
            direction = mode.replace("_steering", "")
            infer_steering(
                steering_direction=direction,
                hf_model=hf_model,
                tokenizer=tokenizer,
                args=inference_args,
                training_args=training_args,
                world_size=world_size,
                device=device,
                rank=rank,
                logger=logger,
                layer=layer,
                overwrite_inference_dump_dir=overwrite_inference_dump_dir,
                train_dir=train_dir,
            )
        elif mode == "latent":
            infer_latent(
                hf_model=hf_model,
                tokenizer=tokenizer,
                args=inference_args,
                training_args=training_args,
                world_size=world_size,
                device=device,
                rank=rank,
                logger=logger,
                train_dir=train_dir,
                dump_dir=overwrite_inference_dump_dir,
            )
        elif mode == "factor":
            infer_steering_factor(
                hf_model=hf_model,
                tokenizer=tokenizer,
                args=inference_args,
                training_args=training_args,
                world_size=world_size,
                device=device,
                rank=rank,
                logger=logger,
                train_dir=train_dir,
                data_dir=data_dir,
            )
        elif mode == "all":
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown mode: `{mode}`.")

    logger.info("All finished")

    dist.destroy_process_group()
    for hdlr in logger_handlers:
        logger.removeHandler(hdlr)


if __name__ == "__main__":
    main()
