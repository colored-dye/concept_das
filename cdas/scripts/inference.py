"""
Predict latents, gather steering factors, or generate with steering.
"""

import os
import datetime
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import pickle
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
    load_config,
    load_metadata,
    partition_concept_ids,
    prepare_df_prompt_only,
    prepare_df_pair,
    save_df,
    separate_contrast_df,
)

import logging

# logger = logging.getLogger(__name__)


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
    mode = "steering"

    # Load steering factors
    factor_path = (train_dir / FACTOR_FILE).resolve()
    if not factor_path.exists():
        raise FileNotFoundError(f"Factor file `{factor_path}` does not exist.")
    factor_df = pd.read_parquet(factor_path)

    # Load saved states
    state = load_state(overwrite_inference_dump_dir, mode, rank)
    last_concept_id_processed = state.get('last_concept_id', None) if state else None

    # Load metadata from trained dir
    metadata_path = train_dir / METADATA_FILE
    metadata = load_metadata(metadata_path)
    concept_ids = list(metadata.keys())
    logger.warning(f"Total number of concepts loaded: {len(concept_ids)}")

    # Partition concept_ids among ranks sequentially
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    # Skip processed concept ids
    if last_concept_id_processed is not None:
        if last_concept_id_processed in my_concept_ids:
            idx = my_concept_ids.index(last_concept_id_processed)
            my_concept_ids = my_concept_ids[idx+1:]
            logger.warning(f"Concept id up to {last_concept_id_processed} processed. Starting from {my_concept_ids[0]}.")

    # Load steering data
    test_data_path = Path(args.test_data_path).resolve()
    logger.warning(f"Using test data: {test_data_path}.")
    raw_df = pd.read_json(test_data_path)
    all_df = {}
    all_df['input'] = raw_df['instruction']
    all_df['original_prompt'] = raw_df['instruction']
    all_df['output'] = raw_df['output']
    all_df = pd.DataFrame(all_df)

    num_of_examples = args.steering_num_of_examples
    if num_of_examples is not None:
        logger.warning(f"Only select first {num_of_examples} examples.")
        num_of_examples = int(num_of_examples)

    prefix_length = get_prefix_length(tokenizer)

    # Format prompts
    all_df = prepare_df_prompt_only(all_df, tokenizer)

    for concept_id in my_concept_ids:
        concept = metadata[concept_id]['concept']
        logger.warning(f"[Concept {concept_id}]: {concept}")
        df = all_df.copy()
        if num_of_examples is not None:
            # df = df.iloc[:num_of_examples]
            # Sample from eval df, following axbench.
            df = all_df.sample(num_of_examples, random_state=int(concept_id))
        df['concept_id'] = [concept_id] * len(df)
        df['concept'] = concept

        # Get factors
        concept_factor_df = factor_df[factor_df['concept_id'] == concept_id]
        concept_factor_labels = (concept_factor_df['category']=='positive').to_numpy()

        for model_name in args.models:
            for factor in args.steering_factors:
                # Prepare steering factors
                def get_acts_end(row):
                    return row[f'{model_name}_acts'][-1]
                acts = concept_factor_df.apply(get_acts_end, axis=1).to_numpy()
                pos_acts = acts[concept_factor_labels == 1]
                neg_acts = acts[concept_factor_labels == 0]
                positive_factor = pos_acts.mean().item()
                negative_factor = neg_acts.mean().item()
                logger.info(f"[Concept {concept_id} || Model {model_name}]: multiplier={factor:.2f}, positive factor={positive_factor:.4f}, negative factor={negative_factor:.4f}.")
                positive_factor *= factor
                negative_factor *= factor

                model_class = getattr(models_module, model_name)
                logger.info(f"Loading {model_class} on {device}.")
                low_rank_dimension = training_args.models[model_name].low_rank_dimension
                if low_rank_dimension is None:
                    low_rank_dimension = 1
                benchmark_model: Model = model_class(
                    model=hf_model,
                    tokenizer=tokenizer,
                    layer=layer,
                    low_rank_dimension=low_rank_dimension,
                    device=device
                )
                benchmark_model.load(
                    dump_dir=train_dir,
                    mode="steering",
                    priority_mode="compute_priority",
                    low_rank_dimension=low_rank_dimension,
                )
                benchmark_model.to(device=device)
                benchmark_model.ax.eval()

                # Incorporate factors into data
                if steering_direction == "negative":
                    logger.warning("Steering direction: negative->positive")

                    df['factor'] = [negative_factor] * len(df)
                elif steering_direction == "positive":
                    logger.warning("Steering direction: positive->negative")

                    df['factor'] = [positive_factor] * len(df)
                elif steering_direction == "two_way":
                    logger.warning("Steering direction: bi-directional")

                    positive_df = df.copy()
                    negative_df = df.copy()
                    positive_df['factor'] = [negative_factor] * len(positive_df)
                    negative_df['factor'] = [positive_factor] * len(negative_df)
                    df = pd.concat([positive_df, negative_df])
                else:
                    raise ValueError(f"Unknown steering direction: `{steering_direction}`.")

                def do_steer(benchmark_model, cur_df):
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

                df = do_steer(benchmark_model, df)

                del benchmark_model
                torch.cuda.empty_cache()

                save_df(overwrite_inference_dump_dir, mode, df, rank)

        current_state = {'last_concept_id': concept_id}
        save_state(overwrite_inference_dump_dir, current_state, mode, rank)
        logger.warning(f"Saved inference results to {overwrite_inference_dump_dir}.")

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
                    [old_df, combined_df], axis=0, ignore_index=True
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
    overwrite_inference_dump_dir: Path,
    layer: int,
):
    mode = "latent"
    dump_dir = overwrite_inference_dump_dir

    prefix_length = get_prefix_length(tokenizer)

    # Load metadata from trained dir
    metadata_path = train_dir / METADATA_FILE
    metadata = load_metadata(metadata_path)
    concept_ids = list(metadata.keys())
    logger.warning(f"Total number of concepts loaded: {len(concept_ids)}")

    # Partition concept_ids among ranks sequentially
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    # Load steering data
    test_data_path = Path(args.test_data_path).resolve()
    logger.warning(f"Using test data: {test_data_path}.")
    all_df = pd.read_parquet(test_data_path)
    num_of_examples = args.latent_num_of_examples
    if num_of_examples is not None:
        logger.warning(f"Only select first {num_of_examples} examples.")
        num_of_examples = int(num_of_examples)

    for concept_id in my_concept_ids:
        logger.warning(f"[Concept {concept_id}]: {metadata[concept_id]['concept']}")
        concept_df = all_df[all_df['concept_id'] == concept_id]
        if num_of_examples is not None:
            concept_df = concept_df.iloc[:num_of_examples]
        df = concept_df
        df = prepare_df_pair(df=df, tokenizer=tokenizer, is_chat_model=True, remove_suffix=False)

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

            def do_latents(benchmark_model, df):
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
            df = do_latents(benchmark_model, df)

            del benchmark_model
            torch.cuda.empty_cache()
        save_df(dump_dir=dump_dir, partition=mode, current_df=df, rank=rank)

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


def main():
    custom_args = [
        {
            "args": ["--mode"],
            "kwargs": {
                "type": str,
                "choices": [
                    "positive_steering",
                    "negative_steering",
                    "two_way_steering",
                    "latent",
                ],
                "nargs": "+",
                "required": True,
                "help": "The inference mode.",
            },
        },
        {
            "args": ["--test_data_path"],
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
                overwrite_inference_dump_dir=overwrite_inference_dump_dir,
                layer=layer,
            )
        else:
            raise ValueError(f"Unknown mode: `{mode}`.")

    logger.info("All finished")

    dist.destroy_process_group()
    for hdlr in logger_handlers:
        logger.removeHandler(hdlr)


if __name__ == "__main__":
    main()
