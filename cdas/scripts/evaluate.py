"""
Evaluate inference results.

Environment variables:
    * OPENAI_API_KEY
    * OPENAI_BASE_URL (Optional)

"""

import datetime
import asyncio
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import pickle
import httpx
from typing import List, Generator, Tuple
import re
from tqdm.auto import tqdm

import torch
import torch.distributed as dist
from transformers import (
    set_seed,
    PreTrainedTokenizer,
)
from openai import AsyncClient

from cdas.args import EvalArgs, TrainingArgs, DatasetArgs
from cdas.chat_templates import (
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE,
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE,
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE,
)
from cdas.constants import (
    CONFIG_FILE,
    EVALUATE_STATE_FILE,
    FACTOR_FILE,
    METADATA_FILE,
    STEERING_FILE,
    MODELS_WITH_FACTOR_FILE,
)
import cdas.models as models_module
from cdas.models import Model
from cdas.utils.model_utils import load_hf_model_tokenizer, get_prefix_length
from cdas.utils.logger_utils import logger_setup
from cdas.utils.api_model_utils import RemoteAPIModel

from cdas.scripts.utils import (
    load_config,
    load_metadata,
    save_df,
    partition_concept_ids,
    separate_contrast_df,
    prepare_df_prompt_only,
    prepare_df_pair,
)

import logging


def load_state(dump_dir, mode):
    """
    Load the state from a file if it exists.
    
    Args:
        dump_dir (str): The directory to load the state file from.
    
    Returns:
        dict: The loaded state dictionary, or None if no state file exists.
    """
    state_path = os.path.join(f"{dump_dir}", f"{mode}_{EVALUATE_STATE_FILE}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def save_state(dump_dir, state, mode):
    if not isinstance(dump_dir, Path):
        dump_dir = Path(dump_dir)
        
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save state
    state_path = os.path.join(dump_dir, f"{mode}_{EVALUATE_STATE_FILE}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)


def harmonic_mean(scores):
    # Return 0 if any score is 0 to maintain strict evaluation
    if 0 in scores:
        return 0.0
    return len(scores) / sum(1/s for s in scores)


def data_generator(data_dir, mode) -> Generator[Tuple[int, DataFrame], None, None]:
    """
    Generator function to read data files and yield data subsets by group_id.
    Pre-loads data in chunks to reduce I/O bottlenecks.

    Args:
        data_dir (str): Path to the data directory.
        mode (str): Mode of operation ('latent' or 'steering').

    Yields:
        (group_id, df_subset): A tuple containing the group_id and subset DataFrame.
    """
    # Pre-load and organize data by concept_id
    concept_data = {}
    if mode == "latent":
        df = pd.read_parquet(os.path.join(data_dir, 'latent_data.parquet'))
    elif "steering" in mode:
        df = pd.read_parquet(os.path.join(data_dir, 'rank_0_steering_data.parquet'))
        # df = pd.read_parquet(os.path.join(data_dir, 'steering_data.parquet'))
    elif mode == "train_data":
        df = pd.read_parquet(os.path.join(data_dir, 'dpo_train_data.parquet'))
    # Group by concept_id and store in dictionary
    for concept_id, group in df.groupby('concept_id'):
        if concept_id not in concept_data:
            concept_data[concept_id] = []
        concept_data[concept_id].append(group)
    
    # Yield concatenated data for each concept_id
    for concept_id in sorted(concept_data.keys()):
        if len(concept_data[concept_id]) > 1:
            df_subset = pd.concat(concept_data[concept_id])
        else:
            df_subset = concept_data[concept_id][0]
        yield (concept_id, df_subset)


def eval_steering_single_df(
    lm: RemoteAPIModel,
    current_df: DataFrame,
    concept: str,
    model_name: str,
    logger: logging.Logger,
    batch_size: int = 32,
):
    pattern = re.compile(r'Rating:\s*(\d+)')
    n_batches = (len(current_df) + batch_size - 1) // batch_size

    results_dict = {
        "concept_score": [],
        "relevance_score": [],
        "fluency_score": [],
        "mean_score": [],
    }
    for batch_idx in range(n_batches):
        batch_prompts = []
        for _, rec in current_df.iloc[
            batch_idx * batch_size : (batch_idx + 1) * batch_size
        ].iterrows():
            batch_prompts.append(
                UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE.format(
                    concept=concept, sentence=rec[f"{model_name}_steered_generation"]
                )
            )
            batch_prompts.append(
                UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE.format(
                    instruction=rec["original_prompt"],
                    sentence=rec[f"{model_name}_steered_generation"],
                )
            )
            batch_prompts.append(
                UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE.format(
                    sentence=rec[f"{model_name}_steered_generation"]
                )
            )

        responses = asyncio.run(
            lm.chat_completions(prompts=batch_prompts, batch_size=batch_size)
        )
        results = []
        for resp in responses:
            match = pattern.findall(resp)
            try:
                v = int(match[0])
            except Exception as e:
                logger.error(e)
                logger.error(f"Response: `{resp}`")
                v = 0
            results.append(v)
        for i in range(0, len(results), 3):
            row = results[i:i+3]
            score = harmonic_mean(row)
            results_dict['concept_score'].append(row[0])
            results_dict['relevance_score'].append(row[1])
            results_dict['fluency_score'].append(row[2])
            results_dict['mean_score'].append(score)

    results_df = DataFrame(results_dict)
    return pd.concat([current_df, results_df], axis=1)


def eval_steering(
    logger: logging.Logger,
    args: EvalArgs,
    dump_dir: Path,
    inference_dump_dir: Path,
):
    """
    Args:
        dump_dir: Place to put evaluation results.
    """
    mode = "steering"
    # Load previous state if exists
    state = load_state(dump_dir, mode=mode)
    last_concept_id = state.get('last_concept_id', None) if state else None

    client = AsyncClient(
        api_key=os.environ.get("OPENAI_API_KEY", None),
        base_url=os.environ.get("OPENAI_BASE_URL", None),
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=1000),
            headers={"Connection": "close"},
        ),
        max_retries=10,
        timeout=60.0,
    )
    lm_model_name = args.lm_model
    logger.warning(f"Evaluator model: {lm_model_name}.")

    lm = RemoteAPIModel(
        model=lm_model_name,
        client=client,
        temperature=0.01,
    )

    df_generator = data_generator(data_dir=inference_dump_dir, mode=mode)
    df_list = list(df_generator)

    # Filter data if we designate range
    if args.start_concept_id == -1:
        start_concept_id = 0
    else:
        start_concept_id = args.start_concept_id
    if args.end_concept_id == -1:
        end_concept_id = max(x[0] for x in df_list)
    else:
        end_concept_id = args.end_concept_id
    logger.warning(f"Evaluation range: {start_concept_id} - {end_concept_id}.")

    for concept_id, concept_df in df_list:
        if concept_id < start_concept_id or concept_id > end_concept_id:
            logger.warning(f"Concept id {concept_id} not in range.")
            continue
        if last_concept_id is not None and concept_id <= last_concept_id:
            logger.warning(f"Skipping concept id {concept_id}.")
            continue

        try:
            concept = concept_df.iloc[0]['concept']
        except:
            concept = concept_df.iloc[0]['input_concept']
        logger.warning(f"[Concept {concept_id}]: {concept}")

        # Deduplication
        concept_df = concept_df.drop_duplicates(
            subset=["input", "concept_id", "factor"]
        )

        for model_name in args.models:
            for strength, current_df in tqdm(concept_df.groupby(
                f"{model_name}_strength", sort=False,
            ), desc=f"[Concept {concept_id} || Model {model_name}]"):
                current_df = current_df.reset_index()

                current_df = eval_steering_single_df(
                    lm=lm,
                    current_df=current_df,
                    concept=concept,
                    model_name=model_name,
                    logger=logger,
                )
                save_df(
                    dump_dir=dump_dir,
                    partition=mode,
                    current_df=current_df,
                    rank=None,
                )
            logger.warning(f"Results saved to {dump_dir}.")
        current_state = {'last_concept_id': concept_id}
        save_state(dump_dir, current_state, mode)


def eval_kl(
    logger: logging.Logger,
    args: EvalArgs,
    training_args: TrainingArgs,
    inference_args: DatasetArgs,
    dump_dir: Path,
    rank: int,
    device: torch.device,
    world_size: int,
    train_dir: Path,
    inference_dump_dir: Path,
):
    """
    Evaluate KL divergence.
    Currently only supports negative steering.
    """
    mode = "kl"

    config = load_config(train_dir)
    if config is None:
        raise ValueError(f"Config file `{CONFIG_FILE}` not found in {train_dir}.")
    layer = config["layer"]

    # Load metadata from trained dir
    metadata_path = train_dir / METADATA_FILE
    metadata = load_metadata(metadata_path)
    concept_ids = list(metadata.keys())

    # Partition concept_ids among ranks sequentially
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    logger.warning(f"Total number of concepts loaded: {len(concept_ids)}")
    if args.contrast_test_data_path is None:
        raise ValueError("--contrast_test_data_path not specified")
    contrast_test_data_path = Path(args.contrast_test_data_path).resolve()
    all_df = pd.read_parquet(contrast_test_data_path)
    num_of_examples = inference_args.steering_num_of_examples
    if num_of_examples is not None:
        logger.warning(f"Only select first {num_of_examples} examples.")
        num_of_examples = int(num_of_examples)

    hf_model, tokenizer = load_hf_model_tokenizer(
        model_name_or_path=inference_args.model_name,
        device=device,
        dtype=torch.bfloat16,
        padding_side="right",
    )
    prefix_length = get_prefix_length(tokenizer)

    for concept_id in my_concept_ids:
        logger.warning(f"[Concept {concept_id}]: {metadata[concept_id]['concept']}")
        concept_df = all_df[all_df['concept_id'] == concept_id]
        if num_of_examples is not None:
            concept_df = concept_df.iloc[:num_of_examples]
        positive_df, negative_df = separate_contrast_df(concept_df)
        positive_df['original_prompt'] = positive_df['input'].copy()

        # Format prompts
        positive_df['output'] = negative_df['output']

        # positive_df = prepare_df_pair(positive_df, tokenizer)
        # negative_df = prepare_df_pair(negative_df, tokenizer)
        positive_df = prepare_df_prompt_only(positive_df, tokenizer)
        negative_df = prepare_df_prompt_only(negative_df, tokenizer)

        for model_name in args.models:
            use_saved_factors = False
            if model_name in MODELS_WITH_FACTOR_FILE:
                logger.warning(f"Using saved factors; loading from {train_dir}")
                # Get factors
                factor_path = (train_dir / FACTOR_FILE).resolve()
                if not factor_path.exists():
                    raise FileNotFoundError(f"Factor file `{factor_path}` does not exist.")
                factor_df = pd.read_parquet(factor_path)
                concept_factor_df = factor_df[factor_df['concept_id'] == 0]
                concept_factor_labels = (concept_factor_df['category']=='positive').to_numpy()
                use_saved_factors = True

            for multiplier in inference_args.steering_factors:
                if use_saved_factors:
                    # Prepare steering factors
                    def get_acts_end(row):
                        return row[f'{model_name}_acts'][-1]
                    acts = concept_factor_df.apply(get_acts_end, axis=1).to_numpy()
                    neg_acts = acts[concept_factor_labels == 0]
                    negative_factor = neg_acts.mean().item()
                    logger.info(f"[Model {model_name}]: multiplier={multiplier:.2f}, negative factor={negative_factor:.4f}.")
                    negative_factor *= multiplier
                    factor = negative_factor
                else:
                    logger.info(f"[Model {model_name}]: multiplier={multiplier:.2f}, negative factor={multiplier:.4f}.")
                    factor = multiplier
                positive_df['factor'] = [factor] * len(positive_df)

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

                kl_div = benchmark_model.get_kl_divergence(
                    base_examples=positive_df,
                    source_examples=negative_df,
                    batch_size=int(inference_args.steering_batch_size),
                    prefix_length=prefix_length,
                    # do_original_kl=True, # get original kl
                )
                logger.warning(f"[Concept id: {concept_id}] negative->positive: Model: {model_name} || KL div: {kl_div.mean():.5f}")


def main():
    custom_args = [
        {
            "args": ["--mode"],
            "kwargs": {
                "type": str,
                "choices": [
                    "steering",
                    "kl",
                ],
                "nargs": "+",
                "required": True,
                "help": "Evaluation mode.",
            },
        },
        {
            "args": ["--start_concept_id"],
            "kwargs": {
                "type": int,
                "default": -1,
            },
        },
        {
            "args": ["--end_concept_id"],
            "kwargs": {
                "type": int,
                "default": -1,
            },
        },
        {
            "args": ["--contrast_test_data_path"],
            "kwargs": {
                "type": str,
                "default": None,
            }
        }
    ]
    args = EvalArgs(custom_args=custom_args, section="evaluate", ignore_unknown=True)
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
    dump_dir = Path(args.dump_dir).resolve()
    train_dir = dump_dir / "train"
    dump_dir = dump_dir / "evaluate"
    if args.overwrite_evaluate_dump_dir is not None:
        dump_dir = Path(args.overwrite_evaluate_dump_dir).resolve()
    logger.warning(f"Dump dir: {dump_dir}")

    inference_dump_dir = Path(args.dump_dir).resolve() / "inference"
    if args.overwrite_inference_dump_dir is not None:
        inference_dump_dir = Path(args.overwrite_inference_dump_dir).resolve()
    logger.warning(f"Inference dump dir: {inference_dump_dir}")

    # hf_model, tokenizer = load_hf_model_tokenizer(
    #     model_name_or_path=args.model_name,
    #     device=device,
    #     dtype=torch.bfloat16,
    #     padding_side="right",
    # )

    for mode in args.mode:
        if mode == "steering":
            eval_steering(
                args=args,
                dump_dir=dump_dir,
                inference_dump_dir=inference_dump_dir,
                logger=logger,
            )
        elif mode == "kl":
            eval_kl(
                args=args,
                training_args=training_args,
                inference_args=inference_args,
                dump_dir=dump_dir,
                train_dir=train_dir,
                inference_dump_dir=inference_dump_dir,
                rank=rank,
                world_size=world_size,
                device=device,
                logger=logger,
            )
        else:
            raise NotImplementedError(f"Mode `{mode}` unknown")

    logger.warning("All finished")
    dist.destroy_process_group()

    for hdlr in logger_handlers:
        logger.removeHandler(hdlr)


if __name__ == "__main__":
    main()
