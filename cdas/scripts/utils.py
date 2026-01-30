"""Common utils for scripts."""

import copy
import os
import json
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from typing import Any, Dict
from transformers import (
    PreTrainedTokenizer
)

from cdas.constants import CONTRAST_TRAIN_DATA_FILE, CONFIG_FILE
from cdas.utils.model_utils import get_suffix_length


def prepare_df_prompt_only(
    df: DataFrame,
    tokenizer: PreTrainedTokenizer,
):
    """Apply chat template to prompts."""
    def apply_chat_template(df: DataFrame, column_name):
        def template_fn(row):
            messages = [{"role": "user", "content": row[column_name]}]
            nobos = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            # skip BOS since tokenizer will automatically add BOS later.
            if nobos[0] == tokenizer.bos_token_id:
                nobos = nobos[1:]
            return tokenizer.decode(nobos)

        df[column_name] = df.apply(template_fn, axis=1)

    for column_name in df.keys():
        if column_name.endswith("input"):
            apply_chat_template(df, column_name)
    return df


def prepare_df_pair(
    df: DataFrame,
    tokenizer: PreTrainedTokenizer,
    is_chat_model: bool = True,
    remove_suffix: bool = True,
):
    """
    Apply chat template to (prompt, response) pairs
    and concat to 'input' column.
    """
    suffix_length, _ = get_suffix_length(tokenizer)
    if is_chat_model:
        def apply_chat_template(row):
            messages = [
                {"role": "user", "content": row["input"]},
                {"role": "assistant", "content": row["output"]}
            ]
            tokens = tokenizer.apply_chat_template(messages, tokenize=True)[1:]
            if remove_suffix:
                tokens = tokens[:-suffix_length]
            else:
                tokens = tokens[:-suffix_length+1]
            return tokenizer.decode(tokens)
        df['input'] = df.apply(apply_chat_template, axis=1)
    else:
        raise ValueError(f"Only chat models are supported.")
    return df


def load_metadata(metadata_path) -> Dict[int, Dict[str, Any]]:
    """
    Load metadata from a JSON lines file.

    We use **dictionary** as container
    so concept ids may start from nonzero values.
    """
    metadata = []
    with open(metadata_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            id = data['concept_id']
            metadata += [(id, data)]
    metadata = sorted(metadata, key=lambda x: x[0])
    return dict(metadata)


def train_data_generator(data_dir):
    """
    Generator function to read multiple data files and yield data subsets by concept_id.
    Processes files in order: contrast_train_data.parquet, contrast_train_data_0.parquet, contrast_train_data_1.parquet, etc.
    """
    # Gather all file paths in the directory
    data_dir = Path(data_dir)
    file_paths = list(data_dir.glob("contrast_train_data*.parquet"))
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) \
        if f.startswith('contrast_train_data') and f.endswith('.parquet') and "combined" not in f]

    # Sort files: 'train_data.parquet' comes first, then 'train_data_X.parquet' sorted by X
    def extract_index(file_name):
        if file_name == CONTRAST_TRAIN_DATA_FILE:
            return -1  # Ensure 'train_data.parquet' comes first
        else:
            # Extract the number X from 'train_data_X.parquet'
            return int(file_name.split('_')[-1].split('.')[0])

    file_paths.sort(key=lambda x: extract_index(os.path.basename(x)))

    for file_path in file_paths:
        df = pd.read_parquet(file_path)
        concept_ids = df['concept_id'].unique()
        concept_ids.sort()
        for concept_id in concept_ids:
            if concept_id >= 0:
                # print(f"Processing concept_id {concept_id}")
                df_subset = df[df['concept_id'] == concept_id]
                yield (concept_id, df_subset)


def partition_list(lst, n):
    """
    Partition a list into n approximately equal slices.

    Args:
        lst (list): The list to partition.
        n (int): The number of partitions.

    Returns:
        list of lists: A list containing n sublists.
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def partition_concept_ids(concept_ids, world_size):
    concept_ids_per_rank = []
    n = len(concept_ids)
    chunk_size = n // world_size
    remainder = n % world_size
    start = 0
    for i in range(world_size):
        end = start + chunk_size + (1 if i < remainder else 0)
        concept_ids_per_rank.append(concept_ids[start:end])
        start = end
    return concept_ids_per_rank


def save_df(dump_dir: Path | str, partition: str, current_df: DataFrame, rank: int = None):
    """This function saves DataFrames per rank per partition (latent or steering)"""
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    if rank is None:
        df_path = os.path.join(dump_dir, f"{partition}_data.parquet")
    else:
        df_path = os.path.join(dump_dir, f"rank_{rank}_{partition}_data.parquet")

    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)

        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df

    combined_df.to_parquet(df_path, engine='pyarrow')


def separate_contrast_df(df: DataFrame):
    """
    Group positive and negative (prompt, response) pairs;
    rename input & output keys;
    add binary labels.

    :return positive_df, negative_df:
    """
    positive_df = {
        "input": [],
        "output": [],
        "category": [],
        "concept_id": [],
        "concept": [],
        "concept_genre": [],
    }
    negative_df = copy.deepcopy(positive_df)
    for _, row in df.iterrows():
        positive_df["input"].append(row["positive_input"])
        positive_df["output"].append(row["positive_output"])
        positive_df["category"].append("positive")
        positive_df["concept_id"].append(row["concept_id"])
        positive_df["concept"].append(row["concept"])
        positive_df["concept_genre"].append(row["concept_genre"])

        negative_df["input"].append(row["negative_input"])
        negative_df["output"].append(row["negative_output"])
        negative_df["category"].append("negative")
        negative_df["concept_id"].append(row["concept_id"])
        negative_df["concept"].append(row["concept"])
        negative_df["concept_genre"].append(row["concept_genre"])
    return DataFrame(positive_df), DataFrame(negative_df)


def load_config(config_path):
    """
    Load metadata from a JSON lines file.
    """
    if not os.path.exists(Path(config_path) / CONFIG_FILE):
        return None
    with open(Path(config_path) / CONFIG_FILE) as f:
        d = json.load(f)
    return d


