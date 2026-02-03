"""
Obtain contrastive data pairs: `(x, y, x_c, y_c)`
by generating `x_c` and `y`.
"""

import argparse
import asyncio
import json
import httpx
from pathlib import Path
import os
import pandas as pd
from openai import AsyncOpenAI
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

import logging

from axbench.models.language_models import LanguageModel

logger = logging.getLogger(__name__)

METADATA_FILE = "metadata.jsonl"

CONCEPT_INSTRUCTION_TEMPLATE = """Please incorporate the following concept in your response to the instruction.

Concept: {concept}

Instruction: {instruction}"""

NEUTRAL_RESPONSE_TEMPLATE = """Given the following instruction:

{instruction}

Your task is to:
1. Provide a response that continues or addresses the instruction naturally.
2. Avoid any mention of '{concept}' in the continuation, regardless of coherence.

**Formatting Guidelines:**

- Return only the response to the instruction.
- Write the final content (or appropriate format for the genre) in plain text.
- Do not include any additional text, explanations, or formatting.

**Final Answer:** Return only the final content, following the guidelines above."""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--do_self_generate", action="store_true")
    parser.add_argument("--max_concepts", type=int, default=None)
    args = parser.parse_args()
    return args


def data_generator(data_dir, use_dpo_loss=False):
    """
    Generator function to read multiple data files and yield data subsets by concept_id.
    Processes files in order: train_data.parquet, train_data_0.parquet, train_data_1.parquet, etc.

    Args:
        data_dir (str): Path to the data directory.

    Yields:
        (concept_id, df_subset): A tuple containing the concept_id and subset DataFrame.
    """
    # Gather all file paths in the directory
    if use_dpo_loss:
        file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) \
            if f.startswith('dpo_train_data') and f.endswith('.parquet') and "combined" not in f]
    else:
        file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) \
            if f.startswith('train_data') and f.endswith('.parquet') and "combined" not in f]

    # Sort files: 'train_data.parquet' comes first, then 'train_data_X.parquet' sorted by X
    def extract_index(file_name):
        if use_dpo_loss:
            if file_name == 'dpo_train_data.parquet':
                return -1  # Ensure 'train_data.parquet' comes first
            else:
                # Extract the number X from 'train_data_X.parquet'
                return int(file_name.split('_')[-1].split('.')[0])
        else:
            if file_name == 'train_data.parquet':
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


def load_metadata(metadata_path):
    """
    Load metadata from a JSON lines file.
    """
    metadata = []
    with open(metadata_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            metadata += [data]  # Return the metadata as is
    return metadata


def augment_dataset(
    metadata: List[Dict[str, Any]],
    df_list: List[Tuple[int, pd.DataFrame]],
    data_dir: str,
    tokenizer: AutoTokenizer,
    local_model_path: str,
    max_length: int,
    do_self_generate: bool,
    save_path: Path,
    batch_size: int = 32,
):
    if do_self_generate:
        lm = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        tokenizer.padding_side = "left"
    else:
        client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            timeout=60.0,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_keepalive_connections=100, max_connections=1000
                ),
                headers={"Connection": "close"},
            ),
            max_retries=3,
        )
        lm = LanguageModel(
            model="gpt-4o-mini",
            client=client,
            temperature=0.3,
            master_data_dir=data_dir,
            use_cache=True,
        )

    new_df = {k: [] for k in df_list[0][1].keys()}
    new_df["input_concept"] = []
    new_df["output_neutral"] = []

    if save_path.exists():
        old_df = pd.read_parquet(save_path)
        for k in new_df:
            new_df[k] = old_df[k].to_list()
        last_processed_id = old_df['concept_id'].max()
        df_list = [
            (concept_id, content) for (concept_id, content) in df_list if concept_id > last_processed_id
        ]
        logger.warning(f"Last processed id: {last_processed_id}. Resuming...")

    for concept_id, concept_df in tqdm(df_list, desc=f"Processing df"):
        concept_id = int(concept_id)
        concept = metadata[concept_id]["concept"]
        concept_instructions = [
            CONCEPT_INSTRUCTION_TEMPLATE.format(concept=concept, instruction=x)
            for x in concept_df["input"]
        ]
        neutral_instructions = [
            NEUTRAL_RESPONSE_TEMPLATE.format(instruction=x, concept=concept)
            for x in concept_df["input"]
        ]
        n_batches = (len(neutral_instructions) - 1) // batch_size + 1
        for batch_idx in tqdm(
            range(n_batches), desc=f"Concept [{concept_id}/{len(df_list)}]"
        ):
            batch = neutral_instructions[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]

            if do_self_generate:

                def apply_chat_template(prompt):
                    messages = [{"role": "user", "content": prompt}]
                    nobos = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )[1:]
                    return tokenizer.decode(nobos)

                batch_prompts = [apply_chat_template(prompt) for prompt in batch]
                ids = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                ).to(lm.device)
                outputs = lm.generate(**ids, max_new_tokens=max_length, do_sample=False)
                outputs = outputs[:, ids.input_ids.shape[1] :]
                neutral_responses = tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
            else:
                neutral_responses = asyncio.run(
                    lm.chat_completions(
                        api_names="",
                        prompts=batch,
                        batch_size=len(batch),
                    )
                )

            if do_self_generate:
                neutral_responses_processed = neutral_responses
            else:
                neutral_responses_processed = []
                for r in neutral_responses:
                    tokens = tokenizer.encode(r, add_special_tokens=False)
                    if len(tokens) > max_length:
                        r = tokenizer.decode(tokens, skip_special_tokens=True)
                    neutral_responses_processed.append(r)

            new_df["output_neutral"].extend(neutral_responses_processed)

        new_df["input_concept"].extend(concept_instructions)
        for k, v in concept_df.items():
            new_df[k].extend(v)

        save_df = pd.DataFrame(new_df)
        save_df.to_parquet(save_path, engine='pyarrow')
    return new_df


def main(args: argparse.Namespace):
    logger.setLevel(logging.WARNING)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    concept_ds_dir = Path(args.dataset_dir)
    # Load dataset and metadata
    metadata_path = concept_ds_dir / METADATA_FILE
    metadata = load_metadata(metadata_path)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    new_metadata_path = save_dir / METADATA_FILE
    with open(new_metadata_path, 'w') as f:
        for rec in metadata:
            f.write(json.dumps(rec) + "\n")

    df_generator = data_generator(concept_ds_dir, use_dpo_loss=False)
    df_list = list(df_generator)

    logger.warning(f"Total size: {sum(len(df) for _, df in df_list)}")

    logger.warning(f"Total number of concept df loaded: {len(df_list)}")
    if args.max_concepts:
        logger.warning(f"Only processing {args.max_concepts} concepts")
        df_list = df_list[:args.max_concepts]

    new_df_save_path = save_dir / "train_data.parquet"
    new_df = augment_dataset(
        metadata=metadata,
        df_list=df_list,
        data_dir=save_dir,
        tokenizer=tokenizer,
        local_model_path=args.model_path,
        do_self_generate=args.do_self_generate,
        max_length=512,
        save_path=new_df_save_path,
    )

    logger.warning(f"Saved to {new_df_save_path}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
