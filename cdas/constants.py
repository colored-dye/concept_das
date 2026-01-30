from typing import Literal


CONTRAST_PAIRS_OPTIONS_TYPE = Literal["neg+pos", "pos+neg"]
"""Configurations for contrastive counterfactual data."""

CONFIG_FILE = "config.json"
METADATA_FILE = "metadata.jsonl"
TRAIN_STATE_FILE = "train_state.pkl"
INFERENCE_STATE_FILE = "inference_state.pkl"
EVALUATE_STATE_FILE = "evaluate_state.pkl"

CONTRAST_TRAIN_DATA_FILE = "contrast_train_data.parquet"
FACTOR_FILE = "factor_data.parquet"
LATENT_FILE = "latent_data.parquet"
STEERING_FILE = "steering_data.parquet"
"""Latents computed on training data."""

MODELS_WITH_FACTOR_FILE = [
    "CDASVector", "DASVector", "PDASVector", "KLDASVector",
]
"""DII-based steering vectors use factors gathered from training data."""
