from dataclasses import dataclass
import torch, einops, os
import pandas as pd
from pandas import DataFrame
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from pyvene import (
    IntervenableModel,
)
from transformers import (
    set_seed,
    PreTrainedModel, PreTrainedTokenizer,
)
import transformers, datasets
from typing import Dict, Optional, Sequence, Union, List, Any

from ..utils.data_utils import make_data_module
from ..utils.model_utils import gather_residual_activations


import logging

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARN,
)
logger = logging.getLogger(__name__)


class BaseModel(object):
    """Base class for all models."""

    def __init__(self, **kwargs):
        pass

    def __str__(self):
        pass

    def make_model(self, **kwargs):
        pass

    def make_dataloader(self, examples, **kwargs):
        pass

    def train(self, examples, **kwargs):
        pass

    def save(self, dump_dir, **kwargs):
        pass

    def load(self, dump_dir, **kwargs):
        pass

    def predict_latent(self, examples, **kwargs):
        pass

    def predict_steer(self, examples, **kwargs):
        pass

    def get_logits(self, concept_id, k=10):
        pass

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        pass

    def to(self, device):
        pass


class Model(BaseModel):
    ax_model: IntervenableModel

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        layer: int,
        training_args=None,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        # abstracting layer
        self.layer = layer
        self.training_args = training_args
        self.max_activations = {}
        # Set default device to GPU if available, otherwise CPU
        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = kwargs.get(
            "dtype", model.dtype,
        )
        self.seed = kwargs.get("seed", 42)
        self.steering_layers = kwargs.get("steering_layers", None)
        self.num_of_layers = len(self.steering_layers) if self.steering_layers else 1
        self.dump_dir = kwargs.get("dump_dir", None)
        self.use_wandb = kwargs.get("use_wandb", False)

    def make_model(self, **kwargs):
        pass

    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, examples, **kwargs)
        g = torch.Generator()
        g.manual_seed(self.seed)
        train_dataloader = DataLoader(
            data_module["train_dataset"],
            shuffle=True,  # we shuffle for examples.
            batch_size=self.training_args.batch_size,
            collate_fn=data_module["data_collator"],
            generator=g,
        )
        return train_dataloader

    def train(self, examples, **kwargs):
        pass

    def save(self, dump_dir, model_name: str = None, do_overwrite_vector = False, **kwargs):
        if model_name is None:
            model_name = self.__str__()
        overwrite_ckpt = do_overwrite_vector
        weight_file = dump_dir / f"{model_name}_weight.pt"
        weight = self.ax.proj.weight.data.cpu()
        if weight_file.exists():
            if overwrite_ckpt is True:
                print(f"Overwriting {weight_file}")
            else:
                weight = torch.cat([torch.load(weight_file), weight], dim=0)
        torch.save(weight, weight_file)

        bias_file = dump_dir / f"{model_name}_bias.pt"
        bias = self.ax.proj.bias.data.cpu()
        if bias_file.exists():
            if overwrite_ckpt is True:
                print(f"Overwriting {weight_file}")
            else:
                bias = torch.cat([torch.load(bias_file), bias], dim=0)
        torch.save(bias, bias_file)

    def load(self, dump_dir=None, **kwargs):
        priority_mode = kwargs.get("priority_mode", "compute_priority")
        self.priority_mode = priority_mode
        if priority_mode == "mem_priority":
            # prioritize MEM
            concept_id = kwargs.get("concept_id")
            model_name = kwargs.get("model_name", self.__str__())
            weight = torch.load(
                f"{dump_dir}/{model_name}_weight.pt",
                map_location=torch.device("cpu"),
                mmap=True,  # Enable memory mapping
            )
            bias = torch.load(
                f"{dump_dir}/{model_name}_bias.pt",
                map_location=torch.device("cpu"),
                mmap=True,  # Enable memory mapping
            )
            weight_rank_1 = weight[concept_id].unsqueeze(0)
            bias_rank_1 = bias[concept_id].unsqueeze(0)
            # load only 1 rank to prevent OOM, and faster inference
            self.make_model(**kwargs)
            self.ax.proj.weight.data = weight_rank_1.to(self.device)
            self.ax.proj.bias.data = bias_rank_1.to(self.device)
        elif priority_mode == "compute_priority":
            # prioritize COMPUTE
            model_name = kwargs.get("model_name", self.__str__())
            print(f"Loading {model_name} from {dump_dir}.")
            weight = torch.load(
                f"{dump_dir}/{model_name}_weight.pt",
                map_location=torch.device("cpu"),
                weights_only=True,
            )
            bias = torch.load(
                f"{dump_dir}/{model_name}_bias.pt",
                map_location=torch.device("cpu"),
                weights_only=True,
            )
            # override low_rank_dimension in kwargs
            kwargs["low_rank_dimension"] = weight.shape[0]
            self.make_model(**kwargs)
            self.ax.proj.weight.data = weight.to(self.device)
            self.ax.proj.bias.data = bias.to(self.device)


    @torch.no_grad
    def get_kl_divergence(
        self,
        base_examples: DataFrame,
        source_examples: DataFrame,
        batch_size: int,
        prefix_length: int,
        do_original_kl: bool = False,
        **kwargs,
    ):
        assert len(base_examples) == len(source_examples)

        self.ax.eval()

        self.tokenizer.padding_side = "left"
        rank = torch.distributed.get_rank()
        progress_bar = tqdm(
            range(0, len(base_examples), batch_size), position=rank, leave=True
        )
        all_divergence = []
        for i in progress_bar:
            base_batch_examples = base_examples.iloc[i : i+batch_size]
            base_input_strings = base_batch_examples["input"].tolist()
            base_mag = torch.tensor(base_batch_examples["factor"].tolist()).to(self.device)
            base_idx = torch.tensor(base_batch_examples["concept_id"].tolist()).to(self.device)
            base_max_acts = torch.tensor(
                [
                    1.0
                    for _ in base_batch_examples["input"].tolist()
                ]
            ).to(self.device)

            base_inputs = self.tokenizer(
                base_input_strings, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            if do_original_kl:
                logger.warning("Computing original KL.")
                cf_base_outputs = self.ax_model.model(**base_inputs)
            else:
                _, cf_base_outputs = self.ax_model(
                    base_inputs,
                    unit_locations=None,
                    subspaces=[
                        {
                            "idx": base_idx,
                            "mag": base_mag,
                            "max_act": base_max_acts,
                            "prefix_length": prefix_length,
                        }
                    ]
                    * self.num_of_layers,
                    use_cache=False, # must
                )

            source_batch_examples = source_examples.iloc[i : i+batch_size]

            source_input_strings = source_batch_examples["input"].tolist()
            source_inputs = self.tokenizer(
                source_input_strings, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            source_outputs = self.ax_model.model(
                **source_inputs,
            )

            cf_logps = cf_base_outputs.logits[:, -1].contiguous().log_softmax(dim=-1)
            sc_logps = source_outputs.logits[:, -1].contiguous().log_softmax(dim=-1)
            cf_logps = cf_logps.view(-1, cf_logps.size(-1))
            sc_logps = sc_logps.view(-1, sc_logps.size(-1))
            kldiv = torch.nn.functional.kl_div(
                input=cf_logps,
                target=sc_logps,
                log_target=True,
                reduction='none',
            )
            all_divergence.extend(kldiv.float().cpu().sum(dim=-1).tolist())

        return torch.tensor(all_divergence)


    @torch.no_grad()
    def predict_steer(
        self,
        examples: DataFrame,
        batch_size: int,
        prefix_length: int,
        eval_output_length: 128,
        temperature=1.0,
        use_synergy=False,
        **kwargs,
    ):
        """
        Generate with steering intervention.

        Args:
            examples: Keys required: input, factor.
        """
        self.ax.eval()
        # set tokenizer padding to left
        self.tokenizer.padding_side = "left"

        # iterate rows in batch
        all_generations = []
        all_strenghts = []
        # Main training loop.
        rank = torch.distributed.get_rank()
        progress_bar = tqdm(
            range(0, len(examples), batch_size), position=rank, leave=True
        )
        for i in range(0, len(examples), batch_size):
            batch_examples = examples.iloc[i : i + batch_size]
            if use_synergy:
                # print("Using steered prompt to evaluate synergy of prompt and lsreft.")
                input_strings = batch_examples["steered_input"].tolist()
            else:
                input_strings = batch_examples["input"].tolist()
            mag = torch.tensor(batch_examples["factor"].tolist()).to(self.device)
            idx = torch.tensor(batch_examples["concept_id"].tolist()).to(self.device)
            max_acts = torch.tensor(
                [
                    1.0
                    for _ in batch_examples["input"].tolist()
                ]
            ).to(self.device)
            # logger.warning(f"Using max activations: {max_acts}")
            # tokenize input_strings
            inputs = self.tokenizer(
                input_strings, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            if temperature > 0.0:
                _, generations = self.ax_model.generate(
                    inputs,
                    unit_locations=None,
                    intervene_on_prompt=True,
                    subspaces=[
                        {
                            "idx": idx,
                            "mag": mag,
                            "max_act": max_acts,
                            "prefix_length": prefix_length,
                        }
                    ]
                    * self.num_of_layers,
                    max_new_tokens=eval_output_length,
                    do_sample=True,
                    temperature=temperature,
                    use_cache=False, # must
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                _, generations = self.ax_model.generate(
                    inputs,
                    unit_locations=None,
                    intervene_on_prompt=True,
                    subspaces=[
                        {
                            "idx": idx,
                            "mag": mag,
                            "max_act": max_acts,
                            "prefix_length": prefix_length,
                        }
                    ]
                    * self.num_of_layers,
                    max_new_tokens=eval_output_length,
                    do_sample=False,
                    use_cache=False, # must
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode and print only the generated text without prompt tokens
            input_lengths = [len(input_ids) for input_ids in inputs.input_ids]
            generated_texts = [
                self.tokenizer.decode(
                    generation[input_length:], skip_special_tokens=True
                )
                for generation, input_length in zip(generations, input_lengths)
            ]
            all_generations += generated_texts

            all_strenghts.extend((mag * max_acts).tolist())
            progress_bar.update(1)

        return {
            "steered_generation": all_generations,
            "strength": all_strenghts,
        }

    def get_logits(self, concept_id, k=10):
        top_logits, neg_logits = [None], [None]
        if concept_id is not None:
            W_U = self.model.lm_head.weight.T
            W_U = (
                W_U
                * (
                    self.model.model.norm.weight
                    + torch.ones_like(self.model.model.norm.weight)
                )[:, None]
            )
            W_U -= einops.reduce(W_U, "d_model d_vocab -> 1 d_vocab", "mean")

            vocab_logits = self.ax.proj.weight.data[concept_id] @ W_U
            top_values, top_indices = vocab_logits.topk(k=k, sorted=True)
            top_tokens = self.tokenizer.batch_decode(top_indices.unsqueeze(dim=-1))
            top_logits = [list(zip(top_tokens, top_values.tolist()))]

            neg_values, neg_indices = vocab_logits.topk(k=k, largest=False, sorted=True)
            neg_tokens = self.tokenizer.batch_decode(neg_indices.unsqueeze(dim=-1))
            neg_logits = [list(zip(neg_tokens, neg_values.tolist()))]
        return top_logits, neg_logits

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        max_activations = {}  # sae_id to max_activation
        # Loop over saved latent files in dump_dir.
        for file in os.listdir(dump_dir):
            if file.startswith("latent_") and file.endswith(".parquet"):
                latent_path = os.path.join(dump_dir, file)
                latent = pd.read_parquet(latent_path)
                # loop through unique sorted concept_id
                for concept_id in sorted(latent["concept_id"].unique()):
                    concept_latent = latent[latent["concept_id"] == concept_id]
                    max_act = concept_latent[f"{self.__str__()}_max_act"].max()
                    max_activations[concept_id] = max_act if max_act > 0 else 50
        self.max_activations = max_activations
        return max_activations

    def to(self, device: torch.device | str):
        """Move model to specified device"""
        self.device = device
        if hasattr(self, "ax"):
            self.ax = self.ax.to(device)
            if hasattr(self, "ax_model"):
                if isinstance(self.ax_model, IntervenableModel):
                    self.ax_model.set_device(device)
                else:
                    self.ax_model = self.ax_model.to(device)
        return self

    @torch.no_grad
    def predict_latent(
        self,
        examples: DataFrame,
        prefix_length: int,
        batch_size: int = 32,
        overwrite_concept_id: int = None,
        return_max_act_only: bool = False,
        **kwargs,
    ):
        self.ax.eval()

        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []

        with tqdm(range(0, len(examples), batch_size), desc="Processing batches") as progress_bar:
            for i in progress_bar:
                batch = examples.iloc[i:i+batch_size]
                inputs = self.tokenizer(
                    batch["input"].tolist(),
                    return_tensors='pt',
                    padding=True,
                    add_special_tokens=True,
                ).to(self.device)
                act_in = gather_residual_activations(
                    model=self.model, target_layer=self.layer, inputs=inputs
                )
                ax_acts_batch = self.ax(act_in[:, prefix_length:])
                seq_lens = inputs['attention_mask'].sum(dim=-1) - prefix_length

                for seq_idx, row in enumerate(batch.itertuples()):
                    # select acts with attention mask
                    acts = (
                        ax_acts_batch[
                            seq_idx,
                            : seq_lens[seq_idx],
                            (
                                overwrite_concept_id
                                if overwrite_concept_id is not None
                                else row.concept_id
                            ),
                        ]
                        .flatten()
                        .float()
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    acts = [round(x, 3) for x in acts]
                    max_act = max(acts)
                    all_max_act.append(max_act)
                    if not return_max_act_only:
                        max_act_indices = [i for i, x in enumerate(acts) if abs(x - max_act)<1e-6]
                        max_act_idx = max_act_indices[0]
                        # Get tokens for this specific sequence
                        tokens = self.tokenizer.tokenize(row.input)
                        if (
                            self.tokenizer.apply_chat_template(
                                [{"role": "user", "content": "a"}]
                            )[0]
                            == self.tokenizer.bos_token_id
                        ):
                            tokens = tokens[prefix_length-1:] # -1 is because it does not prepend BOS token
                        else:
                            tokens = tokens[prefix_length:]
                        max_token = tokens[max_act_idx]
                        all_acts.append(acts)
                        all_max_act_idx.append(max_act_idx)
                        all_max_token.append(max_token)
                        all_tokens.append(tokens)
                # clear memory and cache
                del ax_acts_batch
                del act_in
                torch.cuda.empty_cache()
        if return_max_act_only:
            return {
                "max_act": all_max_act
            }
        return {
            "acts": all_acts,
            "max_act": all_max_act,
            "max_act_idx": all_max_act_idx,
            "max_token": all_max_token,
            "tokens": all_tokens
        }
