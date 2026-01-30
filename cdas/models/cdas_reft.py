from pathlib import Path
import pandas as pd
from pandas import DataFrame
from tqdm.auto import tqdm
from typing import List, Literal

import torch
from torch.utils.data import DataLoader
from transformers import (
    set_seed, get_scheduler,
)
from pyvene import (
    IntervenableConfig,
    IntervenableModel,
)

from ..constants import CONTRAST_PAIRS_OPTIONS_TYPE
from ..utils.model_utils import (
    get_lr,
)

from .cdas_model import (
    CDASModel,
    jensen_shannon_loss,
)
from .reft import make_eval_data_module
from .interventions import (
    LoreftIntervention,
)

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class CDASLoReFT(CDASModel):
    """CDASModel with LoReFT interventions."""

    def __str__(self):
        return "CDASLoReFT"

    def save(self, dump_dir, **kwargs):
        dump_dir = Path(f"{dump_dir}/loreft/{self.concept_id}")
        dump_dir.mkdir(parents=True, exist_ok=True)
        self.ax_model.save(dump_dir, include_model=False)

    def load(self, dump_dir=None, concept_id=None, priority_mode="compute_priority", **kwargs):
        self.priority_mode = priority_mode
        self.concept_id = concept_id
        self.make_model(concept_id=concept_id, **kwargs)
        dump_dir = Path(f"{dump_dir}/loreft/{self.concept_id}")
        self.ax_model.load_intervention(dump_dir, include_model=False)

    def make_model(
        self,
        mode: Literal["train", "steering"],
        concept_id,
        embed_dim: int = None,
        low_rank_dimension: int = 1,
        contrast_pairs: List[CONTRAST_PAIRS_OPTIONS_TYPE] = ["neg+pos", "pos+neg"],
        dropout=0.0,
        **kwargs,
    ):
        """
        Args:
            mode: All modes use the same intervention.
        """
        if embed_dim is None:
            embed_dim = self.model.config.hidden_size

        if mode == "steering":
            reft_layers = (
                self.steering_layers
                if self.steering_layers is not None
                else [self.layer]
            )
        elif mode == "train":
            reft_layers = (
                self.training_args.reft_layers
                if self.training_args.reft_layers is not None
                else [self.layer]
            )

        self.number_of_interventions = len(reft_layers)
        ax = []
        for _ in range(self.number_of_interventions):
            _ax = LoreftIntervention(
                embed_dim=embed_dim,
                low_rank_dimension=low_rank_dimension,
                dropout=dropout,
            )
            _ax.to(device=self.device)
            _ax.train()
            ax.append(_ax)
        self.ax = ax

        ax_config = IntervenableConfig(
            representations=[
                {
                    "layer": l,
                    "component": f"model.layers[{l}].output",
                    "low_rank_dimension": low_rank_dimension,
                    "intervention": self.ax[i],
                }
                for i, l in enumerate(reft_layers)
            ]
        )
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax_model = ax_model
        self.contrast_pairs = ["neg+pos"] # contrast_pairs
        self.intervention_positions = self.training_args.reft_positions
        self.concept_id = concept_id


    def train(self, examples: DataFrame, **kwargs):
        train_dataloader = self.make_contrast_dataloader(examples=examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        logger.warning(f"Trainable parameters: {sum([p.numel() for p in self.ax_model.get_trainable_parameters()])}")
        optimizer = torch.optim.AdamW(
            self.ax_model.get_trainable_parameters(), 
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        num_training_steps = self.training_args.n_epochs * (
            len(train_dataloader) // self.training_args.gradient_accumulation_steps
        )
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        # Main training loop.
        rank = torch.distributed.get_rank()
        progress_bar = tqdm(range(num_training_steps), position=rank, leave=True)
        curr_step = 0

        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                for k, v in batch.items():
                    bs = len(v)
                    break
                minibatch_size = min(self.training_args.batch_size, bs)
                expanded_batch_size = minibatch_size * len(self.contrast_pairs)
                num_minibatches = (expanded_batch_size + minibatch_size - 1) // minibatch_size

                base_inputs = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                    "intervention_locations": [],
                }
                source_inputs = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                    # "intervention_locations": [],
                }
                for i in range(minibatch_size):
                    for pair in self.contrast_pairs:
                        base_inputs["input_ids"].append(batch[f"{pair}_base_input_ids"][i])
                        base_inputs["attention_mask"].append(batch[f"{pair}_base_attention_mask"][i])
                        base_inputs["labels"].append(batch[f"{pair}_base_labels"][i])
                        base_inputs["intervention_locations"].append(batch[f"{pair}_base_intervention_locations"][i])

                        source_inputs["input_ids"].append(batch[f"{pair}_source_input_ids"][i])
                        source_inputs["attention_mask"].append(batch[f"{pair}_source_attention_mask"][i])
                        source_inputs["labels"].append(batch[f"{pair}_source_labels"][i])
                        # source_inputs["intervention_locations"].append(batch[f"{pair}_source_intervention_locations"][i])

                loss_sum = 0
                for mb in range(num_minibatches):
                    start_idx = mb * minibatch_size
                    end_idx = min((mb + 1) * minibatch_size, expanded_batch_size)

                    if start_idx >= expanded_batch_size:
                        break

                    base_minibatch_inputs = {
                        k: torch.stack(base_inputs[k][start_idx:end_idx], dim=0).to(self.device)
                        for k, _ in base_inputs.items()
                    }
                    source_minibatch_inputs = {
                        k: torch.stack(source_inputs[k][start_idx:end_idx], dim=0).to(self.device)
                        for k, _ in source_inputs.items()
                    }
                    if isinstance(self.ax, list):
                        unit_locations = {
                            "sources->base": (
                                None,
                                base_minibatch_inputs["intervention_locations"]
                                .permute(1, 0, 2)
                                .tolist()
                                * len(self.ax),
                            )
                        }
                    else:
                        unit_locations = {
                            "sources->base": (
                                None,
                                base_minibatch_inputs["intervention_locations"]
                                .permute(1, 0, 2)
                                .tolist(),
                            )
                        }

                    _, cf_outputs = self.ax_model(
                        base={
                            "input_ids": base_minibatch_inputs['input_ids'],
                            "attention_mask": base_minibatch_inputs['attention_mask'],
                        },
                        unit_locations=unit_locations,
                        use_cache=False,
                    )

                    with torch.no_grad():
                        source_outputs = self.ax_model.model(
                            input_ids=source_minibatch_inputs['input_ids'],
                            attention_mask=source_minibatch_inputs['attention_mask'],
                        )

                    minibatch_loss = jensen_shannon_loss(
                        cf_logits=cf_outputs.logits,
                        sc_logits=source_outputs.logits,
                        cf_labels=base_minibatch_inputs['labels'],
                        sc_labels=source_minibatch_inputs['labels'],
                    )
                    minibatch_loss = minibatch_loss

                    minibatch_loss = minibatch_loss / (num_minibatches * self.training_args.gradient_accumulation_steps)
                    minibatch_loss.backward()

                    loss_sum += minibatch_loss.detach() * (end_idx - start_idx)
                loss = loss_sum / expanded_batch_size

                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.ax_model.get_trainable_parameters(), 1.0)
                    curr_lr = get_lr(optimizer)
                    # optim
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)

                    progress_bar.set_description(
                        "lr %.6f || loss %.6f" % (
                            curr_lr, loss, ))
                    curr_step += 1

        progress_bar.close()

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        # set tokenizer padding to left
        self.tokenizer.padding_side = "left"
        # depending on the model, we use different concept id columns
        concept_id_col = "concept_id"
        # iterate rows in batch
        batch_size = kwargs.get("batch_size", 64)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)
        all_generations = []
        all_strengths = []
        rank = torch.distributed.get_rank()

        data_module = make_eval_data_module(
            self.tokenizer,
            self.model,
            examples,
            positions=self.intervention_positions,
            num_interventions=self.number_of_interventions,
            nonstop=True,
            share_weights=True,
        )
        eval_dataloader = DataLoader(
            data_module["eval_dataset"],
            shuffle=False,
            batch_size=kwargs.get("batch_size"),
            collate_fn=data_module["data_collator"],
        )

        torch.cuda.empty_cache()
        all_batch_examples = [
            examples.iloc[i : i + batch_size]
            for i in range(0, len(examples), batch_size)
        ]
        progress_bar = tqdm(all_batch_examples, position=rank, leave=True)
        for i, batch in enumerate(eval_dataloader):
            # prepare input
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            unit_locations = {
                "sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist(),
                )
            }
            batch_examples = all_batch_examples[i]
            idx = torch.tensor(batch_examples["concept_id"].tolist()).to(self.device)
            mag = torch.tensor(batch_examples["factor"].tolist()).to(self.device)
            _, generations = self.ax_model.generate(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
                unit_locations=unit_locations,
                intervene_on_prompt=True,
                subspaces=[{"idx": idx, "steering_factor": mag}]
                * self.number_of_interventions,
                max_new_tokens=eval_output_length,
                do_sample=True,
                temperature=temperature,
                use_cache=False,
            )

            # Decode and print only the generated text without prompt tokens
            input_lengths = [len(input_ids) for input_ids in inputs["input_ids"]]
            generated_texts = [
                self.tokenizer.decode(
                    generation[input_length:], skip_special_tokens=True
                )
                for generation, input_length in zip(generations, input_lengths)
            ]
            all_generations += generated_texts
            all_strengths.extend((mag).tolist())
            progress_bar.update(1)

        return {
            "steered_generation": all_generations,
            "strength": all_strengths,
        }
