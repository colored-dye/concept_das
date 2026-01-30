import random
from tqdm.auto import tqdm
import os
import pandas as pd
from pandas import DataFrame
from typing import List, Literal

import torch, einops
from torch.utils.data import DataLoader

from transformers import (
    set_seed, get_scheduler,
)

from pyvene import (
    IntervenableConfig,
    IntervenableModel
)

from .model import Model
from ..constants import CONTRAST_PAIRS_OPTIONS_TYPE
from ..utils.model_utils import (
    set_decoder_norm_to_unit_norm,
    get_lr,
)
from ..utils.data_utils import make_contrast_data_module

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


def kl_loss(
    option: Literal["forward_kl", "reverse_kl"],
    cf_logits,
    sc_logits,
    cf_labels,
    sc_labels,
):
    """
    KL divergence loss.

    Args:
        cf_logits: CounterFactual logits (not shifted) with base inputs.
        sc_logits: SourCe logits (not shifted) with source inputs.
        cf_labels: labels (not shifted) for base inputs.
        sc_labels: labels (not shifted) for source inputs.
    """
    base_labels = cf_labels[:, 1:].contiguous().view(-1)
    source_labels = sc_labels[:, 1:].contiguous().view(-1)
    base_mask = (base_labels != -100)
    source_mask = (source_labels != -100)

    cf_logits = cf_logits[:, :-1].contiguous()
    sc_logits = sc_logits[:, :-1].contiguous()

    cf_logits = cf_logits.view(-1, cf_logits.size(-1))[base_mask]
    sc_logits = sc_logits.view(-1, sc_logits.size(-1))[source_mask]
    cf_logps = cf_logits.log_softmax(dim=-1)
    sc_logps = sc_logits.log_softmax(dim=-1)
    # cf_ps = cf_logits.softmax(dim=-1)
    # sc_ps = sc_logits.softmax(dim=-1)

    # pytorch kl_div computes reverse KL(p || q_\theta) by default
    if option == "reverse_kl":
        loss = torch.nn.functional.kl_div(
            input=cf_logps,
            target=sc_logps,
            reduction='batchmean',
            log_target=True,
        )
    elif option == "forward_kl":
        loss = torch.nn.functional.kl_div(
            input=sc_logps,
            target=cf_logps,
            reduction='batchmean',
            log_target=True,
        )
    else:
        raise ValueError(f"Unknown option: `{option}`")

    return loss


class KLDASModel(Model):
    """
    Ablation for Concept Distributed Alignment Search.
    Use KL divergence objectives.
    """

    ax_model: IntervenableModel
    contrast_pairs: List[CONTRAST_PAIRS_OPTIONS_TYPE]
    """Bi-directional intervention by default."""

    def __str__(self):
        return 'KLDASModel'

    def make_model(self, **kwargs):
        raise NotImplementedError()

    def make_contrast_dataloader(
        self,
        examples: DataFrame,
        source_tokens: str,
        positions: List[str],
        prefix_length: int,
        exclude_bos: bool,
        **kwargs,
    ):
        data_module = make_contrast_data_module(
            tokenizer=self.tokenizer,
            df=examples,
            source_tokens=source_tokens,
            positions=positions,
            prefix_length=prefix_length,
            exclude_bos=exclude_bos,
            **kwargs,
        )
        g = torch.Generator()
        g.manual_seed(self.seed)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, # we shuffle for examples.
            batch_size=self.training_args.batch_size, 
            collate_fn=data_module["data_collator"],
            generator=g)
        return train_dataloader

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

        # old_vec = None
        # sim = 0
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
                    "intervention_locations": [],
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
                        source_inputs["intervention_locations"].append(batch[f"{pair}_source_intervention_locations"][i])

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
                                source_minibatch_inputs["intervention_locations"]
                                .permute(1, 0, 2)
                                .tolist()
                                * len(self.ax),
                                base_minibatch_inputs["intervention_locations"]
                                .permute(1, 0, 2)
                                .tolist()
                                * len(self.ax),
                            )
                        }
                    else:
                        unit_locations = {
                            "sources->base": (
                                source_minibatch_inputs["intervention_locations"]
                                .permute(1, 0, 2)
                                .tolist(),
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
                        sources=[{
                            "input_ids": source_minibatch_inputs['input_ids'],
                            "attention_mask": source_minibatch_inputs['attention_mask'],
                        }],
                        unit_locations=unit_locations,
                        use_cache=False,
                    )

                    with torch.no_grad():
                        source_outputs = self.ax_model.model(
                            input_ids=source_minibatch_inputs['input_ids'],
                            attention_mask=source_minibatch_inputs['attention_mask'],
                        )

                    minibatch_loss = kl_loss(
                        option=self.training_args.loss_type,
                        cf_logits=cf_outputs.logits,
                        sc_logits=source_outputs.logits,
                        cf_labels=base_minibatch_inputs['labels'],
                        sc_labels=source_minibatch_inputs['labels'],
                    )

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

                    # also report geometrical cosine similarity with previous step
                    # with torch.no_grad():
                    #     v = self.ax.proj.weight.clone().detach().view(-1)
                    #     if old_vec is not None:
                    #         sim = torch.dot(v, old_vec)/(v.norm() * old_vec.norm())
                    # old_vec = v

                    progress_bar.set_description(
                        "lr %.6f || loss %.6f" % (
                            curr_lr, loss, ))
                    curr_step += 1

                    set_decoder_norm_to_unit_norm(self.ax)

        progress_bar.close()
