from tqdm.auto import tqdm
import os
import copy
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import List, Literal, Tuple

import torch, einops
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    set_seed, get_scheduler,
)

from pyvene import (
    IntervenableConfig,
    IntervenableModel
)

from .interventions import (
    SubspaceIntervention,
    AdditionIntervention,
    ConceptVectorIntervention
)
from .model import Model
from ..constants import CONTRAST_PAIRS_OPTIONS_TYPE
from ..utils.model_utils import (
    set_decoder_norm_to_unit_norm,
    get_lr,
)
from ..utils.data_utils import make_preference_contrast_data_module

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Ref of Eric's repo: 
    https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L90

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    gemma: float,
    simpo_scaler: float,
    winning_lens: torch.LongTensor,
    losing_lens: torch.LongTensor,
    loss_type: Literal["dpo", "scaled_simpo"] = "scaled_simpo",
    label_smoothing: float = 0.0,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Ref of Eric's repo: 
    https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L45

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        loss_type: different preference loss functions.
            dpo: with reference.
            (scaled_)simpo: no reference.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    # ref_logratios_reverse = reference_rejected_logps - reference_chosen_logps

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if loss_type == "dpo":
        # -log sigmoid ( (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected) )
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    elif loss_type == "scaled_simpo":
        scaled_policy_chosen_logps = (
            torch.max(ref_logratios * simpo_scaler, torch.ones_like(ref_logratios))
            / winning_lens
        ) * policy_chosen_logps
        scaled_policy_rejected_logps = (1.0 / losing_lens) * policy_rejected_logps
        losses = -F.logsigmoid(
            scaled_policy_chosen_logps - scaled_policy_rejected_logps - gemma
        )
    else:
        raise ValueError(f"Loss type {loss_type} not supported")

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


class PDASModel(Model):
    """
    Implementation of steering via Preference-optimization
    and interchange intervention training.

    It is also an ablation for `PreferenceModel` (called "RePS" in the paper)
    regarding the intervention protocol.
    """

    ax_model: IntervenableModel
    contrast_pairs: List[CONTRAST_PAIRS_OPTIONS_TYPE]
    """Bi-directional intervention by default."""

    def __str__(self):
        return 'PDASModel'

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
        data_module = make_preference_contrast_data_module(
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
            data_module["train_dataset"],
            shuffle=True,  # we shuffle for examples.
            batch_size=self.training_args.batch_size,
            collate_fn=data_module["data_collator"],
            generator=g,
        )
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

        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                for k, v in batch.items():
                    bs = len(v)
                    break
                minibatch_size = min(self.training_args.batch_size, bs)
                expanded_batch_size = minibatch_size * len(self.contrast_pairs)
                num_minibatches = (expanded_batch_size + minibatch_size - 1) // minibatch_size

                base_winning_inputs = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                    "intervention_locations": [],
                }
                base_losing_inputs = copy.deepcopy(base_winning_inputs)
                source_winning_inputs = copy.deepcopy(base_winning_inputs)
                source_losing_inputs = copy.deepcopy(base_winning_inputs)

                for i in range(minibatch_size):
                    for pair in self.contrast_pairs:
                        base_winning_inputs["input_ids"].append(batch[f"{pair}_base_winning_input_ids"][i])
                        base_winning_inputs["attention_mask"].append(batch[f"{pair}_base_winning_attention_mask"][i])
                        base_winning_inputs["labels"].append(batch[f"{pair}_base_winning_labels"][i])
                        base_winning_inputs["intervention_locations"].append(batch[f"{pair}_base_winning_intervention_locations"][i])

                        base_losing_inputs["input_ids"].append(batch[f"{pair}_base_losing_input_ids"][i])
                        base_losing_inputs["attention_mask"].append(batch[f"{pair}_base_losing_attention_mask"][i])
                        base_losing_inputs["labels"].append(batch[f"{pair}_base_losing_labels"][i])
                        base_losing_inputs["intervention_locations"].append(batch[f"{pair}_base_losing_intervention_locations"][i])

                        source_winning_inputs["input_ids"].append(batch[f"{pair}_source_winning_input_ids"][i])
                        source_winning_inputs["attention_mask"].append(batch[f"{pair}_source_winning_attention_mask"][i])
                        source_winning_inputs["labels"].append(batch[f"{pair}_source_winning_labels"][i])
                        source_winning_inputs["intervention_locations"].append(batch[f"{pair}_source_winning_intervention_locations"][i])

                        source_losing_inputs["input_ids"].append(batch[f"{pair}_source_losing_input_ids"][i])
                        source_losing_inputs["attention_mask"].append(batch[f"{pair}_source_losing_attention_mask"][i])
                        source_losing_inputs["labels"].append(batch[f"{pair}_source_losing_labels"][i])
                        source_losing_inputs["intervention_locations"].append(batch[f"{pair}_source_losing_intervention_locations"][i])

                loss_sum = 0
                batch_metrics = {}
                for mb in range(num_minibatches):
                    start_idx = mb * minibatch_size
                    end_idx = min((mb + 1) * minibatch_size, expanded_batch_size)

                    if start_idx >= expanded_batch_size:
                        break

                    base_winning_minibatch_inputs = {
                        k: torch.stack(base_winning_inputs[k][start_idx:end_idx], dim=0).to(self.device)
                        for k, _ in base_winning_inputs.items()
                    }
                    base_losing_minibatch_inputs = {
                        k: torch.stack(base_losing_inputs[k][start_idx:end_idx], dim=0).to(self.device)
                        for k, _ in base_losing_inputs.items()
                    }
                    source_winning_minibatch_inputs = {
                        k: torch.stack(source_winning_inputs[k][start_idx:end_idx], dim=0).to(self.device)
                        for k, _ in source_winning_inputs.items()
                    }
                    source_losing_minibatch_inputs = {
                        k: torch.stack(source_losing_inputs[k][start_idx:end_idx], dim=0).to(self.device)
                        for k, _ in source_losing_inputs.items()
                    }

                    winning_unit_locations = {
                        "sources->base": (
                            source_winning_minibatch_inputs["intervention_locations"]
                            .permute(1, 0, 2)
                            .tolist(),
                            base_winning_minibatch_inputs["intervention_locations"]
                            .permute(1, 0, 2)
                            .tolist(),
                        )
                    }
                    ref_winning_outputs, policy_winning_outputs = self.ax_model(
                        base={
                            "input_ids": base_winning_minibatch_inputs['input_ids'],
                            "attention_mask": base_winning_minibatch_inputs['attention_mask'],
                        },
                        sources=[{
                            "input_ids": source_winning_minibatch_inputs['input_ids'],
                            "attention_mask": source_winning_minibatch_inputs['attention_mask'],
                        }],
                        unit_locations=winning_unit_locations,
                        output_original_output=False if self.training_args.loss_type == "dpo" else True,
                        use_cache=False,
                    )
                    chosen_logps = _get_batch_logps(
                        policy_winning_outputs.logits,
                        base_winning_minibatch_inputs["labels"],
                        average_log_prob=False,
                    )
                    # ref_chosen_logps = _get_batch_logps(
                    #     ref_winning_outputs.logits,
                    #     base_winning_minibatch_inputs["labels"],
                    #     average_log_prob=False,
                    # )
                    if self.training_args.loss_type == "dpo":
                        ref_winning_outputs = self.ax_model.model(
                            input_ids=source_winning_minibatch_inputs['input_ids'],
                            attention_mask=source_winning_minibatch_inputs['attention_mask'],
                        )
                        ref_chosen_logps = _get_batch_logps(
                            ref_winning_outputs.logits,
                            source_winning_minibatch_inputs["labels"],
                            average_log_prob=False,
                        )
                    else:
                        ref_chosen_logps = _get_batch_logps(
                            ref_winning_outputs.logits,
                            base_winning_minibatch_inputs["labels"],
                            average_log_prob=False,
                        )

                    losing_unit_locations = {
                        "sources->base": (
                            source_losing_minibatch_inputs["intervention_locations"]
                            .permute(1, 0, 2)
                            .tolist(),
                            base_losing_minibatch_inputs["intervention_locations"]
                            .permute(1, 0, 2)
                            .tolist(),
                        )
                    }
                    ref_losing_outputs, policy_losing_outputs = self.ax_model(
                        base={
                            "input_ids": base_losing_minibatch_inputs['input_ids'],
                            "attention_mask": base_losing_minibatch_inputs['attention_mask'],
                        },
                        sources=[{
                            "input_ids": source_losing_minibatch_inputs['input_ids'],
                            "attention_mask": source_losing_minibatch_inputs['attention_mask'],
                        }],
                        unit_locations=losing_unit_locations,
                        output_original_output=False if self.training_args.loss_type == "dpo" else True,
                        use_cache=False,
                    )
                    rejected_logps = _get_batch_logps(
                        policy_losing_outputs.logits,
                        base_losing_minibatch_inputs["labels"],
                        average_log_prob=False,
                    )
                    # ref_rejected_logps = _get_batch_logps(
                    #     ref_losing_outputs.logits,
                    #     base_losing_minibatch_inputs["labels"],
                    #     average_log_prob=False,
                    # )
                    if self.training_args.loss_type == "dpo":
                        ref_losing_outputs = self.ax_model.model(
                            input_ids=source_losing_minibatch_inputs['input_ids'],
                            attention_mask=source_losing_minibatch_inputs['attention_mask'],
                        )
                        ref_rejected_logps = _get_batch_logps(
                            ref_losing_outputs.logits,
                            source_losing_minibatch_inputs["labels"],
                            average_log_prob=False,
                        )
                    else:
                        ref_rejected_logps = _get_batch_logps(
                            ref_losing_outputs.logits,
                            base_losing_minibatch_inputs["labels"],
                            average_log_prob=False,
                        )

                    def get_lens(labels):
                        mask = labels != -100
                        return mask.sum(dim=-1)
                    winning_lens = get_lens(base_winning_minibatch_inputs["labels"])
                    losing_lens = get_lens(base_losing_minibatch_inputs["labels"])

                    steer_losses, steer_chosen_rewards, steer_rejected_rewards = preference_loss(
                        policy_chosen_logps=chosen_logps,
                        policy_rejected_logps=rejected_logps,
                        reference_chosen_logps=ref_chosen_logps,
                        reference_rejected_logps=ref_rejected_logps,
                        beta=self.training_args.beta,
                        gemma=self.training_args.gemma,
                        simpo_scaler=self.training_args.simpo_scaler,
                        label_smoothing=self.training_args.label_smoothing,
                        winning_lens=winning_lens,
                        losing_lens=losing_lens,
                        loss_type=self.training_args.loss_type,
                    )

                    steer_loss = steer_losses.mean()
                    minibatch_loss = steer_loss

                    minibatch_loss = minibatch_loss / (num_minibatches * self.training_args.gradient_accumulation_steps)
                    minibatch_loss.backward()

                    loss_sum += steer_loss.detach() * (end_idx - start_idx)

                    # Accumulate metrics for this minibatch
                    minibatch_metrics = self._compute_metrics(
                        chosen_logps=chosen_logps,
                        rejected_logps=rejected_logps,
                        ref_chosen_logps=ref_chosen_logps,
                        ref_rejected_logps=ref_rejected_logps,
                        chosen_rewards=steer_chosen_rewards,
                        rejected_rewards=steer_rejected_rewards,
                        losses=steer_losses,
                    )

                    # Accumulate metrics
                    for k, v in minibatch_metrics.items():
                        if k not in batch_metrics:
                            batch_metrics[k] = [v * (end_idx - start_idx)]
                        else:
                            batch_metrics[k].append(v * (end_idx - start_idx))
                loss = loss_sum / expanded_batch_size

                # Calculate the average loss and metrics across all minibatches
                metrics = {}
                for k, v in batch_metrics.items():
                    metrics[k] = sum(v) / expanded_batch_size
                metrics[f'loss/train'] = loss.cpu().float().numpy().tolist()
                metrics[f'loss/steer'] = loss.cpu().float().numpy().tolist()

                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.ax_model.get_trainable_parameters(), 1.0)
                    curr_lr = get_lr(optimizer)
                    # optim
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)

                    progress_bar.set_description(
                        "lr %.6f || loss %.6f || steer acc %.6f" % (
                            curr_lr, loss, metrics.get('rewards_train/steer_accuracies', 0.0)))
                    curr_step += 1

                    set_decoder_norm_to_unit_norm(self.ax)

        progress_bar.close()

    def _compute_metrics(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, 
                         chosen_rewards, rejected_rewards, losses):
        """Helper method to compute metrics for a minibatch"""
        metrics = {}

        # Compute reward accuracies
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics[f'rewards_train/steer_chosen_rewards'] = chosen_rewards.mean().cpu().float().numpy().tolist()
        metrics[f'rewards_train/steer_rejected_rewards'] = rejected_rewards.mean().cpu().float().numpy().tolist()
        metrics[f'rewards_train/steer_margins'] = (chosen_rewards - rejected_rewards).mean().cpu().float().numpy().tolist()
        metrics[f'rewards_train/pos_steer_reward_accuracies'] = np.array(reward_accuracies.cpu().numpy().tolist()[:len(reward_accuracies)//2]).mean()
        metrics[f'rewards_train/neg_steer_reward_accuracies'] = np.array(reward_accuracies.cpu().numpy().tolist()[len(reward_accuracies)//2:]).mean()
        metrics[f'rewards_train/steer_accuracies'] = reward_accuracies.mean().cpu().numpy().tolist()
        metrics[f'logps_train/steer_chosen'] = chosen_logps.detach().mean().cpu().float().numpy().tolist()
        metrics[f'logps_train/steer_rejected'] = rejected_logps.detach().mean().cpu().float().numpy().tolist()
        metrics[f'loss/steer'] = losses.mean().detach().cpu().float().numpy().tolist()

        return metrics
