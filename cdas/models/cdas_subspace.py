from pathlib import Path
from typing import List, Literal

import torch
from torch import nn
from pyvene import (
    IntervenableConfig,
    IntervenableModel,
    TrainableIntervention,
    DistributedRepresentationIntervention,
    SourcelessIntervention,
)

from ..constants import CONTRAST_PAIRS_OPTIONS_TYPE

from .cdas_model import (
    CDASModel,
)
from .mean import LogisticRegressionModel


class TrainingSubspaceIntervention(
    TrainableIntervention,
    DistributedRepresentationIntervention,
):

    def __init__(self, low_rank_dimension, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        proj = torch.nn.Linear(self.embed_dim, low_rank_dimension, bias=True)
        with torch.no_grad():
            torch.nn.init.orthogonal_(proj.weight)
            proj.weight.data = nn.functional.normalize(proj.weight.data, dim=1, p=2)
            proj.bias.fill_(0)
        self.proj = proj

    def forward(self, base, source, subspaces=None, **kwargs):
        rotated_base = torch.matmul(base.to(self.proj.weight.dtype), self.proj.weight.T)
        rotated_source = torch.matmul(source.to(self.proj.weight.dtype), self.proj.weight.T)
        output = base + torch.matmul(
            (rotated_source - rotated_base),
            self.proj.weight,
        )
        return output.to(base.dtype)


class SteeringSubspaceIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Used for inference-time steering via clamping.
    """
    def __init__(self, low_rank_dimension, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        proj = torch.nn.Linear(self.embed_dim, low_rank_dimension, bias=True)
        with torch.no_grad():
            proj.bias.fill_(0)
        self.proj = proj

    def forward(self, base, source=None, subspaces=None):
        assert source is None

        prefix_length = subspaces["prefix_length"]
        if base.shape[1] > 1:
            cached_base_prefix = base[:,:prefix_length].clone()

        dtype = self.proj.weight.dtype
        rotated_base = torch.matmul(base.to(dtype=dtype), self.proj.weight.T)

        factor = subspaces["mag"].unsqueeze(-2)
        diff = factor - rotated_base

        output = base + torch.matmul(diff, self.proj.weight)
        if base.shape[1] > 1:
            output[:, :prefix_length] = cached_base_prefix
        return output.to(dtype=base.dtype)


class CDASSubspace(CDASModel):
    """Rank-r CDASModel."""

    def __str__(self):
        return "CDASSubspace"

    def make_model(
        self,
        mode: Literal["train", "latent", "factor", "steering"],
        embed_dim: int = None,
        low_rank_dimension: int = 2,
        contrast_pairs: List[CONTRAST_PAIRS_OPTIONS_TYPE] = ["neg+pos", "pos+neg"],
        **kwargs,
    ):
        if embed_dim is None:
            embed_dim = self.model.config.hidden_size

        if mode == "train": # clamping with source
            ax = TrainingSubspaceIntervention(
                embed_dim=embed_dim,
                low_rank_dimension=low_rank_dimension,
            )
        elif mode == "latent" or mode == "factor": # No intervention
            ax = LogisticRegressionModel(
                embed_dim=embed_dim,
                low_rank_dimension=low_rank_dimension,
            )
        elif mode == "steering": # clamping
            ax = SteeringSubspaceIntervention(
                embed_dim=embed_dim,
                low_rank_dimension=low_rank_dimension,
            )
        else:
            raise ValueError(f"Unknown mode: `{mode}`.")

        self.ax = ax.to(self.device)
        self.ax.train()

        layers = self.steering_layers if self.steering_layers else [self.layer]
        ax_config = IntervenableConfig(
            representations=[
                {
                    "layer": l,
                    "component": f"model.layers[{l}].output",
                    "low_rank_dimension": low_rank_dimension,
                    "intervention": self.ax,
                }
                for l in layers
            ]
        )
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax_model = ax_model
        self.contrast_pairs = contrast_pairs

    def load(self, dump_dir=None, concept_id=None, priority_mode="compute_priority", **kwargs):
        self.priority_mode = priority_mode
        self.concept_id = concept_id
        model_name = kwargs.get("model_name", self.__str__())
        print(f"Loading {model_name} from {dump_dir}.")

        weight = torch.load(f"{dump_dir}/{model_name}_weight.pt", map_location='cpu', weights_only=True)
        bias = torch.load(f"{dump_dir}/{model_name}_bias.pt", map_location='cpu', weights_only=True)
        kwargs['low_rank_dimension'] = weight.shape[0]
        self.make_model(concept_id=concept_id, **kwargs)
        self.ax.proj.weight.data = weight.to(self.device)
        self.ax.proj.bias.data = bias.to(self.device)
