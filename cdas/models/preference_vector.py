from typing import List, Literal
import torch

from pyvene import (
    IntervenableConfig,
    IntervenableModel,
)

from .interventions import (
    AdditionIntervention,
    AdditionSuppressionIntervention,
    PreferenceVectorIntervention,
)
from .preference_model import PreferenceModel


class PreferenceVector(PreferenceModel):
    """Rank-1 PreferenceModel."""

    # the base class for all preference models
    preference_pairs = [
        "orig_add"
    ]  # "orig_add", "orig_sub", "steered_add", "steered_sub"

    def __str__(self):
        return "PreferenceVector"

    def make_model(
        self,
        mode: str = "latent",
        preference_pairs: List[
            Literal["orig_add", "orig_sub", "steered_add", "steered_sub"]
        ] = ["orig_add"],
        intervention_type: Literal[
            "addition", "addition_suppression"
        ] = "addition_suppression",
        embed_dim: int = None,
        low_rank_dimension=1,
        overwrite_component=None,
        intervention_positions_dropout=0.0,
        dropout=0.0,
        **kwargs,
    ):

        if mode == "steering":
            if embed_dim is None:
                embed_dim = self.model.config.hidden_size
            if intervention_type == "addition":
                ax = AdditionIntervention(
                    low_rank_dimension=low_rank_dimension, embed_dim=embed_dim
                )
            elif intervention_type == "addition_suppression":
                ax = AdditionSuppressionIntervention(
                    low_rank_dimension=low_rank_dimension, embed_dim=embed_dim
                )
            else:
                raise ValueError(f"Intervention type {intervention_type} not supported")
        else:
            if intervention_type == "addition":
                if embed_dim is None:
                    embed_dim = self.model.config.hidden_size
                ax = PreferenceVectorIntervention(
                    low_rank_dimension=low_rank_dimension,
                    dropout=dropout,
                    intervention_positions_dropout=intervention_positions_dropout,
                    embed_dim=embed_dim,
                )
        self.intervention_type = intervention_type
        layers = self.steering_layers if self.steering_layers else [self.layer]
        self.ax = ax.to(self.device)
        self.ax.train()
        ax_config = IntervenableConfig(
            representations=[
                {
                    "layer": l,
                    "component": (
                        f"model.layers[{l}].output"
                        if overwrite_component is None
                        else overwrite_component
                    ),
                    "low_rank_dimension": low_rank_dimension,
                    "intervention": self.ax,
                }
                for l in layers
            ]
        )
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax_model = ax_model

        self.preference_pairs = preference_pairs
