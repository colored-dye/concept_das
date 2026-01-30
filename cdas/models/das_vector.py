from pathlib import Path
from typing import List, Literal

from pyvene import (
    IntervenableConfig,
    IntervenableModel,
)

from ..constants import CONTRAST_PAIRS_OPTIONS_TYPE
from ..utils.model_utils import set_decoder_norm_to_unit_norm

from .das_model import (
    DASModel,
)
from .mean import LogisticRegressionModel
from .interventions import (
    InterchangeSubspaceIntervention,
    ClampingSubspaceIntervention,
)


class DASVector(DASModel):
    """Rank-1 DASModel."""

    def __str__(self):
        return "DASVector"

    def make_model(
        self,
        mode: Literal["train", "latent", "steering"],
        embed_dim: int = None,
        low_rank_dimension: int = 1,
        contrast_pairs: List[CONTRAST_PAIRS_OPTIONS_TYPE] = ["neg+pos", "pos+neg"],
        **kwargs,
    ):
        if embed_dim is None:
            embed_dim = self.model.config.hidden_size

        if mode == "train": # clamping with source
            ax = InterchangeSubspaceIntervention(
                embed_dim=embed_dim,
                low_rank_dimension=low_rank_dimension,
            )
        elif mode == "latent" or mode == "factor": # No intervention
            ax = LogisticRegressionModel(
                embed_dim=embed_dim,
                low_rank_dimension=low_rank_dimension,
            )
        elif mode == "steering": # clamping
            ax = ClampingSubspaceIntervention(
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
