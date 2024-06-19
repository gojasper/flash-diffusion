from typing import Dict, Optional, Union

import torch
from diffusers.models.transformers import SD3Transformer2DModel, Transformer2DModel

from .utils import AdaLayerNormSingle


class DiffusersTransformer2DWrapper(Transformer2DModel):
    """
    Wrapper for the Transformer2DModel from diffusers

    See diffusers' Transformer2DModel for more details.


    Args:

        time_embed_dim (int): The time embedding dimension
        timesteps_embedding_num_channels (int): The number of timesteps embedding channels
        projection_class_embeddings_input_dim (Optional[int]): The input dimension of the vector embeddings.
        use_concat_vector_conditioning (bool): Whether to use concat vector conditioning. If True one TimestepEmbedding will be created
            for each vector conditionings and the outputs will be concatenated and added to the time embedding. If False, a single
            TimestepEmbedding will be created and the outputs will be added to the time embedding during normalization.
        num_vector_conditionings (Optional[int]): The number of vector conditionings. If use_concat_conditioning is True, then this
            parameter must be provided.
    """

    def __init__(
        self,
        time_embed_dim: int = 256,
        timesteps_embedding_num_channels: int = 256,
        projection_class_embeddings_input_dim: Optional[int] = None,
        use_concat_vector_conditioning: bool = False,
        num_vector_conditionings: Optional[int] = None,
        *args,
        **kwargs
    ):
        Transformer2DModel.__init__(self, *args, **kwargs)

        if self.config.norm_type == "ada_norm_single":
            self.adaln_single = AdaLayerNormSingle(
                time_embed_dim=time_embed_dim,
                timesteps_embedding_num_channels=timesteps_embedding_num_channels,
                projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
                use_concat_conditioning=use_concat_vector_conditioning,
                num_vector_conditionings=num_vector_conditionings,
            )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        conditioning: Dict[str, torch.Tensor],
        hidden_states_masks: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        """
        The forward pass of the model

        Args:

            sample (torch.Tensor): The input sample
            timesteps (Union[torch.Tensor, float, int]): The number of timesteps
            conditioning (Dict[str, torch.Tensor]): The conditioning data
            hidden_states_masks (Optional[torch.Tensor]): The hidden states masks. Defaults to None.
        """

        assert isinstance(conditioning, dict), "conditionings must be a dictionary"

        class_labels = conditioning["cond"].get("vector", None)
        crossattn = conditioning["cond"].get("crossattn", None)
        concat = conditioning["cond"].get("concat", None)
        attention_mask = conditioning["cond"].get("attention_mask", None)

        sample_channels = sample.shape[1]

        # concat conditioning
        if concat is not None:
            sample = torch.cat([sample, concat], dim=1)

        return (
            super()
            .forward(
                hidden_states=sample,
                timestep=timestep,
                encoder_hidden_states=crossattn,
                encoder_attention_mask=attention_mask,
                added_cond_kwargs={"vector_conditioning": class_labels},
            )
            .sample[:, :sample_channels]
        )

    def freeze(self):
        """
        Freeze the model
        """
        self.eval()
        for param in self.parameters():
            param.requires_grad = False


class DiffusersSD3Transformer2DWrapper(SD3Transformer2DModel):
    """
    Wrapper for the Transformer2DModel from diffusers

    See diffusers' Transformer2DModel for more details.
    """

    def __init__(self, *args, **kwargs):
        SD3Transformer2DModel.__init__(self, *args, **kwargs)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        conditioning: Dict[str, torch.Tensor],
        hidden_states_masks: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        """
        The forward pass of the model

        Args:

            sample (torch.Tensor): The input sample
            timesteps (Union[torch.Tensor, float, int]): The number of timesteps
            conditioning (Dict[str, torch.Tensor]): The conditioning data
            hidden_states_masks (Optional[torch.Tensor]): The hidden states masks. Defaults to None.
        """

        assert isinstance(conditioning, dict), "conditionings must be a dictionary"

        class_labels = conditioning["cond"].get("vector", None)
        crossattn = conditioning["cond"].get("crossattn", None)
        concat = conditioning["cond"].get("concat", None)
        attention_mask = conditioning["cond"].get("attention_mask", None)

        sample_channels = sample.shape[1]

        # concat conditioning
        if concat is not None:
            sample = torch.cat([sample, concat], dim=1)

        return (
            super()
            .forward(
                hidden_states=sample,
                timestep=timestep,
                encoder_hidden_states=crossattn,
                pooled_projections=class_labels,
            )
            .sample[:, :sample_channels]
        )

    def freeze(self):
        """
        Freeze the model
        """
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
