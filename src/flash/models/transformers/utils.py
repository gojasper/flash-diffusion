from typing import Dict, Optional, Tuple

import torch
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn


class AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Args:

        time_embed_dim (int): The time embedding dimension
        timesteps_embedding_num_channels (int): The number of timesteps embedding channels
        projection_class_embeddings_input_dim (Optional[int]): The input dimension of the vector embeddings.
        use_concat_conditioning (bool): Whether to use concat vector conditioning. If True one TimestepEmbedding will be created
            for each vector conditionings and the outputs will be concatenated and added to the time embedding. If False, a single
            TimestepEmbedding will be created and the outputs will be added to the time embedding.
        num_vector_conditionings (Optional[int]): The number of vector conditionings. If use_concat_conditioning is True, then this
            parameter must be provided.
    """

    def __init__(
        self,
        time_embed_dim,
        timesteps_embedding_num_channels=256,
        projection_class_embeddings_input_dim: Optional[int] = None,
        use_concat_conditioning: bool = False,
        num_vector_conditionings: Optional[int] = None,
    ):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=timesteps_embedding_num_channels,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=timesteps_embedding_num_channels, time_embed_dim=time_embed_dim
        )
        self.projection_class_embeddings_input_dim = (
            projection_class_embeddings_input_dim
        )
        self.use_concat_conditioning = use_concat_conditioning
        self.num_vector_conditionings = num_vector_conditionings

        if self.projection_class_embeddings_input_dim is not None:
            if not use_concat_conditioning:
                self.add_embedding = TimestepEmbedding(
                    in_channels=projection_class_embeddings_input_dim,
                    time_embed_dim=time_embed_dim,
                )
            else:
                assert (
                    num_vector_conditionings is not None
                ), "num_vector_conditionings must be provided if use_concat_conditioning is True"
                self.add_embedding = nn.ModuleList(
                    [
                        TimestepEmbedding(
                            in_channels=projection_class_embeddings_input_dim,
                            time_embed_dim=time_embed_dim // num_vector_conditionings,
                        )
                        for _ in range(num_vector_conditionings)
                    ]
                )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embed_dim, 6 * time_embed_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        timestep_proj = self.time_proj(timestep.reshape(-1)).to(
            timestep.device
        )  # (N, C, H, W)
        timesteps_emb = self.timestep_embedder(timestep_proj)  # (N, D)

        if self.projection_class_embeddings_input_dim is not None:
            vector_conditioning = added_cond_kwargs.get("vector_conditioning", None)
            if isinstance(self.add_embedding, nn.ModuleList):

                vector_cond = torch.chunk(
                    vector_conditioning, self.num_vector_conditionings, dim=1
                )

                timesteps_emb = timesteps_emb + torch.cat(
                    [
                        self.add_embedding[i](vector_cond[i])
                        for i in range(len(self.add_embedding))
                    ],
                    dim=1,
                )
            else:
                timesteps_emb = timesteps_emb + self.add_embedding(vector_conditioning)
        return self.linear(self.silu(timesteps_emb)), timesteps_emb
