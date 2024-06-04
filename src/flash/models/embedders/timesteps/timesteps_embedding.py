from diffusers.models.embeddings import Timesteps

from ..base import BaseConditioner


class TimestepsEmbedder(BaseConditioner):
    """
    This is the TimestepsEmbedder class which defines the TimestepsEmbedder model

    Args:

        config (TimestepsEmbedderConfig): The config class which defines all the required parameters.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.timesteps = Timesteps(
            num_channels=config.num_channels,
            flip_sin_to_cos=config.flip_sin_to_cos,
            downscale_freq_shift=config.downscale_freq_shift,
        )

    def forward(self, batch, force_zero_embedding: bool = False, *args, **kwargs):
        """
        Forward pass of the embedder.

        Args:

            batch (Dict[str, Any]): A dictionary containing the input data.
            force_zero_embedding (bool): If True, the output of the embedder will be a tensor of zeros.

        Returns:

            Dict[str, Any]: The output of the embedder. This embedder outputs a 1-dimensional conditioning (type "vector").
        """
        x = batch[self.input_key]
        x = self.timesteps(x.flatten()).reshape(x.shape[0], -1)

        if force_zero_embedding:
            x = 0 * x

        return {
            self.dim2outputkey[x.dim()]: x,
        }
