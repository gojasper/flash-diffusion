from pydantic.dataclasses import dataclass

from ..base import BaseConditionerConfig


@dataclass
class TimestepsEmbedderConfig(BaseConditionerConfig):
    """
    Embedder configuration for timesteps embedder.

    Args:

        num_channels (int): The number of channels in the model. Defaults to 256.
        flip_sin_to_cos (bool): Whether to flip the sin to cos. Defaults to True.
        downscale_freq_shift (int): The downscale frequency shift. Defaults to 0.
        input_key (str): The key for the input. Defaults to "timesteps".
    """

    num_channels: int = 256
    flip_sin_to_cos: bool = True
    downscale_freq_shift: int = 0
    input_key: str = "timesteps"
