from typing import Literal, Optional

from pydantic.dataclasses import dataclass

from ..base import BaseConditionerConfig


@dataclass
class ClipEmbedderConfig(BaseConditionerConfig):
    """This is the ClipEmbedderConfig class which defines all the useful parameters to instantiate the model

    Args:

        version (str): The version of the model on HF Hub. Defaults to "openai/clip-vit-large-patch14".
        text_embedder_subfolder (str): The subfolder for the text embedder if loaded from an other model. Defaults to "".
        tokenizer_subfolder (str): The subfolder for the tokenizer if loaded from an other model. Defaults to "".
        text_embedder_revision (str): The revision of the text embedder. Defaults to "main".
        tokenizer_revision (str): The revision of the tokenizer. Defaults to "main".
        layer (str): The layer to return. Defaults to "last". Choices are "last", "pooled", "hidden".
        layer_idx (int): The index of the hidden layer to return. Defaults to None.
        always_return_pooled (bool): Whether to always return the pooled output. Defaults to False.
        input_key (str): The key for the input. Defaults to "text".
        pad_token (Optional[str]): Pad token. Defaults to None to use the predefined one from the `version` directory.
        tokenizer_truncation (bool): Whether to truncate the input in the tokenizer. Defaults to True.
        tokenizer_return_length (bool): Whether to return the length of the input in the tokenizer. Defaults to True.

    """

    version: str = "openai/clip-vit-large-patch14"
    text_embedder_subfolder: str = ""
    tokenizer_subfolder: str = ""
    text_embedder_revision: str = "main"
    tokenizer_revision: str = "main"
    layer: Literal["last", "pooled", "hidden"] = "last"
    layer_idx: int = None
    always_return_pooled: bool = False
    input_key: str = "text"
    pad_token: Optional[str] = None
    tokenizer_truncation: bool = True
    tokenizer_return_length: bool = True

    def __post_init__(self):
        super().__post_init__()

        if self.layer == "hidden":
            assert (
                self.layer_idx is not None
            ), "Layer index is required for hidden layer"
            assert (
                0 <= abs(self.layer_idx) <= 12
            ), "Layer index should be between 0 and 12"
