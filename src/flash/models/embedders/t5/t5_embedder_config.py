from typing import Any, Dict, List, Literal, Optional

from pydantic.dataclasses import dataclass

from ..base import BaseConditionerConfig


@dataclass
class T5TextEmbedderConfig(BaseConditionerConfig):
    """This is the T5TextEmbedderConfig class which defines all the useful parameters to instantiate the model

    Args:

        version (str): The version of the model on HF Hub. Defaults to "google/flan-t5-xxl".
        text_embedder_subfolder (str): The subfolder for the text embedder if loaded from an other model. Defaults to "".
        tokenizer_subfolder (str): The subfolder for the tokenizer if loaded from an other model. Defaults to "".
        text_embedder_revision (str): The revision of the text embedder. Defaults to "main".
        tokenizer_revision (str): The revision of the tokenizer. Defaults to "main".
        layer (str): The layer to return. Defaults to "last". Choices are "last", "pooled", "hidden".
        layer_idx (int): The index of the hidden layer to return. Defaults to None.
        input_key (str): The key for the input. Defaults to "text".
        tokenizer_max_length (int): The maximum length of the tokenizer. Defaults to None.
        projection_nn_modules (List[str]): A list of strings representing the modules to be used to project the text embedding.
            Defaults to None *i.e.* no projection.
        projection_nn_modules_kwargs (List[Dict[str, Any]]): A list of dictionaries containing the kwargs for each module.
            Defaults to None.
        returns_attention_mask (bool): Whether to return the attention mask. Defaults to False.
        tokenizer_truncation (bool): Whether to truncate the input in the tokenizer. Defaults to True.
        tokenizer_return_length (bool): Whether to return the length of the input in the tokenizer. Defaults to True.
        tokenizer_add_special_tokens (bool): Whether to add special tokens in the tokenizer. Defaults to True.
    """

    version: str = "google/flan-t5-xxl"
    text_embedder_subfolder: str = ""
    tokenizer_subfolder: str = ""
    text_embedder_revision: str = "main"
    tokenizer_revision: str = "main"
    layer: Literal["last", "hidden"] = "last"
    layer_idx: int = None
    input_key: str = "text"
    tokenizer_max_length: Optional[int] = None
    projection_nn_modules: Optional[List[str]] = None
    projection_nn_modules_kwargs: Optional[List[Dict[str, Any]]] = None
    returns_attention_mask: bool = False
    tokenizer_truncation: bool = True
    tokenizer_return_length: bool = True
    tokenizer_add_special_tokens: bool = True

    def __post_init__(self):
        super().__post_init__()

        if self.layer == "hidden":
            assert (
                self.layer_idx is not None
            ), "Layer index is required for hidden layer"
            assert (
                0 <= abs(self.layer_idx) <= 24
            ), "Layer index should be between 0 and 24"

        if self.projection_nn_modules is None:
            self.projection_nn_modules = []
        if self.projection_nn_modules_kwargs is None:
            self.projection_nn_modules_kwargs = []
        assert len(self.projection_nn_modules) == len(
            self.projection_nn_modules_kwargs
        ), "Number of modules and kwargs should be same"
