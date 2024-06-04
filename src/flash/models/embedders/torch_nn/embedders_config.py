from typing import Any, Dict, List

from pydantic.dataclasses import dataclass

from ..base import BaseConditionerConfig


@dataclass
class TorchNNEmbedderConfig(BaseConditionerConfig):
    """
    Embedder that allows to provides a list of nn modules that are applied in a sequential
    manner to a sample. The output of the last module is returned as the embedding.

    Args:

        input_key (str): The key of the input data in the batch. Defaults to "image".
        nn_modules (List[str]): A list of strings representing the modules to be used. Defaults to None.
        nn_modules_kwargs (List[Dict[str, Any]]): A list of dictionaries containing the kwargs for each module. Defaults to None.
        flatten_output (bool): If True, the output of the embedder will be flattened. Defaults to False.
    """

    nn_modules: List[str] = None
    nn_modules_kwargs: List[Dict[str, Any]] = None
    flatten_output: bool = False
    input_key: str = "image"

    def __post_init__(self):
        super().__post_init__()
        if self.nn_modules is None:
            self.nn_modules = []
        if self.nn_modules_kwargs is None:
            self.nn_modules_kwargs = []
        assert len(self.nn_modules) == len(
            self.nn_modules_kwargs
        ), "Number of modules and kwargs should be same"
