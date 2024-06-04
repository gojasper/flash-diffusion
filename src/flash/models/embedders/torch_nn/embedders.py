import importlib
from typing import Any, Dict

import torch.nn as nn

from ..base import BaseConditioner
from .embedders_config import TorchNNEmbedderConfig


class TorchNNEmbedder(BaseConditioner):
    """
    Embedder that allows to provides a list of nn modules that are applied in a sequential
    manner to an image. The output of the last module is returned as the embedding.

    Args:

        config (TorchNNEmbedderConfig): The configuration of the embedder.
    """

    def __init__(self, config: TorchNNEmbedderConfig):
        super().__init__(config)
        self.config = config
        self.flatten_output = config.flatten_output

        chained_modules = []
        for nn_module, module_kwargs in zip(
            config.nn_modules, config.nn_modules_kwargs
        ):
            nn_mod, module_cls = nn_module.rsplit(".", 1)
            module = getattr(importlib.import_module(nn_mod), module_cls)
            chained_modules.append(module(**module_kwargs))

        self.nn_modules = nn.Sequential(*chained_modules)

    def forward(
        self, batch: Dict[str, Any], force_zero_embedding: bool = False, *args, **kwargs
    ):
        """
        Forward pass of the embedder.

        Args:

            batch (Dict[str, Any]): A dictionary containing the input data.
            force_zero_embedding (bool): If True, the output of the embedder will be a tensor of zeros.
        """

        x = batch[self.input_key]
        x = self.nn_modules(x)

        if force_zero_embedding:
            x = 0 * x

        if self.flatten_output:
            x = x.view(x.size(0), -1)

        return {self.dim2outputkey[x.dim()]: x}
