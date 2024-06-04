from typing import List

import torch
from diffusers import T2IAdapter


class DiffusersT2IAdapterWrapper(T2IAdapter):
    """
    Wrapper for the T2IAdapter from diffusers

    See diffusers' T2IAdapter for more details
    """

    def __init__(self, *args, **kwargs):
        T2IAdapter.__init__(self, *args, **kwargs)

    def forward(self, t2i_adapter_cond: torch.Tensor) -> List[torch.Tensor]:
        return super().forward(t2i_adapter_cond)

    def freeze(self):
        """
        Freeze the model
        """
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
