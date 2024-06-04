from typing import Dict, Union

import torch
from diffusers.models import UNet2DConditionModel, UNet2DModel


class DiffusersUNet2DWrapper(UNet2DModel):
    """
    Wrapper for the UNet2DModel from diffusers

    See diffusers' UNet2DModel for more details
    """

    def __init__(self, *args, **kwargs):
        UNet2DModel.__init__(self, *args, **kwargs)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        conditioning: Dict[str, torch.Tensor] = None,
        *args,
        **kwargs
    ):
        """
        The forward pass of the model

        Args:

            sample (torch.Tensor): The input sample
            timesteps (Union[torch.Tensor, float, int]): The number of timesteps
        """
        if conditioning is not None:
            class_labels = conditioning["cond"].get("vector", None)
            concat = conditioning["cond"].get("concat", None)

        else:
            class_labels = None
            concat = None

        if concat is not None:
            sample = torch.cat([sample, concat], dim=1)

        return super().forward(sample, timestep, class_labels).sample

    def freeze(self):
        """
        Freeze the model
        """
        self.eval()
        for param in self.parameters():
            param.requires_grad = False


class DiffusersUNet2DCondWrapper(UNet2DConditionModel):
    """
    Wrapper for the UNet2DConditionModel from diffusers

    See diffusers' Unet2DConditionModel for more details
    """

    def __init__(self, *args, **kwargs):
        UNet2DConditionModel.__init__(self, *args, **kwargs)
        # BaseModel.__init__(self, config=ModelConfig())

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        conditioning: Dict[str, torch.Tensor],
        down_intrablock_additional_residuals: torch.Tensor = None,
        return_intermediate: bool = False,
        *args,
        **kwargs
    ):
        """
        The forward pass of the model

        Args:

            sample (torch.Tensor): The input sample
            timesteps (Union[torch.Tensor, float, int]): The number of timesteps
            conditioning (Dict[str, torch.Tensor]): The conditioning data
            down_intrablock_additional_residuals (List[torch.Tensor]): Residuals for the down intrablocks.
                These residuals typically are used for the T2I adapters.
        """

        assert isinstance(conditioning, dict), "conditionings must be a dictionary"
        # assert "crossattn" in conditioning["cond"], "crossattn must be in conditionings"

        class_labels = conditioning["cond"].get("vector", None)
        crossattn = conditioning["cond"].get("crossattn", None)
        concat = conditioning["cond"].get("concat", None)

        # concat conditioning
        if concat is not None:
            sample = torch.cat([sample, concat], dim=1)

        # down_intrablock_additional_residuals needs to be cloned, since unet will modify it
        if down_intrablock_additional_residuals is not None:
            down_intrablock_additional_residuals_clone = [
                curr_residuals.clone()
                for curr_residuals in down_intrablock_additional_residuals
            ]
        else:
            down_intrablock_additional_residuals_clone = None

        return (
            super()
            .forward(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=crossattn,
                class_labels=class_labels,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals_clone,
                return_intermediate=return_intermediate,
            )
            .sample
        )

    def freeze(self):
        """
        Freeze the model
        """
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
