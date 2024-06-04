from dataclasses import field
from typing import List, Literal, Optional, Union

from pydantic.dataclasses import dataclass

from ..base import ModelConfig


@dataclass
class FlashDiffusionConfig(ModelConfig):
    """
    Configuration class for the FlashDiffusion model.

    Args:

        K (List[int]): The list of number of timesteps for each stage. Defaults to [32, 32, 32, 32, 32].
        num_iterations_per_K (List[int]): The number of iterations for each stage. Defaults to [5000, 10000, 15000, 20000, 25000].
        guidance_scale_min (List[float]): The minimum guidance scale for each stage. Defaults to 3.0.
        guidance_scale_max (List[float]): The maximum guidance scale for each stage. Defaults to 7.0.
        distill_loss_type (Literal["l2", "l1", "lpips"]): The type of distillation loss to use. Defaults to "l2". Choices are "l2", "l1", "lpips".
        ucg_keys (List[str]): The keys to use for classifier-guidance with the teacher model. Defaults to ["text"].
        timestep_distribution (Literal["gaussian", "uniform", "mixture"]): The distribution of timesteps to use. Defaults to "mixture". Choices are "gaussian", "uniform", "mixture".
        mixture_num_components (List[int]): The number of components in the timestep mixture distribution for each stage. Defaults to 4.
        mixture_var (List[float]): The variance of the timestep mixture distribution for each stage. Defaults to 0.5.
        adapter_conditioning_scale (float): The scale of the adapter conditioning. Defaults to 1.0.
        adapter_input_key (Optional[str]): The key to use for the adapter input. Defaults to None.
        use_dmd_loss (bool): Whether to use the DMD loss. Defaults to False.
        dmd_loss_scale (Union[float, List[float]]): The scale of the DMD loss for each stage. Defaults to 1.0.
        distill_loss_scale (Union[float, List[float]]): The scale of the distillation loss for each stage. Defaults to 1.0.
        adversarial_loss_scale (Union[float, List[float]]): The scale of the adversarial loss for each stage. Defaults to 1.0.
        gan_loss_type (Literal["hinge", "vanilla", "non-saturating", "wgan", "lsgan"]): The type of GAN loss to use. Defaults to "hinge". Choices are "hinge", "vanilla", "non-saturating", "wgan", "lsgan".
        mode_probs (Optional[List[List[float]]]): The mode probabilities for the timestep mixture distribution. Defaults to None.
        use_teacher_as_real (bool): Whether to use the teacher model as the real image. Defaults to False.
        use_empty_prompt (bool): Whether to use an empty prompt for classifier-free guidance (useful for SD1.5 and Pixart-Alpha). Defaults to False.
    """

    K: List[int] = field(default_factory=lambda: [32, 32, 32, 32, 32])
    num_iterations_per_K: List[int] = field(
        default_factory=lambda: [5000, 10000, 15000, 20000, 25000]
    )
    guidance_scale_min: Union[float, List[float]] = 3.0
    guidance_scale_max: Union[float, List[float]] = 7.0
    distill_loss_type: Literal["l2", "l1", "lpips"] = "l2"
    ucg_keys: List[str] = field(default_factory=lambda: ["text"])
    timestep_distribution: Literal["gaussian", "uniform", "mixture"] = "mixture"
    mixture_num_components: Union[int, List[int]] = 4
    mixture_var: Union[float, List[float]] = 0.5
    adapter_conditioning_scale: float = 1.0
    adapter_input_key: Optional[str] = None
    use_dmd_loss: bool = False
    dmd_loss_scale: Union[float, List[float]] = 1.0
    distill_loss_scale: Union[float, List[float]] = 1.0
    adversarial_loss_scale: Union[float, List[float]] = 1.0
    gan_loss_type: Literal["hinge", "vanilla", "non-saturating", "wgan", "lsgan"] = (
        "hinge"
    )
    mode_probs: Optional[List[List[float]]] = None
    use_teacher_as_real: bool = False
    use_empty_prompt: bool = False

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.mixture_num_components, int):
            self.mixture_num_components = [self.mixture_num_components] * len(self.K)

        if isinstance(self.guidance_scale_min, float):
            self.guidance_scale_min = [self.guidance_scale_min] * len(self.K)

        if isinstance(self.guidance_scale_max, float):
            self.guidance_scale_max = [self.guidance_scale_max] * len(self.K)

        if isinstance(self.mixture_num_components, int):
            self.mixture_num_components = [self.mixture_num_components] * len(self.K)

        if isinstance(self.mixture_var, float):
            self.mixture_var = [self.mixture_var] * len(self.K)

        if isinstance(self.distill_loss_scale, float):
            self.distill_loss_scale = [self.distill_loss_scale] * len(self.K)

        if isinstance(self.dmd_loss_scale, float):
            self.dmd_loss_scale = [self.dmd_loss_scale] * len(self.K)

        if isinstance(self.adversarial_loss_scale, float):
            self.adversarial_loss_scale = [self.adversarial_loss_scale] * len(self.K)

        if self.mode_probs is None:
            self.mode_probs = [
                [1 / mixtures] * mixtures for mixtures in self.mixture_num_components
            ]

        for i in range(len(self.K)):
            assert len(self.mode_probs[i]) == self.mixture_num_components[i], (
                f"Number of mode probabilities must match number of mixture components for stage {i}, "
                f"got {len(self.mode_probs[i])} mode probabilities and {self.mixture_num_components[i]} mixture components"
            )

        assert len(self.K) == len(
            self.num_iterations_per_K
        ), f"Number of timesteps must match number of iterations, got {len(self.K)} timesteps and {len(self.num_iterations_per_K)} iterations"

        assert len(self.K) == len(
            self.mode_probs
        ), f"Number of timesteps must match number of mode probabilities, got {len(self.K)} timesteps and {len(self.mode_probs)} mode probabilities"
