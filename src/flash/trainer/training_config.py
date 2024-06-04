from dataclasses import field
from typing import List, Literal, Optional, Union

from pydantic.dataclasses import dataclass

from ..config import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
    """
    Configuration for the training pipeline

    Args:

        experiment_id (str):
            The experiment id for the training run. If not provided, a random id will be generated.
        optimizers_name (List[str]):
            The list of optimizers to use. Default is ["AdamW"]. Choices are "Adam", "AdamW", "Adadelta", "Adagrad", "RMSprop", "SGD"
        optimizers_kwargs (List[Dict[str, Any]])
            The optimizers kwargs. Default is [{}]
        learning_rates (List[float]):
            The learning rates to use for each optimizer. Default is [1e-3]
        lr_schedulers_name (List[str]):
            The learning rate schedulers to use. Default is [None]. Choices are "StepLR", "CosineAnnealingLR",
            "CosineAnnealingWarmRestarts", "ReduceLROnPlateau", "ExponentialLR"
        lr_schedulers_kwargs (List[Dict[str, Any]])
            The learning rate schedulers kwargs. Default is [{}]
        lr_schedulers_interval (List[str]):
            The learning rate scheduler intervals. Default is ["step"]. Choices are "step", "epoch"
        lr_schedulers_frequency (List[int]):
            The learning rate scheduler frequency. Default is 1
        metrics (List[str])
            The metrics to use. Default is None
        tracking_metrics: Optional[List[str]]
            The metrics to track. Default is None
        backup_every (int):
            The frequency to backup the model. Default is 50.
        trainable_params (Union[List[str], List[List[str]]]):
            Regexes indicateing the parameters to train for each optimizer.
            Default is [["./*"]] (i.e. all parameters are trainable)
        log_keys: Union[str, List[str]]:
            The keys to log when sampling from the model. Default is "txt"
        log_samples_model_kwargs (Dict[str, Any]):
            The kwargs for logging samples from the model. Default is {
                "max_samples": 8,
                "num_steps": 20,
                "input_shape": (4, 32, 32),
                "guidance_scale": 7.5,
            }
    """

    experiment_id: Optional[str] = None
    optimizers_name: List[
        Literal["Adam", "AdamW", "Adadelta", "Adagrad", "RMSprop", "SGD"]
    ] = field(default_factory=lambda: ["AdamW"])
    optimizers_kwargs: Optional[Union[List[dict]]] = field(default_factory=lambda: [{}])
    learning_rates: List[float] = field(default_factory=lambda: [1e-3])
    lr_schedulers_name: Optional[
        List[
            Literal[
                "StepLR",
                "CosineAnnealingLR",
                "CosineAnnealingWarmRestarts",
                "ReduceLROnPlateau",
                "ExponentialLR",
                None,
            ]
        ]
    ] = field(default_factory=lambda: [None])
    lr_schedulers_kwargs: Optional[List[dict]] = field(default_factory=lambda: [{}])
    lr_schedulers_interval: Optional[List[Literal["step", "epoch", None]]] = field(
        default_factory=lambda: ["step"]
    )
    lr_schedulers_frequency: Optional[List[Union[int, None]]] = field(
        default_factory=lambda: [1]
    )
    metrics: Optional[List[str]] = None
    tracking_metrics: Optional[List[str]] = None
    backup_every: int = 50
    trainable_params: Optional[List[List[str]]] = field(
        default_factory=lambda: [["./*"]]
    )
    log_keys: Optional[Union[str, List[str]]] = "txt"
    log_samples_model_kwargs: Optional[dict] = field(
        default_factory=lambda: {
            "max_samples": 8,
            "num_steps": 20,
            "input_shape": (4, 32, 32),
            "guidance_scale": 7.5,
        }
    )

    def __post_init__(self):
        # if optimizers_kwargs provided check len
        if self.optimizers_kwargs != [{}]:
            assert len(self.optimizers_name) == len(
                self.optimizers_kwargs
            ), f"The length of optimizers_name ({len(self.optimizers_name)}) must be equal to the length of optimizers_kwargs ({len(self.optimizers_kwargs)})"
        else:
            self.optimizers_kwargs = [{} for _ in self.optimizers_name]

        if self.trainable_params != [[".*"]]:
            assert len(self.optimizers_name) == len(
                self.trainable_params
            ), f"The length of optimizers_name ({len(self.optimizers_name)}) must be equal to the length of trainable_params ({len(self.trainable_params)})"
        else:
            self.trainable_params = [[".*"] for _ in self.optimizers_name]

        # if lr_scheduler_kwargs provided check len
        if self.lr_schedulers_kwargs != [{}]:
            assert len(self.lr_schedulers_name) == len(
                self.lr_schedulers_kwargs
            ), f"The length of lr_schedulers_name ({len(self.lr_schedulers_name)}) must be equal to the length of lr_schedulers_kwargs ({len(self.lr_schedulers_kwargs)})"
            if self.lr_schedulers_frequency != [1]:
                assert len(self.lr_schedulers_name) == len(
                    self.lr_schedulers_frequency
                ), f"The length of lr_schedulers_name ({len(self.lr_schedulers_name)}) must be equal to the length of lr_schedulers_frequency ({len(self.lr_schedulers_frequency)})"
            else:
                self.lr_schedulers_frequency = [1 for _ in self.lr_schedulers_name]

            if self.lr_schedulers_interval != ["step"]:
                assert len(self.lr_schedulers_name) == len(
                    self.lr_schedulers_interval
                ), f"The length of lr_schedulers_name ({len(self.lr_schedulers_name)}) must be equal to the length of lr_schedulers_interval ({len(self.lr_schedulers_interval)})"
            else:
                self.lr_schedulers_interval = ["step" for _ in self.lr_schedulers_name]

        else:
            self.lr_schedulers_kwargs = [{} for _ in self.lr_schedulers_name]

        assert len(self.optimizers_name) == len(
            self.learning_rates
        ), f"The length of optimizers_name ({len(self.optimizers_name)}) must be equal to the length of learning_rates ({len(self.learning_rates)})"

        super().__post_init__()
