import logging
from typing import Any, Dict

import torch
import wandb
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import make_grid

from ..trainer import TrainingPipeline

logging.basicConfig(level=logging.INFO)


class WandbSampleLogger(Callback):
    """
    Logger for logging samples to wandb. This logger is used to log images, text, and metrics to wandb.

    Args:

        log_batch_freq (int): The frequency of logging samples to wandb. Default is 100.
    """

    def __init__(self, log_batch_freq: int = 100):
        super().__init__()
        self.log_batch_freq = log_batch_freq

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: TrainingPipeline,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.log_samples(trainer, pl_module, outputs, batch, batch_idx, split="train")
        self._process_logs(trainer, outputs, split="train")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: TrainingPipeline,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.log_samples(trainer, pl_module, outputs, batch, batch_idx, split="val")
        self._process_logs(trainer, outputs, split="val")

    @rank_zero_only
    @torch.no_grad()
    def log_samples(
        self,
        trainer: Trainer,
        pl_module: TrainingPipeline,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
        split: str = "train",
    ) -> None:
        if hasattr(pl_module, "log_samples"):
            if batch_idx % self.log_batch_freq == 0:
                is_training = pl_module.training
                if is_training:
                    pl_module.eval()

                logs = pl_module.log_samples(batch)
                logs = self._process_logs(trainer, logs, split=split)

                if is_training:
                    pl_module.train()
        else:
            logging.warning(
                "log_img method not found in LightningModule. Skipping image logging."
            )

    @rank_zero_only
    def _process_logs(
        self, trainer, logs: Dict[str, Any], rescale=True, split="train"
    ) -> Dict[str, Any]:
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if value.dim() == 4:
                    images = value
                    if rescale:
                        images = (images + 1.0) / 2.0
                    grid = make_grid(images, nrow=4)
                    grid = grid.permute(1, 2, 0)
                    grid = grid.mul(255).clamp(0, 255).to(torch.uint8)
                    logs[key] = grid.numpy()
                    trainer.logger.experiment.log(
                        {f"{key}/{split}": [wandb.Image(Image.fromarray(logs[key]))]},
                        step=trainer.global_step,
                    )

                # Scalar tensor
                if value.dim() == 1 or value.dim() == 0:
                    value = value.float().numpy()
                    trainer.logger.experiment.log(
                        {f"{key}/{split}": value}, step=trainer.global_step
                    )

            # list of string (e.g. text)
            if isinstance(value, list):
                if isinstance(value[0], str):
                    text_log = [[caption] for caption in value]
                    table = wandb.Table(data=text_log, columns=["text"])
                    trainer.logger.experiment.log(
                        {f"{key}/{split}": table}, step=trainer.global_step
                    )
                if isinstance(value[0], torch.Tensor):
                    for i, v in enumerate(value):
                        if isinstance(v, torch.Tensor):
                            value[i] = v.detach().cpu().numpy()
                        else:
                            value[i] = v
                    trainer.logger.experiment.log(
                        {f"{key}/{split}": value}, step=trainer.global_step
                    )

            # dict of tensors (e.g. metrics)
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        value[k] = v.detach().cpu().numpy()
                trainer.logger.experiment.log(
                    {f"{key}/{split}": value}, step=trainer.global_step
                )

            if isinstance(value, int) or isinstance(value, float):
                trainer.logger.experiment.log(
                    {f"{key}/{split}": value}, step=trainer.global_step
                )

        return logs
