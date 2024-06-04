import importlib
import logging
import re
import time
from typing import Any, Dict

import pytorch_lightning as pl
import torch

from ..models.base.base_model import BaseModel
from .training_config import TrainingConfig

logging.basicConfig(level=logging.INFO)


class TrainingPipeline(pl.LightningModule):
    """
    Main Training Pipeline class

    Args:

        model (BaseModel): The model to train
        pipeline_config (TrainingConfig): The configuration for the training pipeline
        verbose (bool): Whether to print logs in the console. Default is False.
    """

    def __init__(
        self,
        model: BaseModel,
        pipeline_config: TrainingConfig,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.pipeline_config = pipeline_config
        self.log_samples_model_kwargs = pipeline_config.log_samples_model_kwargs

        # save hyperparameters.
        self.save_hyperparameters(ignore="model")
        self.save_hyperparameters({"model_config": model.config.to_dict()})

        # logger.
        self.verbose = verbose

        # setup logging.
        log_keys = pipeline_config.log_keys

        if isinstance(log_keys, str):
            log_keys = [log_keys]

        if log_keys is None:
            log_keys = []

        self.log_keys = log_keys

    def on_train_start(self) -> None:
        if self.global_rank == 0:
            self.timer = time.perf_counter()

    def on_train_batch_end(
        self, outputs: Dict[str, Any], batch: Any, batch_idx: int
    ) -> None:
        if self.global_rank == 0:
            logging.debug("on_train_batch_end")
        self.model.on_train_batch_end(batch)

        average_time_frequency = 10
        if self.global_rank == 0 and batch_idx % average_time_frequency == 0:
            delta = time.perf_counter() - self.timer
            logging.info(
                f"Average time per batch {batch_idx} took {delta / (batch_idx + 1)} seconds"
            )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Setup optimizers and learning rate schedulers.
        """
        optimizers = []
        for i in range(len(self.pipeline_config.optimizers_name)):
            lr = self.pipeline_config.learning_rates[i]
            param_list = []
            n_params = 0
            param_list_ = {"params": []}
            for name, param in self.model.named_parameters():
                for regex in self.pipeline_config.trainable_params[i]:
                    pattern = re.compile(regex)
                    if re.match(pattern, name):
                        if param.requires_grad:
                            param_list_["params"].append(param)
                            n_params += param.numel()

            param_list.append(param_list_)

            logging.info(
                f"Number of trainable parameters for optimizer {i}: {n_params}"
            )

            optimizer_cls = getattr(
                importlib.import_module("torch.optim"),
                self.pipeline_config.optimizers_name[i],
            )
            optimizer = optimizer_cls(
                param_list, lr=lr, **self.pipeline_config.optimizers_kwargs[i]
            )
            optimizers.append(optimizer)

        if len(optimizers) > 1:
            self.automatic_optimization = False

        self.optims = optimizers
        schedulers_config = self.configure_lr_schedulers()

        for name, param in self.model.named_parameters():
            set_grad_false = True
            for regexes in self.pipeline_config.trainable_params:
                for regex in regexes:
                    pattern = re.compile(regex)
                    if re.match(pattern, name):
                        if param.requires_grad:
                            set_grad_false = False
            if set_grad_false:
                param.requires_grad = False

        num_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        logging.info(f"Number of trainable parameters: {num_trainable_params}")

        schedulers_config = self.configure_lr_schedulers()

        if schedulers_config is None:
            return optimizers

        return optimizers, [
            schedulers_config_ for schedulers_config_ in schedulers_config
        ]

    def configure_lr_schedulers(self):
        schedulers_config = []
        for i in range(len(self.pipeline_config.lr_schedulers_name)):
            if self.pipeline_config.lr_schedulers_name[i] is None:
                scheduler = None
                schedulers_config.append(scheduler)
            else:
                scheduler_cls = getattr(
                    importlib.import_module("torch.optim.lr_scheduler"),
                    self.pipeline_config.lr_schedulers_name[i],
                )
                scheduler = scheduler_cls(
                    self.optims[i],
                    **self.pipeline_config.lr_schedulers_kwargs[i],
                )
                lr_scheduler_config = {
                    "scheduler": scheduler,
                    "interval": self.pipeline_config.lr_schedulers_interval[i],
                    "monitor": "val_loss",
                    "frequency": self.pipeline_config.lr_schedulers_frequency[i],
                }
                schedulers_config.append(lr_scheduler_config)

        if all([scheduler is None for scheduler in schedulers_config]):
            return None

        return schedulers_config

    def training_step(self, train_batch: Dict[str, Any], batch_idx: int) -> dict:
        if self.automatic_optimization:
            out = self.model(train_batch, device=self.device)
            loss = out["loss"]
            if "start_timestep" in out:
                start_timestep = out["start_timestep"]
            else:
                start_timestep = None
            logging.info("########################")
            logging.info(f"loss: {loss}")
            logging.info(f"start_timestep: {start_timestep}")
            logging.info("########################")
            return {
                "loss": loss,
                "batch_idx": batch_idx,
                "start_timestep": start_timestep,
            }
        # manual optim for multiple optimizers
        else:
            optimizers = self.optimizers()
            # assert len(loss) == len(
            #     optimizers
            # ), "Number of losses must match number of optimizers"
            outputs = {"batch_idx": batch_idx}
            # flag_opt_num = 0
            for i, opt in enumerate(optimizers):
                # if i == 0 and batch_idx < 10000:
                #     continue
                # if batch_idx % 100 == 0:
                #     flag_opt_num = 1 - flag_opt_num
                # if i != flag_opt_num:
                #     continue

                model_output = self.model(
                    train_batch, device=self.device, step=i, batch_idx=batch_idx
                )
                loss = model_output["loss"]
                if "start_timestep" in model_output:
                    start_timestep = model_output["start_timestep"]
                    outputs["start_timestep"] = start_timestep
                logging.info("########################")
                logging.info(f"loss for optimizer {i}: {loss[i]}")
                logging.info("########################")
                outputs[f"loss_optimizer_{i}"] = loss[i]
                self.toggle_optimizer(optimizers[i])
                opt.zero_grad()
                self.manual_backward(loss[i])
                opt.step()
                self.untoggle_optimizer(optimizers[i])
            return outputs

    def validation_step(self, val_batch: Dict[str, Any], val_idx: int) -> dict:
        loss = self.model(val_batch, device=self.device)["loss"]

        metrics = self.model.compute_metrics(val_batch)

        return {"loss": loss, "metrics": metrics}

    def log_samples(self, batch: Dict[str, Any]):
        logging.info("########################")
        logging.info("log_samples")
        logging.info("########################")
        logs = self.model.log_samples(
            batch,
            device=self.device,
            **self.log_samples_model_kwargs,
        )

        if logs is not None:
            N = min([logs[keys].shape[0] for keys in logs])
        else:
            N = 0

        # Log inputs
        if self.log_keys is not None:
            for key in self.log_keys:
                if key in batch:
                    if N > 0:
                        logs[key] = batch[key][:N]
                    else:
                        logs[key] = batch[key]

        return logs
