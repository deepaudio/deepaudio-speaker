import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import Adam, Adagrad, Adadelta, Adamax, AdamW, SGD, ASGD

from deepaudio.speaker.optim import AdamP, RAdam, Novograd
from deepaudio.speaker.criterion import CRITERION_REGISTRY
from deepaudio.speaker.optim.scheduler import SCHEDULER_REGISTRY


class SpeakerModel(pl.LightningModule):
    def __init__(self, configs: DictConfig, num_classes: int) -> None:
        super(SpeakerModel, self).__init__()
        self.configs = configs
        self.num_classes = num_classes
        self.gradient_clip_val = configs.trainer.gradient_clip_val
        self.current_val_loss = 100.0
        self.build_model()
        self.criterion = self.configure_criterion(configs.criterion.name)

    def build_model(self):
        raise NotImplementedError

    def forward(self, inputs: torch.FloatTensor) -> Tensor:
        raise NotImplementedError

    def training_step(self, batch: tuple, batch_idx: int):
        r"""
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        raise NotImplementedError

    def validation_step(self, batch: tuple, batch_idx: int):
        r"""
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        raise NotImplementedError

    def configure_optimizers(self):
        r"""
        Choose what optimizers and learning-rate schedulers to use in your optimization.


        Returns:
            - **Dictionary** - The first item has multiple optimizers, and the second has multiple LR schedulers
                (or multiple ``lr_dict``).
        """
        SUPPORTED_OPTIMIZERS = {
            "adam": Adam,
            "adamp": AdamP,
            "radam": RAdam,
            "adagrad": Adagrad,
            "adadelta": Adadelta,
            "adamax": Adamax,
            "adamw": AdamW,
            "sgd": SGD,
            "asgd": ASGD,
            "novograd": Novograd,
        }

        assert self.configs.model.optimizer in SUPPORTED_OPTIMIZERS.keys(), \
            f"Unsupported Optimizer: {self.configs.model.optimizer}\n" \
            f"Supported Optimizers: {SUPPORTED_OPTIMIZERS.keys()}"

        self.optimizer = SUPPORTED_OPTIMIZERS[self.configs.model.optimizer](
            self.parameters(),
            lr=self.configs.lr_scheduler.lr,
            weight_decay=1e-5,
        )
        scheduler = SCHEDULER_REGISTRY[self.configs.lr_scheduler.scheduler_name](self.optimizer, self.configs)

        if self.configs.lr_scheduler.scheduler_name == "reduce_lr_on_plateau":
            lr_scheduler = {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
            }
        elif self.configs.lr_scheduler.scheduler_name == "warmup_reduce_lr_on_plateau":
            lr_scheduler = {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'step',
            }
        elif self.configs.lr_scheduler.scheduler_name == "warmup_adaptive_reduce_lr_on_plateau":
            lr_scheduler = {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'step',
            }
        else:
            print('by step')
            lr_scheduler = {
                'scheduler': scheduler,
                'interval': 'step',
            }

        return [self.optimizer], [lr_scheduler]

    def configure_criterion(self, criterion_name: str) -> nn.Module:
        r"""
        Configure criterion for training.

        Args:
            criterion_name (str): name of criterion

        Returns:
            criterion (nn.Module): criterion for training
        """

        return CRITERION_REGISTRY[criterion_name](
            configs=self.configs,
            num_classes=self.num_classes,
            embedding_size=self.configs.model.embed_dim
        )

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
