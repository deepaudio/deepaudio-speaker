import torch
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import DictConfig
from torch.optim import Optimizer

from deepaudio.speaker.dataclass.configurations import LearningRateSchedulerConfigs
from deepaudio.speaker.optim.scheduler import register_scheduler
from deepaudio.speaker.optim.scheduler.lr_scheduler import LearningRateScheduler


@dataclass
class FixLRSchedulerConfigs(LearningRateSchedulerConfigs):
    scheduler_name: str = field(
        default="fix", metadata={"help": "Name of learning rate scheduler."}
    )
    peak_lr: float = field(
        default=1e-04, metadata={"help": "Maximum learning rate."}
    )


@register_scheduler("fix", dataclass=FixLRSchedulerConfigs)
class FixLRScheduler(LearningRateScheduler):
    """
    Warmup learning rate until `total_steps`

    Args:
        optimizer (Optimizer): wrapped optimizer.
        configs (DictConfig): configuration set.
    """
    def __init__(
            self,
            optimizer: Optimizer,
            configs: DictConfig,
    ) -> None:
        super(FixLRScheduler, self).__init__(optimizer, configs.lr_scheduler.peak_lr)
        self.lr = configs.lr_scheduler.peak_lr

    def step(self):
        self.set_lr(self.optimizer, self.lr)
        return self.lr
