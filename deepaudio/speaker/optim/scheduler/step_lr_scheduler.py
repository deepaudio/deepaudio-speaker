import torch
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import DictConfig
from torch.optim import Optimizer

from deepaudio.speaker.dataclass.configurations import LearningRateSchedulerConfigs
from deepaudio.speaker.optim.scheduler import register_scheduler
from deepaudio.speaker.optim.scheduler.lr_scheduler import LearningRateScheduler


@dataclass
class StepLRSchedulerConfigs(LearningRateSchedulerConfigs):
    scheduler_name: str = field(
        default="warmup", metadata={"help": "Name of learning rate scheduler."}
    )
    peak_lr: float = field(
        default=1e-04, metadata={"help": "Maximum learning rate."}
    )
    min_lr: float = field(
        default=1e-7, metadata={"help": "Min learning rate."}
    )
    step_size: int = field(
        default=50, metadata={"help": "Step size to decay"}
    )
    lr_factor: float = field(
        default=0.8, metadata={"help": "Factor by which the learning rate will be reduced. new_lr = lr * factor."}
    )


@register_scheduler("steplr", dataclass=StepLRSchedulerConfigs)
class StepLRScheduler(LearningRateScheduler):
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
        super(StepLRScheduler, self).__init__(optimizer, configs.lr_scheduler.peak_lr)
        self.update_steps = 1
        self.lr = configs.lr_scheduler.peak_lr
        self.step_size = configs.lr_scheduler.step_size
        self.min_lr = configs.lr_scheduler.min_lr
        self.lr_factor = configs.lr_scheduler.lr_factor

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        if self.update_steps % self.step_size == 0:
            lr = self.lr * self.lr_factor
            self.set_lr(self.optimizer, lr)
            self.lr = lr
        self.update_steps += 1
        return self.lr
