import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from omegaconf import DictConfig

from .. import register_criterion
from ..subcenter_aamsoftmax.subcenter_aamsoftmax import SubcenterArcMarginProduct
from .configuration import AdaptiveSubcenterAAMSoftmaxConfigs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


@register_criterion("adaptive_subcenter_aamsoftmax", dataclass=AdaptiveSubcenterAAMSoftmaxConfigs)
class PyannoteAAMSoftmax(nn.Module):
    def __init__(self,
                 configs: DictConfig,
                 num_classes: int,
                 embedding_size: int
                 ) -> None:
        super(PyannoteAAMSoftmax, self).__init__()
        self.configs = configs
        self.classifier_ = SubcenterArcMarginProduct(
            in_features=self.configs.model.embed_dim,
            out_features=num_classes,
            K=configs.model.criterion.K,
            m=configs.criterion.margin,
            s=configs.criterion.scale
        )
        self.loss_ = nn.CrossEntropyLoss()
        self.margin = configs.criterion.margin
        self.warmup_steps = configs.lr_scheduler.warmup_steps if configs.lr_scheduler.scheduler_name.startswith(
            'warmup') else 0
        self.increase_steps = configs.criterion.increase_steps
        self.increase_rate = self.margin / (self.increase_steps - self.warmup_steps)

    def step(self, global_steps):
        if global_steps < self.warmup_steps:
            self.classifier_.margin = 0
        elif global_steps < self.increase_steps:
            self.classifier_.margin = (global_steps - self.warmup_steps) * self.increase_rate
        else:
            self.classifier_.margin = self.margin

    def forward(self, embeddings: Tensor, targets: Tensor) -> Tensor:
        logits = self.classifier_(embeddings, target=targets)
        return self.loss_(logits, targets)
