import math
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from pytorch_metric_learning.losses import ArcFaceLoss

from .. import register_criterion
from ..aamsoftmax.configuration import AAMSoftmaxConfigs


def radian2degree(radian):
    return math.degrees(radian)


@register_criterion("aamsoftmax", dataclass=AAMSoftmaxConfigs)
class AAMSoftmax(nn.Module):
    def __init__(self,
                 configs: DictConfig,
                 num_classes: int,
                 embedding_size: int
                 ) -> None:
        super(AAMSoftmax, self).__init__()
        self.arcface_loss = ArcFaceLoss(
            num_classes,
            embedding_size,
            margin=radian2degree(configs.criterion.margin),
            scale=configs.criterion.scale
        )

    def forward(self, embeddings: Tensor, targets: Tensor) -> Tensor:
        return self.arcface_loss(embeddings, targets)



