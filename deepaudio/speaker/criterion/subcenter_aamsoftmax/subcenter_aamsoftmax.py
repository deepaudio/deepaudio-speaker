import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from omegaconf import DictConfig

from .. import register_criterion
from .configuration import SubcenterAAMSoftmaxConfigs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class SubcenterArcMarginProduct(nn.Module):
    r"""Modified implementation from https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py#L10
        """

    def __init__(self, in_features, out_features, K=3, s=30.0, m=0.50, easy_margin=False):
        super(SubcenterArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = s
        self.margin = m
        self.K = K
        self.weight = Parameter(torch.FloatTensor(out_features * self.K, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin


    def forward(self, input, label):
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if self.K > 1:
            cosine = torch.reshape(cosine, (-1, self.out_features, self.K))
            cosine, _ = torch.max(cosine, axis=2)

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        # cos(phi+m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output


@register_criterion("subcenter_aamsoftmax", dataclass=SubcenterAAMSoftmaxConfigs)
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

    def forward(self, embeddings: Tensor, targets: Tensor) -> Tensor:
        logits = self.classifier_(embeddings, target=targets)
        return self.loss_(logits, targets)
