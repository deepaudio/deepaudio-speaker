import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from omegaconf import DictConfig


from .. import register_criterion
from .configuration import PyannoteAAMSoftmaxConfigs


class ArcLinear(nn.Module):
    """Additive Angular Margin classification module
    Parameters
    ----------
    nfeat : int
        Embedding dimension
    nclass : int
        Number of classes
    margin : float
        Angular margin to penalize distances between embeddings and centers
    scale : float
        Scaling factor for the logits
    """

    def __init__(self, nfeat, nclass, margin, scale):
        super(ArcLinear, self).__init__()
        eps = 1e-4
        self.min_cos = eps - 1
        self.max_cos = 1 - eps
        self.nclass = nclass
        self.margin = margin
        self.scale = scale
        self.W = nn.Parameter(Tensor(nclass, nfeat))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, target=None):
        """Apply the angular margin transformation
        Parameters
        ----------
        x : `torch.Tensor`
            an embedding batch
        target : `torch.Tensor`
            a non one-hot label batch
        Returns
        -------
        fX : `torch.Tensor`
            logits after the angular margin transformation
        """
        # normalize the feature vectors and W
        xnorm = F.normalize(x)
        Wnorm = F.normalize(self.W)
        target = target.long().view(-1, 1)
        # calculate cosθj (the logits)
        cos_theta_j = torch.matmul(xnorm, torch.transpose(Wnorm, 0, 1))
        # get the cosθ corresponding to the classes
        cos_theta_yi = cos_theta_j.gather(1, target)
        # for numerical stability
        cos_theta_yi = cos_theta_yi.clamp(min=self.min_cos, max=self.max_cos)
        # get the angle separating xi and Wyi
        theta_yi = torch.acos(cos_theta_yi)
        # apply the margin to the angle
        cos_theta_yi_margin = torch.cos(theta_yi + self.margin)
        # one hot encode  y
        one_hot = torch.zeros_like(cos_theta_j)
        one_hot.scatter_(1, target, 1.0)
        # project margin differences into cosθj
        return self.scale * (cos_theta_j + one_hot * (cos_theta_yi_margin - cos_theta_yi))

@register_criterion("pyannote_aamsoftmax", dataclass=PyannoteAAMSoftmaxConfigs)
class PyannoteAAMSoftmax(nn.Module):
    def __init__(self,
                 configs: DictConfig,
                 num_classes: int,
                 embedding_size: int
                 ) -> None:
        super(PyannoteAAMSoftmax, self).__init__()
        self.configs=configs
        self.classifier_ = ArcLinear(
            nfeat=self.configs.model.embed_dim,
            nclass=num_classes,
            margin=configs.criterion.margin,
            scale=configs.criterion.scale
        )
        self.logsoftmax_ = nn.LogSoftmax(dim=1)
        self.loss_ = nn.NLLLoss()

    def forward(self, embeddings: Tensor, targets: Tensor) -> Tensor:
        logits = self.logsoftmax_(self.classifier_(embeddings, target=targets))
        return self.loss_(logits, targets)




