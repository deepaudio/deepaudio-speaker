from omegaconf import DictConfig
from torch import Tensor

from deepaudio.speaker.models import register_model
from deepaudio.speaker.models.speaker_embedding_model import SpeakerEmbeddingModel
from deepaudio.speaker.modules.backbones.resnet import ResNet, Bottleneck

from .configurations import Resnet101Configs


@register_model('resnet101', dataclass=Resnet101Configs)
class Resnet101Model(SpeakerEmbeddingModel):
    def __init__(self, configs: DictConfig, num_classes: int):
        super(SpeakerEmbeddingModel, self).__init__(configs, num_classes)

    def build_model(self):
        self.model = ResNet(
            Bottleneck,
            [3, 4, 23, 3],
            feat_dim=self.configs.feature.n_mels,
            embed_dim=self.configs.model.embed_dim,
            squeeze_excitation=self.configs.model.squeeze_excitation
        )
