from omegaconf import DictConfig
from torch import Tensor

from deepaudio.speaker.models import register_model
from deepaudio.speaker.models.speaker_embedding_model import SpeakerEmbeddingModel
from deepaudio.speaker.modules.backbones.clovaai.ResNetSE34L import MainModel

from .configurations import ClovaaiResnetse34lConfigs


@register_model('clovaai_resnetse34l', dataclass=ClovaaiResnetse34lConfigs)
class ClovaaiResnetse34lModel(SpeakerEmbeddingModel):
    def __init__(self, configs: DictConfig, num_classes: int):
        super(ClovaaiResnetse34lModel, self).__init__(configs, num_classes)

    def build_model(self):
        self.model = MainModel(
            configs=self.configs
        )
