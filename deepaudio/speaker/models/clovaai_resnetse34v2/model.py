from omegaconf import DictConfig
from torch import Tensor

from deepaudio.speaker.models import register_model
from deepaudio.speaker.models.speaker_embedding_model import SpeakerEmbeddingModel
from deepaudio.speaker.modules.backbones.clovaai.ResNetSE34V2 import MainModel

from .configurations import ClovaaiResnetse34V2Configs


@register_model('clovaai_resnetse34v2', dataclass=ClovaaiResnetse34V2Configs)
class ECAPAModel(SpeakerEmbeddingModel):
    def __init__(self, configs: DictConfig, num_classes: int):
        super(SpeakerEmbeddingModel, self).__init__(configs, num_classes)

    def build_model(self):
        self.model = MainModel(
            configs=self.configs
        )
