from omegaconf import DictConfig
from torch import Tensor

from deepaudio.speaker.models import register_model
from deepaudio.speaker.models.speaker_embedding_model import SpeakerEmbeddingModel
from deepaudio.speaker.modules.backbones.wespeaker.speaker_model import MainModel

from .configurations import WespeakerModelConfigs


@register_model('wespeaker_model', dataclass=WespeakerModelConfigs)
class WespeakerModel(SpeakerEmbeddingModel):
    def __init__(self, configs: DictConfig, num_classes: int):
        super(WespeakerModel, self).__init__(configs, num_classes)

    def build_model(self):
        self.model = MainModel(
            configs=self.configs
        )
