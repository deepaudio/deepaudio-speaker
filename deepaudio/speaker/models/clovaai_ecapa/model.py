from omegaconf import DictConfig
from torch import Tensor

from deepaudio.speaker.models import register_model
from deepaudio.speaker.models.speaker_embedding_model import SpeakerEmbeddingModel
from deepaudio.speaker.modules.backbones.clovaai.ECAPA_TDNN import MainModel

from .configurations import ClovaaiECAPAConfigs


@register_model('clovaai_ecapa', dataclass=ClovaaiECAPAConfigs)
class ClovaaiECAPAModel(SpeakerEmbeddingModel):
    def __init__(self, configs: DictConfig, num_classes: int):
        super(ClovaaiECAPAModel, self).__init__(configs, num_classes)

    def build_model(self):
        self.model = MainModel(
            configs=self.configs
        )
