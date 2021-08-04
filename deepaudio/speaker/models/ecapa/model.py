from omegaconf import DictConfig
from torch import Tensor

from deepaudio.speaker.models import register_model
from deepaudio.speaker.models.speaker_embedding_model import SpeakerEmbeddingModel
from deepaudio.speaker.modules.backbones.ecapa import ECAPA_TDNN

from .configurations import ECAPAConfigs


@register_model('ecapa', dataclass=ECAPAConfigs)
class ECAPAModel(SpeakerEmbeddingModel):
    def __init__(self, configs: DictConfig, num_classes: int):
        super(SpeakerEmbeddingModel, self).__init__(configs, num_classes)

    def build_model(self):
        self.model = ECAPA_TDNN(
            in_channels=self.configs.feature.n_mels,
            channels=self.configs.model.channels,
            embed_dim=self.configs.model.embed_dim
        )
