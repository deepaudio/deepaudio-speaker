from omegaconf import DictConfig
from torch import Tensor

from deepaudio.speaker.models import register_model
from deepaudio.speaker.models.speaker_embedding_model import SpeakerEmbeddingModel
from deepaudio.speaker.modules.backbones.mmcl.seresnet_asv import MainModel

from .configurations import MMCLSeResnet34Configs


@register_model('mmcl_seresnet34', dataclass=MMCLSeResnet34Configs)
class MMCLSeResnet34Model(SpeakerEmbeddingModel):
    def __init__(self, configs: DictConfig, num_classes: int):
        super(MMCLSeResnet34Model, self).__init__(configs, num_classes)

    def build_model(self):
        self.model = MainModel(
            configs=self.configs
        )
