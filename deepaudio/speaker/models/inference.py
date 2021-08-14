from typing import Text, Union
from pathlib import Path

import torch

from pytorch_lightning.utilities.cloud_io import load as pl_load

from deepaudio.speaker.data.audio_io.with_torchaudio import Audio
from deepaudio.speaker.data.feature import AUDIO_FEATURE_TRANSFORM_REGISTRY
from deepaudio.speaker.models.speaker_embedding_model import SpeakerEmbeddingModel


class Inference:
    def __init__(
            self,
            path_for_pl: Union[Text, Path],
            device: torch.device = None,
            strict: bool = False
    ):
        loaded_ckpt = pl_load(str(path_for_pl))
        configs = loaded_ckpt["configs"]
        self.model = SpeakerEmbeddingModel.from_pretrained(str(path_for_pl), device, strict).eval()
        self.audio = Audio()
        self.feature_extractor = AUDIO_FEATURE_TRANSFORM_REGISTRY[configs.feature.name](configs)

    def make_embedding(self, wav):
        waveform = self.audio(wav)
        feature = self.feature_extractor(waveform)
        return self.model.make_embedding(feature)
