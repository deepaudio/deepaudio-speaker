import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram

from ..utils import CMVN
from .configuration import FBankConfigs
from .. import register_audio_feature_transform

EPSILON = 1e-6


@register_audio_feature_transform("fbank", dataclass=FBankConfigs)
class Fbank(nn.Module):
    def __init__(self, configs):
        super(Fbank, self).__init__()
        win_length = int(configs.feature.sample_rate * configs.feature.frame_duration)
        hop_length = int(configs.feature.sample_rate * configs.feature.frame_shift)
        self.melSpectrogram = MelSpectrogram(sample_rate=configs.feature.sample_rate,
                                             n_mels=configs.feature.n_mels,
                                             n_fft=512,
                                             win_length=win_length,
                                             hop_length=hop_length,
                                             window_fn=torch.hann_window)
        self.cmvn = CMVN(var_norm=configs.feature.var_norm)
        self.input_dim = configs.feature.n_mels

    def forward(self, waveform):
        mel_spectrogram = self.melSpectrogram(waveform)
        mel_spectrogram = torch.log(mel_spectrogram + EPSILON)
        mel_spectrogram = mel_spectrogram.transpose(1, 2)
        return self.cmvn(mel_spectrogram)
