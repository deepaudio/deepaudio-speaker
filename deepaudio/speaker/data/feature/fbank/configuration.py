from dataclasses import dataclass, field

from deepaudio.speaker.dataclass.configurations import DeepMMDataclass

@dataclass
class FBankConfigs(DeepMMDataclass):
    name: str = field(
        default="fbank", metadata={"help": "Name of feature transform."}
    )
    sample_rate: int = field(
        default=16000, metadata={"help": "Sampling rate of audio"}
    )
    frame_duration: float = field(
        default=0.025, metadata={"help": "Frame length for spectrogram"}
    )
    frame_shift: float = field(
        default=0.01, metadata={"help": "Length of hop between STFT"}
    )
    n_mels: int = field(
        default=80, metadata={"help": "Number of mel filterbanks.."}
    )
    var_norm: bool = field(
        default=False, metadata={"help": "Flag for cmvn"}
    )
