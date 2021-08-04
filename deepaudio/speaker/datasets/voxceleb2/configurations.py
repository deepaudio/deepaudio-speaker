from omegaconf import MISSING
from dataclasses import dataclass, field

from deepaudio.speaker.dataclass.configurations import DeepMMDataclass
@dataclass
class Voxceleb2Configs(DeepMMDataclass):
    """ Configuration dataclass that common used """
    name: str = field(
        default="voxceleb2", metadata={"help": "Select dataset for training (librispeech, ksponspeech, aishell, lm)"}
    )
    dataset_path: str = field(
        default="/Users/yin/project/data/aac4", metadata={"help": "Path of dataset"}
    )