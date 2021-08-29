from dataclasses import dataclass, field

from deepaudio.speaker.dataclass.configurations import DeepMMDataclass

@dataclass
class Resnet101Configs(DeepMMDataclass):
    name: str = field(
        default="resnet101", metadata={"help": "Model name"}
    )
    embed_dim: int = field(
        default=256, metadata={"help": "Dimension of embedding."}
    )
    optimizer: str = field(
        default="adam", metadata={"help": "Optimizer for training."}
    )
    min_num_frames: int = field(
        default=300, metadata={"help": "Min num frames."}
    )
    max_num_frames: int = field(
        default=400, metadata={"help": "Max num frames."}
    )
    squeeze_excitation: bool = field(
        default=False, metadata={"help": "Max num frames."}
    )