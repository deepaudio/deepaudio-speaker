from dataclasses import dataclass, field

from deepaudio.speaker.dataclass.configurations import DeepMMDataclass


@dataclass
class WespeakerModelConfigs(DeepMMDataclass):
    name: str = field(
        default="ResNet34", metadata={"help": "Model name"}
    )
    embed_dim: int = field(
        default=256, metadata={"help": "Dimension of embedding."}
    )
    pooling_func: str = field(
        default="TSTP", metadata={"help": "Pooling function for model."}
    )
    optimizer: str = field(
        default="adam", metadata={"help": "Optimizer for training."}
    )
    min_num_frames: int = field(
        default=200, metadata={"help": "Min num frames."}
    )
    max_num_frames: int = field(
        default=300, metadata={"help": "Max num frames."}
    )
    pretrained: bool = field(
        default=False, metadata={"help": "Use pretrained model or not."}
    )
    checkpoint: str = field(
        default="None", metadata={"help": "Checkpoint path."}
    )
