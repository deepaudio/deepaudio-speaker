from dataclasses import dataclass, field

from deepaudio.speaker.dataclass.configurations import DeepMMDataclass

@dataclass
class ClovaaiResnetse34lConfigs(DeepMMDataclass):
    name: str = field(
        default="clovaai_resnetse34l", metadata={"help": "Model name"}
    )
    embed_dim: int = field(
        default=256, metadata={"help": "Dimension of embedding."}
    )
    encoder_type: str = field(
        default="SAP", metadata={"help": "Encoder type."}
    )
    optimizer: str = field(
        default="adam", metadata={"help": "Optimizer for training."}
    )
    min_num_frames: int = field(
        default=200, metadata={"help": "Min num frames."}
    )
    max_num_frames: int = field(
        default=400, metadata={"help": "Max num frames."}
    )