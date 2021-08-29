from dataclasses import dataclass, field

from deepaudio.speaker.dataclass.configurations import DeepMMDataclass

@dataclass
class MMCLSeResnet34Configs(DeepMMDataclass):
    name: str = field(
        default="mmcl_seresnet34", metadata={"help": "Model name"}
    )
    embed_dim: int = field(
        default=256, metadata={"help": "Dimension of embedding."}
    )
    in_channels: int = field(
        default=1, metadata={"help": "In channel."}
    )
    stem_channels: int = field(
        default=32, metadata={"help": "Stem channel."}
    )
    base_channels: int = field(
        default=32, metadata={"help": "Base channel."}
    )
    depth: int = field(
        default=34, metadata={"help": "Depth."}
    )
    out_bn: bool = field(
        default=True, metadata={"help": "Flag for batch normalization in embedding layer."}
    )
    num_stages: int = field(
        default=4, metadata={"help": "Number of stages"}
    )
    out_indices: int = field(
        default=3, metadata={"help": "Out indices"}
    )
    norm_cfg_type: str = field(
        default='BN', metadata={"help": "Norm type"}
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