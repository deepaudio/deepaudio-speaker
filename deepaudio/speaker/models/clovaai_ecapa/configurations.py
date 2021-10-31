from dataclasses import dataclass, field

from deepaudio.speaker.dataclass.configurations import DeepMMDataclass

@dataclass
class ClovaaiECAPAConfigs(DeepMMDataclass):
    name: str = field(
        default="clovaai_ecapa", metadata={"help": "Model name"}
    )
    embed_dim: int = field(
        default=192, metadata={"help": "Dimension of embedding."}
    )
    channels: int = field(
        default=512, metadata={"help": "Dimension of embedding."}
    )
    model_scale: int = field(
        default=8, metadata={"help": "Model scale."}
    )
    context: bool = field(
        default=True, metadata={"help": "Context."}
    )
    summed: bool = field(
        default=True, metadata={"help": "Summed."}
    )
    out_bn: bool = field(
        default=True, metadata={"help": "Flag for batch normalization in embedding layer."}
    )
    encoder_type: str = field(
        default="ECA", metadata={"help": "Encoder type."}
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
    pretrained: bool = field(
        default=False, metadata={"help": "Use pretrained model or not."}
    )
    checkpoint: str = field(
        default="None", metadata={"help": "Checkpoint path."}
    )