from dataclasses import dataclass, field

from ...dataclass.configurations import DeepMMDataclass


@dataclass
class AdaptiveAAMSoftmaxConfigs(DeepMMDataclass):
    name: str = field(
        default="adaptive_aamsoftmax", metadata={"help": "Criterion name for training"}
    )
    margin: float = field(
        default=0.2, metadata={"help": "The angular margin penalty in radians."}
    )

    scale: float = field(
        default=32, metadata={"help": "The scale for loss."}
    )
    increase_steps: int = field(
        default=50000, metadata={"help": "The increase step for margin."}
    )