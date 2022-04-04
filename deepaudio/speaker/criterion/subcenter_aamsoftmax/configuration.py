from dataclasses import dataclass, field

from ...dataclass.configurations import DeepMMDataclass


@dataclass
class SubcenterAAMSoftmaxConfigs(DeepMMDataclass):
    name: str = field(
        default="subcenter_aamsoftmax", metadata={"help": "Criterion name for training"}
    )
    margin: float = field(
        default=0.2, metadata={"help": "The angular margin penalty in radians."}
    )
    K: int = field(
        default=3, metadata={"help": "The number of subcenter."}
    )
    scale: float = field(
        default=32, metadata={"help": "The scale for loss."}
    )