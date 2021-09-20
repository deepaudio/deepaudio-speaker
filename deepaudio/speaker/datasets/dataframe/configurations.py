from omegaconf import MISSING
from dataclasses import dataclass, field

from deepaudio.speaker.dataclass.configurations import DeepMMDataclass
@dataclass
class DataframeConfigs(DeepMMDataclass):
    """ Configuration dataclass that common used """
    name: str = field(
        default="dataframe", metadata={"help": "Select dataset for training (librispeech, ksponspeech, aishell, lm)"}
    )
    database_yml: str = field(
        default="/home/tian/auido_data/deepaudio-database/database.yml", metadata={"help": "Path of database.yml"}
    )
    dataset_name: str = field(
        default="debug", metadata={"help": "Database name. If you want use multiple dataset, please use ',' to split"}
    )
    duration: float = field(
        default=4, metadata={"help": "Sliding window duration."}
    )
    step: float = field(
        default=2, metadata={"help": "Sliding window step."}
    )
    exhaustive: bool = field(
        default=True, metadata={"help": "exhaustive mode."}
    )