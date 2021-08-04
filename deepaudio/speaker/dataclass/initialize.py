from hydra.core.config_store import ConfigStore
from deepaudio.speaker.data.augmentation.configurations import AugmentConfigs
from deepaudio.speaker.datasets.voxceleb2.configurations import Voxceleb2Configs
from .configurations import (
    CPUTrainerConfigs,
    GPUTrainerConfigs,
    TPUTrainerConfigs,
    Fp16GPUTrainerConfigs,
    Fp16TPUTrainerConfigs,
    Fp64CPUTrainerConfigs,
)


SPEAKER_TRAIN_CONFIGS = [
    "feature",
    "augment",
    "dataset",
    "model",
    "criterion",
    "lr_scheduler",
    "trainer",
]


DATASET_DATACLASS_REGISTRY = {
    "voxceleb2": Voxceleb2Configs,
}
TRAINER_DATACLASS_REGISTRY = {
    "cpu": CPUTrainerConfigs,
    "gpu": GPUTrainerConfigs,
    "tpu": TPUTrainerConfigs,
    "gpu-fp16": Fp16GPUTrainerConfigs,
    "tpu-fp16": Fp16TPUTrainerConfigs,
    "cpu-fp64": Fp64CPUTrainerConfigs,
}
AUGMENT_DATACLASS_REGISTRY = {
    "default": AugmentConfigs,
}

def hydra_train_init() -> None:
    r""" initialize ConfigStore for hydra-train """
    from deepaudio.speaker.models import MODEL_DATACLASS_REGISTRY
    from deepaudio.speaker.criterion import CRITERION_DATACLASS_REGISTRY
    from deepaudio.speaker.data.feature import AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY
    from deepaudio.speaker.optim.scheduler import SCHEDULER_DATACLASS_REGISTRY

    registries = {
        "feature": AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY,
        "augment": AUGMENT_DATACLASS_REGISTRY,
        "dataset": DATASET_DATACLASS_REGISTRY,
        "trainer": TRAINER_DATACLASS_REGISTRY,
        "model": MODEL_DATACLASS_REGISTRY,
        "criterion": CRITERION_DATACLASS_REGISTRY,
        "lr_scheduler": SCHEDULER_DATACLASS_REGISTRY,
    }

    cs = ConfigStore.instance()

    for group in SPEAKER_TRAIN_CONFIGS:
        dataclass_registry = registries[group]

        for k, v in dataclass_registry.items():
            cs.store(group=group, name=k, node=v, provider="deepaudio")

