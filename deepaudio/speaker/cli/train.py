import os
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_info

from deepaudio.speaker.datasets import DATA_MODULE_REGISTRY
from deepaudio.speaker.dataclass.initialize import hydra_train_init
from deepaudio.speaker.models import MODEL_REGISTRY
from deepaudio.speaker.utils import parse_configs, get_pl_trainer


@hydra.main(config_path=os.path.join("..", "configs"), config_name="train")
def hydra_main(configs: DictConfig) -> None:
    rank_zero_info(OmegaConf.to_yaml(configs))
    pl.seed_everything(configs.trainer.seed)
    logger, num_devices = parse_configs(configs)

    data_module = DATA_MODULE_REGISTRY[configs.dataset.name](configs)
    data_module.prepare_data()
    model = MODEL_REGISTRY[configs.model.name](configs=configs, num_classes=data_module.num_classes)
    model.build_model()
    trainer = get_pl_trainer(configs, num_devices, logger)
    trainer.fit(model, data_module)


def main():
    hydra_train_init()
    hydra_main()


if __name__ == '__main__':
    main()
