

import logging
import torch
import platform
from typing import Tuple, Union, Iterable
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, LightningLoggerBase, WandbLogger


def _check_environment(use_cuda: bool, logger) -> int:
    r"""
    Check execution envirionment.
    OS, Processor, CUDA version, Pytorch version, ... etc.
    """

    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    logger.info(f"Operating System : {platform.system()} {platform.release()}")
    logger.info(f"Processor : {platform.processor()}")

    num_devices = torch.cuda.device_count()

    if str(device) == 'cuda':
        for idx in range(torch.cuda.device_count()):
            logger.info(f"device : {torch.cuda.get_device_name(idx)}")
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"CUDA version : {torch.version.cuda}")
        logger.info(f"PyTorch version : {torch.__version__}")

    else:
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"PyTorch version : {torch.__version__}")

    return num_devices


def parse_configs(configs: DictConfig) -> Tuple[Union[TensorBoardLogger, bool], int]:
    r"""
    Parsing configuration set.

    Args:
        configs (DictConfig): configuration set.

    Returns:
        logger (Union[TensorBoardLogger, bool]): logger for training
        num_devices (int): the number of cuda device
    """
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(configs))
    num_devices = _check_environment(configs.trainer.use_cuda, logger)

    if configs.trainer.logger == "tensorboard":
        logger = TensorBoardLogger("logs/")
    elif configs.trainer.logger == "wandb":
        logger = WandbLogger(project=f"{configs.model.name}-{configs.dataset.name}", job_type='train')
    else:
        logger = True

    return logger, num_devices


def get_pl_trainer(
        configs: DictConfig,
        num_devices: int,
        logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]
) -> pl.Trainer:
    amp_backend = None

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="deepaudio-{epoch:02d}-{val_loss:.2f}",
        save_top_k=configs.trainer.num_checkpoints,
        mode="min",
    )

    if hasattr(configs.trainer, "amp_backend"):
        amp_backend = "apex" if configs.trainer.amp_backend == "apex" else "native"

    if configs.trainer.name == "cpu":
        trainer = pl.Trainer(accelerator=configs.trainer.accelerator,
                             accumulate_grad_batches=configs.trainer.accumulate_grad_batches,
                             check_val_every_n_epoch=configs.trainer.check_val_every_n_epoch,
                             gradient_clip_val=configs.trainer.gradient_clip_val,
                             logger=logger,
                             auto_scale_batch_size=configs.trainer.auto_scale_batch_size,
                             max_epochs=configs.trainer.max_epochs,
                             callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback])
    elif configs.trainer.name == "gpu":
        trainer = pl.Trainer(accelerator=configs.trainer.accelerator,
                             gpus=num_devices,
                             accumulate_grad_batches=configs.trainer.accumulate_grad_batches,
                             auto_select_gpus=configs.trainer.auto_select_gpus,
                             check_val_every_n_epoch=configs.trainer.check_val_every_n_epoch,
                             gradient_clip_val=configs.trainer.gradient_clip_val,
                             logger=logger,
                             auto_scale_batch_size=configs.trainer.auto_scale_batch_size,
                             max_epochs=configs.trainer.max_epochs,
                             callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback])
    elif configs.trainer.name == "tpu":
        trainer = pl.Trainer(accelerator=configs.trainer.accelerator,
                             tpu_cores=configs.trainer.tpu_cores,
                             accumulate_grad_batches=configs.trainer.accumulate_grad_batches,
                             auto_select_gpus=configs.trainer.auto_select_gpus,
                             check_val_every_n_epoch=configs.trainer.check_val_every_n_epoch,
                             gradient_clip_val=configs.trainer.gradient_clip_val,
                             logger=logger,
                             auto_scale_batch_size=configs.trainer.auto_scale_batch_size,
                             max_epochs=configs.trainer.max_epochs,
                             callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback])
    elif configs.trainer.name == "gpu-fp16":
        trainer = pl.Trainer(precision=configs.trainer.precision,
                             accelerator=configs.trainer.accelerator,
                             gpus=num_devices,
                             accumulate_grad_batches=configs.trainer.accumulate_grad_batches,
                             amp_backend=amp_backend,
                             auto_select_gpus=configs.trainer.auto_select_gpus,
                             check_val_every_n_epoch=configs.trainer.check_val_every_n_epoch,
                             gradient_clip_val=configs.trainer.gradient_clip_val,
                             logger=logger,
                             auto_scale_batch_size=configs.trainer.auto_scale_batch_size,
                             max_epochs=configs.trainer.max_epochs,
                             callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback])
    elif configs.trainer.name == "tpu-fp16":
        trainer = pl.Trainer(precision=configs.trainer.precision,
                             accelerator=configs.trainer.accelerator,
                             tpu_cores=configs.trainer.tpu_configs,
                             accumulate_grad_batches=configs.trainer.accumulate_grad_batches,
                             amp_backend=amp_backend,
                             check_val_every_n_epoch=configs.trainer.check_val_every_n_epoch,
                             gradient_clip_val=configs.trainer.gradient_clip_val,
                             logger=logger,
                             auto_scale_batch_size=configs.trainer.auto_scale_batch_size,
                             max_epochs=configs.trainer.max_epochs,
                             callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback])
    elif configs.trainer.name == "cpu-fp64":
        trainer = pl.Trainer(precision=configs.trainer.precision,
                             accelerator=configs.trainer.accelerator,
                             accumulate_grad_batches=configs.trainer.accumulate_grad_batches,
                             amp_backend=amp_backend,
                             check_val_every_n_epoch=configs.trainer.check_val_every_n_epoch,
                             gradient_clip_val=configs.trainer.gradient_clip_val,
                             logger=logger,
                             auto_scale_batch_size=configs.trainer.auto_scale_batch_size,
                             max_epochs=configs.trainer.max_epochs,
                             callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback])
    else:
        raise ValueError(f"Unsupported trainer: {configs.trainer.name}")

    return trainer
