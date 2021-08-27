# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho and Ruiqing Yin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from omegaconf import MISSING
from dataclasses import dataclass, _MISSING_TYPE, field
from typing import List, Optional, Any


@dataclass
class DeepMMDataclass:
    """ DeepMM base dataclass that supported fetching attributes and metas """

    def _get_all_attributes(self) -> List[str]:
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_meta(self, attribute_name: str, meta: str, default: Optional[Any] = None) -> Any:
        return self.__dataclass_fields__[attribute_name].metadata.get(meta, default)

    def _get_name(self, attribute_name: str) -> str:
        return self.__dataclass_fields__[attribute_name].name

    def _get_default(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith("${"):
                return str(getattr(self, attribute_name))
            elif str(self.__dataclass_fields__[attribute_name].default).startswith("${"):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif getattr(self, attribute_name) != self.__dataclass_fields__[attribute_name].default:
                return getattr(self, attribute_name)

        f = self.__dataclass_fields__[attribute_name]
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    def _get_type(self, attribute_name: str) -> Any:
        return self.__dataclass_fields__[attribute_name].type

    def _get_help(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "help")


@dataclass
class BaseTrainerConfigs(DeepMMDataclass):
    """ Base trainer dataclass """
    seed: int = field(
        default=1, metadata={"help": "Seed for training."}
    )
    accelerator: str = field(
        default="dp", metadata={"help": "Previously known as distributed_backend (dp, ddp, ddp2, etc…)."}
    )
    accumulate_grad_batches: int = field(
        default=1, metadata={"help": "Accumulates grads every k batches or as set up in the dict."}
    )
    num_workers: int = field(
        default=4, metadata={"help": "The number of cpu cores"}
    )
    batch_size: int = field(
        default=32, metadata={"help": "Size of batch"}
    )
    check_val_every_n_epoch: int = field(
        default=1, metadata={"help": "Check val every n train epochs."}
    )
    gradient_clip_val: float = field(
        default=5.0, metadata={"help": "0 means don’t clip."}
    )
    logger: str = field(
        default="tensorboard", metadata={"help": "Training logger. {wandb, tensorboard}"}
    )
    max_epochs: int = field(
        default=20, metadata={"help": "Stop training once this number of epochs is reached."}
    )
    num_checkpoints: int = field(
        default=20, metadata={"help": "Number of checkpoints to be stored."}
    )
    auto_scale_batch_size: str = field(
        default=False, metadata={"help": "If set to True, will initially run a batch size finder trying to find "
                                               "the largest batch size that fits into memory."}
    )

@dataclass
class CPUTrainerConfigs(BaseTrainerConfigs):
    name: str = field(
        default="cpu", metadata={"help": "Trainer name"}
    )
    device: str = field(
        default="cpu", metadata={"help": "Training device."}
    )
    use_cuda: bool = field(
        default=False, metadata={"help": "If set True, will train with GPU"}
    )


@dataclass
class GPUTrainerConfigs(BaseTrainerConfigs):
    """ GPU trainer dataclass """
    name: str = field(
        default="gpu", metadata={"help": "Trainer name"}
    )
    device: str = field(
        default="gpu", metadata={"help": "Training device."}
    )
    use_cuda: bool = field(
        default=True, metadata={"help": "If set True, will train with GPU"}
    )
    auto_select_gpus: bool = field(
        default=True, metadata={"help": "If enabled and gpus is an integer, pick available gpus automatically."}
    )


@dataclass
class TPUTrainerConfigs(BaseTrainerConfigs):
    name: str = field(
        default="tpu", metadata={"help": "Trainer name"}
    )
    device: str = field(
        default="tpu", metadata={"help": "Training device."}
    )
    use_cuda: bool = field(
        default=False, metadata={"help": "If set True, will train with GPU"}
    )
    use_tpu: bool = field(
        default=True, metadata={"help": "If set True, will train with GPU"}
    )
    tpu_cores: int = field(
        default=8, metadata={"help": "Number of TPU cores"}
    )


@dataclass
class Fp16GPUTrainerConfigs(GPUTrainerConfigs):
    name: str = field(
        default="gpu-fp16", metadata={"help": "Trainer name"}
    )
    precision: int = field(
        default=16, metadata={"help": "Double precision (64), full precision (32) or half precision (16). "
                                      "Can be used on CPU, GPU or TPUs."}
    )
    amp_backend: str = field(
        default="apex", metadata={"help": "The mixed precision backend to use (“native” or “apex”)"}
    )


@dataclass
class Fp16TPUTrainerConfigs(TPUTrainerConfigs):
    name: str = field(
        default="tpu-fp16", metadata={"help": "Trainer name"}
    )
    precision: int = field(
        default=16, metadata={"help": "Double precision (64), full precision (32) or half precision (16). "
                                      "Can be used on CPU, GPU or TPUs."}
    )
    amp_backend: str = field(
        default="apex", metadata={"help": "The mixed precision backend to use (“native” or “apex”)"}
    )


@dataclass
class Fp64CPUTrainerConfigs(CPUTrainerConfigs):
    name: str = field(
        default="cpu-fp64", metadata={"help": "Trainer name"}
    )
    precision: int = field(
        default=64, metadata={"help": "Double precision (64), full precision (32) or half precision (16). "
                                      "Can be used on CPU, GPU or TPUs."}
    )
    amp_backend: str = field(
        default="apex", metadata={"help": "The mixed precision backend to use (“native” or “apex”)"}
    )


@dataclass
class LearningRateSchedulerConfigs(DeepMMDataclass):
    """ Super class of learning rate dataclass """
    lr: float = field(
        default=1e-04, metadata={"help": "Learning rate"}
    )



