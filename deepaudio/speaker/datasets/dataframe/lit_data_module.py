from typing import Optional

import random
from omegaconf import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from deepaudio.speaker.data.dataset import SpeakerAudioDataset
from deepaudio.speaker.data.dataloader import SpeakerUttDataLoader
from deepaudio.speaker.data.samplers import ClovaaiSampler

from .utils import get_dataset_items, SpeakerDataframe, split_segment
from .. import register_data_module


@register_data_module('dataframe')
class LightningDataframeDataModule(pl.LightningDataModule):
    def __init__(self, configs: DictConfig):
        super(LightningDataframeDataModule, self).__init__()
        self.configs = configs

    def prepare_data(self):
        dataset_items = get_dataset_items(self.configs.dataset.database_yml,
                                          self.configs.dataset.dataset_name)
        dataset = SpeakerDataframe(dataset_items)
        speaker2items = dataset.speaker2items
        spk2ids = dataset.spk2ids
        self.num_classes = len(spk2ids)
        self.train_utts, self.valid_utts = self._split_train_valid(speaker2items, spk2ids)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = SpeakerAudioDataset(self.configs, self.train_utts)
        self.valid_dataset = SpeakerAudioDataset(self.configs, self.valid_utts)

    def train_dataloader(self) -> DataLoader:
        if self.configs.dataset.sampler == 'clovaai':
            sampler = ClovaaiSampler(self.train_dataset.labels)
        else:
            sampler = None
        return SpeakerUttDataLoader(
            dataset=self.train_dataset,
            num_workers=self.configs.trainer.num_workers,
            min_num_frames=self.configs.model.min_num_frames,
            max_num_frames=self.configs.model.max_num_frames,
            batch_size=self.configs.trainer.batch_size,
            shuffle=True,
            sampler=sampler
        )

    def val_dataloader(self) -> DataLoader:
        return SpeakerUttDataLoader(
            dataset=self.valid_dataset,
            num_workers=self.configs.trainer.num_workers,
            min_num_frames=self.configs.model.min_num_frames,
            max_num_frames=self.configs.model.max_num_frames,
            batch_size=self.configs.trainer.batch_size
        )

    def _spk2wav_utts(self, speaker2items, spk2ids):
        utts = []
        for spk in speaker2items:
            for item in speaker2items[spk]:
                wav, spk, seg = item
                utts.append((str(wav), spk2ids[spk], seg))
                if self.configs.dataset.exhaustive:
                    for subseg in split_segment(seg,
                                                self.configs.dataset.duration,
                                                self.configs.dataset.step):
                        utts.append((str(wav), spk2ids[spk], subseg))
        random.shuffle(utts)
        return utts

    def _split_train_valid(self, speaker2items, spk2ids):
        valid_spk2item = {}
        for spk in speaker2items:
            random.shuffle(speaker2items[spk])
            valid_spk2item[spk] = [speaker2items[spk].pop(0)]
        train_utts = self._spk2wav_utts(speaker2items, spk2ids)
        valid_utts = self._spk2wav_utts(valid_spk2item, spk2ids)
        return train_utts, valid_utts
