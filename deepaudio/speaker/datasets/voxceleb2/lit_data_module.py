from typing import Optional

import random
from omegaconf import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from deepaudio.speaker.data.dataset import SpeakerAudioDataset
from deepaudio.speaker.data.dataloader import SpeakerUttDataLoader

from .preprocess import get_speaker_list, get_speaker_wavs
from .. import register_data_module


@register_data_module('voxceleb2')
class LightningVoxceleb2DataModule(pl.LightningDataModule):
    def __init__(self, configs: DictConfig):
        super(LightningVoxceleb2DataModule, self).__init__()
        self.configs = configs

    def prepare_data(self):
        speakers, spk2id = get_speaker_list(self.configs)
        speaker2wav = get_speaker_wavs(self.configs.dataset.dataset_path, speakers)
        self.num_classes = len(speakers)
        self.train_utts, self.valid_utts = self._split_train_valid(speaker2wav, spk2id)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = SpeakerAudioDataset(self.configs, self.train_utts)
        self.valid_dataset = SpeakerAudioDataset(self.configs, self.valid_utts)

    def train_dataloader(self) -> DataLoader:
        return SpeakerUttDataLoader(
            dataset=self.train_dataset,
            num_workers=self.configs.trainer.num_workers,
            min_num_frames=self.configs.model.min_num_frames,
            max_num_frames=self.configs.model.max_num_frames,
            batch_size=self.configs.trainer.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        return SpeakerUttDataLoader(
            dataset=self.valid_dataset,
            num_workers=self.configs.trainer.num_workers,
            min_num_frames=self.configs.model.min_num_frames,
            max_num_frames=self.configs.model.max_num_frames,
            batch_size=self.configs.trainer.batch_size
        )

    def _spk2wav_utts(self, spk2wav, spk2id):
        utts = []
        for spk in spk2wav:
            for wav in spk2wav[spk]:
                utts.append((str(wav), spk2id[spk], None))
        random.shuffle(utts)
        return utts

    def _split_train_valid(self, speaker2wav, spk2id):
        valid_spk2wav = {}
        for spk in speaker2wav:
            random.shuffle(speaker2wav[spk])
            valid_spk2wav[spk] = [speaker2wav[spk].pop(0)]
        train_utts = self._spk2wav_utts(speaker2wav, spk2id)
        valid_utts = self._spk2wav_utts(valid_spk2wav, spk2id)
        return train_utts, valid_utts
