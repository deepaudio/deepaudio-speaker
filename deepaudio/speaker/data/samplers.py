from typing import List, Union
from omegaconf import DictConfig

import numpy as np
from collections import defaultdict
import torch
import torch.distributed as dist


def round_down(num, divisor):
    return num - (num % divisor)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class ClovaaiSampler(torch.utils.data.Sampler):
    def __init__(self, labels, configs):

        self.data_label = labels
        self.nPerSpeaker = configs.dataset.per_speaker
        self.batch_size = configs.trainer.batch_size
        self.epoch = 0
        self.seed = 42
        self.distributed = False
        if configs.trainer.accelerator == 'ddp':
            self.distributed = True
        self.__iter__()

    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()

        data_dict = defaultdict(list)

        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            data_dict[speaker_label].append(index)

        dictkeys = list(data_dict.keys())
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        for findex, key in enumerate(dictkeys):
            data = data_dict[key]
            numSeg = round_down(len(data), self.nPerSpeaker)

            rp = lol(np.arange(numSeg), self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Mix data in random order
        mixid = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel = []
        mixmap = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        mixed_list = [flattened_list[i] for i in mixmap]

        ## Divide data to each GPU
        if self.distributed:
            total_size = round_down(len(mixed_list), self.batch_size * dist.get_world_size())
            start_index = int((dist.get_rank()) / dist.get_world_size() * total_size)
            end_index = int((dist.get_rank() + 1) / dist.get_world_size() * total_size)
            self.num_samples = end_index - start_index
            return iter(mixed_list[start_index:end_index])
        else:
            total_size = round_down(len(mixed_list), self.batch_size)
            self.num_samples = total_size
            return iter(mixed_list[:total_size])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
