from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader


def _collate_fn(batch, min_num_frames, max_num_frames):
    r"""
    Functions that pad to the maximum sequence length

    Args:
        batch (tuple): tuple contains input and target tensors

    Returns:
        inputs (torch.FloatTensor): tensor contains input tensor and target tensor.
    """
    def get_min_num_frames(batch):
        return min([sample[0].size(0) for sample in batch])

    def get_subsample(feature, num_frames):
        length = feature.size(0)
        if length < num_frames:
            msg = 'Sample is too short'
            raise ValueError(msg)
        elif length == num_frames:
            return feature
        else:
            start = np.random.randint(0, length - num_frames)
            return feature[start:start + num_frames]

    min_num_frames_batch = get_min_num_frames(batch)
    if min_num_frames < min_num_frames_batch:
        num_frames = min_num_frames
    else:
        num_frames = np.random.randint(min_num_frames, max_num_frames)

    X = []
    y = []
    for item in batch:
        feature = item[0]
        X.append(get_subsample(feature, num_frames).unsqueeze(0))
        y.append(item[1])
    return {
        'X': torch.cat(X),
        'y': torch.tensor(y, dtype=torch.int64)
    }


class SpeakerUttDataLoader(DataLoader):
    r"""
    Text Data Loader

    Args:
        dataset (torch.utils.data.Dataset): dataset from which to load the data.
        num_workers (int): how many subprocesses to use for data loading.
    """
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            num_workers: int,
            min_num_frames: int,
            max_num_frames: int,
            batch_size: int,
            **kwargs,
    ) -> None:
        super(SpeakerUttDataLoader, self).__init__(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            **kwargs,
        )
        self.min_num_frames = min_num_frames
        self.max_num_frames = max_num_frames
        self.collate_fn = partial(_collate_fn,
                                  min_num_frames=min_num_frames,
                                  max_num_frames=max_num_frames)
