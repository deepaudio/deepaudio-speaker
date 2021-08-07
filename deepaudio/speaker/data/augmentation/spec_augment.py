import numpy as np
import random
from torch import Tensor


class SpecAugment(object):
    """
    Provides Spec Augment. A simple data augmentation method for speech recognition.
    This concept proposed in https://arxiv.org/abs/1904.08779

    Args:
        freq_mask_para (int): maximum frequency masking length
        time_mask_num (int): how many times to apply time masking
        freq_mask_num (int): how many times to apply frequency masking

    Inputs: feature_vector
        - **feature_vector** (torch.FloatTensor): feature vector from audio file.

    Returns: feature_vector:
        - **feature_vector**: masked feature vector.
    """

    def __init__(self, configs) -> None:
        self.freq_mask_para = configs.augment.freq_mask_para
        self.time_mask_num = configs.augment.time_mask_num
        self.freq_mask_num = configs.augment.freq_mask_num

    def __call__(self, feature: Tensor) -> Tensor:
        """ Provides SpecAugmentation for audio """
        time_axis_length = feature.size(0)
        freq_axis_length = feature.size(1)
        time_mask_para = time_axis_length / 20  # Refer to "Specaugment on large scale dataset" paper

        # time mask
        for _ in range(self.time_mask_num):
            t = int(np.random.uniform(low=0.0, high=time_mask_para))
            t0 = random.randint(0, time_axis_length - t)
            feature[t0: t0 + t, :] = 0

        # freq mask
        for _ in range(self.freq_mask_num):
            f = int(np.random.uniform(low=0.0, high=self.freq_mask_para))
            f0 = random.randint(0, freq_axis_length - f)
            feature[:, f0: f0 + f] = 0

        return feature
