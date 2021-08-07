from torch_audiomentations import AddBackgroundNoise, ApplyImpulseResponse, Compose

from .utils import get_all_wavs


class Noise:
    def __init__(self, configs):
        background_paths = get_all_wavs(configs.augment.noise_dir)
        self.noise = AddBackgroundNoise(background_paths=configs.augment.background_paths,
                                        min_snr_in_db=configs.augment.min_snr_in_db,
                                        max_snr_in_db=configs.augment.max_snr_in_db,
                                        p=1)

    def __call__(self, waveform):
        waveform = waveform.unsqueeze(0)
        return self.noise(waveform).squeeze(0)


class Reverb:
    def __init__(self, configs):
        ir_paths = get_all_wavs(configs.augment.rir_dir)
        self.reverb = ApplyImpulseResponse(ir_paths=ir_paths, p=1)

    def __call__(self, waveform):
        waveform = waveform.unsqueeze(0)
        return self.reverb(waveform).squeeze(0)


class NoiseReverb:
    def __init__(self, configs):
        background_paths = get_all_wavs(configs.augment.noise_dir)
        ir_paths = get_all_wavs(configs.augment.rir_dir)
        self.noise = AddBackgroundNoise(background_paths=background_paths,
                                        min_snr_in_db=configs.augment.min_snr_in_db,
                                        max_snr_in_db=configs.augment.max_snr_in_db,
                                        p=1)
        self.reverb = ApplyImpulseResponse(ir_paths=ir_paths, p=1)
        self.compose = Compose([self.noise, self.reverb], p=1)

    def __call__(self, waveform):
        waveform = waveform.unsqueeze(0)
        return self.compose(waveform).squeeze(0)
