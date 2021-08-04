from dataclasses import dataclass, _MISSING_TYPE, field

from deepaudio.speaker.dataclass.configurations import DeepMMDataclass

@dataclass
class AugmentConfigs(DeepMMDataclass):
    apply_spec_augment: bool = field(
        default=False, metadata={"help": "Flag indication whether to apply spec augment or not"}
    )
    apply_noise_augment: bool = field(
        default=False, metadata={"help": "Flag indication whether to apply noise augment or not "
                                         "Noise augment requires `noise_dataset_path`. "
                                         "`noise_dataset_dir` should be contain audio files."}
    )
    apply_reverb_augment: bool = field(
        default=False, metadata={"help": "Flag indication whether to apply joining augment or not "
                                         "If true, create a new audio file by connecting two audio randomly"}
    )
    apply_noise_reverb_augment: bool = field(
        default=False, metadata={"help": "Flag indication whether to apply spec augment or not"}
    )
    min_snr_in_db: float = field(
        default=3.0, metadata={"help": "Flag indication whether to apply spec augment or not"}
    )
    max_snr_in_db: float = field(
        default=30.0, metadata={"help": "Flag indication whether to apply spec augment or not"}
    )
    freq_mask_para: int = field(
        default=27, metadata={"help": "Hyper Parameter for freq masking to limit freq masking length"}
    )
    freq_mask_num: int = field(
        default=2, metadata={"help": "How many freq-masked area to make"}
    )
    time_mask_num: int = field(
        default=4, metadata={"help": "How many time-masked area to make"}
    )
    noise_dataset_dir: str = field(
        default='None', metadata={"help": "Noise Directory"}
    )
    rir_dataset_dir: str = field(
        default='None', metadata={"help": "Rirs Directory"}
    )
    noise_augment_weight: float = field(
        default=2.0, metadata={"help": "Hyper Parameter for freq masking to limit freq masking length"}
    )
    reverb_augment_weight: float = field(
        default=1.0, metadata={"help": "Hyper Parameter for freq masking to limit freq masking length"}
    )
    noise_reverb_augment_weight: float = field(
        default=2.0, metadata={"help": "Hyper Parameter for freq masking to limit freq masking length"}
    )
    specaugment_weight: float = field(
        default=1.0, metadata={"help": "Hyper Parameter for freq masking to limit freq masking length"}
    )

