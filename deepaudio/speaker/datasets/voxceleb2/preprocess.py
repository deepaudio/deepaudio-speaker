from pathlib import Path

from ..utils import get_subdirs, get_all_wavs


def get_speaker_list(configs):
    data_dir = configs.dataset.dataset_path
    data_dir = configs.dataset.dataset_path
    speaker_dirs = get_subdirs(data_dir)
    speakers = [d.stem for d in speaker_dirs]
    spk2id = {k: v for v, k in enumerate(speakers)}
    return speakers, spk2id


def get_speaker_wavs(data_dir, speakers):
    speaker2wav = {}
    for spk in speakers:
        spk_dir = Path(data_dir) / spk
        speaker2wav[spk] = get_all_wavs(spk_dir)
    return speaker2wav

