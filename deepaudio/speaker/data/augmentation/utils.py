from pathlib import Path


def get_all_wavs(parent_dir):
    return list(Path(parent_dir).glob('**/*.wav'))
