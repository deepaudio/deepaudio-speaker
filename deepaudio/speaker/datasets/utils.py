from pathlib import Path


def get_subdirs(directory):
    directory = Path(directory)
    return directory.glob('*/')


def get_all_wavs(directory):
    directory = Path(directory)
    return list(directory.glob('**/*.wav'))

