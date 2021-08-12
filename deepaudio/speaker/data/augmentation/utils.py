import glob


def get_all_wavs(parent_dir):
    return glob.glob(f'{parent_dir}/**/*.wav', recursive=True)
