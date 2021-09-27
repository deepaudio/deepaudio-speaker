import warnings

from pathlib import Path
from collections import defaultdict
import pandas as pd
import yaml

from pyannote.core import Segment, Timeline, SlidingWindow


def load_dataframe(wav_dir, table_path):
    df = pd.read_csv(table_path, header=None, delimiter=' ')
    df[0] = df[0].apply(lambda x: Path(wav_dir) / f'{x}.wav')
    return df


def load_trial_dataframe(wav_dir, table_path):
    df = pd.read_csv(table_path, header=None, delimiter=' ')
    df[1] = df[1].apply(lambda x: Path(wav_dir) / f'{x}')
    df[2] = df[2].apply(lambda x: Path(wav_dir) / f'{x}')
    trials = []
    for row in df.iterrows():
        y, enroll, test = row[1]
        trials.append((enroll, test, y))
    return trials


def get_speaker_from_dataframe(dataframe):
    return set(dataframe[3])


def get_spk_id(speakers):
    sorted_speakers = sorted(list(speakers))
    return {spk: i for i, spk in enumerate(sorted_speakers)}


def split_segment(segment, duration, step):
    if segment.duration < duration + step:
        return Timeline([segment])
    else:
        segs = []
        sw = SlidingWindow(start=segment.start, duration=duration, step=step)
        for s in sw:
            if s in segment:
                segs.append(s)
            else:
                break
        if s.start < segment.end < s.end:
            segs.append(Segment(segment.end - duration, segment.end))
    return Timeline(segs)


def get_dataset_items(database_yml, dataset_names, category='train'):
    dataset_items = []
    dataset_names = dataset_names.split(',')
    dataset_names = [n.strip() for n in dataset_names]
    with open(database_yml) as fp:
        dataset = yaml.load(fp, Loader=yaml.FullLoader)
    for name in dataset_names:
        dataset_items.append(get_dataset_item(dataset, name, category))
    return dataset_items


def get_dataset_item(dataset, name, category='train'):
    dataset_item = dataset['Datasets']['SpeakerDataset'][category].get(name, None)
    if dataset_item is None:
        msg = f'{name} does not exist'
        raise ValueError(msg)
    return dataset_item['wav_dir'], dataset_item['list_path']


class SpeakerDataframe:
    def __init__(self, dataset_items,
                 strict=False,
                 segment_min_duration=0,
                 speaker_min_duration=0):
        self.strict = strict
        self.segment_min_duration = segment_min_duration
        self.speaker_min_duration = speaker_min_duration
        dfs = [load_dataframe(*item) for item in dataset_items]
        self.check_speakers(dfs)
        self.load_speaker2items(dfs)

    def check_speakers(self, dataframes):
        all_spks = [get_speaker_from_dataframe(df) for df in dataframes]
        if len(all_spks) > 1 and len(set.intersection(*all_spks)) > 0:
            msg = 'Different datasets contain same speakers'
            if self.strict:
                raise ValueError(msg)
            else:
                warnings.warn(msg)

    def load_speaker2items(self, dataframes):
        self._speaker2items = defaultdict(list)
        self.spk2duration = defaultdict(int)
        for df in dataframes:
            for _, row in df.iterrows():
                wav, start, end, spk = row
                if (end - start) < self.segment_min_duration:
                    continue
                self._speaker2items[spk].append((wav, spk, Segment(start, end)))
                self.spk2duration[spk] += end - start

        for spk in self.spk2duration:
            if self.spk2duration[spk] < self.speaker_min_duration:
                self._speaker2items.pop(spk)

        self._spk_ids = get_spk_id(self._speaker2items.keys())

    @property
    def spk2ids(self):
        return self._spk_ids

    @property
    def speaker2items(self):
        return self._speaker2items
