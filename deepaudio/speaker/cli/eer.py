from deepaudio.speaker.datasets.dataframe.utils import load_trial_dataframe, get_dataset_items
from deepaudio.speaker.models.inference import Inference
from deepaudio.speaker.metrics.eer import model_eer

trial_meta = get_dataset_items('/home/amax/audio/deepaudio-database/database.yml',
                               'voxceleb1_o', 'trial')
print(trial_meta[0])
wav_dir, trial_path = trial_meta[0]
trials = load_trial_dataframe(wav_dir, trial_path)
inference = Inference('/home/amax/audio/deepaudio-speaker/outputs/2021-10-26/00-37-08/logs/default/version_0/checkpoints/deepaudio-epoch=19-val_loss=2.33.ckpt')
print(model_eer(inference, trials))