import numpy as np
from sklearn.metrics import roc_curve

from .utils import get_all_wavs, get_all_embeddings


def compute_eer(y, y_pred, pos_label=1):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=pos_label)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer, eer_threshold


def model_eer(model, trials, wav_dict):
    wav_trials = get_all_wavs(trials, wav_dict)
    embedding_trials = get_all_embeddings(model, wav_trials)


