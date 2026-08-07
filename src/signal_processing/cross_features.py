'''
@TODO: move this code to other file.
'''
import numpy as np
from typing import Dict, Any


def safe_mean(x: np.ndarray) -> np.floating:
    return np.nanmean(x) if len(x) > 0 else np.nan


def safe_std(x: np.ndarray) -> np.floating:
    return np.nanstd(x) if len(x) > 0 else np.nan


def extract_cross_features(ecg_peaks: np.ndarray, ppg_peaks: np.ndarray, fs: int) -> Dict[str, float]:

    features = {}

    try:
        # PTT (ECG R → PPG peak)
        n = min(len(ecg_peaks), len(ppg_peaks))
        delays = (ppg_peaks[:n] - ecg_peaks[:n]) / fs

        features["ptt_mean"] = safe_mean(delays)
        features["ptt_std"] = safe_std(delays)

    except Exception:
        features["ptt_mean"] = np.nan
        features["ptt_std"] = np.nan

    return features
