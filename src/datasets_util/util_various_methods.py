from pathlib import Path
import os
import sys
import warnings
import traceback
from collections import defaultdict
import numpy as np
from itertools import combinations
import pandas as pd


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else None
    traceback.print_stack(file=log)
    warnings._showwarning_orig(message, category, filename, lineno, file, line)


def initialize_script_environment():
    # ===========================
    # Setup Script Directorys
    # ===========================
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    # ===========================
    # Warning with Traceback
    # Helps to identify where warnings are coming from
    # ===========================

    warnings._showwarning_orig = warnings.showwarning
    warnings.showwarning = warn_with_traceback

    return SCRIPT_DIR, PROJECT_ROOT


def accumulate_mse_dicts(list_of_mse_dicts):
    """
    Input:
        list_of_mse_dicts = [
            { (det1, det2): mse_value, ... },   # subject 1
            { (det1, det2): mse_value, ... },   # subject 2
            ...
        ]

    Output:
        avg_mse_dict = { (det1, det2): mean_mse }
    """

    # Stores: key → sum of MSEs
    mse_sum = defaultdict(float)

    # Stores: key → number of occurrences
    mse_count = defaultdict(int)

    # -------- Accumulate --------
    for mse_dict in list_of_mse_dicts:
        for pair, mse in mse_dict.items():
            mse_sum[pair] += mse
            mse_count[pair] += 1

    # -------- Compute averages --------
    avg_mse_dict = {
        pair: mse_sum[pair] / mse_count[pair]
        for pair in mse_sum
    }

    return avg_mse_dict

# ---------------------------------------------------------
# MSE between peak locations
# ---------------------------------------------------------


def old_mse(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.nan

    # align sequences by min length
    L = min(len(a), len(b))
    return np.mean((a[:L] - b[:L]) ** 2)


def mse(a, b):
    """
    1) Remove mean
    2) Compute cross-correlation
    3) Align a and b by the shift that maximizes |xcorr|
    4) Return the MSE over the overlapping region
    """

    a = np.asarray(a)
    b = np.asarray(b)

    if len(a) == 0 or len(b) == 0:
        return np.nan, None, 0

    # 1. mean removal
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)

    # 2. cross-correlation (full range)
    xcorr = np.correlate(a0, b0, mode="full")

    # lags: negative means b is shifted right
    lags = np.arange(-(len(b)-1), len(a))

    # 3. find lag with maximum absolute correlation
    best_index = np.argmax(np.abs(xcorr))
    best_lag = lags[best_index]

    # 4. extract the overlapping region given the best lag
    if best_lag >= 0:
        # b starts at position `best_lag` of a
        a_seg = a[best_lag:]
        b_seg = b[:len(a_seg)]
    else:
        # b starts before a
        a_seg = a[:len(a)+best_lag]
        b_seg = b[-best_lag:len(b)]

    # make sure lengths match
    L = min(len(a_seg), len(b_seg))
    if L == 0:
        return np.nan, best_lag, 0

    mse_val = np.mean((a_seg[:L] - b_seg[:L])**2)

    return mse_val


def pairwise_mse(results, key_name="peaks"):
    '''
    input results is a list of dicts like:
    [ {"name": detector_name, "peaks": [...], ...}, ... ]
    '''
    pairs = {}
    for (r1, r2) in combinations(results, 2):
        m = mse(r1[key_name], r2[key_name])
        pairs[(r1["name"], r2["name"])] = m
    return pairs


def find_wave_file(subject_folder):
    """Locate wave.csv in subject directory."""
    for file_ in os.listdir(subject_folder):
        if file_.endswith('wave.csv'):
            return os.path.join(subject_folder, file_)
    return None


def load_signal(wave_path):
    """Read PPG signal from CSV."""
    try:
        df = pd.read_csv(wave_path)
        signal = np.asarray(df['Wave']).flatten()
        return signal
    except Exception as e:
        print(f"Error loading signal from {wave_path}: {e}")
        return None
