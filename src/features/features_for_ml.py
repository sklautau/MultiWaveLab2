'''
Multimodal machine learning for physiological signals.
Session-consistent: one feature vector per window (e.g., 30–60 s)
Modality-aware: ECG ≠ PPG ≠ Bioimp
Robust to missing modalities
Low leakage risk (no direct use of labels like GLC in features)
Compatible with tabular ML + deep learning

Important:
If you have multiple files with the SAME modality
(e.g., two ECG files): Only the LAST one is kept—previous ones are overwritten.

[ METADATA | ECG | PPG | BIOIMP | CROSS_MODAL | QUALITY ]
'''

import os

import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import Dict, Any, Tuple, Generator, Optional

from datasets_util.naming_conventions import DatasetConfig
from datasets_util.waveform_files import read_sigmf_file
from datasets_util.groups import ColumnsGroup, select_best_in_group

# Input data is from the following files:
WAVEFORM_ID = "filtered"


def load_signal(path: str) -> np.ndarray:
    """
    Read your .sigmf-data or CSV and return numpy array
    """
    # read csv or sigmf-data file and return signal as numpy array
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        return np.asarray(df['mag_filtered']).flatten()
    elif path.endswith(".sigmf-meta") or path.endswith(".sigmf-data"):
        # implement reading .sigmf-data file and return signal as numpy array
        # you can use read_sigmf_file from datasets_util.waveforms
        signal, metadata = read_sigmf_file(path)
        return np.asarray(signal)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def safe_mean(x: np.ndarray) -> float:
    return np.nanmean(x) if len(x) > 0 else np.nan


def safe_std(x: np.ndarray) -> float:
    return np.nanstd(x) if len(x) > 0 else np.nan


def old_sliding_windows(signal: np.ndarray, fs: int, window_sec: int = 60, overlap: float = 0.5) -> Generator[np.ndarray, None, None]:
    step = int(window_sec * fs * (1 - overlap))
    size = int(window_sec * fs)

    for start in range(0, len(signal) - size + 1, step):
        yield signal[start:start + size]


def extract_ecg_features(ecg: np.ndarray, fs: int) -> Dict[str, Any]:

    try:
        signals, info = nk.ecg_process(ecg, sampling_rate=fs)
        rpeaks = info["ECG_R_Peaks"]

        if False:  # you can enable this block to extract more features using neurokit2
            nk.hrv_frequency(rpeaks, sampling_rate=fs)

            # Nonlinear HRV features
            nk.hrv_nonlinear(rpeaks, sampling_rate=fs)

            # Morphological (wave-based features)
            signals, info = nk.ecg_process(ecg, sampling_rate=fs)
            _, waves = nk.ecg_delineate(
                ecg, info["ECG_R_Peaks"], sampling_rate=fs)

            signals, info = nk.ecg_process(ecg, sampling_rate=fs)

            rpeaks = info["ECG_R_Peaks"]

            features = nk.hrv(rpeaks, sampling_rate=fs)

            features["mean_hr"] = signals["ECG_Rate"].mean()
            features["std_hr"] = signals["ECG_Rate"].std()

        # --- HRV ---
        hrv = nk.hrv(rpeaks, sampling_rate=fs)

        features = {
            "ecg_hr_mean": safe_mean(signals["ECG_Rate"]),
            "ecg_hr_std": safe_std(signals["ECG_Rate"]),
        }

        # Frequency-domain (HRV spectral)
        # flatten HRV
        for col in hrv.columns:
            features[f"ecg_{col.lower()}"] = hrv.iloc[0][col]

        # --- RR ---
        rr = np.diff(rpeaks) / fs
        features["ecg_rr_mean"] = safe_mean(rr)
        features["ecg_rr_std"] = safe_std(rr)

        # --- SQI proxy ---
        features["ecg_sqi"] = np.mean(
            (signals["ECG_Rate"] > 40) & (signals["ECG_Rate"] < 180))

        return features

    except Exception as e:
        raise RuntimeError(f"ECG feature extraction failed: {e}") from e

    return features


def extract_ppg_features(ppg: np.ndarray, fs: int) -> Dict[str, Any]:
    """
    Extracts features from PPG signal.

    Args:
        ppg (np.ndarray): The PPG signal.
        fs (int): The sampling rate of the PPG signal.

    Returns:
        Dict[str, Any]: A dictionary containing the extracted features. The keys are the feature names,
        and the values are the corresponding feature values. The available features are:
        - ppg_hr_mean: The mean heart rate.
        - ppg_hr_std: The standard deviation of the heart rate.
        - ppg_amp_mean: The mean amplitude of the PPG signal.
        - ppg_amp_std: The standard deviation of the PPG amplitude.
        - ppg_peak_interval_std: The standard deviation of the peak intervals.
        - ppg_d1_max_mean: The mean absolute value of the first derivative of the PPG signal.
        - ppg_d2_max_mean: The mean absolute value of the second derivative of the PPG signal.
        - ppg_sqi: The signal quality index (SQI) of the PPG signal.

    If an error occurs during feature extraction, a RuntimeError is raised.
    """
    try:
        signals, info = nk.ppg_process(ppg, sampling_rate=fs)

        peaks = info["PPG_Peaks"]

        features = {
            "ppg_hr_mean": safe_mean(signals["PPG_Rate"]),
            "ppg_hr_std": safe_std(signals["PPG_Rate"]),
        }

        # amplitude
        features["ppg_amp_mean"] = safe_mean(signals["PPG_Clean"])
        features["ppg_amp_std"] = safe_std(signals["PPG_Clean"])

        # peak intervals
        rr = np.diff(peaks) / fs
        features["ppg_peak_interval_std"] = safe_std(rr)

        # derivatives
        d1 = np.gradient(signals["PPG_Clean"])
        d2 = np.gradient(d1)

        features["ppg_d1_max_mean"] = safe_mean(np.abs(d1))
        features["ppg_d2_max_mean"] = safe_mean(np.abs(d2))

        # SQI (native)
        features["ppg_sqi"] = safe_mean(nk.ppg_quality(signals["PPG_Clean"]))

        return features

    except Exception as e:
        raise RuntimeError(f"PPG feature extraction failed: {e}") from e

    return features


def extract_bioimp_features(bioimp: np.ndarray, fs: int) -> Dict[str, Any]:

    try:
        features = {}

        features["bioimp_mean"] = safe_mean(bioimp)
        features["bioimp_std"] = safe_std(bioimp)

        # slope
        slope = np.gradient(bioimp)
        features["bioimp_slope_mean"] = safe_mean(slope)
        features["bioimp_slope_std"] = safe_std(slope)

        features["bioimp_peak_to_peak"] = np.max(bioimp) - np.min(bioimp)

        return features

    except Exception as e:
        raise RuntimeError(
            f"Bioimpedance feature extraction failed: {e}") from e

    return features


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


def extract_features_from_record(record_df: pd.DataFrame, fs_dict: Dict[str, int], datasetConfig: DatasetConfig) -> Dict[str, Any]:
    """
    record_df: subset of df for ONE session
    fs_dict: {"ecg": 250, "ppg": 100, "bioimp": 50}
    """

    features = {}

    ecg_signal = None
    ppg_signal = None
    bioimp_signal = None

    ecg_peaks = None
    ppg_peaks = None

    # Track which modalities we've already processed
    # If multiple files exist for the same modality, only the last one will be used
    seen_modalities = {}

    # ----------------------------------
    # Load signals (you plug your loader)
    # ----------------------------------
    for _, row in record_df.iterrows():

        modality = row["modality"]
        path = row["relative_path"]

        file_id = row["file_id"]
        path = datasetConfig.get_gen_complete_path(file_id, WAVEFORM_ID)
        print(f"Processing {file_id}: ", path)

        signal = load_signal(path)  # <-- YOU IMPLEMENT THIS

        # Warn if we've already seen this modality (will overwrite)
        if modality in seen_modalities:
            print(
                f"WARNING: Multiple files for modality '{modality}' detected!")
            print(
                f"  Previous file_id: {seen_modalities[modality]['prev_file_id']}")
            print(f"  Previous file: {seen_modalities[modality]['prev_path']}")
            print(f"  Current file_id: {file_id}")
            print(f"  Current file: {path}")
            print(f"  --> OVERWRITING previous signal with current one\n")

        if modality == "ecg":
            ecg_signal = signal
            seen_modalities[modality] = {
                'prev_file_id': file_id, 'prev_path': path}
        elif modality == "ppg":
            ppg_signal = signal
            seen_modalities[modality] = {
                'prev_file_id': file_id, 'prev_path': path}
        elif modality == "bioimp":
            bioimp_signal = signal
            seen_modalities[modality] = {
                'prev_file_id': file_id, 'prev_path': path}

    # ----------------------------------
    # ECG
    # ----------------------------------
    if ecg_signal is not None:
        ecg_feat = extract_ecg_features(ecg_signal, fs_dict["ecg"])
        features.update(ecg_feat)
        features["has_ecg"] = 1

        # recompute peaks for cross features
        try:
            _, info = nk.ecg_process(ecg_signal, sampling_rate=fs_dict["ecg"])
            ecg_peaks = info["ECG_R_Peaks"]
        except Exception:
            ecg_peaks = None
    else:
        features["has_ecg"] = 0
        ecg_peaks = None

    # ----------------------------------
    # PPG
    # ----------------------------------
    if ppg_signal is not None:
        ppg_feat = extract_ppg_features(ppg_signal, fs_dict["ppg"])
        features.update(ppg_feat)
        features["has_ppg"] = 1

        try:
            _, info = nk.ppg_process(ppg_signal, sampling_rate=fs_dict["ppg"])
            ppg_peaks = info["PPG_Peaks"]
        except Exception:
            ppg_peaks = None
    else:
        features["has_ppg"] = 0
        ppg_peaks = None

    # ----------------------------------
    # BIOIMP
    # ----------------------------------
    if bioimp_signal is not None:
        bioimp_feat = extract_bioimp_features(bioimp_signal, fs_dict["bioimp"])
        features.update(bioimp_feat)
        features["has_bioimp"] = 1
    else:
        features["has_bioimp"] = 0

    # ----------------------------------
    # CROSS MODAL
    # ----------------------------------
    if ecg_peaks is not None and ppg_peaks is not None:
        features.update(extract_cross_features(
            ecg_peaks, ppg_peaks, fs_dict["ecg"]))

    return features


def build_feature_dataset(df_merged: pd.DataFrame, fs_dict: Dict[str, int], datasetConfig: DatasetConfig) -> pd.DataFrame:

    all_features = []

    columnsGroup = ColumnsGroup(
        df_merged, ["participant_id", "session_id", "datetime"])

    grouped = columnsGroup.df_groupby
    print(f"Processing {grouped.ngroups} groups...")

    for (pid, sid, dt), group in grouped:
        # group is a DataFrame containing all rows for this participant/session/datetime combination
        print("pid:", pid, "sid:", sid, "dt:", dt, "group:", group)
        # show number of records in this group
        print(f"  Number of records in this group: {len(group)}")
        feats = extract_features_from_record(group, fs_dict, datasetConfig)

        feats["participant_id"] = pid
        feats["session_id"] = sid
        feats["datetime"] = dt
        feats["GLC"] = group["GLC"].iloc[0]

        all_features.append(feats)

    return pd.DataFrame(all_features)


if __name__ == "__main__":
    fs_dict = {
        "ecg": 250,
        "ppg": 100,
        "bioimp": 50
    }

    dataset_config_file = "multimodal_dataset_folders.json"

    datasetConfig = DatasetConfig(dataset_config_file)

    # a deep copy is made below, to allow external modifications
    df = datasetConfig.get_dataset_info_dataframe()

    df_features = build_feature_dataset(df, fs_dict, datasetConfig)

    print(df_features.head())

    # save to file
    output_file = str(datasetConfig.generated_waveforms_path) + \
        "/machinelearning/multimodal_features.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_features.to_csv(output_file, index=False)

    print(f"Feature dataset saved to: {output_file}")
