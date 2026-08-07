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
from datasets_util.waveform_files import load_signal
from datasets_util.groups import ColumnsGroup, select_best_in_group
from signal_processing.ecg import extract_ecg_features
from signal_processing.ppg import extract_ppg_features
from signal_processing.bioimpedance import extract_bioimp_features
from signal_processing.cross_features import extract_cross_features

# Input data is from the following files:
WAVEFORM_ID = "filtered"


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

        # load CSV or SigMF file and return signal as numpy array
        signal = load_signal(path)

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
    '''
    Build a feature dataset where each row corresponds to a
    unique combination of participant/session/datetime,
    and contains features extracted from the corresponding signals.
    When there are multiple files for the same modality within a group,
    only the last one is used (with a warning).
    '''

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
        filter_based_on_quality = False
        if filter_based_on_quality:
            # get only best quality record for each modality in this group
            group = select_best_in_group(group)
            exit(-1)
        feats = extract_features_from_record(group, fs_dict, datasetConfig)

        feats["participant_id"] = pid
        feats["session_id"] = sid
        feats["datetime"] = dt
        feats["GLC"] = group["GLC"].iloc[0]

        all_features.append(feats)

    return pd.DataFrame(all_features)


if __name__ == "__main__":
    fs_dict = {
        "ecg": 500,
        "ppg": 500,
        "bioimp": np.NaN  # set to NaN if not available or not applicable
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
