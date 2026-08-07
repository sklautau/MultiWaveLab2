'''
Multimodal machine learning for physiological signals.
Modality-aware: ECG ≠ PPG ≠ Bioimp
Robust to missing modalities
Low leakage risk (no direct use of labels like GLC in features)
Compatible with tabular ML + deep learning

Strategy to deal with multiple segments per modality:
A group is defined as all records for a specific participant/session/datetime combination.
Then, for each group the best segment per modality is selected based on quality,
and features are extracted from those segments.

[ METADATA | ECG | PPG | BIOIMP | CROSS_MODAL | QUALITY ]
'''

import argparse
import os

import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import Dict, Any, Tuple, Generator, Optional

from datasets_util.naming_conventions import DatasetConfig
from datasets_util.waveform_files import load_signal
from datasets_util.groups import ColumnsGroup, select_best_in_group
from segments.segments_core import SegmentManager
from signal_processing.ecg import extract_ecg_features
from signal_processing.ppg import extract_ppg_features
from signal_processing.bioimpedance import extract_bioimp_features
from signal_processing.cross_features import extract_cross_features


# Input data is from the following files:
WAVEFORM_ID = "filtered"
BEST_SEGMENTS_INFO_FILE = "best_segments_info.csv"
MULTIMODAL_FEATURES_FILE = "multimodal_features.csv"
INPUT_ALL_SEGMENTS_FILE = "all_segments_with_quality.csv"


def extract_features_from_group(dataframe: pd.DataFrame, fs_dict: Dict[str, int], datasetConfig: DatasetConfig) -> Dict[str, Any]:
    """
    dataframe: A DataFrame with one row per modality for a specific participant/session/datetime combination
    fs_dict: {"ecg": 250, "ppg": 100, "bioimp": 50}
    """
    # check that dataframe has exactly 3 rows, one per modality, and that the modality column contains only valid modalities (ecg, ppg, bioimp)
    if len(dataframe) != 3:
        raise ValueError(
            "dataframe must have exactly 3 rows, one per modality")

    features = {}
    valid_modalities = {"ecg", "ppg", "bioimp"}

    ecg_signal = None
    ppg_signal = None
    bioimp_signal = None

    ecg_peaks = None
    ppg_peaks = None

    for _, row in dataframe.iterrows():
        modality = row["modality"]
        if modality not in valid_modalities:
            raise ValueError(
                f"record_df contains invalid modality: {modality}")

        file_id = row["file_id"]
        path = datasetConfig.get_gen_complete_path(file_id, WAVEFORM_ID)
        print(f"Processing {file_id}: ", path)

        # load CSV or SigMF file and return signal as numpy array
        signal = load_signal(path)

        if modality == "ecg":
            ecg_signal = signal
        elif modality == "ppg":
            ppg_signal = signal
        elif modality == "bioimp":
            bioimp_signal = signal

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


def build_feature_dataset(segmentManager: SegmentManager, fs_dict: Dict[str, int], datasetConfig: DatasetConfig) -> (pd.DataFrame, pd.DataFrame):
    '''Extract features for each session, using only the best quality segment for each modality.'''
    df = best_segment_per_group(segmentManager, datasetConfig)
    all_features = []

    # need to group by participant/session/datetime again, because best_segment_per_group returns one row per modality, and we need to combine them into one row per participant/session/datetime for feature extraction
    columnsGroup = ColumnsGroup(
        df, ["participant_id", "session_id", "datetime"])

    # loop over the groups, extract features for each group, and append to all_features
    for (participant_id, session_id, day_time), group in columnsGroup.df_groupby:
        print("pid:", participant_id, "sid:", session_id,
              "dt:", day_time, "group:", group)

        feats = extract_features_from_group(
            group, fs_dict, datasetConfig)

        feats["participant_id"] = group["participant_id"].iloc[0]
        feats["session_id"] = group["session_id"].iloc[0]
        feats["datetime"] = group["datetime"].iloc[0]
        # assuming GLC is the same for all rows in the group, we can take the first one
        feats["GLC"] = group["GLC"].iloc[0]

        all_features.append(feats)

    return pd.DataFrame(all_features), df


def best_segment_per_group(segmentManager: SegmentManager, datasetConfig: DatasetConfig) -> pd.DataFrame:
    '''
    Create a new dataframe with one row per participant/session/datetime combination,
    containing only the best quality segment for each modality.
    '''
    segments_df = segmentManager.get_segments_dataframe()
    df_merged = datasetConfig.get_dataset_info_dataframe()

    columnsGroup = ColumnsGroup(
        df_merged, ["participant_id", "session_id", "datetime"])

    grouped = columnsGroup.df_groupby
    print(f"Processing {grouped.ngroups} groups...")

    best_rows = []
    num_skipped_groups = 0  # counter
    for (participant_id, session_id, day_time), group in grouped:
        # group is a DataFrame containing all rows for this participant/session/datetime combination
        print("pid:", participant_id, "sid:", session_id,
              "dt:", day_time, "group:", group)
        # show number of records in this group
        print(f"  Number of records in this group: {len(group)}")

        # select best segment for each modality/file_id in this group
        # catch ValueException if no segments are found for some of the required modalities, and print a warning with the group info
        try:
            best_segments_group = select_best_in_group(
                group, segments_df)
            # print("Best segments columns:", best_segments_group.columns)

        except ValueError as e:
            print("############ WARNING ############")
            print(
                f"Skipping group (pid={participant_id}, sid={session_id}, dt={day_time}): {e}")
            num_skipped_groups += 1
            print("############ WARNING ############\n")
            continue

        # append best_segments_group to best_rows
        best_rows.append(best_segments_group)

    print("Total groups processed :", grouped.ngroups)
    print(f"Total groups skipped: {num_skipped_groups}")

    # create a single DataFrame from the list of best rows
    df_best_segments = pd.concat(best_rows, ignore_index=True)

    # save best segments info to file
    output_file_segments = str(
        datasetConfig.machine_learning_path / BEST_SEGMENTS_INFO_FILE)
    df_best_segments.to_csv(output_file_segments, index=False)
    print(f"Best segments info saved to: {output_file_segments}")

    return df_best_segments


if __name__ == "__main__":
    fs_dict = {
        "ecg": 500,
        "ppg": 500,
        "bioimp": np.nan  # set to NaN if not available or not applicable
    }

    # Create a DatasetConfig instance to access dataset paths and info
    # Create a DatasetConfig instance
    # dataset_config_file = "multimodal_dataset_folders.json"

    parser = argparse.ArgumentParser(
        description="Read and process a JSON configuration file."
    )

    parser.add_argument(
        "json_file",
        metavar="JSON_FILE",
        help="Path to the input JSON file."
    )
    # parse command line
    args = parser.parse_args()

    # Create a DatasetConfig instance
    dataset_config_file = args.json_file

    datasetConfig = DatasetConfig(dataset_config_file)

    # generate segments file
    dataset_segments = INPUT_ALL_SEGMENTS_FILE

    # Load the segments information to be able to filter for good quality segments
    segments_path = os.path.join(
        datasetConfig.segments_path, dataset_segments)
    segmentManager = SegmentManager(datasetConfig, segments_path)

    df_features, df_best_segments = build_feature_dataset(
        segmentManager, fs_dict, datasetConfig)

    # save features to file in provided ML folder
    output_file = str(datasetConfig.machine_learning_path /
                      MULTIMODAL_FEATURES_FILE)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_features.to_csv(output_file, index=False)

    print(f"Feature dataset saved to: {output_file}")
