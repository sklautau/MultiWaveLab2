'''
Multimodal machine learning for physiological signals.
Modality-aware with support for: ECG, PPG and Bioimpedance.

New strategy to extract features from set:
For each segment in given dataframe, extract the feature based on its modality.
Then, create a new dataframe for each modality (three dataframes in current
case), with the extracted features,
such that they can have the same columns, and where each row has the
corresponding feature values. Include in each row, all important information,
such as patient_id, quality_indicator, segment ID, etc, so that we can later merge the dataframes based
on these columns, and have a concatenaded dataframe with all features for each segment,
which can then be used for machine learning.

# Set command line argument to True to use fixed-duration windows (e.g., 5 seconds) instead of variable-duration segments.
USE_FIXED_DURATION_WINDOWS = True or False
'''

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import sys


from datasets_util.naming_conventions import DatasetConfig
from datasets_util.waveform_files import load_signal
from segments.segments_core import SegmentManager
from signal_processing.ecg import extract_ecg_features
from signal_processing.ppg import extract_ppg_features
from signal_processing.bioimpedance import extract_bioimp_features


# Input data is from the following files:
WAVEFORM_ID = "filtered"

# If False, errors will be logged but the pipeline will continue, returning NaN or empty values for features that failed to extract.
RAISE_EXCEPTION_ON_PPG_PROCESSING = True

# Metadata columns to include in all modality-specific dataframes
'''METADATA_COLUMNS = [
    "participant_id",
    "session_id",
    "datetime",
    "file_id",
    "modality",
    "segment_id",
    "window_id",
    "quality_indicator",
    "GLC"
]
'''
METADATA_COLUMNS = [line.strip() for line in open(
    "../input_ieb1/metadata/metadata_columns.txt").read().splitlines()[1:]]


def parse_bool_arg(value: str) -> bool:
    """Parse common string representations of booleans."""
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value '{value}'. Use one of: true/false, 1/0, yes/no")


def extract_features_for_segment(row: pd.Series, fs_dict: Dict[str, int], datasetConfig: DatasetConfig) -> Any:
    """
    Extract features for a single segment based on its modality.

    Args:
        row: A row from segments dataframe with segment info
        fs_dict: Sampling rates dict {"ecg": 250, "ppg": 100, "bioimp": 50}
        datasetConfig: Dataset configuration object

    Returns:
        Dictionary with extracted features
    """

    modality = row["modality"]
    file_id = row["file_id"]

    # to DEBUG
    # if modality != "ppg":
    #    return  # Skip non-PPG modalities for debugging

    # Load signal
    path = datasetConfig.get_gen_complete_path(file_id, WAVEFORM_ID)
    print(f"Processing {file_id} ({modality}): {path}")

    try:
        signal = load_signal(path)
    except Exception as e:
        print(f"Error loading signal: {e}")
        return None

    # Segment the file based on start and duration and validate bounds.
    start_sample = int(row["start_sample"])
    duration = int(row["duration"])
    n_samples = len(signal)

    if duration <= 0:
        print(
            f"Skipping {file_id} ({modality}): non-positive duration={duration} for segment/window {row.get('segment_id', 'unknown')}")
        return None

    if start_sample < 0 or start_sample >= n_samples:
        print(
            f"Skipping {file_id} ({modality}): start_sample={start_sample} is outside signal length={n_samples}")
        return None

    end_sample = min(start_sample + duration, n_samples)
    signal = signal[start_sample:end_sample]

    if len(signal) == 0:
        print(
            f"Skipping {file_id} ({modality}): empty slice [{start_sample}:{end_sample}] from signal length={n_samples}")
        return None
    # print(f"row: {row}")
    # print(f"Signal shape after segmentation: {signal.shape}")

    features = {}

    # Extract features based on modality
    try:
        if modality == "ecg":
            features = extract_ecg_features(signal, fs_dict["ecg"])
        elif modality == "ppg":
            features = extract_ppg_features(
                signal, fs_dict["ppg"], per_pulse=True)
        elif modality == "bioimp":
            features = extract_bioimp_features(signal, fs_dict["bioimp"])
        else:
            print(f"Unknown modality: {modality}")
            return None
    except Exception as e:
        if RAISE_EXCEPTION_ON_PPG_PROCESSING:
            raise Exception(e)
        else:
            print(
                f"Error in extract_features_for_segment() for {modality}. Message is: {e}")
            return None

    return features


def _append_metadata_to_features(features: Any, segment_metadata: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Attach metadata to a dict or a list of per-pulse dicts."""
    if isinstance(features, list):
        rows: list[Dict[str, Any]] = []
        for feature_row in features:
            if not isinstance(feature_row, dict):
                continue
            row = dict(feature_row)
            row.update(segment_metadata)
            rows.append(row)
        return rows

    if isinstance(features, dict):
        row = dict(features)
        row.update(segment_metadata)
        return [row]

    return []


def build_modality_dataframes(segmentManager: SegmentManager, fs_dict: Dict[str, int], datasetConfig: DatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract features for each segment and create separate dataframes per modality.

    Returns:
        Tuple of (df_ecg, df_ppg, df_bioimp)
    """
    segments_df = segmentManager.get_segments_dataframe()
    df_dataset_info = datasetConfig.get_dataset_info_dataframe()

    # debug
    # segments_df.to_csv("segments_info.csv", index=False)

    # Merge segments with dataset info to get all metadata
    df_merged = segments_df.merge(
        df_dataset_info[["file_id", "participant_id",
                         "session_id", "datetime", "GLC"]],
        on="file_id",
        how="left"
    )

    # debug
    # df_merged.to_csv("merged_segments_info.csv", index=False)

    # Initialize lists for each modality
    ecg_features = []
    ppg_features = []
    bioimp_features = []

    num_processed = 0
    num_errors = 0

    # Process each segment
    for idx, row in df_merged.iterrows():
        modality = row["modality"]

        # Extract features
        features = extract_features_for_segment(row, fs_dict, datasetConfig)

        if features is None:
            num_errors += 1
            continue

        # Add metadata to features
        segment_metadata = {col: row[col]
                            for col in METADATA_COLUMNS if col in row}
        feature_rows = _append_metadata_to_features(features, segment_metadata)

        # Append to appropriate modality list
        if modality == "ecg":
            ecg_features.extend(feature_rows)
        elif modality == "ppg":
            ppg_features.extend(feature_rows)
        elif modality == "bioimp":
            bioimp_features.extend(feature_rows)

        num_processed += 1
        if num_processed % 10 == 0:
            print(f"Processed {num_processed} segments...")

    print(f"\nTotal segments processed: {num_processed}")
    print(f"Total errors: {num_errors}")

    # Create dataframes
    df_ecg = pd.DataFrame(ecg_features) if ecg_features else pd.DataFrame()
    df_ppg = pd.DataFrame(ppg_features) if ppg_features else pd.DataFrame()
    df_bioimp = pd.DataFrame(
        bioimp_features) if bioimp_features else pd.DataFrame()

    print(f"\nECG dataframe shape: {df_ecg.shape}")
    print(f"PPG dataframe shape: {df_ppg.shape}")
    print(f"Bioimpedance dataframe shape: {df_bioimp.shape}")

    return df_ecg, df_ppg, df_bioimp


def save_modality_dataframes(df_ecg: pd.DataFrame, df_ppg: pd.DataFrame, df_bioimp: pd.DataFrame, datasetConfig: DatasetConfig):
    """Save modality-specific dataframes to CSV files."""
    output_dir = datasetConfig.machine_learning_path

    if not df_ecg.empty:
        output_file = os.path.join(output_dir, "features_ecg_segments.csv")
        df_ecg.to_csv(output_file, index=False)
        print(f"ECG features saved to: {output_file}")

    if not df_ppg.empty:
        output_file = os.path.join(output_dir, "features_ppg_segments.csv")
        df_ppg.to_csv(output_file, index=False)
        print(f"PPG features saved to: {output_file}")

    if not df_bioimp.empty:
        output_file = os.path.join(output_dir, "features_bioimp_segments.csv")
        df_bioimp.to_csv(output_file, index=False)
        print(f"Bioimpedance features saved to: {output_file}")


if __name__ == "__main__":
    # read USE_FIXED_DURATION_WINDOWS from command line argument if provided
    USE_FIXED_DURATION_WINDOWS = False  # default value
    print(f"Command line arguments: {sys.argv}")
    if len(sys.argv) > 1:
        USE_FIXED_DURATION_WINDOWS = parse_bool_arg(sys.argv[1])

    # Create a DatasetConfig instance
    dataset_config_file = "multimodal_dataset_folders.json"
    datasetConfig = DatasetConfig(dataset_config_file)

    # Use sampling frequencies from dataset configuration (JSON).
    ppg_fs = datasetConfig.get_ppg_fs()
    ecg_fs = int(datasetConfig.config_dictionary.get("ECG_FS", 500))

    fs_dict = {
        "ecg": ecg_fs,
        "ppg": ppg_fs,
        "bioimp": np.nan  # set to NaN if not available or not applicable
    }

    if USE_FIXED_DURATION_WINDOWS:
        input_file_name = "all_fixed_duration_windows_with_quality.csv"
    else:
        # input_file_name = "all_segments_with_quality.csv"
        input_file_name = "best_segments_per_file.csv"
    print(
        f"Using input file: {input_file_name} because USE_FIXED_DURATION_WINDOWS={USE_FIXED_DURATION_WINDOWS}")

    # Load segments or windows information
    input_path = os.path.join(datasetConfig.segments_path, input_file_name)
    segmentManager = SegmentManager(datasetConfig, input_path)

    # Extract features for each modality
    df_ecg, df_ppg, df_bioimp = build_modality_dataframes(
        segmentManager, fs_dict, datasetConfig)

    # Save dataframes
    save_modality_dataframes(df_ecg, df_ppg, df_bioimp, datasetConfig)

    print("\nFeature extraction complete!")
