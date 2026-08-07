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
import shutil
import traceback
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import argparse
import json
from pathlib import Path

from datasets_util.naming_conventions import LOGIDENTIFIER, DatasetConfig
from datasets_util.waveform_files import load_signal
from segments.segments_core import SegmentManager, Segmenter
from features.clean_features_dataframe import cleaning_and_imputation
from signal_processing.ecg import extract_ecg_features
from signal_processing.ppg import extract_ppg_features
from signal_processing.bioimpedance import extract_bioimp_features
from datasets_util.naming_conventions import MANDATORY_METADATA_COLUMNS

# Input data is from the following files:
WAVEFORM_ID = "filtered"

# If False, errors will be logged but the pipeline will continue, returning NaN or empty values for features that failed to extract.
RAISE_EXCEPTION_ON_PPG_PROCESSING = True

# Metadata columns to include in all modality-specific dataframes
'''
METADATA_COLUMNS = [
    "participant_id",
    "start_sample",
    "duration",
    "session_id",
    "datetime",
    "file_id",
    "modality",
    "segment_id",
    "quality_indicator",
    "GLC"
]
METADATA_COLUMNS = [line.strip() for line in open(
    "../input_ieb1/metadata/metadata_columns.txt").read().splitlines()[1:]]
'''


def convert_features_of_each_glucose_measurement_to_single_vector(df: pd.DataFrame, metadata_columns: list[str]) -> pd.DataFrame:
    """
    Convert features of each group of (participant, session, date-time), which
    correspond to a group that has a distinct glocuse measurement,
    into a single vector by aggregating features.

    Args:
        df: DataFrame containing features and metadata.
        metadata_columns: List of metadata columns to retain.

    Returns:
        DataFrame with aggregated features for each group.
    """

    if df.empty or len(df.columns.to_list()) == 0:
        print("Warning: DataFrame is empty. No features to aggregate.")
        return df

    group_columns = ["participant_id", "session_id", "datetime"]
    missing_group_columns = [
        col for col in group_columns if col not in df.columns]
    if missing_group_columns:
        raise ValueError(
            f"Missing required columns for glucose-measurement aggregation: {missing_group_columns}")

    if "GLC" not in df.columns:
        raise ValueError(
            "Column 'GLC' is required to aggregate features by glucose measurement.")

    metadata_columns_present = [
        col for col in metadata_columns if col in df.columns]
    metadata_columns_set = set(metadata_columns_present)

    # Check whether each (participant_id, session_id, datetime) group has a single GLC value.
    groups_with_multiple_glc = df.groupby(
        group_columns)['GLC'].nunique(dropna=False)
    groups_with_multiple_glc = groups_with_multiple_glc[groups_with_multiple_glc > 1]
    if not groups_with_multiple_glc.empty:
        offending_groups = [
            tuple(group_key) if isinstance(group_key, tuple) else (group_key,)
            for group_key in groups_with_multiple_glc.index
        ]
        raise ValueError(
            f"Warning: The following groups have multiple GLC values: {offending_groups}")

    # Aggregate feature columns only; keep metadata columns with a stable representative value.
    feature_columns = [
        col for col in df.columns
        if col not in metadata_columns_set and col not in group_columns
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    aggregation_rules = {col: "mean" for col in feature_columns}
    for col in metadata_columns_present:
        if col not in group_columns:
            aggregation_rules[col] = "first"

    # Preserve the glucose value after the uniqueness check above.
    aggregation_rules["GLC"] = "first"

    aggregated_df = df.groupby(
        group_columns, as_index=False).agg(aggregation_rules)

    return aggregated_df


def convert_features_of_each_fileid_to_single_vector(df: pd.DataFrame, metadata_columns: list[str]) -> pd.DataFrame:
    """
    Convert features of each participant into a single vector by aggregating features.

    Args:
        df: DataFrame containing features and metadata.
        metadata_columns: List of metadata columns to retain.

    Returns:
        DataFrame with aggregated features for each participant.
    """

    if len(df.columns.to_list()) == 0:
        print("Warning: DataFrame is empty. No features to aggregate.")
        return df

    # check whether each file_id has a single value for GLC target
        raise ValueError(
            "Column 'file_id' is required to aggregate features into a single vector.")

    metadata_columns_present = [
        col for col in metadata_columns if col in df.columns]
    # Keep file_id as grouping key and aggregate only feature columns.
    feature_columns = [
        col for col in df.columns
        if col not in metadata_columns_present and col != 'file_id'
    ]
    aggregated_df = df.groupby('file_id')[feature_columns].mean().reset_index()

    # Merge back the metadata columns
    metadata_df_columns = list(
        dict.fromkeys(['file_id'] + [col for col in metadata_columns_present if col != 'file_id']))
    metadata_df = df[metadata_df_columns].drop_duplicates(subset=['file_id'])
    final_df = pd.merge(aggregated_df, metadata_df, on='file_id', how='left')

    return final_df


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
    if modality == "ecg":
        input_waveform_id = datasetConfig.get_value(
            "ECG_FEATURE_INPUT_WAVEFORM")
    elif modality == "ppg":
        input_waveform_id = datasetConfig.get_value(
            "PPG_FEATURE_INPUT_WAVEFORM")
    elif modality.startswith("bioimp"):
        input_waveform_id = datasetConfig.get_value(
            "BIOIMP_FEATURE_INPUT_WAVEFORM")
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # Load signal
    path = datasetConfig.get_gen_complete_path(file_id, input_waveform_id)
    print(f"Processing {file_id} ({modality}): {path}")

    try:
        signal = load_signal(path)
    except Exception as e:
        print(f"Error loading signal: {e}")
        traceback.print_exc()
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

    features = {}  # initialize features as empty dict in case of error

    # Extract features based on modality
    try:
        if modality == "ecg":
            features = extract_ecg_features(datasetConfig,
                                            signal, fs_dict["ecg"])
        elif modality == "ppg":
            features = extract_ppg_features(datasetConfig,
                                            signal, fs_dict["ppg"])
        elif modality == "bioimp":
            features = extract_bioimp_features(datasetConfig,
                                               signal, fs_dict["bioimp"])
        else:
            print(f"Unknown modality: {modality}")
            return None
    except Exception as e:
        print(
            f"Error in extract_features_for_segment() for {modality}. Message is: {e}")
        traceback.print_exc()
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
    Extract a single feature vector for each segment provided by
    SegmentManager and create separate dataframes per modality.

    Returns:
        Tuple of (df_ecg, df_ppg, df_bioimp)
    """
    segments_df = segmentManager.get_segments_dataframe()
    dataset_info_df = datasetConfig.get_dataset_info_dataframe()
    # print both heads
    print("Segments dataframe head:")
    print(segments_df.head())
    print("Dataset info dataframe head:")
    print(dataset_info_df.head())

    # get the metadata columns we are interested in keeping in the output CSV features file
    # including the target column (GLC)
    metadata_columns = datasetConfig.get_chosen_metadata_columns()
    # Merge segments with dataset info to get all metadata
    metadata_in_segments = datasetConfig.get_segments_columns()
    # all metadata columns, avoiding to repeat colums
    all_metadata_columns = list(dict.fromkeys(
        metadata_in_segments + metadata_columns))
    if "file_id" not in all_metadata_columns:
        all_metadata_columns.append("file_id")
    # Find columns from dataset info that are not in segment columns
    metadata_columns_to_be_added = [
        col for col in metadata_columns
        if col not in metadata_in_segments and col != "file_id"
    ]
    metadata_columns_to_be_added.append("file_id")
    metadata_columns_to_be_added = list(
        dict.fromkeys(metadata_columns_to_be_added))

    '''
    The block below is doing a left join in pandas:
    - Left table: segments_df
    - Right table: df_dataset_info filtered to only these columns: file_id, participant_id, session_id, datetime, GLC
    - Join key: file_id
    - Join type: left, meaning all rows from segments_df are kept

    What it accomplishes:

    1. For each segment row, it looks up the matching file_id in dataset info.
    2. If a match exists, it appends participant_id, session_id, datetime, and GLC to that segment row.
    3. If no match exists, those appended columns become NaN, but the segment row still remains (because of how="left").

    Why that column subset is used:

    - It limits the merge to only metadata needed downstream.
    - It avoids bringing unnecessary columns.
    - It reduces chances of column name collisions.

    Important behavior to be aware of:

    - If df_dataset_info has duplicate file_id values, the merge can duplicate rows from segments_df (one-to-many expansion).
    - If file_id is unique in df_dataset_info, each segment row gets at most one metadata match.
    '''
    df_merged = segments_df.merge(
        dataset_info_df[metadata_columns_to_be_added],
        on="file_id",
        how="left"  # left join is not going to create a row if a file_id is not in segments_df
    )
    print("Metadata columns that were added=", metadata_columns_to_be_added)
    print(f"Merged dataframe shape: {df_merged.shape}")

    # Initialize lists for each modality.
    # Each list will hold dictionaries of features and metadata for that modality.
    ecg_features = []
    ppg_features = []
    bioimp_features = []

    num_processed = 0
    num_errors = 0

    print(f"\nMerged dataframe head:")
    print(df_merged.head())

    # Process each segment
    for idx, row in df_merged.iterrows():
        modality = row["modality"]

        # Extract features
        features = extract_features_for_segment(row, fs_dict, datasetConfig)

        if features is None:
            num_errors += 1
            continue

        # Add metadata to features and create a new dictionary
        # with features and metadata for this segment
        segment_metadata = {col: row[col]
                            for col in all_metadata_columns if col in row}
        feature_rows = _append_metadata_to_features(features, segment_metadata)

        # Append to appropriate modality list
        # considering that feature_rows could be a single dict
        # or a list of dicts (e.g., for per-pulse features)
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

    print("Finished computing features for all modalities:")
    print(f"\nECG dataframe shape: {df_ecg.shape}")
    print(f"PPG dataframe shape: {df_ppg.shape}")
    print(f"Bioimpedance dataframe shape: {df_bioimp.shape}")

    should_aggregate_per_file = datasetConfig.get_value(
        "AGGREGATE_FEATURES_PER_FILE", False)
    should_aggregate_per_glc_measurement = datasetConfig.get_value(
        "AGGREGATE_FEATURES_PER_GLUCOSE_MEASUREMENT", False)
    if should_aggregate_per_file and should_aggregate_per_glc_measurement:
        raise ValueError(
            "Both AGGREGATE_FEATURES_PER_FILE and AGGREGATE_FEATURES_PER_GLUCOSE_MEASUREMENT are set to True. Please choose only one.")

    if should_aggregate_per_file:
        print("Aggregating all features of a given file_id into single vector, for all modalities:")
        df_ecg = convert_features_of_each_fileid_to_single_vector(
            df_ecg, all_metadata_columns)
        df_ppg = convert_features_of_each_fileid_to_single_vector(
            df_ppg, all_metadata_columns)
        df_bioimp = convert_features_of_each_fileid_to_single_vector(
            df_bioimp, all_metadata_columns)
    elif should_aggregate_per_glc_measurement:
        print(
            "Aggregating all features of a given (participant, session, datetime) into single vector, for all modalities:")
        df_ecg = convert_features_of_each_glucose_measurement_to_single_vector(
            df_ecg, all_metadata_columns)
        df_ppg = convert_features_of_each_glucose_measurement_to_single_vector(
            df_ppg, all_metadata_columns)
        df_bioimp = convert_features_of_each_glucose_measurement_to_single_vector(
            df_bioimp, all_metadata_columns)

    if should_aggregate_per_file or should_aggregate_per_glc_measurement:
        print(LOGIDENTIFIER +
              f"Aggregated ECG dataframe shape: {df_ecg.shape}")
        print(LOGIDENTIFIER +
              f"Aggregated PPG dataframe shape: {df_ppg.shape}")
        print(LOGIDENTIFIER +
              f"Aggregated Bioimpedance dataframe shape: {df_bioimp.shape}")

    return df_ecg, df_ppg, df_bioimp


def clean_and_save_modality_dataframes(df_ecg: pd.DataFrame, df_ppg: pd.DataFrame, df_bioimp: pd.DataFrame, datasetConfig: DatasetConfig):
    """Clean and save modality-specific dataframes to CSV files."""
    output_dir = datasetConfig.features_path

    if not df_ecg.empty:
        print(LOGIDENTIFIER + "Cleaning ECG features...")
        df_ecg = cleaning_and_imputation(df_ecg, datasetConfig, modality="ecg")
        df_ecg = make_sure_mandatory_columns_exist(df_ecg)
        output_file = os.path.join(
            output_dir, datasetConfig.get_features_file_name(modality="ecg"))
        df_ecg.to_csv(output_file, index=False)
        print(f"ECG features saved to: {output_file}")

    if not df_ppg.empty:
        print(LOGIDENTIFIER + "Cleaning PPG features...")
        df_ppg = cleaning_and_imputation(df_ppg, datasetConfig, modality="ppg")
        df_ppg = make_sure_mandatory_columns_exist(df_ppg)
        output_file = os.path.join(
            output_dir, datasetConfig.get_features_file_name(modality="ppg"))
        df_ppg.to_csv(output_file, index=False)
        print(f"PPG features saved to: {output_file}")

    if not df_bioimp.empty:
        print(LOGIDENTIFIER + "Cleaning Bioimpedance features...")
        df_bioimp = cleaning_and_imputation(
            df_bioimp, datasetConfig, modality="bioimp")
        df_bioimp = make_sure_mandatory_columns_exist(df_bioimp)
        output_file = os.path.join(
            output_dir, datasetConfig.get_features_file_name(modality="bioimp"))
        df_bioimp.to_csv(output_file, index=False)
        print(f"Bioimpedance features saved to: {output_file}")


def make_sure_mandatory_columns_exist(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure that mandatory columns exist in the dataframe."""
    if "session_id" not in df.columns:
        # create a new column called session_id with all values equal to 1
        df["session_id"] = 1
    if "datetime" not in df.columns:
        # create a new column called datetime
        df["datetime"] = -1

    # all the others are mandatory:
    for col in MANDATORY_METADATA_COLUMNS:
        if col not in df.columns:
            raise Exception(
                f"Mandatory column '{col}' is missing in the dataframe. Please check the input data and ensure that all the following (required) metadata columns are present: {MANDATORY_METADATA_COLUMNS}")

    return df


if __name__ == "__main__":
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

    # Use sampling frequencies from dataset configuration (JSON).
    ppg_fs = datasetConfig.get_ppg_fs()
    ecg_fs = int(datasetConfig.config_dictionary.get("ECG_FS", 500))

    fs_dict = {
        "ecg": ecg_fs,
        "ppg": ppg_fs,
        "bioimp": np.nan  # set to NaN if not available or not applicable
    }

    # find the used segmenter, which gives the output file name for the segments
    # segmenter_file = datasetConfig.get_segmenter_file_name()
    # segmenter = Segmenter(segmenter_file, dataset_config_file)
    # segments_file_name = segmenter.get_segments_file_name()
    segments_file_name = datasetConfig.get_segments_file_name()
    print(f"Using input file: {segments_file_name}")

    # Load segments or windows information
    # input_path = os.path.join(datasetConfig.segments_path, input_file_name)
    segmentManager = SegmentManager(datasetConfig, segments_file_name)

    # Extract features for each modality
    df_ecg, df_ppg, df_bioimp = build_modality_dataframes(
        segmentManager, fs_dict, datasetConfig)

    # Save dataframes
    clean_and_save_modality_dataframes(
        df_ecg, df_ppg, df_bioimp, datasetConfig)

    print("\nFeature extraction complete!")
