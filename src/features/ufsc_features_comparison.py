'''
Compares our PPG feature extraction with the features
extracted at UFSC.
'''

import argparse
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

from features.features_per_modality import save_modality_dataframes

# Input data is from the following files:
WAVEFORM_ID = "filtered"
# If False, errors will be logged but the pipeline will continue, returning NaN or empty values for features that failed to extract.
RAISE_EXCEPTION_ON_PPG_PROCESSING = True
REQUIRED_FS = 60  # Required sampling frequency for PPG signals (in Hz)

FEATURES_FILE_FOR_COMPARISON = r"..\tcc_guilherme\files\dataRecord_spectrogram_v5.csv"
SEGMENTS_FILE_FOR_COMPARISON = r"..\tcc_guilherme\good_segments.csv"
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


def compare_dataframes_by_column_name(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = True,
) -> dict:
    """
    Compare two pandas DataFrames by matching columns with the same names,
    regardless of column order.

    Returns a dictionary with:
      - columns_only_in_df1
      - columns_only_in_df2
      - matched_columns
      - shape_info
      - comparison_per_column
      - all_matched_columns_equal
    """

    cols1 = list(df1.columns)
    cols2 = list(df2.columns)

    set1 = set(cols1)
    set2 = set(cols2)

    matched_columns = [col for col in cols1 if col in set2]
    columns_only_in_df1 = [col for col in cols1 if col not in set2]
    columns_only_in_df2 = [col for col in cols2 if col not in set1]

    results = {}

    same_number_of_rows = len(df1) == len(df2)

    for col in matched_columns:
        if not same_number_of_rows:
            results[col] = {
                "status": "not_compared",
                "reason": "different number of rows",
                "df1_rows": len(df1),
                "df2_rows": len(df2),
            }
            continue

        x = pd.to_numeric(df1[col], errors="coerce").to_numpy()
        y = pd.to_numeric(df2[col], errors="coerce").to_numpy()

        close_mask = np.isclose(
            x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)
        diff = x - y

        results[col] = {
            "status": "compared",
            "equal": bool(np.all(close_mask)),
            "num_equal": int(np.sum(close_mask)),
            "num_different": int(np.sum(~close_mask)),
            "max_abs_error": float(np.nanmax(np.abs(diff))) if len(diff) else 0.0,
            "mean_abs_error": float(np.nanmean(np.abs(diff))) if len(diff) else 0.0,
        }

    all_equal = all(
        info.get("equal", False)
        for info in results.values()
        if info["status"] == "compared"
    )

    return {
        "columns_only_in_df1": columns_only_in_df1,
        "columns_only_in_df2": columns_only_in_df2,
        "matched_columns": matched_columns,
        "shape_info": {
            "df1_shape": df1.shape,
            "df2_shape": df2.shape,
            "same_number_of_rows": same_number_of_rows,
        },
        "comparison_per_column": results,
        "all_matched_columns_equal": all_equal,
    }


def read_file_and_get_features(row: pd.Series, fs: float, datasetConfig: DatasetConfig) -> Optional[Dict[str, Any] | list[Dict[str, Any]]]:
    """
    Extract features for a single segment.

    Args:
        row: A row from segments dataframe with segment info
        fs: Sampling rate
        datasetConfig: Dataset configuration object

    Returns:
        Dictionary with extracted features
    """

    modality = row["modality"]
    file_id = row["file_id"]

    if modality != "ppg":
        raise ValueError(
            f"Expected modality 'ppg', but got '{modality}' for file_id '{file_id}'")

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
        features = extract_ppg_features(
            signal, int(REQUIRED_FS), per_pulse=True)
    except Exception as e:
        if RAISE_EXCEPTION_ON_PPG_PROCESSING:
            raise Exception(e)
        else:
            print(
                f"Error in extract_features_for_segment() for {modality}. Message is: {e}")
            return None

    return features


def _append_metadata_to_features(features: Any, segment_metadata: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Attach metadata to a dict or to each dict in a list of per-pulse features."""
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


def build_ppg_dataframe(segmentManager: SegmentManager, datasetConfig: DatasetConfig) -> pd.DataFrame:
    """
    Extract features.
    """
    segments_df = segmentManager.get_segments_dataframe()
    df_dataset_info = datasetConfig.get_dataset_info_dataframe()

    # Merge segments with dataset info to get all metadata
    df_merged = segments_df.merge(
        df_dataset_info[["file_id", "participant_id",
                         "session_id", "datetime", "GLC"]],
        on="file_id",
        how="left"
    )

    # Initialize list
    ppg_features = []

    num_processed = 0
    num_errors = 0

    # Process each segment
    for idx, row in df_merged.iterrows():
        modality = row["modality"]

        # Extract features
        features = read_file_and_get_features(row, REQUIRED_FS, datasetConfig)

        if features is None:
            num_errors += 1
            continue

        # Add metadata to features
        segment_metadata = {col: row[col]
                            for col in METADATA_COLUMNS if col in row}
        feature_rows = _append_metadata_to_features(features, segment_metadata)

        # Append to list
        ppg_features.extend(feature_rows)

        num_processed += 1
        if num_processed % 10 == 0:
            print(f"Processed {num_processed} segments...")

    print(f"\nTotal segments processed: {num_processed}")
    print(f"Total errors: {num_errors}")

    # Create dataframe
    df_ppg = pd.DataFrame(ppg_features) if ppg_features else pd.DataFrame()

    print(f"PPG dataframe shape: {df_ppg.shape}")

    return df_ppg


def obtain_ufsc_features_and_save():
    """
    Obtain UFSC features and save them.
    """
    # Create a DatasetConfig instance

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

    # Load segments or windows information
    segmentManager = SegmentManager(
        datasetConfig, SEGMENTS_FILE_FOR_COMPARISON)

    # Extract features for each modality
    df_ppg = build_ppg_dataframe(
        segmentManager, datasetConfig)

    # Save dataframes
    save_modality_dataframes(pd.DataFrame(), df_ppg,
                             pd.DataFrame(), datasetConfig)


if __name__ == "__main__":
    # obtain_ufsc_features_and_save()

    # open the CSV for comparison and read it into a DataFrame
    df_comparison = pd.read_csv(FEATURES_FILE_FOR_COMPARISON)
    df_ppg_saved = pd.read_csv(r"..\output_ieb1\ml\features_ppg_segments.csv")

    # find the df with largest number of rows and discard the excess
    # rows in the other df to make them have the same number of rows
    if len(df_comparison) > len(df_ppg_saved):
        df_comparison = df_comparison.iloc[:len(df_ppg_saved)]
    elif len(df_ppg_saved) > len(df_comparison):
        df_ppg_saved = df_ppg_saved.iloc[:len(df_comparison)]

    df_comparison = df_comparison.iloc[:9]
    df_ppg_saved = df_ppg_saved.iloc[:9]

    # most columns if df_ppg_saved have a name with a prefix "ppg_u5_" or "ppg_i6"
    # and we need to rename these columns by removing these prefixes
    # Explicit method for the two known prefixes
    rename_dict = {}
    for column in df_ppg_saved.columns:
        if column.startswith("ppg_u5_"):
            new_name = column[7:]  # Remove "ppg_u5_" (7 characters)
        elif column.startswith("ppg_i6_"):
            new_name = column[7:]  # Remove "ppg_i6_" (7 characters)
        else:
            new_name = column
        rename_dict[column] = new_name

    df_ppg_saved.rename(columns=rename_dict, inplace=True)

    report = compare_dataframes_by_column_name(
        df_ppg_saved, df_comparison, rtol=1e-6, atol=1e-9)

    print("Columns only in df_ppg:", report["columns_only_in_df1"])
    print("Columns only in df_comparison:", report["columns_only_in_df2"])
    print("Matched columns:", report["matched_columns"])

    # order the matched columns by mean_abs_error in descending order
    matched_columns_info = report["comparison_per_column"]
    matched_columns_info = dict(sorted(matched_columns_info.items(
    ), key=lambda x: x[1]['mean_abs_error'], reverse=True))
    print("\nMatched columns ordered by mean_abs_error (descending):")
    for col, info in matched_columns_info.items():
        if info["status"] == "compared":
            print(
                f"{col}: mean_abs_error={info['mean_abs_error']}, max_abs_error={info['max_abs_error']}, num_equal={info['num_equal']}, num_different={info['num_different']}")
        else:
            print(f"{col}: {info['reason']}")
    # print(report)

    pd.DataFrame(report["comparison_per_column"]).T

    print("\nComparison of feature extraction complete!")
