'''
Fuse multimodality features into a single machine learning dataset.

Considering that the number of features may differ among the modalities
and that some features may be missing for some files, use the following
heuristic strategy to concatenate features creating a multimodal row:
1) Fuse:
    1.1) For each group of participant_id, session_id, time_date, count how
    many features are available for each modality. The largest number Nmax will
    dictate how many rows will be created for that group. For example, if for
    a given group there are Nmax=6 features for PPG, 3 for ECG and 1 for bioimpedance,
    then Nmax=6 rows will be created for that group, with the available features for
    each modality.
    1.2) Create each row such that all modalities contribute with Nmax features,
    by repeating features as needed. For example, if for a given group there are
    Nmax=6 features for PPG, 3 for ECG and 1 for bioimpedance, then the 3 ECG features
    will be repeated twice and the 1 bioimpedance feature will be repeated 6 times to
    create 6 rows for that group.
    1.3) Count and warn (indicate via stdout) the case in which a group has no features
    for a modality, and skip that group, since it will not contribute to the machine
    learning model.
2) Save a single DataFrame csv with fused features and all available metadata
   (participant_id, session_id, time_date, etc.) for all files.
'''

import argparse
import os
import shutil
import pandas as pd
from typing import Dict, List, Tuple
from datasets_util.naming_conventions import DatasetConfig
from features.clean_features_dataframe import old_remove_missing_data
from datasets_util.naming_conventions import MANDATORY_METADATA_COLUMNS
from datasets_util.naming_conventions import LOGIDENTIFIER

# OUTPUT_MULTIMODAL_FEATURES_FILE = "multimodal_features_with_metadata.csv"
# OUTPUT_MULTIMODAL_WITHOUT_METADATA_FILE = "multimodal_features_ml.csv"
'''
# Columns to exclude specifically from the ML-ready CSV.
ML_EXCLUDED_COLUMNS = [
    "ecg_sqi",
    "quality_indicator",
]

# Metadata columns to preserve in output
SOME_METADATA_COLUMNS = [
    "participant_id",
    "session_id",
    "datetime",
    "GLC"
]
'''


def get_feature_columns_for_modality(df_modality: pd.DataFrame, metadata_columns: List[str]) -> List[str]:
    """Get feature columns (excluding metadata) and metadata columns for specific modality."""
    # metadata columns has all metadata columns, including the target column (GLC)

    # Get feature columns from each modality
    feature_cols = [
        col for col in df_modality.columns if col not in metadata_columns]

    return feature_cols


def get_metadata_columns(datasetConfig: DatasetConfig) -> List[str]:
    """Get metadata columns."""
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

    return all_metadata_columns


def old_get_feature_and_metadata_columns(datasetConfig: DatasetConfig,
                                         df_ecg: pd.DataFrame,
                                         df_ppg: pd.DataFrame,
                                         df_bioimp: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Get feature columns (excluding metadata) and metadata columns."""
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

    # Get feature columns from each modality
    feature_cols_ecg = [
        col for col in df_ecg.columns if col not in all_metadata_columns]
    feature_cols_ppg = [
        col for col in df_ppg.columns if col not in all_metadata_columns]
    feature_cols_bioimp = [
        col for col in df_bioimp.columns if col not in all_metadata_columns]

    # Combine all feature columns assuming they are unique across modalities
    all_feature_columns = feature_cols_ecg + feature_cols_ppg + feature_cols_bioimp

    return all_metadata_columns, all_feature_columns


def old_clean_by_removing_nan(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    """
    Clean a modality-specific dataframe by removing rows/columns with NaN/inf values.
    Drop columns that are not relevant for ML.
    """
    if df.empty:
        return df

    print(f"\nCleaning {modality} dataframe...")
    print(f"  Original shape: {df.shape}")

    df_clean = df.copy()

    # Replace inf with NaN
    df_clean.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

    # Drop non-feature columns first
    cols_to_drop = ["has_ppg", "has_ecg", "ppg_error", "has_bioimp",
                    "segment_id", "modality", "file_id"]
    df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')

    # Drop columns with any NaN values (problematic features)
    df_clean = old_remove_missing_data(
        df_clean, drop_rows=False, drop_cols=True)

    # Drop rows with NaN in remaining columns
    df_clean.dropna(axis=0, how="any", inplace=True)

    print(f"  Cleaned shape: {df_clean.shape}")

    if df_clean.isna().any().any():
        print(f"  Warning: {modality} still contains NaN values")
    if df_clean.isin([float('inf'), float('-inf')]).any().any():
        print(f"  Warning: {modality} still contains inf values")

    return df_clean


def rename_metadata_columns_to_care_for_modality(datasetConfig: DatasetConfig, df: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Rename metadata columns to avoid conflicts."""
    if df.empty:
        return df

    metadata_in_segments = datasetConfig.get_segments_columns()

    # check if all rows of column  "modality" have the specified modality, if not, raise an error
    if "modality" in df.columns:
        unique_modalities = df["modality"].unique()
        if len(unique_modalities) > 1 or unique_modalities[0] != modality:
            raise ValueError(
                f"DataFrame contains multiple modalities: {unique_modalities}, expected only '{modality}'")

    metadata_without_modality = [
        col for col in metadata_in_segments if col != "modality"]

    if modality == "bioimp":
        modality = "bio"
    rename_map = dict()
    for col in metadata_without_modality:
        rename_map[col] = modality + "_" + col

    return df.rename(columns=rename_map)


def load_modality_dataframes(datasetConfig: DatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ppg_input_file = os.path.join(
        feature_dir, datasetConfig.get_features_file_name(modality="ppg"))
    ecg_input_file = os.path.join(
        feature_dir, datasetConfig.get_features_file_name(modality="ecg"))
    bioimp_input_file = os.path.join(
        feature_dir, datasetConfig.get_features_file_name(modality="bioimp"))

    """Load dataframes for each modality."""
    print("Loading modality dataframes...")
    print("Reading ECG features from:", ecg_input_file)
    df_ecg = pd.read_csv(ecg_input_file) if os.path.exists(
        ecg_input_file) else pd.DataFrame()
    print("Reading PPG features from:", ppg_input_file)
    df_ppg = pd.read_csv(ppg_input_file) if os.path.exists(
        ppg_input_file) else pd.DataFrame()
    print("Reading Bioimpedance features from:", bioimp_input_file)
    df_bioimp = pd.read_csv(bioimp_input_file) if os.path.exists(
        bioimp_input_file) else pd.DataFrame()

    return df_ecg, df_ppg, df_bioimp


def old_get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature columns (excluding metadata)."""
    return [col for col in df.columns if col not in MANDATORY_METADATA_COLUMNS]


def repeat_features(df: pd.DataFrame, n_times: int, feature_cols: List[str]) -> pd.DataFrame:
    """Repeat rows n_times by cycling through them."""
    if df.empty or n_times <= 0:
        return df.iloc[0:0].copy()

    n_rows = len(df)

    if n_rows == 0:
        return pd.DataFrame()

    # Repeat rows cyclically
    repeated_indices = [i % n_rows for i in range(n_times)]
    return df.iloc[repeated_indices][feature_cols].reset_index(drop=True)


def fuse_modality_features(
    modality_frames: Dict[str, pd.DataFrame],
    required_modalities: List[str],
    metadata_columns: List[str],
) -> pd.DataFrame:
    """Concatenate features from required modalities using a single strategy."""
    if all(modality_frames.get(mod, pd.DataFrame()).empty for mod in required_modalities):
        raise ValueError("All required modality dataframes are empty")

    print("\nConcatenating features from all modalities...")

    group_keys = set()
    for modality in required_modalities:
        df_modality = modality_frames.get(modality, pd.DataFrame())
        print(f"Processing {len(df_modality)} rows for modality '{modality}'")
        print(df_modality.head())
        if not df_modality.empty:
            group_keys.update(
                df_modality[MANDATORY_METADATA_COLUMNS]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )

    print(f"Found {len(group_keys)} unique groups")

    feature_cols_by_modality = {
        modality: get_feature_columns_for_modality(
            modality_frames.get(modality, pd.DataFrame()),
            metadata_columns,
        )
        for modality in required_modalities
    }

    metadata_source_modality = "ppg" if "ppg" in required_modalities else required_modalities[0]
    metadata_source_df = modality_frames.get(
        metadata_source_modality, pd.DataFrame())
    # Keep shared metadata only. Segment-related metadata and generic "modality"
    # must come from modality-specific prefixed columns (e.g., ecg_segment_id).
    segment_base_cols = {
        "segment_id",
        "file_id",
        "start_sample",
        "duration",
        "quality_indicator",
        "modality",
    }
    metadata_source_cols = [
        col
        for col in metadata_columns
        if col in metadata_source_df.columns and col not in segment_base_cols
    ]

    label_map = {
        "ecg": "ECG",
        "ppg": "PPG",
        "bioimp": "Bioimpedance",
    }

    num_skipped_groups = 0
    fused_rows = []

    for group_key in sorted(group_keys):
        participant_id, session_id, datetime, glc = group_key

        grouped_rows: Dict[str, pd.DataFrame] = {}
        missing_modalities: List[str] = []

        for modality in required_modalities:
            df_modality = modality_frames.get(modality, pd.DataFrame())
            if df_modality.empty:
                grouped_rows[modality] = pd.DataFrame()
                missing_modalities.append(label_map.get(modality, modality))
                continue

            mask = (
                (df_modality["participant_id"] == participant_id)
                & (df_modality["session_id"] == session_id)
                & (df_modality["datetime"] == datetime)
            )
            grouped_rows[modality] = df_modality[mask] if mask.any(
            ) else pd.DataFrame()
            if grouped_rows[modality].empty:
                missing_modalities.append(label_map.get(modality, modality))

        if missing_modalities:
            print(
                f"Skipping group (pid={participant_id}, sid={session_id}, dt={datetime}): "
                f"missing {', '.join(missing_modalities)}"
            )
            num_skipped_groups += 1
            continue

        nmax = max(len(grouped_rows[modality])
                   for modality in required_modalities)

        repeated_features = {
            modality: repeat_features(
                grouped_rows[modality],
                nmax,
                feature_cols_by_modality[modality],
            )
            for modality in required_modalities
        }

        metadata_repeated = repeat_features(
            grouped_rows[metadata_source_modality],
            nmax,
            metadata_source_cols,
        )

        for i in range(nmax):
            row = metadata_repeated.iloc[i].to_dict(
            ) if not metadata_repeated.empty else {}
            row.update(
                {
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "datetime": datetime,
                    "GLC": glc,
                }
            )

            for modality in required_modalities:
                if not repeated_features[modality].empty:
                    row.update(repeated_features[modality].iloc[i].to_dict())

            fused_rows.append(row)

    print(f"Total groups skipped: {num_skipped_groups}")
    print(f"Total fused rows created: {len(fused_rows)}")

    output_df = pd.DataFrame(fused_rows)
    if "modality" in output_df.columns:
        output_df = output_df.drop(columns=["modality"])
    if "relative_path" in output_df.columns:
        output_df = output_df.drop(columns=["relative_path"])
    if "ecg_sqi" in output_df.columns:
        output_df = output_df.drop(columns=["ecg_sqi"])

    # add a new column called quality_indicator with the
    # average of the quality_indicator columns of each modality, if they exist
    quality_indicator_cols = [
        col for col in output_df.columns if col.endswith("quality_indicator")]
    if quality_indicator_cols:
        output_df["quality_indicator"] = output_df[quality_indicator_cols].mean(
            axis=1)
        # if want to drop the individual quality_indicator columns:
        # output_df = output_df.drop(columns=quality_indicator_cols)

    # Keep metadata columns on the left and feature columns on the right.
    segment_suffixes = (
        "segment_id",
        "file_id",
        "start_sample",
        "duration",
        "quality_indicator",
    )
    shared_left_order = ["participant_id", "session_id", "datetime", "GLC"]

    metadata_left = []
    for col in output_df.columns:
        if col in shared_left_order:
            metadata_left.append(col)
            continue
        if col in metadata_columns and col != "modality":
            metadata_left.append(col)
            continue
        if any(col.endswith(f"_{suffix}") for suffix in segment_suffixes):
            metadata_left.append(col)

    # Preserve preferred order for shared metadata, then keep remaining metadata
    # in their original appearance order.
    shared_left_existing = [
        col for col in shared_left_order if col in metadata_left]
    other_metadata_left = [
        col for col in metadata_left if col not in shared_left_existing]
    metadata_left = shared_left_existing + other_metadata_left

    feature_right = [
        col for col in output_df.columns if col not in metadata_left]
    output_df = output_df[metadata_left + feature_right]

    return output_df


if __name__ == "__main__":
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

    feature_dir = datasetConfig.features_path

    dataset_name = datasetConfig.get_value("DATASET_NAME")

    # filter out modalities according to the datasetConfig, even if they are listed in the segmenter config
    user_selected_modalities = datasetConfig.modalities

    # There is no fusion of features for IEB-1, because it only has PPG features, so
    # just copy the PPG features file into the final multimodal features file.
    # Another situation is when the user has selected only one modality of IEB-2 or 3,
    # in which case we also just copy the features file into the final multimodal features file.
    if len(user_selected_modalities) == 1:
        input_file = os.path.join(
            feature_dir, datasetConfig.get_features_file_name(modality=user_selected_modalities[0]))
        output_file = os.path.join(
            feature_dir, datasetConfig.get_features_file_name(modality=None))
        # copy one file into another file with a different name, to be used in the next step of the pipeline
        shutil.copyfile(input_file, output_file)
        print(f"Fused features saved to multimodality file: {output_file}")
        exit(0)

    # Load and clean modality dataframes
    df_ecg, df_ppg, df_bioimp = load_modality_dataframes(datasetConfig)

    # Rename metadata columns to avoid conflicts when merging dataframes
    # by adding the modality prefix to the metadata columns, e.g.,
    # for a row of column seg_id called seg_id01 rename it to ecg_seg_id01
    df_ecg = rename_metadata_columns_to_care_for_modality(
        datasetConfig, df_ecg, "ecg")
    df_ppg = rename_metadata_columns_to_care_for_modality(
        datasetConfig, df_ppg, "ppg")
    df_bioimp = rename_metadata_columns_to_care_for_modality(
        datasetConfig, df_bioimp, "bioimp")

    metadata_columns = get_metadata_columns(datasetConfig)

    # create dictionary of modality dataframes
    modality_frames = {
        "ecg": df_ecg,
        "ppg": df_ppg,
        "bioimp": df_bioimp,
    }
    # remove modalities that are not selected by the user
    modality_frames = {
        k: v for k, v in modality_frames.items() if k in user_selected_modalities}

    # Fuse features
    print("Fusing features for", dataset_name, "dataset...")
    df_fused = fuse_modality_features(
        modality_frames=modality_frames,
        required_modalities=user_selected_modalities,
        metadata_columns=metadata_columns,
    )

    # Save fused features
    output_multimodal_file = os.path.join(
        feature_dir, datasetConfig.get_features_file_name(modality=None))
    print(
        f"Saving fused features to multimodality file: {output_multimodal_file}")
    df_fused.to_csv(output_multimodal_file, index=False)
    print(LOGIDENTIFIER +
          f"Multimodality dataframe has dimension: {df_fused.shape}")

    print("\nFusion of multimodality features complete!")
