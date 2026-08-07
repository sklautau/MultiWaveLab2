'''
Create a participant-level SQI summary dataframe to support train/test split design.

The script reads:
1) Dataset config JSON (for dataset metadata and output paths), and
2) Segments CSV (with quality_indicator values already computed).

It outputs one row per participant_id with:
participant_id, GLC_mean, GLC_std,
sqi_ppg_mean, sqi_ppg_std,
sqi_ecg_mean, sqi_ecg_std,
sqi_biompedance_mean, sqi_bioimpedance_std,
sqi_combined_score, sqi_rank, num_valid_segments.

Missing modalities are handled gracefully (output NaN for missing columns).
'''

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from segments.segments_core import Segmenter
from datasets_util.to_latex import to_latex_table


try:
    from segments.segments_core import SegmentManager
    from datasets_util.naming_conventions import DatasetConfig
except ModuleNotFoundError:
    # Fallback for direct execution when project root is not yet in sys.path.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from segments.segments_core import SegmentManager
    from datasets_util.naming_conventions import DatasetConfig


# Ensure project root is in path for imports when running this script directly.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _normalize_modality(modality: str) -> str:
    value = str(modality).strip().lower()
    '''
    old code:
    if value.startswith("bioimp"):
        return "bioimpedance"
    if value == "ppg":
        return "ppg"
    if value == "ecg":
        return "ecg"
    '''
    return value


def _resolve_glc_column(dataset_info: pd.DataFrame) -> str:
    candidates = ["GLC", "glc", "Glc"]
    for candidate in candidates:
        if candidate in dataset_info.columns:
            return candidate

    # Case-insensitive fallback.
    lower_to_original = {str(col).lower(): col for col in dataset_info.columns}
    if "glc" in lower_to_original:
        return str(lower_to_original["glc"])

    raise ValueError(
        "Could not find glucose column in dataset info. Expected a column named GLC/glc."
    )


def _compute_glc_stats_per_participant(dataset_info: pd.DataFrame, glc_column: str) -> pd.DataFrame:
    required_cols = {"participant_id", glc_column}
    missing = sorted(required_cols - set(dataset_info.columns))
    if missing:
        raise ValueError(
            f"Dataset info is missing required columns for GLC stats: {missing}")

    optional_cols = [col for col in ["session_id",
                                     "datetime"] if col in dataset_info.columns]
    glc_df = dataset_info[["participant_id",
                           glc_column] + optional_cols].copy()
    glc_df["participant_id"] = glc_df["participant_id"].astype(str)
    glc_df[glc_column] = pd.to_numeric(glc_df[glc_column], errors="coerce")
    glc_df = glc_df.dropna(subset=[glc_column])

    dedup_cols = [
        col for col in ["participant_id", "session_id", "datetime", glc_column]
        if col in glc_df.columns
    ]
    if dedup_cols:
        glc_df = glc_df.drop_duplicates(subset=dedup_cols)

    glc_stats = (
        glc_df.groupby("participant_id", as_index=False)[glc_column]
        .agg(
            GLC_mean="mean",
            GLC_std=lambda s: s.std(ddof=0),
        )
    )
    return glc_stats


def _compute_sqi_stats_per_participant(
    dataset_info: pd.DataFrame,
    segments_df: pd.DataFrame,
) -> pd.DataFrame:
    required_dataset_cols = {"file_id", "participant_id"}
    missing_dataset_cols = sorted(
        required_dataset_cols - set(dataset_info.columns))
    if missing_dataset_cols:
        raise ValueError(
            f"Dataset info is missing required columns: {missing_dataset_cols}"
        )

    required_segment_cols = {"file_id", "modality", "quality_indicator"}
    missing_segment_cols = sorted(
        required_segment_cols - set(segments_df.columns))
    if missing_segment_cols:
        raise ValueError(
            f"Segments file is missing required columns: {missing_segment_cols}"
        )

    map_df = dataset_info[["file_id", "participant_id"]
                          ].drop_duplicates(subset=["file_id"]).copy()
    map_df["file_id"] = map_df["file_id"].astype(str)
    map_df["participant_id"] = map_df["participant_id"].astype(str)

    sqi_df = segments_df[["file_id", "modality", "quality_indicator"]].copy()
    sqi_df["file_id"] = sqi_df["file_id"].astype(str)
    sqi_df["modality"] = sqi_df["modality"].apply(_normalize_modality)
    sqi_df["quality_indicator"] = pd.to_numeric(
        sqi_df["quality_indicator"], errors="coerce")
    sqi_df = sqi_df.dropna(subset=["quality_indicator"])

    merged = sqi_df.merge(map_df, on="file_id", how="left")
    missing_pid = int(merged["participant_id"].isna().sum())
    if missing_pid > 0:
        raise ValueError(
            f"Could not map participant_id for {missing_pid} segments rows. "
            "Check file_id consistency between segments and dataset metadata."
        )

    count_df = (
        merged.groupby("participant_id", as_index=False)
        .size()
        .rename(columns={"size": "num_valid_segments"})
    )
    count_df["num_valid_segments"] = count_df["num_valid_segments"].astype(
        "Int64")

    stats_long = (
        merged.groupby(["participant_id", "modality"],
                       as_index=False)["quality_indicator"]
        .agg(
            sqi_mean="mean",
            sqi_std=lambda s: s.std(ddof=0),
        )
    )

    mean_wide = (
        stats_long.pivot(index="participant_id",
                         columns="modality", values="sqi_mean")
        .rename(columns={
            "ppg": "sqi_ppg_mean",
            "ecg": "sqi_ecg_mean",
            "bioimp": "sqi_biompedance_mean",
        })
    )

    std_wide = (
        stats_long.pivot(index="participant_id",
                         columns="modality", values="sqi_std")
        .rename(columns={
            "ppg": "sqi_ppg_std",
            "ecg": "sqi_ecg_std",
            "bioimp": "sqi_bioimpedance_std",
        })
    )

    sqi_stats = mean_wide.join(std_wide, how="outer").reset_index()
    sqi_stats = sqi_stats.merge(count_df, on="participant_id", how="left")
    return sqi_stats


def calculate_and_save_sqis_per_participant_id(
    dataset_config_file: str,
    segments_file: str,
    output_csv: str | None = None,
) -> pd.DataFrame:
    dataset_config = DatasetConfig(dataset_config_file)
    dataset_info = dataset_config.get_dataset_info_dataframe().copy()

    segment_manager = SegmentManager(dataset_config, segments_file)
    segments_df = segment_manager.get_segments_dataframe().copy()

    glc_column = _resolve_glc_column(dataset_info)
    glc_stats = _compute_glc_stats_per_participant(dataset_info, glc_column)
    sqi_stats = _compute_sqi_stats_per_participant(dataset_info, segments_df)

    result = glc_stats.merge(sqi_stats, on="participant_id", how="outer")

    sqi_mean_cols = ["sqi_ppg_mean", "sqi_ecg_mean", "sqi_biompedance_mean"]
    for col in sqi_mean_cols:
        if col not in result.columns:
            result[col] = pd.NA

    # Combined SQI score computed across available modalities.
    result["sqi_combined_score"] = result[sqi_mean_cols].mean(
        axis=1, skipna=True)
    # Higher combined SQI receives better rank (1 is best).
    result["sqi_rank"] = result["sqi_combined_score"].rank(
        method="dense", ascending=False
    ).astype("Int64")

    expected_columns = [
        "participant_id",
        "GLC_mean",
        "GLC_std",
        "sqi_ppg_mean",
        "sqi_ppg_std",
        "sqi_ecg_mean",
        "sqi_ecg_std",
        "sqi_biompedance_mean",
        "sqi_bioimpedance_std",
        "sqi_combined_score",
        "sqi_rank",
        "num_valid_segments",
    ]
    for col in expected_columns:
        if col not in result.columns:
            result[col] = pd.NA

    result = result[expected_columns].sort_values(
        "participant_id").reset_index(drop=True)

    if output_csv is None:
        output_csv = os.path.join(
            dataset_config.get_dataset_machine_learning_path(),
            "sqi_per_participant_for_split.csv",
        )

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Saved SQI split dataframe: {output_csv}")
    print(f"Rows (participants): {len(result)}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create participant-level SQI/GLC summary dataframe for train/test split design."
        )
    )
    parser.add_argument(
        "json_file",
        metavar="JSON_FILE",
        help="Path to the input JSON file."
    )

    args = parser.parse_args()
    # Create a DatasetConfig instance
    dataset_config_file = args.json_file
    dataset_config = DatasetConfig(dataset_config_file)

    # find the used segmenter, which gives the output file name for the segments
    segmenter_file = dataset_config.get_segmenter_file_name()
    segmenter = Segmenter(segmenter_file, dataset_config_file)
    segments_file_name = dataset_config.get_segments_file_name()

    # get the filename from complete Path
    prefix_for_results_file = dataset_config.get_folder_and_prefix_from_json_config_file_name()
    prefix_for_results_file = os.path.basename(prefix_for_results_file)
    # add the dataset name to the prefix for the results file
    prefix_for_results_file = dataset_config.get_value(
        "DATASET_NAME") + "_" + prefix_for_results_file
    file_name = os.path.join(
        dataset_config.get_dataset_machine_learning_path(),  prefix_for_results_file + "_sqi_statistics_per_participant.csv")
    # fix \ and / in the file name
    file_name = os.path.normpath(file_name)

    calculate_and_save_sqis_per_participant_id(
        dataset_config_file=dataset_config_file,
        segments_file=segments_file_name,
        output_csv=file_name,
    )

    # find number of participants in the dataset that have num_valid_segments > 0
    df = pd.read_csv(file_name)
    num_participants = df[df["num_valid_segments"] > 0].shape[0]

    # to save as latex
    # sort by sqi_rank in decreasing order (best rank first)
    df = df.sort_values(by="sqi_rank", ascending=True)

    # remove sqi_rank column from the dataframe
    df = df.drop(columns=["sqi_rank"])

    # rename columns
    df = df.rename(columns={
        "participant_id": "ID",
        "GLC_mean": "GLC_m",
        "GLC_std": "GLC_s",
        "sqi_ppg_mean": "PPG_m",
        "sqi_ppg_std": "PPG_s",
        "sqi_ecg_mean": "ECG_m",
        "sqi_ecg_std": "ECG_s",
        "sqi_biompedance_mean": "Bio_m",
        "sqi_bioimpedance_std": "Bio_s",
        "sqi_combined_score": "SQI",
        # "sqi_rank": "Rank",
        "num_valid_segments": "Segs"
    })

    caption = "Participant-level statistics for segment file " + segments_file_name + \
        " and dataset " + \
        str(dataset_config.get_value("DATASET_NAME")).upper() + "."
    latex = to_latex_table(df, file_name, caption=caption)
    print(latex)

    print(f"Number of participants with valid segments: {num_participants}")
