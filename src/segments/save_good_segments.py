'''
Write file with segments of good quality
'''
import argparse

import numpy as np

from datasets_util.naming_conventions import DatasetConfig
from segments.segments_core import create_segments_dataframe

# set to True to use fixed duration windows instead of best segments
USE_FIXED_DURATION_WINDOWS = False

if USE_FIXED_DURATION_WINDOWS:
    OUTPUT_ALL_SEGMENTS_FILE = "all_fixed_duration_windows_with_quality.csv"
else:
    OUTPUT_ALL_SEGMENTS_FILE = "best_segments_per_file.csv"
print(
    f"Using input file: {OUTPUT_ALL_SEGMENTS_FILE} because USE_FIXED_DURATION_WINDOWS={USE_FIXED_DURATION_WINDOWS}")

MIN_THRESHOLD_PPG = 0.95  # quality threshold for good quality segments
MIN_THRESHOLD_ECG = 0.4  # quality threshold for good quality segments
MIN_DURATION = 10  # minimum duration in seconds for good quality segments

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
    ecg_fs = datasetConfig.get_ecg_fs()

    old_fs_dict = {
        "ecg": ecg_fs,
        "ppg": ppg_fs,
        "bioimp": np.nan  # set to NaN if not available or not applicable
    }

    # generate segments file
    dataset_segments = OUTPUT_ALL_SEGMENTS_FILE
    output_file_name = create_segments_dataframe(
        dataset_config_file, dataset_segments,
        MIN_THRESHOLD_ECG, MIN_THRESHOLD_PPG, MIN_DURATION)

    print("Assuming the thresholds:")
    print(f"minimum ECG threshold: {MIN_THRESHOLD_ECG}")
    print(f"minimum PPG threshold: {MIN_THRESHOLD_PPG}")
    print(f"minimum duration: {MIN_DURATION}s")
