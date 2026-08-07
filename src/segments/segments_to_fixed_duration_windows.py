'''
Slice segments into fixed-duration windows (e.g., 5 seconds)
'''
import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import Dict, Any, Tuple

from datasets_util.naming_conventions import DatasetConfig
from datasets_util.waveform_files import load_signal
from segments.segments_core import SegmentManager, windows_with_all_samples_above_threshold
from signal_processing.ecg import extract_ecg_features
from signal_processing.ppg import extract_ppg_features
from signal_processing.bioimpedance import extract_bioimp_features

INPUT_FILE_NAME = "all_segments_with_quality.csv"
WINDOW_SIZE_SECONDS = 10.0
WINDOW_SHIFT_SECONDS = 10.0

if __name__ == "__main__":
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

    # Load segments information
    dataset_segments = INPUT_FILE_NAME
    segments_path = os.path.join(datasetConfig.segments_path, dataset_segments)
    segmentManager = SegmentManager(datasetConfig, segments_path)

    segments_df = segmentManager.get_segments_dataframe()

    # Here we slice the segments into windows of fixed
    # duration or use the original segments as they are
    # Convert segments to fixed-duration windows

    # define output windows dataframe with the same columns as segments_df plus a new column "window_id"
    window_columns = segments_df.columns.tolist() + ["window_id"]
    windows_df = pd.DataFrame(columns=window_columns)

    print("Slicing segments into fixed-duration windows...")
    next_window_id = 0
    windows_per_segment = []
    for _, segment_row in segments_df.iterrows():
        modality = segment_row["modality"]
        fs = fs_dict.get(modality, np.nan)
        if modality == "ecg" or modality == "ppg":
            this_segment_windows = windows_with_all_samples_above_threshold(
                segment_row,
                window_size_seconds=WINDOW_SIZE_SECONDS,
                window_shift_seconds=WINDOW_SHIFT_SECONDS,
                fs=fs,
                window_id_start=next_window_id
            )
        elif modality.startswith("bioimp"):
            # For bioimpedance, we have a single window per segment, so we just copy the segment row and add a window_id
            this_segment_windows = segment_row.to_frame().T.copy()
            this_segment_windows["window_id"] = f"win_id{next_window_id}"
        else:
            # Unsupported modalities contribute no windows but keep type consistent for concat
            raise Exception(f"Unsupported modality: {modality}")

        windows_per_segment.append(this_segment_windows)
        next_window_id += len(this_segment_windows)

        windows_df = pd.concat(windows_per_segment, ignore_index=True)

    # Save the new windows dataframe to a CSV file
    output_windows_csv = os.path.join(
        datasetConfig.segments_path, "fixed_duration_windows.csv")
    windows_df.to_csv(output_windows_csv, index=False)
    print(f"Fixed-duration windows saved to: {output_windows_csv}")
