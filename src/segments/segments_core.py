'''
Utilities to read and write Dataframe files with segments.
The segments are defined as contiguous time intervals of good quality data, based on a quality indicator (e.g., signal quality index) and a minimum duration threshold.
The output is a CSV file with the following columns:

segment_id,file_id,modality,start_sample,duration,quality_indicator
seg_id0,file_id2,ppg,0,80840,0.9932449460029602
seg_id1,file_id2,ppg,81183,14800,0.9863582849502563
...

where segment_id is a unique identifier for each segment across all files in the dataset, file_id is the identifier of the file from which the segment was extracted, modality is the type of signal (e.g., ppg, ecg, bioimp), start_sample is the index of the first sample of
the segment (inclusive), duration is the length of the segment in samples, and
quality_indicator is the median or average signal quality index for the segment
in the given modality.
'''

import json
from pathlib import Path
from typing import Any, Dict, Union, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import neurokit2 as nk
import os
from pathlib import PurePosixPath

from datasets_util.naming_conventions import DatasetConfig
from datasets_util.waveform_files import read_sigmf_file

# SegmentRow = Dict[str, Any]
# SegmentRows = List[SegmentRow]  # List[Dict[str, Any]]

PREFIX_SEGMENT_ID = "seg_id"
DEBUGGING = False  # True to enable debug prints


def is_it_a_filename(filename: str) -> bool:
    '''
    is_it_a_filename("myfile.txt")                 # True
    is_it_a_filename("./myfile.txt")               # True
    is_it_a_filename(".\\myfile.txt")              # True
    is_it_a_filename("../subfolder1/myfile.txt")   # False
    is_it_a_filename("c:/fold/subfolder1/myfile.txt")  # False
    is_it_a_filename("/tmp/myfile.txt")            # False
    is_current_directory("subfolder/myfile.txt")       # False
    '''
    filename = filename.replace("\\", "/")
    parent = PurePosixPath(filename).parent
    return str(parent) == "."


class Segmenter:
    '''
        Initialize from either:
        - a dictionary
        - a JSON file path (str or Path)
    '''

    def __init__(self, segmenter: Union[str, Path], dataset_config_file: str):
        # now initialize the segments dataframe based on the input type
        if isinstance(segmenter, (str, Path)):
            segmenter = Path(segmenter).as_posix()
            # convert to forward slashes for cross-platform compatibility
            segmenter = segmenter.replace("\\", "/")
            config_path = Path(segmenter)
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Config file not found: {segmenter}")
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            raise TypeError(
                "Segmenter must be either a dict or a path to a JSON file"
            )
        # Store raw dict for later retrieval or saving
        #     What defines a segmenter?
        # mandatory, so use indexing, not get():
        # self.sqi_input_waveform_id = config_dict["sqi_input_waveform_id"]

        # remove the extension of file if it has one
        self.sqi_input_waveform_id = Path(dataset_config_file).stem

        # get the modalities from DatasetConfig, not from the segmenter config file
        dataset_config = DatasetConfig(dataset_config_file)
        self.modalities = dataset_config.modalities

        self.category = config_dict["category"]
        self.type = config_dict["type"]

        # make sure self.modalities is a list of strings
        if not isinstance(self.modalities, list):
            # cast to a list of strings
            self.modalities = [str(self.modalities)]

        if self.category not in ["all", "selected"]:
            raise ValueError(
                f"category must be either 'all' or 'selected', got: {self.category}")
        if self.type not in ["generic", "window", "pulse", "wave"]:
            raise ValueError(
                f"type must be one of 'generic', 'window', 'pulse', or 'wave', got: {self.type}")
        # check whether modalities is ["ppg","ecg","bioimp"] or a subset of those
        valid_modalities = ["ppg", "ecg", "bioimp"]
        if not all(mod in valid_modalities for mod in self.modalities):
            raise ValueError(
                f"modalities must be a list containing any of {valid_modalities}, got: {self.modalities}")

        # optional
        self.wave_min_duration = config_dict.get("wave_min_duration", -1)
        self.mean_or_median = config_dict.get("mean_or_median", "median")
        # check that mean_or_median is either "mean" or "median"
        if self.mean_or_median not in ["mean", "median"]:
            raise ValueError(
                f"mean_or_median must be either 'mean' or 'median', got: {self.mean_or_median}")

        if self.category == "selected":
            # all thresholds in modalities must be specified for generic segmenter
            for mod in self.modalities:
                threshold_key = f"{mod}_sqi_threshold"
                if threshold_key not in config_dict or config_dict[threshold_key] < 0:
                    raise ValueError(
                        f"{threshold_key} must be specified for generic segmenter")
                if mod == "ppg":
                    self.ppg_sqi_threshold = config_dict[threshold_key]
                elif mod == "ecg":
                    self.ecg_sqi_threshold = config_dict[threshold_key]
                elif mod == "bioimp":
                    self.bioimp_sqi_threshold = config_dict[threshold_key]

        if self.type == "window":
            self.window_size_seconds = config_dict.get("window_size_seconds")
            # default is no window overlapping:
            self.window_shift_seconds = config_dict.get(
                "window_shift_seconds", self.window_size_seconds)
            if self.window_size_seconds is None or self.window_size_seconds <= 0:
                raise ValueError(
                    f"window_size_seconds must be specified and positive for window segmenter, got: {self.window_size_seconds}")
            if self.window_shift_seconds <= 0:
                raise ValueError(
                    f"window_shift_seconds must be positive for window segmenter, got: {self.window_shift_seconds}")
        elif self.type == "generic":
            if self.wave_min_duration < 0:
                raise ValueError(
                    "wave_min_duration must be specified to select segments")

        if self.category == "all":
            default_name = self.type + "_" + self.category + "_" + \
                str(self.sqi_input_waveform_id) + "_segments.csv"
        else:
            default_name = self.type + "_" + self.category + "_" + \
                str(self.sqi_input_waveform_id) + "_" + \
                str(self.ppg_sqi_threshold) + "_segments.csv"
        self.output_file_name = config_dict.get(
            "output_file_name", default_name)

        self.config_dictionary = config_dict


class SegmentManager:
    '''
    Class to extract segments of good quality data from the dataset.
    For instance, it can be based on a quality indicator and a minimum duration threshold.
    '''

    def __init__(self, datasetConfig: DatasetConfig,
                 segments: Union[pd.DataFrame, str, Path]):
        self._datasetConfig = datasetConfig
        # now initialize the segments dataframe based on the input type
        if isinstance(segments, pd.DataFrame):
            self._segments_dataframe = segments
        elif isinstance(segments, (str, Path)):
            self._segments_dataframe = self.load_segments(str(segments))
        else:
            raise ValueError(
                "Invalid segments type. Expected DataFrame, str, or Path.")

    def get_segments_dataframe(self) -> pd.DataFrame:
        """Return the segments info as a pandas DataFrame."""
        # make a deep copy to prevent external modifications
        return self._segments_dataframe.copy()

    def get_segments_of_file_id(self, file_id: str) -> pd.DataFrame:
        return self._segments_dataframe.loc[self._segments_dataframe['file_id'] == file_id]

    def load_segments(self, segments_file: str) -> pd.DataFrame:

        print(f"Loading segments from file: {segments_file}")

        # Load segments from file
        segments_path = Path(segments_file)
        segments = None
        if segments_path.exists():
            segments = pd.read_csv(segments_file)
        else:
            complete_path = Path(
                self._datasetConfig.get_dataset_segments_path()) / segments_file
            if complete_path.exists():
                segments = pd.read_csv(complete_path)
            else:
                raise FileNotFoundError(
                    f"Segments file not found as {segments_file} nor {complete_path}")
        return segments

    def best_segment(self, file_id: str, modality: str) -> tuple[Dict[str, Any], float]:
        print(f"Finding best segment for file_id: {file_id}...")
        df = self.get_segments_of_file_id(file_id)
        return best_segment(df, file_id, modality)


@staticmethod
def best_segment(all_segments_dataframe, file_id: str, modality: str) -> tuple[Dict[str, Any], float]:
    print(
        f"best_segment static method: Finding best segment for file_id {file_id}...")
    segments_dataframe = all_segments_dataframe.loc[all_segments_dataframe['file_id'] == file_id]
    # filter by modality
    segments_dataframe = segments_dataframe[segments_dataframe["modality"] == modality]
    if segments_dataframe.empty:
        raise ValueError(
            f"No segments found for file_id {file_id} and modality {modality}")
    best_row = segments_dataframe.loc[segments_dataframe['quality_indicator'].idxmax(
    )]
    largest_quality = best_row['quality_indicator']
    # print(
    #    f"AAA Best segment for file_id {file_id}: segment_id {best_row['segment_id']}, modality {best_row['modality']}, quality_indicator {best_row['quality_indicator']}")
    return best_row.to_dict(), largest_quality


def format_segment_row(
    segment_index: int,
    file_id: str,
    start: int,
    duration: int,
    modality: str,
    quality_indicator: float
) -> Dict[str, Any]:
    dictionary = {
        "segment_id": PREFIX_SEGMENT_ID + str(segment_index),
        "file_id": file_id,
        "modality": modality,
        "start_sample": start,  # onset in NeuroKit2 convention (inclusive)
        "duration": duration,  # duration in samples
        "quality_indicator": quality_indicator
    }
    return dictionary


def extract_bioimpedance_segments(
    quality_per_sample: np.ndarray,
    file_id: str,
    current_index: int = 0,
) -> Dict[str, Any]:
    modality = "bioimp"
    # for bioimpedance, we only have one segment for the whole file, so we can just return a single row with the quality indicator for the whole file
    start = 0
    duration = len(quality_per_sample)
    # TODO read it from a segmenter
    use_median_over_mean = True
    if use_median_over_mean:
        quality_indicator = float(np.median(quality_per_sample))
    else:
        quality_indicator = float(np.mean(quality_per_sample))
    # in general, it stores all segments for this file, but in the bioimpedance case, we only have one segment for the whole file
    this_file_segments_data = format_segment_row(
        current_index, file_id, start, duration, modality, quality_indicator)
    return this_file_segments_data


def segments_with_all_samples_above_threshold(
    quality_indicator: np.ndarray,
    file_id: str,
    fs: int,
    modality: str,
    current_index: int = 0,
    min_quality: float = 0.5,
    min_len_seconds: float = 5.0,
    use_median_over_mean: bool = True
) -> list[Dict[str, Any]]:
    quality_indicator = np.asarray(quality_indicator)

    # Convert threshold from seconds to samples
    min_len = int(min_len_seconds * fs)

    # boolean mask of valid quality samples
    good = quality_indicator >= min_quality

    this_file_segments_data = list()  # all segments for this file
    valid_segment_index = current_index
    start = None

    for i in range(len(good)):
        if good[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i
                duration = end - start

                if use_median_over_mean:
                    quality_indicator_value = float(
                        np.median(quality_indicator[start:end]))
                else:
                    quality_indicator_value = float(
                        np.mean(quality_indicator[start:end]))

                if duration >= min_len:
                    new_row = format_segment_row(
                        valid_segment_index, file_id, start, duration, modality, quality_indicator_value)
                    # use append when adding a single element (not extend)
                    this_file_segments_data.append(new_row)
                    valid_segment_index += 1
                start = None

    # If segment continues until the last sample
    if start is not None:
        end = len(good)
        duration = end - start
        if duration >= min_len:

            if use_median_over_mean:
                quality_indicator_value = float(
                    np.median(quality_indicator[start:end]))
            else:
                quality_indicator_value = float(
                    np.mean(quality_indicator[start:end]))

            new_row = format_segment_row(
                valid_segment_index, file_id, start, duration, modality, quality_indicator_value)
            # append new_row to this_file_segments_data
            # use append when adding a single element (not extend)
            this_file_segments_data.append(new_row)
            valid_segment_index += 1

    # return a list of dictionaries with all segments for this file

    return this_file_segments_data


def number_of_windows(segment_duration_samples: int, window_size_samples: int, window_shift_samples: int) -> int:
    '''
    Calculate the number of windows of size L seconds with shift S seconds that can be extracted from a segment of duration D seconds.
    The formula is: M = floor((N - L) / S) + 1
    See: https://ai6g.org/books/dsp/BlockorWindowProcessing.html#-the-top-representation-shows-nonoverlapping-windows-of-l-samples-with-both-nonwindowed-indexing-xn-and-windowed-indexing-xkm-the-bottom-representation-shows-overlapping-windows-with-l-and-shift-s-sample-using-nonwindowed-indexing
    '''
    N = int(segment_duration_samples)  # segment duration in samples
    L = int(window_size_samples)  # window size in samples
    S = int(window_shift_samples)  # window shift in samples
    M = np.floor((N - L) / S).astype(int) + 1  # number of windows per segment
    return M


@staticmethod
def windows_with_all_samples_above_threshold(sqi_signal: np.ndarray,
                                             segment_row: dict[str, Any],
                                             window_size_samples: int,
                                             window_shift_samples: int,
                                             segment_counter: int,
                                             use_median_over_mean: bool = True) -> list[Dict[str, Any]]:
    '''
    Split one segment row into fixed-duration windows with given shift.
    Returns a new dataframe with each row representing a special
    segment with fixed-duration, which is called "window".
    The output dataframe keeps the same columns as input.
    '''
    segment_start_sample = int(segment_row["start_sample"])
    segment_duration_samples = int(segment_row["duration"])

    # find number of windows that can be extracted from this segment
    num_windows = number_of_windows(
        segment_duration_samples, window_size_samples, window_shift_samples)

    all_windows_rows = []

    # Check if segment is too short for even one window
    if num_windows < 1 or segment_duration_samples < window_size_samples:
        print(f"Warning: Segment {segment_row['segment_id']} (file_id={segment_row['file_id']}, "
              f"modality={segment_row['modality']}) is too short for a window. "
              f"Duration: {segment_duration_samples:.2f} samples, Window size: {window_size_samples:.2f} samples")
        return list()  # return empty list

    # Create one row per window
    for window_idx in range(num_windows):
        window_start_sample = segment_start_sample + window_idx * window_shift_samples
        if use_median_over_mean:
            quality_indicator = float(np.median(
                sqi_signal[window_start_sample:window_start_sample + window_size_samples]))
        else:
            quality_indicator = float(np.mean(
                sqi_signal[window_start_sample:window_start_sample + window_size_samples]))

        # print(f"Window {window_idx}: segment_counter={segment_counter}")
        window_row = format_segment_row(
            segment_counter + window_idx,
            segment_row["file_id"],
            window_start_sample,
            window_size_samples,
            segment_row["modality"],
            quality_indicator
        )

        # use append when adding a single element (not extend)
        all_windows_rows.append(window_row)

    # windows_dataframe = pd.DataFrame(windows_rows)
    if DEBUGGING:
        print(
            f"Created {len(all_windows_rows)} windows from segment {segment_row['segment_id']} of {segment_row['file_id']} ({segment_row['modality']})")
    return all_windows_rows


if __name__ == "__main__":
    # input_file = "../output_ieb1/segments/segmenters/segmenter1.json"
    input_file = "../input_ieb1/segments/segmenters/segmenter_window.json"
    segmenter = Segmenter(input_file, "some_waveform_id")
