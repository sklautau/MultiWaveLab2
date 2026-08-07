'''
Create files with segments.
'''
import numpy as np
from pathlib import Path
from typing import Any, Dict, Union, List, Optional
import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import neurokit2 as nk
import os

from datasets_util.naming_conventions import LOGIDENTIFIER, DatasetConfig
from segments.segments_core import Segmenter, windows_with_all_samples_above_threshold, segments_with_all_samples_above_threshold, extract_bioimpedance_segments
from datasets_util.waveform_files import read_sigmf_file


from datasets_util.naming_conventions import DatasetConfig
from run_scripts.segments_statistics import _aggregate_total_seconds_per_participant
# from datasets_util.segments import create_segments_dataframe


def _get_sqi_waveform_id_for_modality(datasetConfig: DatasetConfig, segmenter: Segmenter, modality: str) -> str:
    """Resolve SQI waveform id per modality, preferring dataset JSON keys and falling back to segmenter config."""
    fallback = segmenter.sqi_input_waveform_id

    if modality == "ppg":
        return datasetConfig.get_value("PPG_SQI_OUTPUT_WAVEFORM", fallback)
    if modality == "ecg":
        return datasetConfig.get_value("ECG_SQI_OUTPUT_WAVEFORM", fallback)
    if modality.startswith("bioimp"):
        return datasetConfig.get_value("BIOIMP_SQI_OUTPUT_WAVEFORM", fallback)

    return fallback


def segment_all_modalities(datasetConfig: DatasetConfig) -> pd.DataFrame:
    '''
    Segment all modalities listed in datasetConfig, even if there is redundancy
    and the segmenter also lists them. But using the datasetConfig config is preferred,
    since it allows to filter out modalities that we do not want to be part
    of the evaluation.
    '''
    # read the complete dataset info dataframe
    dataset_info_df = datasetConfig.get_dataset_info_dataframe()
    print(f"Total files in dataset: {len(dataset_info_df)}")

    segmenter_file_name = datasetConfig.get_segmenter_file_name()
    segmenter = Segmenter(
        segmenter_file_name, dataset_config_file)
    print(
        f"Successfully loaded segmenter configuration from {segmenter_file_name}")

    # filter out modalities according to the datasetConfig, even if they are listed in the segmenter config
    user_selected_modalities = datasetConfig.modalities

    print("User selected modalities:", user_selected_modalities)
    dataset_info_df = dataset_info_df[dataset_info_df["modality"].isin(
        user_selected_modalities)]
    print("This simulation will process the following modalities:",
          user_selected_modalities)
    print(
        f"Total files with the mentioned modalities to be used in this simulation: {len(dataset_info_df)}")

    if segmenter.type == "wave":
        min_ppg_threshold = 0
        min_ecg_threshold = 0
    else:
        if "ecg" in user_selected_modalities:
            min_ecg_threshold = segmenter.ecg_sqi_threshold
        else:
            min_ecg_threshold = np.nan
        if "ppg" in user_selected_modalities:
            min_ppg_threshold = segmenter.ppg_sqi_threshold
        else:
            min_ppg_threshold = np.nan

    use_median_over_mean = segmenter.mean_or_median

    # debug:
    # Filter only bioimpedance
    # df_bio = dataset_info_df[dataset_info_df["modality"].str.contains("bioimp")].copy()

    output_dataframe = None  # initialize output dataframe to store segments from all files
    segment_counter = 0  # global counter for all segments across all files and modalities
    # specific counters for each modality:
    good_ecg_segments = 0
    good_ppg_segments = 0
    bioimpedance_segments = 0

    # list to hold all segments across all modalities and files
    all_segments_for_all_files = []

    for _, row in dataset_info_df.iterrows():
        # process each file in the dataset
        file_id = row["file_id"]
        modality = row["modality"]
        participant_id = row["participant_id"]

        sqi_waveform_id = _get_sqi_waveform_id_for_modality(
            datasetConfig, segmenter, modality)

        if modality.startswith("bioimp"):
            complete_path = datasetConfig.get_gen_complete_path(
                file_id, sqi_waveform_id, new_extension="csv")
        else:
            # PPG/ECG SQI waveforms are stored as SigMF
            complete_path = datasetConfig.get_gen_complete_path(
                file_id, sqi_waveform_id)

        print(
            f"Processing file {file_id} with modality {modality} of patient {participant_id} at path {complete_path}")

        # switch depending on modality
        if modality == "ppg" or modality == "ecg":
            # read the signal and metadata
            sqi_signal, metadata = read_sigmf_file(complete_path)
            fs = metadata["global"]["core:sample_rate"]

            # identify good quality segments based on the threshold
            if modality == "ppg":
                min_quality = min_ppg_threshold
                if fs != datasetConfig.get_value("PPG_FS"):
                    raise ValueError(
                        f"PPG sample rate {fs} does not match expected {datasetConfig.get_value('PPG_FS')}")
            else:  # modality == "ecg":
                min_quality = min_ecg_threshold
                if fs != datasetConfig.get_value("ECG_FS"):
                    raise ValueError(
                        f"ECG sample rate {fs} does not match expected {datasetConfig.get_value('ECG_FS')}")

            if segmenter.type == "wave" and min_quality > 0:
                raise ValueError(
                    f"For wave-based segmentation, the minimum quality threshold must be 0 for all modalities")

            # types: ["generic", "window", "pulse", "wave"]
            if segmenter.type == "generic":
                min_duration = segmenter.wave_min_duration
                this_segments_list = segments_with_all_samples_above_threshold(sqi_signal, file_id,
                                                                               fs, modality,
                                                                               current_index=segment_counter,
                                                                               min_quality=min_quality,
                                                                               min_len_seconds=min_duration
                                                                               )
                # save it to the list of all segments
                all_segments_for_all_files.extend(this_segments_list)
                num_segments_for_this_file = len(this_segments_list)
                segment_counter += num_segments_for_this_file
            elif segmenter.type == "window":
                # assume minimum duration is the same as the window size for this type of segmenter
                min_duration = segmenter.window_size_seconds
                # first we find the segments with all samples above the threshold,
                # also requiring they have the minimum duration, then we break them
                # into fixed-duration windows with given shift
                # pass 0 as the current index, since we will update the segment_counter after processing all windows for this file
                this_segments_list = segments_with_all_samples_above_threshold(sqi_signal, file_id,
                                                                               fs, modality,
                                                                               current_index=0,
                                                                               min_quality=min_quality,
                                                                               min_len_seconds=min_duration
                                                                               )
                # now convert into fixed-duration windows with given shift
                window_size_seconds = segmenter.window_size_seconds
                window_shift_seconds = segmenter.window_shift_seconds
                # Convert time to samples
                window_size_samples = int(window_size_seconds * fs)
                window_shift_samples = int(window_shift_seconds * fs)
                # expand each row of the input dataframe into multiple rows for each window, with new column "window_id"
                # loop over rows of this_segments_dataframe
                # create a list to hold the dataframes for each new window
                all_windows_for_this_segment = []
                for segment_row in this_segments_list:
                    this_windows_list = windows_with_all_samples_above_threshold(sqi_signal,
                                                                                 segment_row,
                                                                                 window_size_samples,
                                                                                 window_shift_samples,
                                                                                 segment_counter,
                                                                                 use_median_over_mean)
                    # concatenate this_windows_list to all_windows_for_this_segment
                    # do not use append, because we want to produce a flat list of windows
                    all_windows_for_this_segment.extend(this_windows_list)
                    # update the global segment counter for all files
                    segment_counter += len(this_windows_list)
                # now we have all windows for this segment, save them to the list of all segments
                # save it to the list of all segments
                all_segments_for_all_files.extend(all_windows_for_this_segment)
                num_segments_for_this_file = len(all_windows_for_this_segment)
            elif segmenter.type == "pulse":
                raise NotImplementedError(
                    "Pulse-based segmentation is not yet implemented")
            elif segmenter.type == "wave":
                min_duration = 0  # no minimum duration for wave-based segmentation
                min_quality = 0  # no minimum quality for wave-based segmentation
                this_segments_list = segments_with_all_samples_above_threshold(sqi_signal, file_id,
                                                                               fs, modality,
                                                                               current_index=segment_counter,
                                                                               min_quality=min_quality,
                                                                               min_len_seconds=min_duration
                                                                               )
                # save it to the list of all segments
                all_segments_for_all_files.extend(this_segments_list)
                num_segments_for_this_file = len(this_segments_list)
                # update the global segment counter for all files
                segment_counter += num_segments_for_this_file
            else:
                raise ValueError(
                    f"Segmenter type {segmenter.type} is not supported")

            if num_segments_for_this_file > 0:
                print(
                    f"  Found {num_segments_for_this_file} good segments in file {file_id} of participant {participant_id}")
            else:
                print(
                    f"  Did not find any good segments in file {file_id} of participant {participant_id}")

            if modality == "ppg":
                good_ppg_segments += num_segments_for_this_file
            elif modality == "ecg":
                good_ecg_segments += num_segments_for_this_file
        elif modality == "bioimp":
            # read the csv with header frequency,quality
            quality_indicator = pd.read_csv(complete_path)["quality"].values
            quality_indicator = np.asarray(quality_indicator)

            # we are not applying a threshold for bioimpedance segments

            this_dictionary = extract_bioimpedance_segments(
                quality_indicator, file_id, current_index=segment_counter)
            # this_dataframe is a dataframe with one row for the
            # whole file
            all_segments_for_all_files.append(this_dictionary)

            num_segments_for_this_file = 1  # only one segment for the whole file
            bioimpedance_segments += num_segments_for_this_file  # update counter
        else:
            raise ValueError(f"Modality {modality} is not supported")

    print(LOGIDENTIFIER +
          f"Total good ECG segments: {good_ecg_segments}")
    print(LOGIDENTIFIER +
          f"Total good PPG segments: {good_ppg_segments}")
    print(LOGIDENTIFIER +
          f"Total bioimpedance segments: {bioimpedance_segments}")
    print("Obs: this software version is not requiring a threshold quality for bioimpedance segments, so all files will have one segment for the whole file.")

    output_dataframe = pd.DataFrame(all_segments_for_all_files)
    print(LOGIDENTIFIER +
          f"Total segments across all files: {len(output_dataframe)}")

    return output_dataframe


def create_segments_dataframe(dataset_config_file: str, output_file_name: str, min_ecg_threshold: float, min_ppg_threshold: float, min_duration: float) -> str:

    datasetConfig = DatasetConfig(dataset_config_file)
    segmenter = Segmenter(
        datasetConfig.get_segmenter_file_name(), dataset_config_file)

    events_dataframe = segment_all_modalities(
        datasetConfig, segmenter)

    # save the combined dataframe with all segments from all files
    output_combined_path = os.path.join(
        datasetConfig.get_dataset_segments_path(), output_file_name)
    events_dataframe.to_csv(output_combined_path, index=False)
    print(
        f"Wrote combined segments dataframe to CSV file {output_combined_path}")
    return output_file_name


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

    # Create a DatasetConfig instance to access dataset paths and info
    dataset_config_file = args.json_file

    datasetConfig = DatasetConfig(dataset_config_file)
    print("Successfully loaded dataset configuration from", dataset_config_file)

    segments_df = segment_all_modalities(datasetConfig)

    output_file_name = datasetConfig.get_segments_file_name()
    # check whether output_file_name already exists
    if Path(output_file_name).exists():
        print(
            f"Warning: file already exists: {output_file_name}. It will be overwritten.")

    segments_df.to_csv(output_file_name, index=False)
    print(f"Segments dataframe saved to: {output_file_name}")

    if False:  # to debug
        print("Aggregating total seconds per participant for ECG...")
        df = _aggregate_total_seconds_per_participant(
            datasetConfig, segments_df, "ecg", "total")
        print(df)
        print("Aggregating total seconds per participant for PPG...")
        df = _aggregate_total_seconds_per_participant(
            datasetConfig, segments_df, "ppg", "total")
        print(df)
