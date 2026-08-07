"""
PPG signal processing
Non-morphological - IMF intrinsic mode functions, energy-based features, entropy-based features, etc.

Morphological - notch-independent morphological features

Morphological - pulse-based, depends on notch
"""

import argparse
import traceback

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis
from scipy.signal import find_peaks
from scipy.signal import welch
import neurokit2 as nk
import matplotlib.pyplot as plt
from typing import Dict, Any, cast
import pandas as pd

from datasets_util.naming_conventions import DatasetConfig
from datasets_util.waveform_files import read_sigmf_file
from datasets_util.waveform_files import save_sigmf_signal_rf32_le
from datasets_util.util_features import append_identification
from features.emd_imf_features import extract_imf_features_from_signal
from signal_processing.cross_features import safe_mean, safe_std
from signal_processing.ppg_quality import estimate_sqi_neurokit_tm
from signal_processing.ppg_quality import estimate_sqi_custom_version
from signal_processing.ppg_quality import estimate_single_sqi_value
from signal_processing.ppg_quality import plot_sqi_comparison
# from signal_processing.ppg_features_by_ufsc import extract_features_from_average_pulse
from signal_processing.ppg_features_by_ufsc import extract_all_UFSC_features_dataframe
from signal_processing.pyppg_methods import estimate_sqi_pyppg
from signal_processing.signal_utils import remove_outliers, estimate_amplitude_bits, plot_quantization_levels, plot_histogram, _energy_in_band, _reverse_signal, BandpassButterworthFilter, LowpassButterworthFilter, savitzky_golay_filtering
# , _calculate_entropy, _calculate_spectral_entropy, _calculate_spectral_kurtosis, _calculate_spectral_skewness
from signal_processing.ppg_features_by_ufsc import obtain_isolated_pulses, extract_isolated_pulse_features_by_ufsc
from datasets_util.util_visualize_plots import plot_input_output_waveforms, plot_input_output_psds, plot_individual_beats

# ======================================================
# Parameters
# ======================================================
SHOULD_PLOT = False  # False to disable plotting
# False to disable plotting within remove_outliers()
SHOULD_PLOT_OUTLIERS = SHOULD_PLOT
# If True, will plot SQI statistics for all processed files at the end of ppg_quality_pipeline
SHOULD_PLOT_SQI_STATISTICS = SHOULD_PLOT
PLOT_SLOW_HISTOGRAM_OF_UNIQUE_VALUES = False  # slower histogram

DEBUGGING = False  # True to enable debug prints

# If False, errors will be logged but the pipeline will continue, returning NaN or empty values for features that failed to extract.
RAISE_EXCEPTION_ON_PPG_PROCESSING = False

# Enable detailed debug messages from UFSC fallback extractor.
DEBUG_UFSC_FEATURE_EXTRACTION = False

if False:  # choose to plot all participants or selected ones
    SUBJECT_IDS_TO_PLOT = None  # use None to plot all subjects
else:
    SUBJECT_IDS_TO_PLOT = {
        "ulac_01",
        "ulac_02",
        "ulac_03",
        "ulac_04",
        "ulac_05",
        "ulac_06",
        "ulac_07",
        "ulac_08",
        "hu_18",  # "best" IEB_1 subject 1
        "hu_15",  # 2nd "best" IEB_1 subject 1
        "hu_35",  # worst still not excluded
        "hu_76",  # before worst still not excluded
        "hu_45",  # excluded: before "worst" IEB_1 subject 1
        "hu_39",  # excluded: "worst" IEB_1 subject 1
        # "ieb_01",  # IEB_3 subject 1
        "ieb_02",
        "ieb_03",
        "ieb_04",
        "ieb_05",
        "ieb_06",
        "ieb_07",
        "ieb_08",
    }

# Filters
bandpass_05_to_4Hz = None
lowpass_20Hz = None
# ======================================================
# UTILITY METHODS
# ======================================================


def simple_ppg_peak_detector(ppg, fs):
    """
    Simple fallback PPG peak detector.

    Parameters
    ----------
    ppg : array_like
        PPG waveform.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    peaks : np.ndarray
        Indices of detected systolic peaks.
    """

    ppg = np.asarray(ppg)

    # Remove DC offset
    ppg = ppg - np.median(ppg)

    # Robust amplitude estimate
    mad = np.median(np.abs(ppg))
    prominence = max(0.5 * mad, 1e-6)

    # Physiological heart-rate limits
    # max HR = 200 bpm -> minimum distance = 60/200 s
    min_distance = int(0.30 * fs)

    peaks, _ = find_peaks(
        ppg,
        distance=min_distance,
        prominence=prominence,
    )

    return peaks


def plot_morphology_statistics(signals, info, title="Morphology Statistics"):
    results = morphology_statistics(signals, info)

    x = np.arange(len(results["mean_beat"]))

    plt.figure(figsize=(6, 4))
    plt.plot(x, results["mean_beat"], color="k", lw=2, label="Mean beat")
    plt.fill_between(
        x,
        results["mean_beat"] - results["std_beat"],
        results["mean_beat"] + results["std_beat"],
        alpha=0.3,
        label="±1 SD",
    )
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def morphology_statistics(signals: pd.DataFrame, info: dict):
    """
    Compute pointwise morphology statistics of PPG beats.

    Parameters
    ----------
    signals : DataFrame
        Output of nk.ppg_process().
    info : dict
        Output of nk.ppg_process().

    Returns
    -------
    results : dict
        Dictionary containing morphology statistics.
    """

    epochs = nk.ppg_segment(
        signals["PPG_Clean"].values,
        peaks=info["PPG_Peaks"],
        sampling_rate=info["sampling_rate"],
        show=False,
    )

    beats = []

    for epoch in epochs.values():

        # NeuroKit versions differ on the waveform column name
        waveform_col = (
            "Signal"
            if "Signal" in epoch.columns
            else "PPG_Clean"
        )

        beats.append(epoch[waveform_col].to_numpy())

    beats = np.vstack(beats)

    # Average morphology
    mean_beat = np.nanmean(beats, axis=0)

    # Pointwise variability
    std_beat = np.nanstd(beats, axis=0)

    # Number of valid beats contributing to each sample
    n_valid = np.sum(~np.isnan(beats), axis=0)

    # Coefficient of variation (avoid divide-by-zero)
    cv_beat = np.divide(
        std_beat,
        np.abs(mean_beat),
        out=np.zeros_like(std_beat),
        where=np.abs(mean_beat) > 1e-12,
    )

    # One scalar summarizing morphology variability
    rms_std = np.sqrt(np.nanmean(std_beat**2))
    print("RMS of pointwise std of beats:", rms_std)

    return {
        "beats": beats,
        "mean_beat": mean_beat,
        "std_beat": std_beat,
        "cv_beat": cv_beat,
        "n_valid": n_valid,
        "rms_std": rms_std,
    }


def our_neurokit_ppg_plot(signals, info, title="Photoplethysmography (PPG)",
                          subtitle1="PPG signal and peaks",
                          subtitle2="Heart Rate",
                          subtitle3="Individual beats"):
    nk.ppg_plot(signals, info)

    # Find heart rate statistics
    rate = signals["PPG_Rate"]
    average_rate = safe_mean(rate)
    std_rate = safe_std(rate)
    subtitle2 = subtitle2 + \
        f" (mean: {average_rate:.1f}, std: {std_rate:.1f} bpm)"

    fig = plt.gcf()
    axes = fig.axes
    fig.suptitle(title, fontsize=16)

    '''
    Modify original labels and titles:

    Axis 0
    Title : PPG signal and peaks
    Xlabel: Time (seconds)
    Ylabel:
    Axis 1
    Title : Heart Rate
    Xlabel: Time (seconds)
    Ylabel: Beats per minute (bpm)
    Axis 2
    Title : Individual beats (average heart rate: 61.8 bpm)
    Xlabel: Time (seconds)
    Ylabel: ppg
    '''
    axes[0].set_ylabel("")
    axes[1].set_ylabel("Beats per minute (bpm)")
    axes[2].set_ylabel("ppg")

    axes[0].set_ylabel("")
    axes[1].set_ylabel("Beats per minute (bpm)")
    axes[2].set_ylabel("ppg")

    axes[0].set_title(subtitle1)
    axes[1].set_title(subtitle2)
    axes[2].set_title(subtitle3)
    plt.tight_layout()


def create_ppg_signals_df(ppg_raw, ppg_clean, sampling_rate,
                          peak_method="elgendi",
                          quality_method="templatematch") -> tuple[pd.DataFrame, dict]:
    """
    Create the DataFrame expected by nk.ppg_plot() without calling
    nk.ppg_process().
    """

    # Detect peaks
    peaks_df, info = nk.ppg_peaks(
        ppg_clean,
        sampling_rate=sampling_rate,
        method=peak_method,
    )

    # Instantaneous heart rate (interpolated to every sample)
    rate = nk.signal_rate(
        info["PPG_Peaks"],
        sampling_rate=sampling_rate,
        desired_length=len(ppg_clean),
    )

    # Signal quality
    quality = nk.ppg_quality(
        ppg_clean,
        peaks=info["PPG_Peaks"],
        sampling_rate=sampling_rate,
        method=quality_method,
        ppg_raw=ppg_raw,
    )

    # Assemble DataFrame
    signals = pd.DataFrame({
        "PPG_Raw": np.asarray(ppg_raw),
        "PPG_Clean": np.asarray(ppg_clean),
        "PPG_Rate": rate,
        "PPG_Quality": quality,
        "PPG_Peaks": peaks_df["PPG_Peaks"].astype(int),
    })

    info["sampling_rate"] = sampling_rate

    return signals, info


def old_collapse_feature_container(features) -> Dict[str, Any]:
    """Convert per-pulse feature containers into a single flat feature dictionary."""
    if features is None:
        return {}

    if isinstance(features, dict):
        return features

    if isinstance(features, pd.DataFrame):
        if features.empty:
            return {}

        numeric = features.select_dtypes(include=[np.number])
        if numeric.empty:
            return {}

        return {str(key): value for key, value in numeric.mean(axis=0).to_dict().items()}

    if isinstance(features, list):
        if not features:
            return {}

        if all(isinstance(item, dict) for item in features):
            frame = pd.DataFrame(features)
            if frame.empty:
                return {}

            numeric = frame.select_dtypes(include=[np.number])
            if numeric.empty:
                return {}

            return {str(key): value for key, value in numeric.mean(axis=0).to_dict().items()}

    return {}


def _to_feature_rows(features) -> list[Dict[str, Any]]:
    """Normalize feature containers to a list of dictionaries, one per row."""
    if features is None:
        return []

    if isinstance(features, dict):
        return [features]

    if isinstance(features, pd.DataFrame):
        if features.empty:
            return []
        rows: list[Dict[str, Any]] = []
        for row in features.to_dict(orient="records"):
            row_dict: Dict[str, Any] = {
                str(key): value for key, value in cast(Dict[Any, Any], row).items()
            }
            rows.append(row_dict)
        return rows

    if isinstance(features, list):
        return [row for row in features if isinstance(row, dict)]

    return []


def _calculate_sqi_statistics(sqi_signals: list):
    # get the overall mean and std of all SQI signals combined
    all_sqi_values = np.concatenate(sqi_signals)
    print(
        f"Overall Mean SQI = {np.mean(all_sqi_values):.4f}, Overall Std SQI = {np.std(all_sqi_values):.4f}")
    print(f"Overall SQI median = {np.median(all_sqi_values):.4f}")
    print(
        f"Overall SQI min = {np.min(all_sqi_values):.4f}, Overall SQI max = {np.max(all_sqi_values):.4f}")
    print(
        f"Overall SQI 25th percentile = {np.percentile(all_sqi_values, 25):.4f}, Overall SQI 75th percentile = {np.percentile(all_sqi_values, 75):.4f}")
    print(
        f"Overall SQI 10th percentile = {np.percentile(all_sqi_values, 10):.4f}, Overall SQI 90th percentile = {np.percentile(all_sqi_values, 90):.4f}")
    print(
        f"Overall SQI 5th percentile = {np.percentile(all_sqi_values, 5):.4f}, Overall SQI 95th percentile = {np.percentile(all_sqi_values, 95):.4f}")
    print(
        f"Overall SQI 1st percentile = {np.percentile(all_sqi_values, 1):.4f}, Overall SQI 99th percentile = {np.percentile(all_sqi_values, 99):.4f}")

    # get the mean and std of each SQI signal and print them
    for i, sqi_signal in enumerate(sqi_signals):
        mean_sqi = np.mean(sqi_signal)
        std_sqi = np.std(sqi_signal)
        print(
            f"Signal {i+1}: Mean SQI = {mean_sqi:.4f}, Std SQI = {std_sqi:.4f}")

    # calculate and plot the histogram of SQI signals using one color per list entry
    plt.figure(figsize=(10, 6))
    for i, sqi_signal in enumerate(sqi_signals):
        plt.hist(sqi_signal, bins=30, alpha=0.7, label=f'Signal {i+1}')
    plt.xlabel('SQI Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of SQI Signals')
    plt.legend()
    plt.grid()
    plt.show()


def __detect_beats_neurokit(ppg_signal, fs):
    """
    Detect peaks and onsets using NeuroKit WITHOUT filtering.
    """

    signal = np.asarray(ppg_signal).astype(float)

    # DO NOT call nk.ppg_clean
    try:
        info = nk.ppg_findpeaks(signal, sampling_rate=fs, method="elgendi")
    except Exception as e:
        print(f"Error in nk.ppg_findpeaks: {e}")
        traceback.print_exc()
        return None

    peaks = info.get("PPG_Peaks", [])

    # Some methods return onsets (Charlton)
    onsets = info.get("PPG_Onsets", None)

    # fallback: estimate onsets via local minima
    if onsets is None or len(onsets) == 0:
        inverted = -signal
        onsets = nk.signal_findpeaks(inverted)["Peaks"]

    beats = []

    for i in range(len(peaks)):
        p = peaks[i]

        prev_onsets = onsets[onsets < p]
        next_onsets = onsets[onsets > p]

        if len(prev_onsets) == 0 or len(next_onsets) == 0:
            continue

        start = prev_onsets[-1]
        end = next_onsets[0]

        if end > start:
            beats.append((start, p, end))

    return beats


def __butter_lowpass_filter(signal, fs):
    # ======================================================
    # Butterworth filter
    # ======================================================
    global lowpass_20Hz
    if lowpass_20Hz is None:
        # design filter only if needed
        lowpass_20Hz = LowpassButterworthFilter(
            order=6, cutoff=20.0, fs=fs)
    if lowpass_20Hz.fs != fs:  # check if sampling frequencies match
        print(
            f"WARNING: Redesigning filter. Filter sampling frequency {lowpass_20Hz.fs} does not match signal sampling frequency {fs}")
        lowpass_20Hz = LowpassButterworthFilter(
            order=6, cutoff=20.0, fs=fs)

    filtered = filtfilt(lowpass_20Hz.Bz, lowpass_20Hz.Az, signal)
    return filtered


def __butter_bandpass_filter(signal, fs):
    # ======================================================
    # Butterworth filter
    # ======================================================
    global bandpass_05_to_4Hz
    if bandpass_05_to_4Hz is None:
        # design filter only if needed
        bandpass_05_to_4Hz = BandpassButterworthFilter(
            order=4, low=0.5, high=4.0, fs=fs)
    if bandpass_05_to_4Hz.fs != fs:  # check if sampling frequencies match
        print(
            f"WARNING: Redesigning filter. Filter sampling frequency {bandpass_05_to_4Hz.fs} does not match signal sampling frequency {fs}")
        bandpass_05_to_4Hz = BandpassButterworthFilter(
            order=4, low=0.5, high=4.0, fs=fs)

    filtered = filtfilt(bandpass_05_to_4Hz.Bz, bandpass_05_to_4Hz.Az, signal)
    return filtered


def _debug_processing_files(dataset_config_file):
    waveform_id = "quality"
    input_waveform_id = "filtered"

    # Example: process a dataset
    datasetConfig = DatasetConfig(dataset_config_file)
    required_ppg_fs = datasetConfig.get_ppg_fs()

    # a deep copy is made below, to allow external modifications
    df = datasetConfig.get_dataset_info_dataframe()

    # Filter only ppg signals. It is safe to modify df because df is a deep copy
    df = df[df["modality"].str.contains("ppg")]

    file_counter = 0
    for counter, row in df.iterrows():
        file_id = row["file_id"]
        raw_complete_path = datasetConfig.get_raw_complete_path(file_id)
        # full_path = os.path.join(base_path, raw_relative_path)
        print(f"File ID: {file_id}, Complete path: {raw_complete_path}")

        filtered_signal, input_signal, metadata = ppg_bandpass_waveform_processing(
            raw_complete_path, required_ppg_fs)

        # features = extract_features_group1_peak_based(
        #    filtered_signal, None, fs)
        # print("AAAAAAAAAAAAAA ", features)


def _normalize_peaks(peaks):
    '''Make sure we deal with np.array'''
    if peaks is None:
        return None

    # pandas Series => numpy
    peaks = np.asarray(peaks)

    if peaks.ndim == 1 and peaks.dtype == bool:
        # boolean mask → indices
        return np.where(peaks)[0]

    # already indices
    if np.issubdtype(peaks.dtype, np.integer):
        return peaks

    # fallback (e.g., float or weird types)
    try:
        return peaks.astype(int)
    except Exception:
        return None


# ======================================================
# METHODS TO EXECUTE PROCESSING PIPELINE
# ======================================================


def ppg_bandpass_waveform_processing(input_waveform: np.ndarray, fs: int) -> np.ndarray:

    output_waveform = __butter_bandpass_filter(input_waveform, fs)

    return output_waveform


def ppg_reverse_waveform_processing(input_waveform: np.ndarray, fs: int) -> np.ndarray:
    #  If signals are reversed in acquisition
    input_waveform = _reverse_signal(input_waveform)

    output_waveform = ppg_bandpass_waveform_processing(input_waveform, fs)

    return output_waveform


def ppg_inversion_waveform_processing(input_waveform: np.ndarray, fs: int) -> np.ndarray:
    # 1) invert the amplitude
    input_waveform = input_waveform * -1.0

    # 2) Remove outliers to prevent spreading of artifacts during filtering
    signal_duration = len(input_waveform) / fs
    outlier_window = 1.0 * signal_duration * 1000  # signal duration in ms
    input_waveform_removed_outliers, outlier_mask = remove_outliers(
        input_waveform, fs=fs, win_ms=outlier_window, k=15.0, should_plot=SHOULD_PLOT_OUTLIERS)
    if np.any(outlier_mask):
        print(
            f"  WARNING:  Detected {np.sum(outlier_mask)} outliers in PPG signal, replacing with linear interpolation")
    input_waveform = input_waveform_removed_outliers

    # 3) Bandpass filtering
    output_waveform = ppg_bandpass_waveform_processing(input_waveform, fs)

    return output_waveform


def ppg_savitzky_waveform_processing(input_waveform: np.ndarray, fs: int) -> np.ndarray:
    temp_waveform = __butter_bandpass_filter(input_waveform, fs)

    output_waveform = savitzky_golay_filtering(temp_waveform)

    return output_waveform


def ppg_neurokit_waveform_processing(input_waveform: np.ndarray, fs: int) -> np.ndarray:

    # See https://neuropsychology.github.io/NeuroKit/functions/ppg.html#neurokit2.ppg.ppg_clean
    output_waveform = nk.ppg_clean(
        input_waveform, sampling_rate=fs, method='elgendi')

    # Not needed because len(output_waveform) == len(input_waveform)
    # if len(output_waveform) != len(input_waveform):
    #    raise Exception(
    #        f"WARNING: NeuroKit2 ppg_clean returned a different length ({len(output_waveform)}) than the input ({len(input_waveform)}). Resizing output to match input.")
    #    output_waveform = np.resize(output_waveform, len(input_waveform))

    return np.array(output_waveform, dtype='float')


def ppg_lowpass_waveform_processing(input_waveform: np.ndarray, fs: int) -> np.ndarray:
    # 1) invert the amplitude
    # input_waveform = input_waveform * -1.0

    # 2) Remove outliers to prevent spreading of artifacts during filtering
    input_waveform_removed_outliers, outlier_mask = remove_outliers(
        input_waveform, fs=fs, win_ms=2000, k=10.0, should_plot=SHOULD_PLOT_OUTLIERS)
    if np.any(outlier_mask):
        print(
            f"  WARNING:  Detected {np.sum(outlier_mask)} outliers in PPG signal, replacing with linear interpolation")

    # 3) Lowpass filtering
    output_waveform = __butter_lowpass_filter(
        input_waveform_removed_outliers, fs)

    # 4) Remove DC level
    output_waveform -= np.mean(output_waveform)

    return output_waveform


def ppg_sqi_nkcustom_waveform_processing(ppg_signal: np.ndarray, fs: int) -> np.ndarray:
    '''
    Estimate Signal Quality Index (SQI) for PPG signals using custom method,
    based on neurokit2, generating one SQI value for each PPG sample.
    '''

    # Remove outliers to prevent spreading of artifacts during filtering
    # input_waveform_removed_outliers, outlier_mask = remove_outliers(
    #    ppg_signal, fs=fs, win_ms=len(ppg_signal-100)/fs*1000, k=4.0, should_plot=False)
    # if np.any(outlier_mask):
    #    print(
    #        f"  WARNING:  Detected {np.sum(outlier_mask)} outliers in PPG signal, replacing with linear interpolation")

    if False:
        plot_input_output_waveforms(
            ppg_signal, input_waveform_removed_outliers, fs, "", title1="Raw signal", title2="Outlier removed")

    # sqi_signal = estimate_sqi_custom_version(ppg_signal, fs)
    sqi_signal = estimate_sqi_custom_version(
        ppg_signal, fs)

    return sqi_signal


def ppg_sqi_pyppg_waveform_processing(ppg_signal: np.ndarray, fs: int) -> np.ndarray:
    '''
    Estimate Signal Quality Index (SQI) for PPG signals using pyppg method,
    generating one SQI value for each PPG sample.
    '''
    try:
        sqi_signal = estimate_sqi_pyppg(ppg_signal, fs)
    except Exception as e:
        traceback.print_exc()
        print(f"Error in pyppg SQI estimation: {e}")
        sqi_signal = np.full_like(ppg_signal, 0.0)  # Return NaN array on error

    return sqi_signal


def ppg_sqi_sumall_waveform_processing(ppg_signal: np.ndarray, fs: int) -> np.ndarray:
    '''
    Estimate Signal Quality Index (SQI) for PPG signals using a sum of all methods,
    generating one SQI value for each PPG sample.
    '''
    # get all SQI identificators from QUALITY_PIPELINES
    sqi_identifiers = [
        key for key in QUALITY_PIPELINES.keys() if key != "sqi_sumall"]
    num_sqi_pipelines = len(sqi_identifiers)
    if num_sqi_pipelines < 1:
        raise ValueError(
            f"Not enough SQI pipelines to combine. Found {num_sqi_pipelines} pipelines.")
    # initialize sqi_signal to zeros
    sqi_signal = np.zeros_like(ppg_signal, dtype=float)
    for sqi_id in sqi_identifiers:
        this_sqi_signal = QUALITY_PIPELINES[sqi_id](ppg_signal, fs)
        # print("\n\n", sqi_id, "XXX", this_sqi_signal.__class__,
        #      this_sqi_signal.shape, this_sqi_signal.dtype, "####")
        sqi_signal += this_sqi_signal
        # print("Aqui", sqi_signal, "\n", np.isnan(
        #    sqi_signal).any(), np.isinf(sqi_signal).any())

    # Combine the SQI signals by taking the average
    sqi_signal /= num_sqi_pipelines
    return sqi_signal


def ppg_sqi_neurokit_waveform_processing(ppg_signal: np.ndarray, fs: int) -> np.ndarray:
    '''
    Estimate Signal Quality Index (SQI) for PPG signals using NeuroKit2 method,
    generating one SQI value for each PPG sample.
    '''

    sqi_signal = estimate_sqi_neurokit_tm(ppg_signal, fs)

    return sqi_signal


def ppg_signal_processing_pipeline(dataset_config_file: str, input_waveform_id: str,
                                   pipeline: str,
                                   output_waveform_id: str) -> None:
    '''
    Use this method for both waveform and SQI processing pipelines. The pipeline is selected by the "pipeline" argument.
    We construct the function name dynamically from the string.
    Requirement: the functions are defined in this same module.
    globals() is part of the Python language and works the same in Jupyter notebooks, Colab, etc.
    '''

    # check if the pipeline is valid
    is_valid_signal_pipeline = pipeline in SIGNAL_PIPELINES
    is_valid_quality_pipeline = pipeline in QUALITY_PIPELINES
    if is_valid_quality_pipeline and is_valid_signal_pipeline:
        raise ValueError(
            f"Pipeline '{pipeline}' is defined in both SIGNAL_PIPELINES and QUALITY_PIPELINES. Please change the code to choose a unique name for each pipeline."
        )
    if is_valid_signal_pipeline:
        try:
            processing_function = SIGNAL_PIPELINES[pipeline]
        except KeyError:
            valid = ", ".join(SIGNAL_PIPELINES.keys())
            raise ValueError(
                f"Unknown waveform_id '{pipeline}'. "
                f"Valid options are: {valid}"
            )
    elif is_valid_quality_pipeline:
        try:
            processing_function = QUALITY_PIPELINES[pipeline]
        except KeyError:
            valid = ", ".join(QUALITY_PIPELINES.keys())
            raise ValueError(
                f"Unknown waveform_id '{pipeline}'. "
                f"Valid options are: {valid}"
            )
    else:
        valid_signal = ", ".join(SIGNAL_PIPELINES.keys())
        valid_quality = ", ".join(QUALITY_PIPELINES.keys())
        raise ValueError(
            f"Unknown waveform_id '{pipeline}'. "
            f"Valid signal processing options are: {valid_signal}. "
            f"Valid quality processing options are: {valid_quality}."
        )

    # by definition, a signal pipeline requires input_waveform_id='raw', while a quality pipeline can accept any input waveform
    if is_valid_signal_pipeline and input_waveform_id != "raw":
        raise ValueError(
            f"Signal processing pipeline '{pipeline}' requires input_waveform_id='raw', but got '{input_waveform_id}'."
        )

    # Process the dataset
    datasetConfig = DatasetConfig(dataset_config_file)
    required_ppg_fs = datasetConfig.get_ppg_fs()

    # make a deep copy to prevent external modifications
    df = datasetConfig.get_dataset_info_dataframe()

    print('df["modality"].value_counts()', df["modality"].value_counts())

    # Filter only ppg signals. It is safe to modify df because df is a deep copy
    df = df[df["modality"].str.contains("ppg")]

    file_counter = 0
    for counter, row in df.iterrows():
        file_id = row["file_id"]
        participant_id = row["participant_id"]

        if input_waveform_id == "raw":
            input_complete_path = datasetConfig.get_raw_complete_path(file_id)
        else:
            input_complete_path = datasetConfig.get_gen_complete_path(
                file_id, input_waveform_id)
        print(f"Input PPG: {input_complete_path}, (file_id={file_id})")

        input_waveform, input_metadata = read_sigmf_file(input_complete_path)
        input_waveform = np.array(input_waveform, dtype='float')

        fs = input_metadata["global"]["core:sample_rate"]
        if fs != required_ppg_fs:
            raise ValueError(
                f"Expected sampling frequency {required_ppg_fs} Hz, but got {fs} Hz in file {ppg_filename}")

        # print the waveform duration in seconds
        duration_sec = len(input_waveform) / fs
        if duration_sec < 15.0:
            print("WARNING: ", input_complete_path,
                  f" has duration less than 15 seconds: {duration_sec:.2f} seconds")

        if DEBUGGING:
            # Estimate amplitude bits (for quantization analysis)
            quantization_info = estimate_amplitude_bits(input_waveform)
            observed_bits = quantization_info["observed_bits"]
            print("Number of bits to quantize amplitude values =", observed_bits)

        if SHOULD_PLOT and ((SUBJECT_IDS_TO_PLOT is None) or (participant_id in SUBJECT_IDS_TO_PLOT)):
            num_chars = 60
            plot_both = PLOT_SLOW_HISTOGRAM_OF_UNIQUE_VALUES
            plot_histogram(
                input_waveform, title=f"{input_complete_path[-num_chars:]} in {pipeline}", bins=128, grid=True, show_me=not plot_both)
            if plot_both:
                # takes longer time to compute
                plot_quantization_levels(
                    input_waveform, title=f"PPG Signal Quantization Levels (file: {input_complete_path})", show_me=True)

        # execute the pipeline
        output_waveform = processing_function(input_waveform, required_ppg_fs)

        output_filename = datasetConfig.get_gen_complete_path(
            file_id, output_waveform_id)

        # create output directory if it doesn't exist
        output_dir = os.path.dirname(output_filename)
        os.makedirs(output_dir, exist_ok=True)
        # Save to proper location
        save_sigmf_signal_rf32_le(
            output_waveform, input_metadata, output_filename)
        print(f"Output PPG: {output_filename} (file_id={file_id})")

        file_counter += 1
        # --------------------------------------------------
        # Plot only selected subjects
        # --------------------------------------------------
        if SHOULD_PLOT and ((SUBJECT_IDS_TO_PLOT is None) or (participant_id in SUBJECT_IDS_TO_PLOT)):
            plot_input_output_waveforms(
                input_waveform,
                output_waveform,
                required_ppg_fs,
                title=f"Subject {participant_id} in the {pipeline} pipeline",
                title1=input_waveform_id,
                title2=pipeline
            )

            # now the PSDs
            plot_input_output_psds(
                input_waveform,
                output_waveform,
                required_ppg_fs,
                title=f"PSDs for {participant_id} in {pipeline} pipeline",
                title1=input_waveform_id,
                title2=pipeline
            )

            this_title = f"Subject {participant_id} in the {pipeline} pipeline"

            if output_waveform_id.startswith("sqi"):
                plot_sqi_comparison(input_waveform, fs=fs, title=this_title)
            else:
                # generate signals to use nk.ppg_plot
                signals, info = create_ppg_signals_df(
                    input_waveform, output_waveform, required_ppg_fs)

                # plot only individual beats
                plot_individual_beats(signals, info, this_title)

                plot_morphology_statistics(signals, info, this_title)

                # the plot below is redundant given the previous one
                # plot all 3 panels (PPG signal, heart rate, individual beats) using our custom function
                # our_neurokit_ppg_plot(signals, info, this_title)
                # nk.ppg_plot(signals, info)
                # plt.show()

    print(
        f"\nFinished processing {file_counter} files from {counter+1} rows in the dataset.")


def old_delete_me_ppg_quality_pipeline(dataset_config_file: str) -> None:
    waveform_id = "quality"
    input_waveform_id = "filtered"

    # Example: process a dataset
    datasetConfig = DatasetConfig(dataset_config_file)
    required_ppg_fs = datasetConfig.get_ppg_fs()

    # a deep copy is made below, to allow external modifications
    df = datasetConfig.get_dataset_info_dataframe()

    # Filter only ppg signals. It is safe to modify df because df is a deep copy
    df = df[df["modality"].str.contains("ppg")]

    file_counter = 0
    sqi_statistics = []
    for counter, row in df.iterrows():
        file_id = row["file_id"]
        participant_id = row["participant_id"]

        filtered_complete_path = datasetConfig.get_gen_complete_path(
            file_id, input_waveform_id)
        print(
            f"File ID: {file_id}, Patient ID: {participant_id}, Complete path: {filtered_complete_path}")

        sqi_signal, input_signal, metadata = ppg_quality_waveform_processing(
            filtered_complete_path, required_ppg_fs)

        sqi_statistics.append(sqi_signal)

        output_filename = datasetConfig.get_gen_complete_path(
            file_id, waveform_id)
        # print("Relative path for file_id", file_id, ":", relative_path)

        # create output directory if it doesn't exist
        output_dir = os.path.dirname(output_filename)
        os.makedirs(output_dir, exist_ok=True)
        # Save to proper location
        save_sigmf_signal_rf32_le(sqi_signal, metadata, output_filename)
        print(
            f"Saved processed data for file_id={file_id} to {output_filename}")

        file_counter += 1

        if SHOULD_PLOT and ((SUBJECT_IDS_TO_PLOT is None) or (participant_id in SUBJECT_IDS_TO_PLOT)):
            plot_input_output_waveforms(
                input_signal,
                sqi_signal,
                required_ppg_fs,
                title=f"Subject {participant_id}",
                title1="Filtered PPG signal",
                title2="SQI"
            )

        if SHOULD_PLOT:
            # try to plot and catch exceptions (e.g., if sqi_signal is empty or has issues)
            try:
                signals, info = nk.ppg_process(
                    sqi_signal, sampling_rate=required_ppg_fs)
                nk.ppg_plot(signals, info)

                plt.show()
            except Exception as e:
                print(
                    f"   ########## Error occurred while estimating SQI: {e}")
                continue

    print(f"\nFinished processing {file_counter} files.")

    if SHOULD_PLOT_SQI_STATISTICS:
        # Calculate and plot SQI statistics for all processed files
        _calculate_sqi_statistics(sqi_statistics)


# ======================================================
# FEATURE EXTRACTION
# ======================================================

def ppg_feature_extraction_pipeline(datasetConfig: DatasetConfig,
                                    ppg: np.ndarray,
                                    fs: int) -> None:
    '''
    We construct the function name dynamically from the string.
    Requirement: the functions are defined in this same module, at the end of the module.
    '''

    try:
        processing_function = FEATURE_PIPELINES[pipeline]
    except KeyError:
        valid = ", ".join(FEATURE_PIPELINES.keys())
        raise ValueError(
            f"Unknown waveform_id '{pipeline}'. "
            f"Valid options are: {valid}"
        )

    # Process the dataset
    datasetConfig = DatasetConfig(dataset_config_file)
    required_ppg_fs = datasetConfig.get_ppg_fs()

    # make a deep copy to prevent external modifications
    df = datasetConfig.get_dataset_info_dataframe()

    print('df["modality"].value_counts()', df["modality"].value_counts())

    # Filter only ppg signals. It is safe to modify df because df is a deep copy
    df = df[df["modality"].str.contains("ppg")]

    file_counter = 0
    for counter, row in df.iterrows():
        file_id = row["file_id"]
        # execute the pipeline
        output_waveform = processing_function(input_waveform, required_ppg_fs)

        output_filename = datasetConfig.get_gen_complete_path(
            file_id, output_waveform_id)

        # create output directory if it doesn't exist
        output_dir = os.path.dirname(output_filename)
        os.makedirs(output_dir, exist_ok=True)
        # Save to proper location
        save_sigmf_signal_rf32_le(
            output_waveform, input_metadata, output_filename)
        print(f"Output PPG: {output_filename} (file_id={file_id})")

        file_counter += 1


def extract_ppg_features(
    datasetConfig: DatasetConfig,
    ppg: np.ndarray,
    fs: int
) -> Dict[str, Any] | list[Dict[str, Any]]:
    '''
    We construct the function name dynamically from the string.
    Requirement: the functions are defined in this same module, at the end of the module.
    '''
    pipeline = datasetConfig.get_value("PPG_FEATURE_EXTRACTION_PIPELINE")

    try:
        processing_function = FEATURE_PIPELINES[pipeline]
    except KeyError:
        valid = ", ".join(FEATURE_PIPELINES.keys())
        raise ValueError(
            f"Unknown waveform_id '{pipeline}'. "
            f"Valid options are: {valid}"
        )

    return processing_function(ppg, fs)


def ppg_single_vector_feature_extraction(ppg: np.ndarray, fs: int):
    features = ppg_all_feature_extraction(ppg, fs)
    return features


def ppg_all_feature_extraction(ppg: np.ndarray, fs: int):

    if False:
        # if want to visualize input signal
        plot_input_output_waveforms(
            ppg, ppg, fs, "", title1="Raw signal", title2="Raw signal (repeated)")

    try:
        # Because the peaks created by nk2 are used in multiple feature extraction groups,
        # we call nk.ppg_process() only once and reuse the peaks for all feature groups.
        # From:
        # https://neuropsychology.github.io/NeuroKit/examples/ecg_hrv/ecg_hrv.html
        # Use Neurokit2 to find PPG events
        signals, info = nk.ppg_process(ppg, sampling_rate=fs)

        if SHOULD_PLOT:
            nk.ppg_plot(signals, info)
            plt.show()

        # Get peaks and normalize to numpy array (after nk.ppg_process)
        peaks = _normalize_peaks(info.get("PPG_Peaks", None))
        if peaks is not None:
            peaks = np.sort(peaks)
    except Exception as e:
        if RAISE_EXCEPTION_ON_PPG_PROCESSING:
            raise Exception(e)
        else:
            print("WARNING: Neurokit2 ppg_process failed!!!")
            traceback.print_exc()
            return None

            peaks = simple_ppg_peak_detector(ppg, fs)
            # Assemble DataFrame
            signals = pd.DataFrame({
                "PPG_Raw": np.asarray(ppg),
                "PPG_Clean": np.asarray(ppg),
                "PPG_Peaks": peaks.astype(int),
            })
            info["sampling_rate"] = fs

            if peaks is not None:
                peaks = np.sort(peaks)

    # Calculate each group of features and rename the features such that
    # all of them have the preamble ppg followed by a group identification
    # append_identification expects a dictionary and iterates with .items().
    # Some feature will use the signal obtained by nk2, while others operate
    # directly on the original ppg signal. The features are later concatenated
    # into a single dictionary.

    def _safe_extract_group(group_name: str, extractor, *args, **kwargs):
        try:
            result = extractor(*args, **kwargs)
            if result is None:
                return {}
            return result
        except Exception as exc:
            print(
                f"WARNING: PPG feature group '{group_name}' failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return {f"ppg_{group_name}_failed": np.nan}

    # Peaks-based features based on Neurokit2 signal processing
    features1_peaks = _safe_extract_group(
        "p1", extract_features_group1_peaks, signals, peaks, fs)
    features1_peaks = append_identification(features1_peaks, "ppg", "p1")

    # Beats-based features based on original (not processed by nk2) ppg signal
    features2_beats = _safe_extract_group(
        "b2", extract_features_group2_beats, ppg, fs)
    features2_beats = append_identification(features2_beats, "ppg", "b2")

    # Power spectral density (PSD)-based features applied to original ppg
    features3_spectral = _safe_extract_group(
        "s3", extract_features_group3_spectral, ppg, fs)
    features3_spectral = append_identification(
        features3_spectral, "ppg", "s3")

    # HRV features
    features4_hrv = _safe_extract_group(
        "h4", extract_features_group4_prv, ppg, peaks, fs)
    features4_hrv = append_identification(features4_hrv, "ppg", "h4")

    # Previous work at UFSC used features5_ufsc and features6_imf
    features5_ufsc = _safe_extract_group(
        "u5", extract_features_group5_ufsc, ppg, fs)
    features6_imf = _safe_extract_group(
        "i6", extract_features_group6_imf, ppg, fs)
    # append to both UFSC and IMF
    features5_ufsc = append_identification(features5_ufsc, "ppg", "u5")
    features6_imf = append_identification(features6_imf, "ppg", "i6")

    concatenated_dictionary = {
        **features1_peaks,
        **features2_beats,
        **features3_spectral,
        **features4_hrv,
        **features5_ufsc,
        **features6_imf
    }

    return concatenated_dictionary


def extract_features_group6_imf(ppg: np.ndarray, fs: int) -> Dict[str, Any]:
    # Features using Empirical Mode Decomposition (EMD) and Intrinsic Mode Functions (IMFs)
    max_imfs = 1  # only the main IMF is used for feature extraction, but this can be adjusted as needed
    features6_imf = extract_imf_features_from_signal(
        ppg, fs=fs, max_imfs=max_imfs)

    # extractor may return None or an empty DataFrame for low-quality segments.
    if features6_imf is None:
        features6_imf = {}
    elif isinstance(features6_imf, pd.DataFrame):
        if features6_imf.empty:
            features6_imf = {}
        else:
            features6_imf = {
                str(k): v for k, v in features6_imf.iloc[0].to_dict().items()
            }
    elif isinstance(features6_imf, dict):
        features6_imf = features6_imf
    else:
        features6_imf = {}

    return features6_imf


def extract_features_group5_ufsc(ppg, fs) -> Dict[str, Any]:
    """Extract UFSC features and normalize the result to a single-row dict.

    The UFSC extractor can yield None, a pandas DataFrame, or a dict depending on
    signal quality and extraction path. This helper always returns a plain dict.
    """
    # Features from legacy UFSC code, which is not robust to low quality PPG:
    # features5_ufsc = extract_features_from_average_pulse(ppg, fs)

    features5_ufsc_raw = extract_all_UFSC_features_dataframe(
        np.array(ppg),
        fs,
        verbose=DEBUG_UFSC_FEATURE_EXTRACTION,
    )

    # Normalize possible extractor outputs (None/DataFrame/dict) to a dict.
    if features5_ufsc_raw is None:
        features5_ufsc = {}
    elif isinstance(features5_ufsc_raw, pd.DataFrame):
        if features5_ufsc_raw.empty:
            features5_ufsc = {}
        else:
            features5_ufsc = {
                str(k): v for k, v in features5_ufsc_raw.iloc[0].to_dict().items()
            }
    elif isinstance(features5_ufsc_raw, dict):
        features5_ufsc = features5_ufsc_raw
    else:
        features5_ufsc = {}

    # if features5_ufsc is not empty, remove
    if features5_ufsc:
        features_to_be_removed = ['ppg_u5_m1_HRV_SDANN2', 'ppg_u5_m1_HRV_SDNNI2',
                                  'ppg_u5_m1_HRV_SDANN5', 'ppg_u5_m1_HRV_SDNNI5',
                                  'ppg_u5_m1_HRV_SDANN1', 'ppg_u5_m1_HRV_SDNNI1']
        for feature in features_to_be_removed:
            if feature in features5_ufsc:
                del features5_ufsc[feature]

    return features5_ufsc


def extract_features_group1_peaks(signals, peaks, fs: int) -> Dict[str, Any]:
    '''
    Extract peaks-based features from PPG signals using
    method nk.ppg_process
    '''
    features = {
        "ppg_hr_mean": safe_mean(signals["PPG_Rate"]),
        "ppg_hr_std": safe_std(signals["PPG_Rate"]),
    }

    # Amplitude
    clean = signals["PPG_Clean"]
    features["ppg_amp_mean"] = safe_mean(clean)
    features["ppg_amp_std"] = safe_std(clean)

    # Peak intervals
    if peaks is not None and len(peaks) >= 2:
        rr = np.diff(peaks) / fs
        features["ppg_peak_interval_std"] = safe_std(rr)
    else:
        features["ppg_peak_interval_std"] = np.nan

    # Derivatives
    d1 = np.gradient(clean)
    d2 = np.gradient(d1)

    features["ppg_d1_max_mean"] = safe_mean(np.abs(d1))
    features["ppg_d2_max_mean"] = safe_mean(np.abs(d2))

    # SQI (robust)
    if False:
        if peaks is not None and len(peaks) >= 3:
            try:
                sqi_values = nk.ppg_quality(clean)
                features["ppg_sqi"] = safe_mean(
                    sqi_values) if len(sqi_values) > 0 else np.nan
            except Exception:
                features["ppg_sqi"] = np.nan
        else:
            features["ppg_sqi"] = np.nan

    return features


def compute_metrics_for_all_beats(ppg, beats, fs) -> pd.DataFrame:
    # ======================================================
    # BEAT Metrics
    # ======================================================
    rows = []

    for start, peak, end in beats:
        beat = ppg[start:end]

        if len(beat) < 3:
            continue

        baseline = np.min(beat)
        amp = ppg[peak] - baseline

        T = (end - start) / fs
        Tr = (peak - start) / fs
        Tr_norm = Tr / T if T > 0 else np.nan

        auc = np.trapezoid(beat - baseline, dx=1/fs)

        half = baseline + amp / 2
        above = np.where(beat >= half)[0]
        width = (above[-1] - above[0]) / fs if len(above) >= 2 else np.nan

        rise_slope = amp / Tr if Tr > 0 else np.nan

        rows.append({
            "amplitude": amp,
            "T": T,
            "Tr": Tr,
            "Tr_norm": Tr_norm,
            "AUC": auc,
            "width": width,
            "rise_slope": rise_slope
        })

    return pd.DataFrame(rows)


def extract_features_group2_beats(segment, fs=500, min_beats=5):
    segment = np.asarray(segment)

    if segment.ndim != 1:
        raise ValueError(f"Expected 1-D segment, got shape {segment.shape}")

    beats = __detect_beats_neurokit(segment, fs)
    if beats is None:
        print("WARNING: No beats detected by neurokit in segment!")
        row = {"N_BEATS": 0}
        return row

    beat_df = compute_metrics_for_all_beats(segment, beats, fs)

    numeric_df = beat_df.select_dtypes(include=[np.number])

    row = {"N_BEATS": len(numeric_df)}

    try:
        if len(numeric_df) >= min_beats:
            for c in numeric_df.columns:
                values = numeric_df[c].dropna()

                if len(values) >= min_beats:
                    median = values.median()
                    mean = values.mean()
                    std = values.std()

                    row[f"{c}_median"] = median
                    row[f"{c}_cv"] = std / (abs(mean) + 1e-8)
                else:
                    row[f"{c}_median"] = np.nan
                    row[f"{c}_cv"] = np.nan
        else:
            for c in numeric_df.columns:
                row[f"{c}_median"] = np.nan
                row[f"{c}_cv"] = np.nan

    except Exception:
        if RAISE_EXCEPTION_ON_PPG_PROCESSING:
            raise
        print("########## Error while aggregating beat-level PPG features")

    return row


def extract_features_group3_spectral(ppg, fs):
    '''
    Spectral features.
    We cannot trust raw spectral power features unless every PPG segment has
    been amplitude-normalized and quality-controlled. Otherwise, the model may
    learn measurement conditions rather than glucose-related information.
    '''
    ppg = np.asarray(ppg, dtype=float)

    if ppg.ndim != 1:
        raise ValueError(f"Expected 1-D PPG signal, got shape {ppg.shape}")

    if len(ppg) < 3:
        raise ValueError(f"PPG signal too short: len={len(ppg)}")

    if not np.all(np.isfinite(ppg)):
        raise ValueError("PPG signal contains NaN or Inf values")

    # Remove DC level to avoid dominance of baseline in PSD
    ppg = ppg - np.mean(ppg)

    eps = 1e-12

    # Welch parameters
    min_cycles = 5
    f_min = 0.7
    nperseg_min = int(min_cycles * fs / f_min)

    # Target frequency resolution around 0.05 Hz
    nperseg_target = int(fs / 0.05)

    nperseg = min(len(ppg), max(nperseg_min, nperseg_target))

    if nperseg < 8:
        raise ValueError(f"nperseg too small: {nperseg}")

    psd_freqs, psd = welch(
        ppg,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )

    # Restrict analysis to physiological PPG band
    analysis_band = (psd_freqs >= 0.5) & (psd_freqs <= 5.0)

    if not analysis_band.any():
        raise ValueError("No valid PPG spectral analysis band found")

    freqs_b = psd_freqs[analysis_band]
    psd_b = psd[analysis_band]

    # Pulse / heart-rate band for fundamental
    pulse_band = (psd_freqs >= 0.6) & (psd_freqs <= 3.0)

    if not pulse_band.any():
        raise ValueError("No pulse band found")

    psd_pulse = psd[pulse_band]
    freqs_pulse = psd_freqs[pulse_band]

    idx_f0 = np.argmax(psd_pulse)
    f0 = freqs_pulse[idx_f0]
    max_power_1st = psd_pulse[idx_f0]
    f_max_1st = f0

    # Band powers
    total_power = np.trapezoid(psd_b, freqs_b) + eps

    E1 = _energy_in_band(0.9 * f0, 1.1 * f0, psd, psd_freqs)
    E2 = _energy_in_band(1.8 * f0, 2.2 * f0, psd, psd_freqs)

    E1_norm = E1 / total_power if np.isfinite(E1) else np.nan
    E2_norm = E2 / total_power if np.isfinite(E2) else np.nan
    harmonic_ratio = E2 / \
        (E1 + eps) if np.isfinite(E1) and np.isfinite(E2) else np.nan

    # Second harmonic peak
    band_2nd = (psd_freqs >= 1.8 * f0) & (psd_freqs <= 2.2 * f0)

    if band_2nd.any():
        psd_2nd = psd[band_2nd]
        freqs_2nd = psd_freqs[band_2nd]

        idx_2nd = np.argmax(psd_2nd)
        f_max_2nd = freqs_2nd[idx_2nd]
        max_power_2nd = psd_2nd[idx_2nd]
    else:
        f_max_2nd = np.nan
        max_power_2nd = np.nan

    # Normalized PSD inside analysis band
    psd_sum_b = np.sum(psd_b) + eps
    psd_norm_b = psd_b / psd_sum_b

    spectral_centroid = np.sum(freqs_b * psd_b) / psd_sum_b

    spectral_entropy = -np.sum(psd_norm_b * np.log(psd_norm_b + eps))
    spectral_entropy_norm = spectral_entropy / np.log(len(psd_norm_b))

    mean_psd = np.mean(psd_b)
    std_psd = np.std(psd_b)
    var_psd = np.var(psd_b)

    log_psd = np.log10(psd_b + eps)
    kurtosis_psd = kurtosis(log_psd, fisher=True, bias=False)

    return {
        # "f0": f0, # f0 coincides with F_MAX_1st
        "E1_norm": E1_norm,
        "E2_norm": E2_norm,
        "harmonic_ratio": harmonic_ratio,
        "spectral_centroid": spectral_centroid,
        "spectral_entropy": spectral_entropy_norm,
        "MAX_POWER_1st": max_power_1st,
        "F_MAX_1st": f_max_1st,
        "MAX_POWER_2nd": max_power_2nd,
        "F_MAX_2nd": f_max_2nd,
        "MEAN_PSD": mean_psd,
        "STD_PSD": std_psd,
        "VAR_PSD": var_psd,
        "KUR_PSD": kurtosis_psd,
        # "TOTAL_POWER_0p5_5Hz": total_power, # highly-correlated to MEAN_PSD
    }


def old_extract_features_group4_hrv(ppg_frame, peaks, fs, show=False):
    # Compute HRV features as dataframe with 1 row and 25 features
    # https://neuropsychology.github.io/NeuroKit/functions/hrv.html#neurokit2.hrv.hrv_time

    hrv_features_df = nk.hrv_time(peaks, sampling_rate=fs, show=False)

    if (show):
        print(peaks)
        print(hrv_features_df)
        plt.plot(ppg_frame)
        plt.scatter(peaks, ppg_frame[peaks], c='r')
        plt.show()

    # Convert to dictionary
    if hrv_features_df is not None and not hrv_features_df.empty:
        hrv_dict = hrv_features_df.iloc[0].to_dict()

    return hrv_dict


def extract_features_group4_prv(ppg_frame, peaks, fs, show=False, min_peaks=5):
    '''
    This method uses Neurokit2 to compute "HRV" features from PPG peaks.
    Because these are not strictly HRV features from ECG, we call them PRV features: pulse rate variability features.
    '''
    ppg_frame = np.asarray(ppg_frame, dtype=float)

    if ppg_frame.ndim != 1:
        raise ValueError(
            f"Expected 1-D PPG signal, got shape {ppg_frame.shape}")

    if not np.all(np.isfinite(ppg_frame)):
        raise ValueError("PPG signal contains NaN or Inf values")

    peaks = np.asarray(peaks, dtype=int)

    if peaks.ndim != 1:
        raise ValueError(f"Expected 1-D peaks array, got shape {peaks.shape}")

    peaks = np.unique(peaks)

    if len(peaks) < min_peaks:
        return {"PRV_N_PEAKS": len(peaks)}

    if np.any(peaks < 0) or np.any(peaks >= len(ppg_frame)):
        raise ValueError("Peaks contain indices outside PPG signal range")

    if np.any(np.diff(peaks) <= 0):
        raise ValueError("Peaks must be sorted and unique")

    # Basic interval sanity check
    ibi_sec = np.diff(peaks) / fs
    heart_rate_bpm = 60.0 / ibi_sec

    valid_hr = (heart_rate_bpm >= 40) & (heart_rate_bpm <= 180)

    if np.mean(valid_hr) < 0.8:
        return {
            "PRV_N_PEAKS": len(peaks),
            "PRV_VALID_INTERVAL_RATIO": np.mean(valid_hr),
        }

    hrv_features_df = nk.hrv_time(peaks, sampling_rate=fs, show=show)

    prv_dict = {"PRV_N_PEAKS": len(peaks)}
    prv_dict["PRV_VALID_INTERVAL_RATIO"] = np.mean(valid_hr)

    if hrv_features_df is not None and not hrv_features_df.empty:
        for key, value in hrv_features_df.iloc[0].to_dict().items():
            prv_dict[f"PRV_{key}"] = value

    features_to_be_removed = ['PRV_HRV_SDANN2', 'PRV_HRV_SDNNI2', 'PRV_HRV_SDANN5',
                              'PRV_HRV_SDNNI5', 'PRV_HRV_SDANN1', 'PRV_HRV_SDNNI1',
                              'PRV_VALID_INTERVAL_RATIO']

    for feature in features_to_be_removed:
        if feature in prv_dict:
            del prv_dict[feature]

    if show:
        print(peaks)
        print(hrv_features_df)
        plt.plot(ppg_frame)
        plt.scatter(peaks, ppg_frame[peaks], c="r")
        plt.title("PPG peaks used for PRV features")
        plt.show()

    return prv_dict


def old_test_all_feature_and_quality_pipelines(dataset_config_file: str) -> None:
    '''DEPRECATED:
    a limitation here is that the output folder needs to be the name of the pipeline'''

    # all signal pipelines:
    for pipeline in list(SIGNAL_PIPELINES.keys()):
        print(f"Testing pipeline: {pipeline}")
        ppg_signal_processing_pipeline(
            dataset_config_file, pipeline=pipeline,
            input_waveform_id="raw", output_waveform_id=pipeline
        )
    # all quality pipelines:
    for pipeline in list(QUALITY_PIPELINES.keys()):
        print(f"Testing pipeline: {pipeline}")
        input_waveform_id = datasetConfig.get_value(
            "PPG_SQI_INPUT_WAVEFORM")
        ppg_signal_processing_pipeline(
            dataset_config_file, pipeline=pipeline,
            input_waveform_id=input_waveform_id, output_waveform_id=pipeline
        )


# ======================================================
# Processing functions already defined in this module
# Define the dictionary after all the functions
# ======================================================
SIGNAL_PIPELINES = {
    "bandpass": ppg_bandpass_waveform_processing,
    "inversion": ppg_inversion_waveform_processing,
    # "reverse": ppg_reverse_waveform_processing,
    "savitzky": ppg_savitzky_waveform_processing,
    "neurokit": ppg_neurokit_waveform_processing,
    "lowpass": ppg_lowpass_waveform_processing
}

QUALITY_PIPELINES = {
    "sqi_sumall": ppg_sqi_sumall_waveform_processing,
    "sqi_pyppg": ppg_sqi_pyppg_waveform_processing,
    "sqi_nkcustom": ppg_sqi_nkcustom_waveform_processing,
    "sqi_neurokit": ppg_sqi_neurokit_waveform_processing
}

FEATURE_PIPELINES = {
    "single_vector": ppg_single_vector_feature_extraction,
    "all": ppg_all_feature_extraction,
}

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

    dataset_config_file = args.json_file

    datasetConfig = DatasetConfig(dataset_config_file)

    if "ppg" not in datasetConfig.modalities:
        print(
            "No PPG modality found in the dataset configuration. Skipping PPG processing.")
        exit(0)

    required_ppg_fs = datasetConfig.get_ppg_fs()

    # signal processing pipeline for PPG signals
    signal_pipeline = datasetConfig.get_value(
        "PPG_SIGNAL_PROCESSING_PIPELINE")

    output_waveform_id = datasetConfig.get_value(
        "PPG_SIGNAL_OUTPUT_WAVEFORM", signal_pipeline)

    print("Processing PPG signals with pipeline", signal_pipeline,
          "to generate output_waveform_id",
          output_waveform_id)
    ppg_signal_processing_pipeline(
        dataset_config_file, input_waveform_id="raw", pipeline=signal_pipeline, output_waveform_id=output_waveform_id
    )

    # SQI processing pipeline for PPG signals
    sqi_pipeline = datasetConfig.get_value(
        "PPG_SQI_PROCESSING_PIPELINE", "no_sqi")
    if sqi_pipeline == "no_sqi":
        print("Skipping SQI waveform creation for PPG signals.")
    else:
        input_waveform_id = datasetConfig.get_value(
            "PPG_SQI_INPUT_WAVEFORM")
        output_waveform_id = datasetConfig.get_value(
            "PPG_SQI_OUTPUT_WAVEFORM", sqi_pipeline)
        print("Creating SQI waveforms for PPG signals using pipeline", sqi_pipeline, "and input_waveform_id:",
              input_waveform_id, "and output_sqi_id:", sqi_pipeline)
        ppg_signal_processing_pipeline(
            dataset_config_file, input_waveform_id=input_waveform_id,
            pipeline=sqi_pipeline,
            output_waveform_id=output_waveform_id
        )
