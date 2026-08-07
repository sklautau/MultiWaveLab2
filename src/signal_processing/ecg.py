"""
ECG signal processing.
This module provides functions for processing ECG signals, including filtering,
outlier removal, and feature extraction. It also includes pipelines for
processing ECG files in a dataset.
"""

import argparse
import traceback
import numpy as np
import os
import matplotlib.pyplot as plt
import neurokit2 as nk
from typing import Dict, Any
from scipy.signal import welch
from scipy.stats import kurtosis


from datasets_util.util_features import append_identification
from datasets_util.naming_conventions import DatasetConfig
from datasets_util.waveform_files import read_sigmf_file
from datasets_util.waveform_files import save_sigmf_signal_rf32_le
from signal_processing.cross_features import safe_mean, safe_std
from signal_processing.signal_utils import remove_outliers, estimate_amplitude_bits, plot_quantization_levels, plot_histogram, plot_histogram_robust, _energy_in_band, _reverse_signal
# _calculate_entropy, _calculate_spectral_entropy, _calculate_spectral_kurtosis, _calculate_spectral_skewness

# ======================================================
# Parameters
# ======================================================
SHOULD_PLOT = False  # False to disable plotting
# for naming convention when saving processed files (e.g., "file_id_filtered.csv")
REQUIRED_FS = 500  # assumed sampling frequency (Hz)
DEBUGGING = False  # True to enable debug prints

SUBJECT_IDS_TO_PLOT = {
    # "ieb_02", # several files
    "ieb_07"
    # "ieb_02",
    # "ieb_03",
    # "ieb_10",
}

FILE_IDS_TO_PLOT = {
    "file_id20",
    # "file_id80", # bad signal
    "file_id90"
}


def twos_complement_to_signed(x, bits):
    """
    Convert unsigned integers encoded in two's complement to signed integers.

    Parameters
    ----------
    x : scalar or array-like
        Unsigned integer values.

    bits : int
        Number of bits of the representation.

    Returns
    -------
    ndarray or scalar
        Signed integer values.
    """
    x = np.asarray(x)

    if np.any(x < 0):
        raise ValueError("Input must contain only non-negative integers.")

    max_unsigned = (1 << bits) - 1
    if np.any(x > max_unsigned):
        raise ValueError(
            f"Values exceed the maximum representable with {bits} bits ({max_unsigned})."
        )

    sign_bit = 1 << (bits - 1)

    return np.where(x < sign_bit, x, x - (1 << bits))


def _load_debug_signal_csv(file_path: str) -> np.ndarray:
    """Load a 1D numeric ECG signal from CSV/TSV text files with robust encoding parsing."""
    last_error = None
    encodings = ("utf-8", "utf-8-sig", "utf-16", "latin1")
    delimiters = (",", ";", "\t", None)

    for encoding in encodings:
        for delimiter in delimiters:
            try:
                data = np.genfromtxt(
                    file_path,
                    delimiter=delimiter,
                    encoding=encoding,
                    invalid_raise=False,
                )
                arr = np.asarray(data, dtype=float).reshape(-1)
                arr = arr[~np.isnan(arr)]

                if arr.size > 0:
                    return arr
            except Exception as exc:
                last_error = exc

    raise ValueError(
        f"Failed to parse numeric debug ECG file: {file_path}. Last error: {last_error}"
    )

# ======================================================
# Visualization
# ======================================================


def __plot_input_output_waveforms(input_wav, output_wav, fs, title, title1="Raw signal", title2="Filtered signal") -> None:
    t = np.arange(len(input_wav)) / fs

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

    axes[0].plot(t, input_wav, lw=0.8, color="#79B1CE")
    axes[0].set_title(title1)
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, output_wav, lw=0.8, color="#332288")
    axes[1].set_title(title2)
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    axes[0].set_xlabel("Time (s)")
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


def ecg_sqi_neurokit_waveform_processing(input_signal: np.ndarray, fs: int) -> np.ndarray:
    try:
        # "dissimilarity" or "templatematch" or "ho2025" or others (see https://neuropsychology.github.io/NeuroKit/functions/ecg.html#ecg-quality)
        # method = "templatematch"
        method = "averageQRS"
        ecq_quality = nk.ecg.ecg_quality(input_signal, sampling_rate=fs,
                                         method=method)
    except Exception as e:
        # `templatematch` and `dissimilarity` require at least one detected peak.
        # fallback to zeros if error (e.g., no detected peaks) occurs
        ecq_quality = np.zeros_like(input_signal)
        print(f"****Error**** estimating ECG SQI: {e}")
        traceback.print_exc()

    return ecq_quality


def ecg_neurokit_waveform_processing(raw: np.ndarray, fs: int) -> np.ndarray:

    if DEBUGGING:
        debug_csv_path = r"C:\git_sofis\tcc_guilherme\ecg.csv"
        # Optional debug override: only replaces raw when file parsing succeeds.
        if os.path.exists(debug_csv_path):
            try:
                raw = _load_debug_signal_csv(debug_csv_path)
                print(
                    f"Loaded debug ECG from {debug_csv_path} with {len(raw)} samples")
            except Exception as exc:
                print(
                    f"Skipping debug CSV override for {debug_csv_path}: {exc}")
        else:
            print(
                f"Debug CSV not found, keeping SigMF ECG signal: {debug_csv_path}")

        raw = twos_complement_to_signed(raw, bits=20)

        # Estimate amplitude bits (for quantization analysis)
        quantization_info = estimate_amplitude_bits(raw)
        observed_bits = quantization_info["observed_bits"]
        print("Number of bits to quantize amplitude values =", observed_bits)

    if SHOULD_PLOT:

        plot_both = False
        plot_histogram_robust(raw, bins=100, percentile_range=(0.2, 99.8))
        plot_histogram(
            raw, title="ECG Signal Histogram", bins="100", grid=True, show_me=not plot_both, logy=True)
        if plot_both:
            # takes longer time to compute
            plot_quantization_levels(
                raw, title="ECG Signal Quantization Levels", show_me=True)

    #  If signals are reversed in acquisition, uncomment:
    # raw = np.flipud(raw)

    # First, remove outliers to prevent spreading of artifacts during filtering
    raw_removed_outliers, outlier_mask = remove_outliers(raw, fs=fs, k=8.0)

    if SHOULD_PLOT:
        __plot_input_output_waveforms(
            raw,
            raw_removed_outliers,
            fs,
            title="Result of outlier removal for ECG signal",
            title1="Raw ECG signal",
            title2="ECG signal after outlier removal"
        )

    # see https://neuropsychology.github.io/NeuroKit/functions/ecg.html#ecg-process
    # extract the cleaned ECG signal from the DataFrame
    # note that ecg_process has a "cleaning" stage, and
    # we also applied a bandpass Butterworth filter
    # I will disable it because it's breaking for file_id80, but you can experiment with it if you want
    if False:
        signals_df, info = nk.ecg.ecg_process(raw_removed_outliers, sampling_rate=fs,
                                              method="neurokit")
        filt = signals_df["ECG_Clean"]
    else:
        filt = nk.ecg.ecg_clean(raw_removed_outliers, sampling_rate=fs,
                                method="neurokit")
    return np.asarray(filt)


def ecg_signal_processing_pipeline(dataset_config_file: str, input_waveform_id: str,
                                   pipeline: str,
                                   output_waveform_id: str) -> None:
    """
    Use this method for both waveform and SQI processing pipelines. The pipeline is selected by the "pipeline" argument.
    """

    is_valid_signal_pipeline = pipeline in SIGNAL_PIPELINES
    is_valid_quality_pipeline = pipeline in QUALITY_PIPELINES

    if is_valid_quality_pipeline and is_valid_signal_pipeline:
        raise ValueError(
            f"Pipeline '{pipeline}' is defined in both SIGNAL_PIPELINES and QUALITY_PIPELINES. Please choose unique names."
        )

    if is_valid_signal_pipeline:
        processing_function = SIGNAL_PIPELINES[pipeline]
    elif is_valid_quality_pipeline:
        processing_function = QUALITY_PIPELINES[pipeline]
    else:
        valid_signal = ", ".join(SIGNAL_PIPELINES.keys())
        valid_quality = ", ".join(QUALITY_PIPELINES.keys())
        raise ValueError(
            f"Unknown pipeline '{pipeline}'. "
            f"Valid signal processing options are: {valid_signal}. "
            f"Valid quality processing options are: {valid_quality}."
        )

    if is_valid_signal_pipeline and input_waveform_id != "raw":
        raise ValueError(
            f"Signal processing pipeline '{pipeline}' requires input_waveform_id='raw', but got '{input_waveform_id}'."
        )

    datasetConfig = DatasetConfig(dataset_config_file)
    required_ecg_fs = int(datasetConfig.get_value("ECG_FS", REQUIRED_FS))

    df = datasetConfig.get_dataset_info_dataframe()
    df = df[df["modality"].str.contains("ecg")]

    file_counter = 0
    for counter, row in df.iterrows():
        file_id = row["file_id"]
        participant_id = row["participant_id"]

        if input_waveform_id == "raw":
            input_complete_path = datasetConfig.get_raw_complete_path(file_id)
        else:
            input_complete_path = datasetConfig.get_gen_complete_path(
                file_id, input_waveform_id)
        print(f"Input ECG: {input_complete_path} (file_id={file_id})")

        input_waveform, input_metadata = read_sigmf_file(input_complete_path)
        input_waveform = np.asarray(input_waveform, dtype=float)

        fs = input_metadata["global"]["core:sample_rate"]
        if fs != required_ecg_fs:
            raise ValueError(
                f"Expected sampling frequency {required_ecg_fs} Hz, but got {fs} Hz in file {input_complete_path}")

        output_waveform = processing_function(input_waveform, required_ecg_fs)

        output_filename = datasetConfig.get_gen_complete_path(
            file_id, output_waveform_id)

        output_dir = os.path.dirname(output_filename)
        os.makedirs(output_dir, exist_ok=True)

        save_sigmf_signal_rf32_le(
            output_waveform, input_metadata, output_filename)
        print(f"Output ECG: {output_filename} (file_id={file_id})")

        file_counter += 1

        if SHOULD_PLOT and (participant_id in SUBJECT_IDS_TO_PLOT or file_id in FILE_IDS_TO_PLOT):
            __plot_input_output_waveforms(
                input_waveform,
                output_waveform,
                required_ecg_fs,
                title=f"Subject {participant_id} in the {output_waveform_id} pipeline",
                title1=input_waveform_id,
                title2=output_waveform_id
            )

    print(
        f"\nFinished processing {file_counter} files from {counter+1} rows in the dataset.")


def old_ecg_filtering_pipeline(dataset_config_file: str) -> None:
    """Backward-compatible wrapper for ECG signal processing."""
    ecg_signal_processing_pipeline(
        dataset_config_file,
        input_waveform_id="raw",
        pipeline="bandpass",
        output_waveform_id="filtered",
    )


def old_ecg_quality_pipeline(dataset_config_file: str) -> None:
    """Backward-compatible wrapper for ECG SQI processing."""
    ecg_signal_processing_pipeline(
        dataset_config_file,
        input_waveform_id="filtered",
        pipeline="sqi_sumall",
        output_waveform_id="quality",
    )


def extract_features_group3_spectral(ecg, fs):
    '''
    Spectral features.
    We cannot trust raw spectral power features unless every PPG segment has
    been amplitude-normalized and quality-controlled. Otherwise, the model may
    learn measurement conditions rather than glucose-related information.
    '''
    ecg = np.asarray(ecg, dtype=float)

    if ecg.ndim != 1:
        raise ValueError(f"Expected 1-D PPG signal, got shape {ecg.shape}")

    if len(ecg) < 3:
        raise ValueError(f"PPG signal too short: len={len(ecg)}")

    if not np.all(np.isfinite(ecg)):
        raise ValueError("PPG signal contains NaN or Inf values")

    # Remove DC level to avoid dominance of baseline in PSD
    ecg = ecg - np.mean(ecg)

    eps = 1e-12

    # Welch parameters
    min_cycles = 5
    f_min = 0.7
    nperseg_min = int(min_cycles * fs / f_min)

    # Target frequency resolution around 0.05 Hz
    nperseg_target = int(fs / 0.05)

    nperseg = min(len(ecg), max(nperseg_min, nperseg_target))

    if nperseg < 8:
        raise ValueError(f"nperseg too small: {nperseg}")

    psd_freqs, psd = welch(
        ecg,
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


def ecg_all_feature_extracion(ecg: np.ndarray, fs: int) -> Dict[str, Any]:
    '''
    All ECG processing depends on NeuroKit2.
    Warning: This function uses NeuroKit2 to extract features from ECG signals. NeuroKit2 may raise
    warnings for certain signals, especially if they are of low quality or have artifacts. These warnings are caught and ignored
    in this function to ensure that feature extraction continues without interruption.

    Extracts ECG-derived features from a 1-D ECG signal using NeuroKit2.
    The method returns a dictionary containing:
        - heart-rate mean and standard deviation
        - NeuroKit HRV features
        - RR-interval mean and standard deviation
        - a simple ECG quality indicator
    '''
    ecg = np.asarray(ecg, dtype=float).ravel()
    ecg = ecg[np.isfinite(ecg)]

    # Keep a stable set of core ECG features even when NeuroKit fails.
    features: Dict[str, Any] = {
        "ecg_hr_std": np.nan,
        "ecg_rr_mean": np.nan,
        "ecg_rr_std": np.nan,
        "ecg_sqi": np.nan,
    }

    if ecg.size < 3:
        return features

    try:
        signals, info = nk.ecg_process(ecg, sampling_rate=fs)
        rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)

        if rpeaks.size >= 2:
            if DEBUGGING:
                print("Number of R peaks:", len(rpeaks))
                duration_sec = (int(rpeaks[-1]) - int(rpeaks[0])) / fs
                print("Duration (s):", duration_sec)

            rr = np.diff(rpeaks) / float(fs)
            features["ecg_rr_mean"] = safe_mean(rr)
            features["ecg_rr_std"] = safe_std(rr)

        ecg_rate = signals["ECG_Rate"].to_numpy(dtype=float)
        features["ecg_hr_std"] = safe_std(ecg_rate)
        # features["ecg_sqi"] = np.mean((ecg_rate > 40) & (ecg_rate < 180))

        if rpeaks.size >= 3:
            import warnings
            from neurokit2.misc import NeuroKitWarning

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*DFA_alpha2 related indices will not be calculated.*",
                    category=NeuroKitWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=".*invalid value encountered in scalar divide.*",
                    category=RuntimeWarning,
                    module=r".*entropy_multiscale",
                )
                warnings.filterwarnings(
                    "ignore",
                    message=".*divide by zero encountered in divide.*",
                    category=RuntimeWarning,
                    module=r".*optim_complexity_k",
                )
                warnings.filterwarnings(
                    "ignore",
                    message=".*invalid value encountered in multiply.*",
                    category=RuntimeWarning,
                    module=r".*optim_complexity_k",
                )

                hrv = nk.hrv(rpeaks, sampling_rate=fs)
                for col in hrv.columns:
                    features[f"ecg_{col.lower()}"] = hrv.iloc[0][col]

    except Exception as e:
        # NeuroKit can fail on short/noisy segments. Keep pipeline running and return NaN-based core features.
        if DEBUGGING:
            print(
                f"Warning: ECG NeuroKit feature extraction fallback used: {e}")

    try:
        spectral_features = extract_features_group3_spectral(ecg, fs)
        spectral_features = append_identification(
            spectral_features, "ecg", "s2")
        features.update(spectral_features)
    except Exception as e:
        if DEBUGGING:
            print(
                f"Warning: ECG spectral features could not be extracted: {e}")

    return features


def extract_ecg_features(
    datasetConfig: DatasetConfig,
    ecg: np.ndarray,
    fs: int
) -> Dict[str, Any]:
    pipeline = datasetConfig.get_value(
        "ECG_FEATURE_EXTRACTION_PIPELINE", "all")

    try:
        processing_function = FEATURE_PIPELINES[pipeline]
    except KeyError:
        valid = ", ".join(FEATURE_PIPELINES.keys())
        raise ValueError(
            f"Unknown waveform_id '{pipeline}'. "
            f"Valid options are: {valid}"
        )

    return processing_function(ecg, fs)


SIGNAL_PIPELINES = {
    "neurokit": ecg_neurokit_waveform_processing
}

QUALITY_PIPELINES = {
    "sqi_neurokit": ecg_sqi_neurokit_waveform_processing
}

FEATURE_PIPELINES = {
    "all": ecg_all_feature_extracion
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
    if "ecg" not in datasetConfig.modalities:
        print(
            "No ECG modality found in the dataset configuration. Skipping ECG processing.")
        exit(0)

    signal_pipeline = datasetConfig.get_value("ECG_SIGNAL_PROCESSING_PIPELINE")
    output_waveform_id = datasetConfig.get_value(
        "ECG_SIGNAL_OUTPUT_WAVEFORM", signal_pipeline)

    print("Processing ECG signals with pipeline", signal_pipeline,
          "to generate output_waveform_id", output_waveform_id)
    ecg_signal_processing_pipeline(
        dataset_config_file,
        input_waveform_id="raw",
        pipeline=signal_pipeline,
        output_waveform_id=output_waveform_id,
    )

    sqi_pipeline = datasetConfig.get_value(
        "ECG_SQI_PROCESSING_PIPELINE", "no_sqi")
    if sqi_pipeline == "no_sqi":
        print("Skipping SQI waveform creation for ECG signals.")
    else:
        input_waveform_id = datasetConfig.get_value(
            "ECG_SQI_INPUT_WAVEFORM", "raw")
        output_waveform_id = datasetConfig.get_value(
            "ECG_SQI_OUTPUT_WAVEFORM", sqi_pipeline)
        print("Creating SQI waveforms for ECG signals using pipeline", sqi_pipeline,
              "and input_waveform_id:", input_waveform_id,
              "and output_waveform_id:", output_waveform_id)
        ecg_signal_processing_pipeline(
            dataset_config_file,
            input_waveform_id=input_waveform_id,
            pipeline=sqi_pipeline,
            output_waveform_id=output_waveform_id,
        )
