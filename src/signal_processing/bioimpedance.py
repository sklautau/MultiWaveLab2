"""
Signal processing over bioimpedance data (time and frequency domain).

Rather than simply suppressing warnings, the feature extractor is made robust by construction. That means every feature should first check whether the mathematical conditions for computing it are satisfied. For example:
correlation: both variables must have nonzero variance;
polynomial fitting: enough distinct points and sufficient variation;
Hjorth mobility/complexity: denominator variances must be nonzero;
ratios: denominator must be larger than a small threshold;
centroid/spread: total power must be positive.

This approach avoids warnings, produces meaningful NaN values when a feature is undefined,
"""

import argparse

import pandas as pd
import numpy as np
from io import StringIO
import os
import matplotlib.pyplot as plt
from typing import Dict, Any
from scipy import signal, stats

from datasets_util.naming_conventions import DatasetConfig
from signal_processing.cross_features import safe_mean, safe_std
from signal_processing.signal_utils import savitzky_golay_filtering

SHOULD_PLOT = False  # Set to True to enable plotting of raw vs filtered signals
APPLY_PHASE_UNWRAP = True  # Set to True to apply phase unwrapping before smoothing
# values obtained by observing IEB3 bioimp values:
MIN_SQI_MSE = 2e-6
MAX_SQI_MSE = 0.002


def _safe_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.ravel()
    return x


def _safe_stat(func, x: np.ndarray, default: float = np.nan) -> float:
    try:
        x = np.asarray(x)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return default
        return float(func(x))
    except Exception:
        return default


def safe_corrcoef(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    if np.std(x) < 1e-12:
        return np.nan

    if np.std(y) < 1e-12:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a / (b + eps))


def _add_basic_stats(
    features: Dict[str, Any],
    prefix: str,
    x: np.ndarray,
) -> None:
    x = np.asarray(x)
    x = x[np.isfinite(x)]

    if x.size == 0:
        for name in [
            "mean", "std", "min", "max", "ptp", "median",
            "iqr", "rms", "skew", "kurtosis", "energy",
        ]:
            features[f"{prefix}_{name}"] = np.nan
        return

    features[f"{prefix}_mean"] = float(np.mean(x))
    features[f"{prefix}_std"] = float(np.std(x))
    features[f"{prefix}_min"] = float(np.min(x))
    features[f"{prefix}_max"] = float(np.max(x))


def old_add_basic_stats(
    features: Dict[str, Any],
    prefix: str,
    x: np.ndarray,
) -> None:
    x = np.asarray(x)
    x = x[np.isfinite(x)]

    if x.size == 0:
        for name in [
            "mean", "std", "min", "max", "ptp", "median",
            "iqr", "rms", "skew", "kurtosis", "energy",
        ]:
            features[f"{prefix}_{name}"] = np.nan
        return

    features[f"{prefix}_mean"] = float(np.mean(x))
    features[f"{prefix}_std"] = float(np.std(x))
    features[f"{prefix}_min"] = float(np.min(x))
    features[f"{prefix}_max"] = float(np.max(x))


def _poly_features(
    features: Dict[str, Any],
    prefix: str,
    freq: np.ndarray,
    x: np.ndarray,
) -> None:
    mask = np.isfinite(freq) & np.isfinite(x)
    freq = freq[mask]
    x = x[mask]

    if x.size < 2:
        features[f"{prefix}_linear_slope"] = np.nan
        features[f"{prefix}_linear_intercept"] = np.nan
        features[f"{prefix}_quad_curvature"] = np.nan
        return

    p1 = np.polyfit(freq, x, 1)
    features[f"{prefix}_linear_slope"] = float(p1[0])
    features[f"{prefix}_linear_intercept"] = float(p1[1])

    if x.size >= 3:
        p2 = np.polyfit(freq, x, 2)
        features[f"{prefix}_quad_curvature"] = float(p2[0])
    else:
        features[f"{prefix}_quad_curvature"] = np.nan


def old_bioimp_all_feature_extracion(bioimp: np.ndarray, fs: int) -> Dict[str, Any]:
    """
    Extract generic frequency-domain bioimpedance features.

    Parameters
    ----------
    bioimp : np.ndarray
        Complex-valued bioimpedance frequency response Z(f).
        Shape should be (n_frequencies,). Values may be complex.
    fs : int
        Sampling frequency or maximum frequency used to construct a surrogate
        frequency axis. If the true frequency vector is available, it is better
        to adapt this function to receive it explicitly.

    Returns
    -------
    Dict[str, Any]
        Dictionary with scalar features.
    """
    features: Dict[str, Any] = {}

    z = _safe_array(bioimp).astype(np.complex128)
    n = z.size

    if n == 0:
        return {"bioimp_valid": False, "bioimp_n_points": 0}

    features["bioimp_valid"] = True
    features["bioimp_n_points"] = int(n)

    eps = 1e-12

    # Surrogate frequency axis.
    # Prefer replacing this with the true measured frequencies if available.
    freq = np.linspace(0.0, float(fs) / 2.0, n)
    freq_idx = np.arange(n, dtype=float)

    real = np.real(z)
    imag = np.imag(z)
    mag = np.abs(z)
    phase = np.unwrap(np.angle(z))

    log_mag = np.log(mag + eps)
    mag_db = 20.0 * np.log10(mag + eps)

    # Basic component statistics
    _add_basic_stats(features, "bioimp_real", real)
    _add_basic_stats(features, "bioimp_imag", imag)
    _add_basic_stats(features, "bioimp_mag", mag)
    _add_basic_stats(features, "bioimp_phase", phase)
    _add_basic_stats(features, "bioimp_log_mag", log_mag)
    _add_basic_stats(features, "bioimp_mag_db", mag_db)

    # Original-style global features
    features["bioimp_mean"] = complex(np.mean(z))
    features["bioimp_std"] = float(np.std(z))
    features["bioimp_peak_to_peak_mag"] = float(np.ptp(mag))

    # Derivatives
    for name, x in [
        ("real", real),
        ("imag", imag),
        ("mag", mag),
        ("phase", phase),
        ("log_mag", log_mag),
    ]:
        d1 = np.gradient(x)
        d2 = np.gradient(d1)

        _add_basic_stats(features, f"bioimp_d1_{name}", d1)
        _add_basic_stats(features, f"bioimp_d2_{name}", d2)

    # Trend and curvature
    for name, x in [
        ("real", real),
        ("imag", imag),
        ("mag", mag),
        ("phase", phase),
        ("log_mag", log_mag),
        ("mag_db", mag_db),
    ]:
        _poly_features(features, f"bioimp_{name}_vs_freq", freq, x)
        _poly_features(features, f"bioimp_{name}_vs_index", freq_idx, x)

    # Entropy-like descriptors from magnitude energy
    power = mag**2
    total_power = float(np.sum(power))

    if total_power > eps:
        p = power / total_power
        features["bioimp_entropy"] = float(stats.entropy(p + eps))
        features["bioimp_norm_entropy"] = float(
            stats.entropy(p + eps) / np.log(n))
        features["bioimp_centroid"] = float(np.sum(freq * power) / total_power)
        features["bioimp_centroid_index"] = float(
            np.sum(freq_idx * power) / total_power)

        centroid = features["bioimp_centroid"]
        centroid_idx = features["bioimp_centroid_index"]

        features["bioimp_spread"] = float(
            np.sqrt(np.sum(((freq - centroid) ** 2) * power) / total_power)
        )
        features["bioimp_spread_index"] = float(
            np.sqrt(np.sum(((freq_idx - centroid_idx) ** 2) * power) / total_power)
        )
    else:
        features["bioimp_entropy"] = np.nan
        features["bioimp_norm_entropy"] = np.nan
        features["bioimp_centroid"] = np.nan
        features["bioimp_centroid_index"] = np.nan
        features["bioimp_spread"] = np.nan
        features["bioimp_spread_index"] = np.nan

    # Areas under curves
    features["bioimp_auc_mag"] = float(np.trapz(mag, freq))
    features["bioimp_auc_phase"] = float(np.trapz(phase, freq))
    features["bioimp_auc_real"] = float(np.trapz(real, freq))
    features["bioimp_auc_imag"] = float(np.trapz(imag, freq))
    features["bioimp_auc_abs_imag"] = float(np.trapz(np.abs(imag), freq))

    # Dynamic range
    features["bioimp_mag_dynamic_range"] = _safe_div(
        float(np.max(mag)), float(np.min(mag)))
    features["bioimp_mag_dynamic_range_db"] = float(
        20.0 * np.log10(features["bioimp_mag_dynamic_range"] + eps)
    )

    # Phase-angle specific features
    features["bioimp_phase_angle_mean_deg"] = float(np.mean(np.degrees(phase)))
    features["bioimp_phase_angle_std_deg"] = float(np.std(np.degrees(phase)))
    features["bioimp_phase_angle_max_deg"] = float(np.max(np.degrees(phase)))
    features["bioimp_phase_angle_min_deg"] = float(np.min(np.degrees(phase)))

    # Characteristic frequency-like descriptors
    idx_max_mag = int(np.argmax(mag))
    idx_min_mag = int(np.argmin(mag))
    idx_max_abs_imag = int(np.argmax(np.abs(imag)))
    idx_most_negative_imag = int(np.argmin(imag))
    idx_max_phase = int(np.argmax(phase))
    idx_min_phase = int(np.argmin(phase))

    features["bioimp_freq_max_mag"] = float(freq[idx_max_mag])
    features["bioimp_freq_min_mag"] = float(freq[idx_min_mag])
    features["bioimp_freq_max_abs_imag"] = float(freq[idx_max_abs_imag])
    features["bioimp_freq_most_negative_imag"] = float(
        freq[idx_most_negative_imag])
    features["bioimp_freq_max_phase"] = float(freq[idx_max_phase])
    features["bioimp_freq_min_phase"] = float(freq[idx_min_phase])

    features["bioimp_max_mag"] = float(mag[idx_max_mag])
    features["bioimp_min_mag"] = float(mag[idx_min_mag])
    features["bioimp_max_abs_imag"] = float(np.abs(imag[idx_max_abs_imag]))
    features["bioimp_most_negative_imag"] = float(imag[idx_most_negative_imag])
    features["bioimp_max_phase"] = float(phase[idx_max_phase])
    features["bioimp_min_phase"] = float(phase[idx_min_phase])

    # Arc length / shape complexity
    if n >= 2:
        features["bioimp_mag_arc_length"] = float(
            np.sum(np.sqrt(np.diff(freq) ** 2 + np.diff(mag) ** 2))
        )
        features["bioimp_phase_arc_length"] = float(
            np.sum(np.sqrt(np.diff(freq) ** 2 + np.diff(phase) ** 2))
        )
        features["bioimp_complex_plane_arc_length"] = float(
            np.sum(np.abs(np.diff(z)))
        )
    else:
        features["bioimp_mag_arc_length"] = np.nan
        features["bioimp_phase_arc_length"] = np.nan
        features["bioimp_complex_plane_arc_length"] = np.nan

    # Hjorth parameters on magnitude and phase
    for name, x in [("mag", mag), ("phase", phase)]:
        activity = float(np.var(x))
        d1 = np.diff(x)
        d2 = np.diff(d1)

        if activity > eps and d1.size > 0 and np.var(d1) > eps:
            mobility = float(np.sqrt(np.var(d1) / activity))
            complexity = float(np.sqrt(np.var(d2) / np.var(d1)) /
                               (mobility + eps)) if d2.size > 0 else np.nan
        else:
            mobility = np.nan
            complexity = np.nan

        features[f"bioimp_{name}_hjorth_activity"] = activity
        features[f"bioimp_{name}_hjorth_mobility"] = mobility
        features[f"bioimp_{name}_hjorth_complexity"] = complexity

    # Peak features
    peaks, props = signal.find_peaks(mag)

    features["bioimp_num_mag_peaks"] = int(len(peaks))

    if len(peaks) > 0:
        features["bioimp_mag_peak_mean"] = float(np.mean(mag[peaks]))
        features["bioimp_mag_peak_max"] = float(np.max(mag[peaks]))
        features["bioimp_freq_first_mag_peak"] = float(freq[peaks[0]])
        features["bioimp_freq_strongest_mag_peak"] = float(
            freq[peaks[np.argmax(mag[peaks])]])
    else:
        features["bioimp_mag_peak_mean"] = np.nan
        features["bioimp_mag_peak_max"] = np.nan
        features["bioimp_freq_first_mag_peak"] = np.nan
        features["bioimp_freq_strongest_mag_peak"] = np.nan

    # Band features: low / mid / high thirds
    thirds = np.array_split(np.arange(n), 3)

    band_values = {}
    for band_name, idx in zip(["low", "mid", "high"], thirds):
        if idx.size == 0:
            band_values[band_name] = np.nan
            features[f"bioimp_mag_{band_name}_mean"] = np.nan
            features[f"bioimp_real_{band_name}_mean"] = np.nan
            features[f"bioimp_imag_{band_name}_mean"] = np.nan
            features[f"bioimp_phase_{band_name}_mean"] = np.nan
            continue

        band_values[band_name] = float(np.mean(mag[idx]))
        features[f"bioimp_mag_{band_name}_mean"] = float(np.mean(mag[idx]))
        features[f"bioimp_real_{band_name}_mean"] = float(np.mean(real[idx]))
        features[f"bioimp_imag_{band_name}_mean"] = float(np.mean(imag[idx]))
        features[f"bioimp_phase_{band_name}_mean"] = float(np.mean(phase[idx]))

    features["bioimp_mag_low_high_ratio"] = _safe_div(
        band_values["low"], band_values["high"]
    )
    features["bioimp_mag_mid_high_ratio"] = _safe_div(
        band_values["mid"], band_values["high"]
    )
    features["bioimp_mag_low_mid_ratio"] = _safe_div(
        band_values["low"], band_values["mid"]
    )

    # Cole-plot-like geometry: real-imag plane
    features["bioimp_real_imag_corr"] = safe_corrcoef(real, imag)

    features["bioimp_complex_centroid_real"] = float(np.mean(real))
    features["bioimp_complex_centroid_imag"] = float(np.mean(imag))
    features["bioimp_complex_radius_mean"] = float(
        np.mean(np.sqrt((real - np.mean(real)) **
                2 + (imag - np.mean(imag)) ** 2))
    )

    # Simple equivalent-circuit-inspired descriptors
    # These are not true Cole-model fitted parameters, but useful proxies.
    features["bioimp_R_low_freq"] = float(real[0])
    features["bioimp_R_high_freq"] = float(real[-1])
    features["bioimp_R_low_minus_high"] = float(real[0] - real[-1])
    features["bioimp_R_low_high_ratio"] = _safe_div(
        float(real[0]), float(real[-1]))

    features["bioimp_X_low_freq"] = float(imag[0])
    features["bioimp_X_high_freq"] = float(imag[-1])
    features["bioimp_X_low_minus_high"] = float(imag[0] - imag[-1])

    # Ratios at selected relative frequency positions
    positions = {
        "p10": 0.10,
        "p25": 0.25,
        "p50": 0.50,
        "p75": 0.75,
        "p90": 0.90,
    }

    selected = {}
    for label, pos in positions.items():
        idx = int(round(pos * (n - 1)))
        selected[label] = idx

        features[f"bioimp_mag_{label}"] = float(mag[idx])
        features[f"bioimp_real_{label}"] = float(real[idx])
        features[f"bioimp_imag_{label}"] = float(imag[idx])
        features[f"bioimp_phase_{label}"] = float(phase[idx])

    features["bioimp_mag_p10_p90_ratio"] = _safe_div(
        features["bioimp_mag_p10"], features["bioimp_mag_p90"]
    )
    features["bioimp_mag_p25_p75_ratio"] = _safe_div(
        features["bioimp_mag_p25"], features["bioimp_mag_p75"]
    )
    features["bioimp_real_p10_p90_ratio"] = _safe_div(
        features["bioimp_real_p10"], features["bioimp_real_p90"]
    )
    features["bioimp_real_p25_p75_ratio"] = _safe_div(
        features["bioimp_real_p25"], features["bioimp_real_p75"]
    )

    # Remove non-finite values to make downstream ML pipelines safer
    for key, value in list(features.items()):
        if isinstance(value, complex):
            features[f"{key}_real"] = float(np.real(value))
            features[f"{key}_imag"] = float(np.imag(value))
            del features[key]
        elif isinstance(value, float) and not np.isfinite(value):
            features[key] = np.nan

    return features


def bioimp_all_feature_extracion(bioimp: np.ndarray, fs: int) -> Dict[str, Any]:
    """
    Extract generic frequency-domain bioimpedance features.

    Parameters
    ----------
    bioimp : np.ndarray
        Complex-valued bioimpedance frequency response Z(f).
        Shape should be (n_frequencies,). Values may be complex.
    fs : int
        Sampling frequency or maximum frequency used to construct a surrogate
        frequency axis. If the true frequency vector is available, it is better
        to adapt this function to receive it explicitly.

    Returns
    -------
    Dict[str, Any]
        Dictionary with scalar features.
    """
    features: Dict[str, Any] = {}

    z = _safe_array(bioimp).astype(np.complex128)
    n = z.size

    if n == 0:
        return {"bioimp_valid": False, "bioimp_n_points": 0}

    features["bioimp_valid"] = True
    features["bioimp_n_points"] = int(n)

    eps = 1e-12

    # Surrogate frequency axis.
    # Prefer replacing this with the true measured frequencies if available.
    freq = np.linspace(0.0, float(fs) / 2.0, n)
    freq_idx = np.arange(n, dtype=float)

    real = np.real(z)
    imag = np.imag(z)
    mag = np.abs(z)
    phase = np.unwrap(np.angle(z))

    log_mag = np.log(mag + eps)
    mag_db = 20.0 * np.log10(mag + eps)

    # Basic component statistics
    _add_basic_stats(features, "bioimp_real", real)
    _add_basic_stats(features, "bioimp_imag", imag)
    _add_basic_stats(features, "bioimp_mag", mag)
    _add_basic_stats(features, "bioimp_phase", phase)
    _add_basic_stats(features, "bioimp_log_mag", log_mag)
    _add_basic_stats(features, "bioimp_mag_db", mag_db)

    # Original-style global features
    features["bioimp_mean"] = complex(np.mean(z))
    features["bioimp_std"] = float(np.std(z))
    features["bioimp_peak_to_peak_mag"] = float(np.ptp(mag))

    # Remove non-finite values to make downstream ML pipelines safer
    for key, value in list(features.items()):
        if isinstance(value, complex):
            features[f"{key}_real"] = float(np.real(value))
            features[f"{key}_imag"] = float(np.imag(value))
            del features[key]
        elif isinstance(value, float) and not np.isfinite(value):
            features[key] = np.nan

    return features


def __plot_quality(df_raw, df_quality, modality, file_id=None):
    """
    Plots raw vs filtered magnitude and phase.

    Args:
        df_raw: original dataframe (must contain Zmag, Zphi and t or f)
        df_filtered: processed dataframe (must contain mag_filtered, phase_filtered)
        modality: 'bioimp'
        file_id: optional (for title)
    """

    # --------------------------------------------------
    # Select axis
    # --------------------------------------------------
    x = df_raw["f"].values
    x_label = "Frequency (Hz)"

    # --------------------------------------------------
    # Extract signals
    # --------------------------------------------------
    mag_raw = df_raw["Zmag"].values
    phase_raw = df_raw["Zphi"].values

    quality = df_quality["quality"].values

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Magnitude
    plt.subplot(2, 1, 1)
    plt.plot(x, mag_raw, label="Bioimpedance", alpha=0.6)
    if "bioimp" in modality:
        plt.xscale("log")
    plt.ylabel("Impedance Magnitude (Ohm)")
    plt.title(f"File: {file_id} | Magnitude")
    plt.legend()
    plt.grid(True)

    # Phase
    plt.subplot(2, 1, 2)
    plt.plot(x, quality, label="Quality", linewidth=2)
    if "bioimp" in modality:
        plt.xscale("log")
    plt.xlabel(x_label)
    plt.ylabel("Quality")
    plt.title("Quality (1=best)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def __plot_raw_vs_filtered(df_raw, df_filtered, modality, file_id=None):
    """
    Plots raw vs filtered magnitude and phase.

    Args:
        df_raw: original dataframe (must contain Zmag, Zphi and t or f)
        df_filtered: processed dataframe (must contain mag_filtered, phase_filtered)
        modality: 'bioimp'
        file_id: optional (for title)
    """

    # --------------------------------------------------
    # Select axis
    # --------------------------------------------------
    x = df_raw["f"].values
    x_label = "Frequency (Hz)"

    # --------------------------------------------------
    # Extract signals
    # --------------------------------------------------
    mag_raw = df_raw["Zmag"].values
    phase_raw = df_raw["Zphi"].values

    mag_f = df_filtered["mag_filtered"].values
    phase_f = df_filtered["phase_filtered"].values

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Magnitude
    plt.subplot(2, 1, 1)
    plt.plot(x, mag_raw, label="Raw", alpha=0.6)
    plt.plot(x, mag_f, label="Filtered", linewidth=2)
    if "bioimp" in modality:
        plt.xscale("log")
    plt.ylabel("Impedance Magnitude (Ohm)")
    plt.title(f"File: {file_id} | Magnitude")
    plt.legend()
    plt.grid(True)

    # Phase
    plt.subplot(2, 1, 2)
    plt.plot(x, phase_raw, label="Raw", alpha=0.6)
    plt.plot(x, phase_f, label="Filtered", linewidth=2)
    if "bioimp" in modality:
        plt.xscale("log")
    plt.xlabel(x_label)
    plt.ylabel("Phase (degrees)")
    plt.title("Phase")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Loader for bioimpedance files (your format)
# --------------------------------------------------


def load_bioimp_file(path):
    header_lines = []
    data_lines = []
    found_table = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("mnum"):
                found_table = True

            if found_table:
                data_lines.append(line)
            else:
                if stripped:
                    header_lines.append(stripped)

    if not data_lines:
        raise ValueError(f"No measurement table found in {path}")

    df = pd.read_csv(StringIO("".join(data_lines)), sep=None, engine="python")
    return header_lines, df


# --------------------------------------------------
# Process a single file
# --------------------------------------------------
def _infer_axis(df: pd.DataFrame, modality: str) -> tuple[np.ndarray, str]:

    if "f" in df.columns:
        return np.asarray(df["f"].values), "frequency"
    if "frequency" in df.columns:
        return np.asarray(df["frequency"].values), "frequency"

    raise ValueError(
        f"Could not infer axis for modality '{modality}'. Expected one of columns: t, time, f, frequency.")


def bioimp_savitzky_waveform_processing(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    x, x_name = _infer_axis(df, modality)

    if "Zmag" not in df.columns:
        raise ValueError("Input dataframe must contain column 'Zmag'.")
    if "Zphi" not in df.columns:
        raise ValueError("Input dataframe must contain column 'Zphi'.")

    mag = np.asarray(df["Zmag"].values)
    phase = np.asarray(df["Zphi"].values)

    if APPLY_PHASE_UNWRAP:
        phase = np.rad2deg(np.unwrap(np.deg2rad(phase)))

    # Smooth
    mag_f = savitzky_golay_filtering(mag)
    phase_f = savitzky_golay_filtering(phase)

    # Build output dataframe
    df_out = pd.DataFrame({
        x_name: x,
        "mag_filtered": mag_f,
        "phase_filtered": phase_f
    })

    return df_out


def bioimp_sqi_mse_waveform_processing(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    # use global variables to track min and max SQI MSE values
    global MIN_SQI_MSE, MAX_SQI_MSE

    if "mag_filtered" in df.columns:
        mag = np.asarray(df["mag_filtered"].values)
    elif "Zmag" in df.columns:
        mag = np.asarray(df["Zmag"].values)
    else:
        raise ValueError(
            "Input dataframe must contain 'mag_filtered' or 'Zmag' column to compute quality.")

    # First normalize mag to [0,1] for quality calculation
    mag_range = np.max(mag) - np.min(mag)
    if mag_range <= 0:
        mag_norm = np.zeros_like(mag)
    else:
        mag_norm = (mag - np.min(mag)) / mag_range

    # Smooth
    mag_f = savitzky_golay_filtering(mag_norm)

    # define quality based on mean-squared error (mse) between raw and smoothed signals
    # recall the magnitudes were already normalized to [0,1] for this calculation
    # so this error is a normalized error
    rmse = np.array((np.array(mag_norm) - np.array(mag_f)) ** 2.0)
    error_average_power = np.mean(rmse)
    # map minimum value to SQI 0.5 and maximum to 1
    single_quality_value = (2*MAX_SQI_MSE - error_average_power -
                            MIN_SQI_MSE) / (2.0*(MAX_SQI_MSE - MIN_SQI_MSE))
    # if quality is negative due to high rmse, set to 0
    if single_quality_value < 0:
        single_quality_value = 0.0
    # if quality is above 1 due to low rmse, set to 1
    if single_quality_value > 1:
        single_quality_value = 1.0
    # all quality values are the same for the entire waveform, so create an array of the same value
    quality = np.full_like(mag_norm, single_quality_value)

    x, x_name = _infer_axis(df, modality)

    # Build output dataframe
    df_out = pd.DataFrame({
        x_name: x,
        "quality": quality
    })
    return df_out


# --------------------------------------------------
# Main pipelines
# --------------------------------------------------


def bioimpedance_signal_processing_pipeline(dataset_config_file: str, input_waveform_id: str,
                                            pipeline: str,
                                            output_waveform_id: str) -> None:
    """Use this method for both waveform and SQI processing pipelines."""

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
    base_path = datasetConfig.get_dataset_root_path()

    df = datasetConfig.get_dataset_info_dataframe()
    df_bio = df[df["modality"].str.contains("bioimp")].copy()

    counter = 0
    for _, row in df_bio.iterrows():
        file_id = row["file_id"]
        modality = row["modality"]

        if input_waveform_id == "raw":
            raw_relative_path = datasetConfig.get_raw_relative_path(file_id)
            input_complete_path = os.path.join(base_path, raw_relative_path)
            _, input_df = load_bioimp_file(input_complete_path)
        else:
            input_complete_path = datasetConfig.get_gen_complete_path(
                file_id, input_waveform_id, new_extension="csv")
            input_df = pd.read_csv(input_complete_path)

        print(f"Input Bioimpedance: {input_complete_path} (file_id={file_id})")

        df_out = processing_function(input_df, modality)

        output_filename = datasetConfig.get_gen_complete_path(
            file_id, output_waveform_id, new_extension="csv")

        output_dir = os.path.dirname(output_filename)
        os.makedirs(output_dir, exist_ok=True)

        df_out.to_csv(output_filename, index=False)
        print(f"Output Bioimpedance: {output_filename} (file_id={file_id})")

        if SHOULD_PLOT:
            if "quality" in df_out.columns:
                __plot_quality(input_df, df_out, modality, file_id=file_id)
            elif "mag_filtered" in df_out.columns and "phase_filtered" in df_out.columns:
                __plot_raw_vs_filtered(
                    input_df, df_out, modality, file_id=file_id)

        counter += 1

    print(f"\nFinished processing {counter} files.")


def old_bioimpedance_frequency_filtering_pipeline(dataset_config_file: str):
    """Backward-compatible wrapper for bioimpedance signal processing."""
    bioimpedance_signal_processing_pipeline(
        dataset_config_file,
        input_waveform_id="raw",
        pipeline="bandpass",
        output_waveform_id="filtered",
    )


def old_bioimpedance_frequency_quality_pipeline(dataset_config_file: str):
    """Backward-compatible wrapper for bioimpedance quality processing."""
    bioimpedance_signal_processing_pipeline(
        dataset_config_file,
        input_waveform_id="filtered",
        pipeline="sqi_sumall",
        output_waveform_id="quality",
    )


def extract_bioimp_features(
    datasetConfig: DatasetConfig,
    bioimp: np.ndarray,
    fs: int
) -> Dict[str, Any]:
    pipeline = datasetConfig.get_value(
        "BIOIMP_FEATURE_EXTRACTION_PIPELINE", "all")

    try:
        processing_function = FEATURE_PIPELINES[pipeline]
    except KeyError:
        valid = ", ".join(FEATURE_PIPELINES.keys())
        raise ValueError(
            f"Unknown waveform_id '{pipeline}'. "
            f"Valid options are: {valid}"
        )

    return processing_function(bioimp, fs)


def old_extract_bioimp_features(bioimp: np.ndarray, fs: int) -> Dict[str, Any]:

    try:
        features = {}

        features["bioimp_mean"] = safe_mean(bioimp)
        features["bioimp_std"] = safe_std(bioimp)

        # slope
        slope = np.gradient(bioimp)
        features["bioimp_slope_mean"] = safe_mean(slope)
        features["bioimp_slope_std"] = safe_std(slope)

        features["bioimp_peak_to_peak"] = np.max(bioimp) - np.min(bioimp)

        return features

    except Exception as e:
        raise RuntimeError(
            f"Bioimpedance feature extraction failed: {e}") from e

    return features


SIGNAL_PIPELINES = {
    "savitzky": bioimp_savitzky_waveform_processing
}

QUALITY_PIPELINES = {
    "sqi_mse": bioimp_sqi_mse_waveform_processing
}

FEATURE_PIPELINES = {
    "all": bioimp_all_feature_extracion
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

    if "bioimp" not in datasetConfig.modalities:
        print(
            "No Bioimpedance modality found in the dataset configuration. Skipping Bioimpedance processing.")
        exit(0)

    signal_pipeline = datasetConfig.get_value(
        "BIOIMP_SIGNAL_PROCESSING_PIPELINE")
    output_waveform_id = datasetConfig.get_value(
        "BIOIMP_SIGNAL_OUTPUT_WAVEFORM", signal_pipeline)

    print("Processing Bioimpedance signals with pipeline", signal_pipeline,
          "to generate output_waveform_id", output_waveform_id)
    bioimpedance_signal_processing_pipeline(
        dataset_config_file,
        input_waveform_id="raw",
        pipeline=signal_pipeline,
        output_waveform_id=output_waveform_id,
    )

    sqi_pipeline = datasetConfig.get_value(
        "BIOIMP_SQI_PROCESSING_PIPELINE", "no_sqi")
    if sqi_pipeline == "no_sqi":
        print("Skipping SQI waveform creation for Bioimpedance signals.")
    else:
        input_waveform_id = datasetConfig.get_value(
            "BIOIMP_SQI_INPUT_WAVEFORM", "raw")
        output_waveform_id = datasetConfig.get_value(
            "BIOIMP_SQI_OUTPUT_WAVEFORM", sqi_pipeline)
        print("Creating SQI waveforms for Bioimpedance signals using pipeline", sqi_pipeline,
              "and input_waveform_id:", input_waveform_id,
              "and output_waveform_id:", output_waveform_id)
        bioimpedance_signal_processing_pipeline(
            dataset_config_file,
            input_waveform_id=input_waveform_id,
            pipeline=sqi_pipeline,
            output_waveform_id=output_waveform_id,
        )
