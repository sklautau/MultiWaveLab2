'''
Features based on EMD/IMF decomposition.
Based on package PyEMD.EMD: https://pyemd.readthedocs.io/en/latest/intro.html

PyEMD is a Python implementation of Empirical Mode Decomposition (EMD) and its variations. One of the most popular expansion is Ensemble Empirical Mode Decomposition (EEMD), which utilises an ensemble of noise-assisted executions.

Decompose signal into a set of components. These components are called Intrinsic Mode Functions (IMF)
to highlight that they contain an intrinsic (self) property which is a specific oscillation (mode).
These are generic oscillations; their frequency and amplitude can change, however, they are distinct
within analyzed signal.

Also compute statistics per IMF.
Important choices: np.var uses ddof=0,
and scipy.stats.kurtosis uses Fisher kurtosis

Example of feature names:
IMF_1_mean        mean
IMF_1_var         variance
IMF_1_std         standard deviation
IMF_1_ptp         peak-to-peak
IMF_1_skewness    skewness
IMF_1_kurtosis    kurtosis
IMF_1_dom_freq    dominant frequency
IMF_1_total_power total PSD power
IMF_1_ae_mean     mean amplitude envelope
IMF_1_if_mean     mean instantaneous frequency
IMF_1_zcr         zero-crossing rate
IMF_1_extrema     number of extrema
IMF_1_psd_mean    mean PSD
IMF_1_psd_var     PSD variance
IMF_1_sc          spectral centroid
IMF_1_se          spectral entropy
IMF_1_sf          spectral flatness
'''
import numpy as np
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis


def decompose_signal_to_imfs(signal, max_imfs=None):
    """
    Decompose one signal into IMFs using PyEMD.
    """
    try:
        from PyEMD import EMD
    except ImportError as exc:
        raise ImportError(
            "PyEMD is required. Install with: pip install EMD-signal"
        ) from exc

    x = np.asarray(signal, dtype=float).ravel()
    x = x[np.isfinite(x)]

    if x.size < 10:
        raise ValueError("Signal is too short.")

    emd = EMD()
    imfs = emd.emd(x)

    if max_imfs is not None:
        imfs = imfs[:max_imfs]

    return imfs


def _safe_psd(x, fs):
    x = np.asarray(x, dtype=float).ravel()
    nperseg = min(len(x), 256)

    # by default, Welch uses nperseg=min(len(imf), 256), and scaling="density".
    freqs, psd = sp_signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        detrend=False,
        scaling="density",
    )

    return freqs, psd


def _safe_entropy_from_psd(psd, eps=1e-12):
    psd = np.asarray(psd, dtype=float)
    p = psd / (np.sum(psd) + eps)
    return -np.sum(p * np.log2(p + eps))


def _safe_spectral_flatness(psd, eps=1e-12):
    psd = np.asarray(psd, dtype=float)
    return np.exp(np.mean(np.log(psd + eps))) / (np.mean(psd) + eps)


def _zero_crossing_rate(x):
    x = np.asarray(x, dtype=float)
    signs = np.sign(x)

    # Replace exact zeros by previous nonzero sign when possible
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]

    return np.sum(np.diff(signs) != 0) / max(len(x), 1)


def _count_extrema(x):
    x = np.asarray(x, dtype=float)
    peaks, _ = sp_signal.find_peaks(x)
    troughs, _ = sp_signal.find_peaks(-x)
    return len(peaks) + len(troughs)


def extract_single_imf_features(imf, fs, prefix):
    """
    Extract features from one IMF.

    Feature names follow the pattern:
    IMF_1_mean, IMF_1_var, ...
    """
    eps = 1e-12

    x = np.asarray(imf, dtype=float).ravel()
    x = x[np.isfinite(x)]

    if x.size < 10:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_var": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_ptp": np.nan,
            f"{prefix}_skewness": np.nan,
            f"{prefix}_kurtosis": np.nan,
            f"{prefix}_dom_freq": np.nan,
            f"{prefix}_total_power": np.nan,
            f"{prefix}_ae_mean": np.nan,
            f"{prefix}_if_mean": np.nan,
            f"{prefix}_zcr": np.nan,
            f"{prefix}_extrema": np.nan,
            f"{prefix}_psd_mean": np.nan,
            f"{prefix}_psd_var": np.nan,
            f"{prefix}_sc": np.nan,
            f"{prefix}_se": np.nan,
            f"{prefix}_sf": np.nan,
        }

    freqs, psd = _safe_psd(x, fs)

    analytic = sp_signal.hilbert(x)
    amplitude_envelope = np.abs(analytic)

    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) * fs / (2 * np.pi)

    total_psd_power = np.sum(psd)
    dom_freq = freqs[np.argmax(psd)] if len(psd) > 0 else np.nan

    spectral_centroid = (
        np.sum(freqs * psd) / (total_psd_power + eps)
        if len(freqs) == len(psd)
        else np.nan
    )

    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_var": float(np.var(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_ptp": float(np.ptp(x)),
        f"{prefix}_skewness": float(skew(x, bias=True)),
        f"{prefix}_kurtosis": float(kurtosis(x, bias=True, fisher=True)),
        f"{prefix}_dom_freq": float(dom_freq),
        f"{prefix}_total_power": float(total_psd_power),
        f"{prefix}_ae_mean": float(np.mean(amplitude_envelope)),
        f"{prefix}_if_mean": float(np.mean(inst_freq)) if len(inst_freq) > 0 else np.nan,
        f"{prefix}_zcr": float(_zero_crossing_rate(x)),
        f"{prefix}_extrema": float(_count_extrema(x)),
        f"{prefix}_psd_mean": float(np.mean(psd)),
        f"{prefix}_psd_var": float(np.var(psd)),
        f"{prefix}_sc": float(spectral_centroid),
        f"{prefix}_se": float(_safe_entropy_from_psd(psd)),
        f"{prefix}_sf": float(_safe_spectral_flatness(psd)),
    }


def extract_imf_features_from_imfs(imfs, fs=60, max_imfs=None):
    """
    Extract IMF_1_..., IMF_2_..., etc. from already computed IMFs.
    """
    imfs = np.asarray(imfs, dtype=float)

    if imfs.ndim != 2:
        raise ValueError("Expected imfs with shape (n_imfs, n_samples).")

    if max_imfs is not None:
        imfs = imfs[:max_imfs]

    features = {}

    for k, imf in enumerate(imfs, start=1):
        prefix = f"IMF_{k}"
        features.update(extract_single_imf_features(imf, fs, prefix))

    return features


def extract_imf_features_from_signal(signal, fs=60, max_imfs=5):
    """
    Full pipeline: Signal -> EMD -> IMF features.
    """
    imfs = decompose_signal_to_imfs(signal, max_imfs=max_imfs)
    return extract_imf_features_from_imfs(imfs, fs=fs, max_imfs=max_imfs)


if __name__ == "__main__":

    signal = np.random.rand(300)  # Example signal with 300 samples

    features = extract_imf_features_from_signal(
        signal,
        fs=60,
        max_imfs=5,
    )

    print("Extracted All IMF features:")
    for key, value in features.items():
        print(f"{key}: {value}")
