'''
Methods to determine the quality of a PPG signal.
Neurokit2 quality (SQI) is fragile by design
It relies on template matching. Sensitive to:
    noise
    missed peaks
    short segments
In practice, SQI often fails on:
    motion-corrupted PPG
    short windows (< ~10-15 beats)
'''
import numpy as np
from scipy.signal import welch
import neurokit2 as nk


# If False, errors will be logged but the pipeline will continue, returning NaN or empty values for features that failed to extract.
RAISE_EXCEPTION_ON_PPG_PROCESSING = True


def _sqi_peaks(peaks, fs, duration):
    if peaks is None or len(peaks) < 3:
        return 0.0

    hr = len(peaks) / duration * 60
    if hr < 40 or hr > 180:
        return 0.2

    return min(1.0, len(peaks) / (duration * 1.2))


def _sqi_rr(peaks, fs):
    if peaks is None or len(peaks) < 3:
        return 0.0

    rr = np.diff(peaks) / fs
    if len(rr) < 2:
        return 0.0

    cv = np.std(rr) / (np.mean(rr) + 1e-12)
    return np.exp(-5 * cv)


def _sqi_spectral(psd, freqs):
    band = (freqs >= 0.6) & (freqs <= 3.0)
    if not band.any():
        return 0.0

    psd_band = psd[band]
    freqs_band = freqs[band]

    idx = np.argmax(psd_band)
    f0 = freqs_band[idx]

    main = (freqs >= 0.9 * f0) & (freqs <= 1.1 * f0)

    E_main = np.sum(psd[main])
    E_total = np.sum(psd_band) + 1e-12

    return E_main / E_total


def _sqi_snr(psd, freqs):
    signal_band = (freqs >= 0.6) & (freqs <= 3.0)
    noise_band = (freqs >= 5.0) & (freqs <= 15.0)

    E_signal = np.sum(psd[signal_band])
    E_noise = np.sum(psd[noise_band]) + 1e-12

    snr = E_signal / E_noise
    return 1 - np.exp(-snr)


def _compute_window_sqi(segment, fs, peaks_win):
    duration = len(segment) / fs

    # adaptive Welch
    nperseg = min(len(segment), int(fs * 8))
    if nperseg < 8:  # too short
        return 0.0

    freqs, psd = welch(segment, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    # components
    Q_spec = _sqi_spectral(psd, freqs)
    Q_snr = _sqi_snr(psd, freqs)

    # peak-based components (optional)
    if peaks_win is not None and len(peaks_win) >= 3:
        Q_peaks = _sqi_peaks(peaks_win, fs, duration)
        Q_rr = _sqi_rr(peaks_win, fs)

        w1, w2, w3, w4 = 0.25, 0.25, 0.3, 0.2
        sqi = w1 * Q_peaks + w2 * Q_rr + w3 * Q_spec + w4 * Q_snr
    else:
        # fallback: rely only on spectral info
        sqi = 0.6 * Q_spec + 0.4 * Q_snr

    return float(np.clip(sqi, 0, 1))


def estimate_sqi_custom_version(
    ppg,
    fs,
    peaks=None,
    win_sec=8,
    hop_sec=1,
    smooth=True
):
    """
    Compute per-sample SQI for a PPG signal.

    Returns:
        np.ndarray of shape (len(ppg),) with values in [0, 1]
    """
    # import locally to avoid circular dependency because
    # ppg.py => imports from ppg_quality.py and
    # ppg_quality.py => imports from ppg.py
    from signal_processing.ppg import _normalize_peaks

    ppg = np.asarray(ppg)
    N = len(ppg)

    # Use Neurokit2 to find PPG events
    signals, info = nk.ppg_process(ppg, sampling_rate=fs)
    # Get peaks and normalize to numpy array (after nk.ppg_process)
    peaks = _normalize_peaks(info.get("PPG_Peaks", None))
    if peaks is not None:
        peaks = np.sort(peaks)

    # adaptive window
    win = int(win_sec * fs)
    if win > N:
        win = max(int(N * 0.8), int(2 * fs))  # fallback

    hop = int(hop_sec * fs)
    hop = max(1, hop)

    sqi_ts = np.zeros(N, dtype=float)
    weight = np.zeros(N, dtype=float)

    # precompute Gaussian window
    t = np.linspace(-1, 1, win)
    gauss = np.exp(-3 * t**2)

    for start in range(0, N - win + 1, hop):
        end = start + win
        segment = ppg[start:end]

        # extract peaks in window
        # inside loop
        if peaks is not None and len(peaks) > 0:
            mask = (peaks >= start) & (peaks < end)
            peaks_win = peaks[mask] - start
            if len(peaks_win) == 0:
                peaks_win = None
        else:
            peaks_win = None
        sqi = _compute_window_sqi(segment, fs, peaks_win)

        # weighted overlap-add
        sqi_ts[start:end] += sqi * gauss
        weight[start:end] += gauss

    # normalize
    valid = weight > 0
    sqi_ts[valid] /= weight[valid]

    # interpolate uncovered regions
    if not np.all(valid):
        idx = np.where(valid)[0]
        if len(idx) > 1:
            sqi_ts = np.interp(np.arange(N), idx, sqi_ts[idx])
        else:
            sqi_ts[:] = 0.0

    # optional smoothing
    if smooth:
        kernel_size = int(fs * 0.5)  # 0.5 sec smoothing
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            sqi_ts = np.convolve(sqi_ts, kernel, mode="same")

    return np.clip(sqi_ts, 0, 1)


def estimate_single_sqi_value(ppg, fs) -> np.floating:

    # Use Neurokit2 to find PPG events
    signals, info = nk.ppg_process(ppg, sampling_rate=fs)
    # Get peaks and normalize to numpy array (after nk.ppg_process)
    peaks = _normalize_peaks(info.get("PPG_Peaks", None))
    if peaks is not None:
        peaks = np.sort(peaks)

    duration = len(ppg) / fs

    # PSD (Welch setup)
    nperseg = min(len(ppg), int(fs * 8))
    freqs, psd = welch(ppg, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    # Components
    Q_peaks = _sqi_peaks(peaks, fs, duration)
    Q_rr = _sqi_rr(peaks, fs)
    Q_spec = _sqi_spectral(psd, freqs)
    Q_snr = _sqi_snr(psd, freqs)

    # Weights (tunable)
    w1, w2, w3, w4 = 0.25, 0.25, 0.3, 0.2

    sqi = w1*Q_peaks + w2*Q_rr + w3*Q_spec + w4*Q_snr

    return sqi


def estimate_sqi_neurokit_tm(ppg_signal, fs) -> np.ndarray:
    '''
    The ppg_signal can be raw because neurokit2.ppg_process() will clean it internally.
    '''
    try:
        # Choose the method here:
        method_quality = "templatematch"  # "templatematch" or "ho2025"
        sqi_dict = _sqi_neurokit_tm(
            ppg_signal, fs, method_quality=method_quality)
    except ValueError as e:
        if RAISE_EXCEPTION_ON_PPG_PROCESSING:
            raise ValueError(e)
        else:
            print(f"   ########## Error occurred while estimating SQI: {e}")
            sqi_signal = np.zeros_like(ppg_signal)
    else:
        # executed only if the try block did not raise the ValueError
        #
        sqi_signal = sqi_dict['sqi_for_each_signal_sample'].to_numpy()

    if not isinstance(sqi_signal, np.ndarray):
        raise ValueError(
            f"SQI signal is not a numpy array. Got {type(sqi_signal)} instead.")
    return sqi_signal


def _sqi_neurokit_tm(ppg, fs, method_quality="templatematch", ppg_raw=None):
    """
    Estimate Signal Quality Index (SQI) for a PPG signal using NeuroKit2.
    Recent NeuroKit2 versions include a PPG SQI function and ppg_process() can return a continuous PPG quality index (0–1) (methods "templatematch" and "disimilarity")
    https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/ppg/ppg_process.html
    neurokit2.ppg.ppg_process() calls ppg_quality() and places the result in the output DataFrame under the column "PPG_Quality". So running signals, info = nk.ppg_process(ppg, sampling_rate=fs) with the default/appropriate method_quality will return signals["PPG_Quality"].
    Neuropsychology
    The PPG pipeline exposes method_quality in ppg_process() so you can choose "templatematch" (default) or "disimilarity".
    # When using method_quality="templatematch", an error occurs in NeuroKit2:
    # lib\site-packages\neurokit2\signal\signal_period.py:83: NeuroKitWarning:
    # Too few peaks detected to compute the rate. Returning empty vector.
    # ValueError: cannot convert float NaN to integer
    # I tried using method_quality="dissimilarity" instead, but the SQI values are too small.
    # I then tried ho2025 and others, but they also give very small SQI values. So I will stick to "templatematch"
    # and handle the error by returning 0 values for the SQI signal.
    """
    # Process PPG. Some quality methods (e.g., perfusion, relative_power)
    # require explicit access to raw waveform values via ppg_raw.
    ppg_process_kwargs = {}
    if ppg_raw is not None:
        ppg_process_kwargs["ppg_raw"] = np.asarray(ppg_raw, dtype=float)

    try:
        signals, info = nk.ppg_process(ppg, sampling_rate=fs,
                                       method="elgendi",            # optional
                                       method_quality=method_quality,
                                       **ppg_process_kwargs)
    except Exception as exc:
        if "ppg_raw must be provided" in str(exc) and ppg_raw is None:
            signals, info = nk.ppg_process(
                ppg,
                sampling_rate=fs,
                method="elgendi",
                method_quality=method_quality,
                ppg_raw=np.asarray(ppg, dtype=float),
            )
        else:
            raise

    # the signals DataFrame contains several columns
    # print(signals.keys())
    # ['PPG_Raw', 'PPG_Clean', 'PPG_Rate', 'PPG_Quality', 'PPG_Peaks']

    # 'PPG_Quality' column (same length as signal) — continuous 0..1
    sqi_for_each_signal_sample = signals["PPG_Quality"]

    # clip to [0, 1]
    sqi_for_each_signal_sample = np.clip(sqi_for_each_signal_sample, 0, 1)
    # sqi_per_pulse = np.clip(sqi_per_pulse, 0, 1)

    sqi_dict = {"name": "NeuroKit2 - TM",
                "sqi_per_pulse": np.array([]), "sqi_for_each_signal_sample": sqi_for_each_signal_sample}

    return sqi_dict


def plot_sqi_comparison(ppg, fs, title="SQI Comparison", show=True):
    """
    Plot the PPG waveform superimposed with the SQI signals from both
    estimate_sqi_neurokit_tm and estimate_sqi_custom_version for visual comparison.

    Parameters
    ----------
    ppg : array-like
        Raw PPG signal.
    fs : float
        Sampling frequency in Hz.
    title : str
        Title for the figure.
    show : bool
        Whether to call plt.show(). Set False when embedding in a larger figure.
    """
    import matplotlib.pyplot as plt

    ppg = np.asarray(ppg, dtype=float)
    t = np.arange(len(ppg)) / fs

    # Compute SQI signals
    sqi_nk = np.zeros_like(ppg)
    sqi_custom = np.zeros_like(ppg)

    try:
        sqi_nk = np.asarray(estimate_sqi_neurokit_tm(ppg, fs), dtype=float)
    except Exception as e:
        print(f"estimate_sqi_neurokit_tm failed: {e}")

    try:
        sqi_custom = np.asarray(
            estimate_sqi_custom_version(ppg, fs), dtype=float)
    except Exception as e:
        print(f"estimate_sqi_custom_version failed: {e}")

    # Normalise PPG to [0, 1] for overlay
    ppg_range = np.ptp(ppg)
    ppg_norm = (ppg - ppg.min()) / (ppg_range if ppg_range > 0 else 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- Top: PPG waveform ---
    axes[0].plot(t, ppg_norm, color="steelblue",
                 linewidth=0.8, label="PPG (norm.)")
    axes[0].set_ylabel("Amplitude (n.u.)", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Bottom: SQI comparison ---
    axes[1].plot(t, sqi_nk, color="darkorange", linewidth=1.2,
                 label="SQI – NeuroKit2 TM")
    axes[1].plot(t, sqi_custom, color="seagreen", linewidth=1.2,
                 linestyle="--", label="SQI – Custom (spectral + peaks)")
    axes[1].axhline(0.5, color="red", linewidth=0.8,
                    linestyle=":", label="Threshold 0.5")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("SQI [0 – 1]", fontsize=11)
    axes[1].set_xlabel("Time (s)", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()

    return fig


if __name__ == "main":
    ppg = None
    fs = 500
    signals = None
    sqi2 = estimate_sqi_custom_version(ppg, fs, signals["PPG_Peaks"])

    print(sqi2[0:10])
