'''
Utility functions for signal processing tasks, including outlier removal.
'''
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import medfilt
from dataclasses import dataclass, field
from scipy.signal import butter
from typing import Any, Dict
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def _reverse_signal(raw):
    '''Similar to np.flipud, but checks for 1-D array and returns a copy'''
    raw = np.asarray(raw)

    if raw.ndim != 1:
        raise ValueError(
            f"Expected a 1-D array of shape (N,), but got shape {raw.shape}"
        )

    return raw[::-1]


def _energy_in_band(min_f, max_f, psd, f):
    m = (f >= min_f) & (f <= max_f)
    return np.trapezoid(psd[m], f[m]) if m.any() else np.nan


def plot_histogram_robust(
    x,
    bins=100,
    percentile_range=(1, 99),
    title="Robust histogram",
    xlabel="Amplitude",
    show_outliers=True,
):
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise ValueError("Input signal has no finite values.")

    lo, hi = np.percentile(x, percentile_range)
    x_inside = x[(x >= lo) & (x <= hi)]

    plt.figure(figsize=(8, 4))
    plt.hist(x_inside, bins=bins)

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if show_outliers:
        n_low = np.sum(x < lo)
        n_high = np.sum(x > hi)
        plt.text(
            0.02,
            0.95,
            f"Range: [{lo:.4g}, {hi:.4g}]\n"
            f"Low outliers: {n_low}\n"
            f"High outliers: {n_high}",
            transform=plt.gca().transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.show()


def plot_histogram(
    x,
    bins="auto",
    density=False,
    xlabel="Amplitude",
    ylabel=None,
    title="Histogram",
    logy=False,
    grid=True,
    show_me=True
):
    """
    Plot a histogram of a signal.

    Parameters
    ----------
    x : array-like
        Input signal.

    bins : int or str, default="auto"
        Number of histogram bins or a NumPy histogram rule
        ("auto", "fd", "sturges", etc.).

    density : bool, default=False
        If True, plot probability density instead of counts.

    xlabel : str
        Label for x-axis.

    ylabel : str or None
        Label for y-axis. If None, uses "Count" or "Probability density".

    title : str
        Figure title.

    logy : bool, default=False
        Use logarithmic y-axis.

    grid : bool, default=True
        Show grid.
    """

    x = np.asarray(x).ravel()

    plt.figure(figsize=(8, 4))
    plt.hist(
        x,
        bins=bins,
        density=density,
        edgecolor="black",
        linewidth=0.7,
    )

    plt.xlabel(xlabel)

    if ylabel is None:
        ylabel = "Probability Density" if density else "Count"

    plt.ylabel(ylabel)
    plt.title(title)

    if logy:
        plt.yscale("log")

    if grid:
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if show_me:
        plt.show()


def plot_quantization_levels(x, title="Quantized Amplitude Levels", show_me=True):
    """
    Plot the distribution of the unique amplitude levels.
    """
    x = np.asarray(x).ravel()

    levels, counts = np.unique(x, return_counts=True)
    print("Debug")
    print(f"  Unique amplitude levels: {levels}")
    print(f"  Counts: {counts}")

    plt.figure(figsize=(10, 4))
    plt.bar(levels, counts, width=np.min(
        np.diff(levels)) if len(levels) > 1 else 1)
    plt.xlabel("Amplitude Level")
    plt.ylabel("Occurrences")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if show_me:
        plt.show()


def estimate_amplitude_bits(x: np.ndarray) -> Dict[str, Any]:
    """
    Estimate the number of bits required to represent the amplitudes
    of an already quantized signal.

    Parameters
    ----------
    x : ndarray
        Input signal.

    Returns
    -------
    dict
        Dictionary containing:
        - unique_levels
        - observed_bits
        - theoretical_bits
        - quantization_step
        - min_value
        - max_value
        - uniformly_quantized
    """

    x = np.asarray(x).ravel()

    if x.size == 0:
        raise ValueError("Empty signal.")

    # Sort unique values
    levels = np.unique(x)
    n_levels = len(levels)

    # Estimate from number of used levels
    observed_bits = int(np.ceil(np.log2(n_levels))) if n_levels > 1 else 0

    # Estimate quantization step
    if n_levels > 1:
        diffs = np.diff(levels)

        # Smallest spacing
        q = np.min(diffs)

        # Check whether levels are uniformly spaced
        uniformly_quantized = np.allclose(
            diffs,
            q,
            rtol=1e-6,
            atol=max(abs(q), 1.0) * 1e-9,
        )

        theoretical_levels = int(round((levels[-1] - levels[0]) / q)) + 1
        theoretical_bits = int(np.ceil(np.log2(theoretical_levels)))
    else:
        q = np.nan
        uniformly_quantized = True
        theoretical_bits = 0

    print(f"  Signal has {n_levels} unique levels, observed bits: {observed_bits}, theoretical bits: {theoretical_bits}, quantization step: {q}, uniformly quantized: {uniformly_quantized}")
    print(f"  Signal min: {levels[0]}, max: {levels[-1]}")
    print(
        f"  quantization step: {q}, uniformly quantized: {uniformly_quantized}")
    return {
        "unique_levels": n_levels,
        "observed_bits": observed_bits,
        "theoretical_bits": theoretical_bits,
        "quantization_step": q,
        "min_value": levels[0],
        "max_value": levels[-1],
        "uniformly_quantized": uniformly_quantized,
    }

# ======================================================
# Remove outliers using a median-based method
# ======================================================


def plot_signal_outliers(x: ArrayLike, x_no_outliers: ArrayLike, mask: ArrayLike, title: str = "Signal with Outliers"):
    """
    Plot original signal and highlight the detected outliers,
    and also the interpolated signal after outlier removal.

    Parameters
    ----------
    x : array_like
        Input signal.
    mask : array_like
        Boolean array indicating the positions of the detected outliers.
    title : str, optional
        Title of the plot. Default is "Signal with Outliers".
    """
    import matplotlib.pyplot as plt

    x = np.asarray(x)
    mask = np.asarray(mask, dtype=bool)

    plt.figure(figsize=(12, 6))
    plt.plot(x, label='Signal', color='blue')
    plt.plot(x_no_outliers, label='Signal after Outlier Removal', color='green')
    plt.scatter(np.where(mask)[0], x[mask],
                color='red', label='Outliers', zorder=5)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()


def remove_outliers(
    input_signal: ArrayLike,
    fs: float,
    win_ms: float = 200.0,
    k: float = 10.0,
    should_plot: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """
    Remove impulsive amplitude outliers from a signal using a
    Hampel-like filter based on a moving median and median absolute
    deviation (MAD).

    Samples whose deviation from the local median exceeds ``k`` times
    the local robust standard deviation estimate are classified as
    outliers and replaced by values obtained through linear
    interpolation between neighboring valid samples. This preprocessing
    step is useful for suppressing large digitization artifacts before
    applying conventional filters.

    Parameters
    ----------
    input_signal : array_like
        Input signal.
    fs : float
        Sampling frequency in Hz.
    win_ms : float, optional
        Length of the moving window in milliseconds used to estimate
        the local median and MAD. The corresponding number of samples
        is automatically rounded to the nearest odd integer.
        Default is 200 ms.
    k : float, optional
        Threshold multiplier applied to the local robust standard
        deviation estimate. Samples satisfying

            abs(input_signal[n] - median[n]) > k * sigma[n]

        are considered outliers. Larger values make the detector less
        sensitive. Default is 6.0.

    Returns
    -------
    y : ndarray
        Signal after replacing detected outliers by linear
        interpolation.
    mask : ndarray of bool
        Boolean array indicating the positions of the detected
        outliers. Elements equal to ``True`` correspond to samples
        that were replaced.

    Notes
    -----
    The robust standard deviation estimate is computed as

    .. math::

        \\sigma[n] = 1.4826 \\, \\mathrm{MAD}[n],

    where

    .. math::

        \\mathrm{MAD}[n] = \\mathrm{median}(|input_signal[n]-m[n]|),

    and :math:`m[n]` is the moving median.

    This method is particularly effective for removing isolated spikes
    caused by ADC errors, packet losses, or other impulsive artifacts,
    and it should generally be applied before bandpass filtering to
    avoid spreading the energy of the outliers over time.

    Examples
    --------
    >>> x_clean, mask = remove_outliers(x, fs=500)
    >>> y = scipy.signal.filtfilt(b, a, x_clean)

    See Also
    --------
    scipy.signal.medfilt : Median filter.
    numpy.interp : One-dimensional linear interpolation.
    """
    x = np.asarray(input_signal, dtype=np.float64)
    x = x - np.mean(x)  # Remove DC offset

    # Implement a Hampel filter to remove impulsive outliers from the signal:
    # Compute moving median m[n].
    # Compute residuals r[n].
    # Estimate local noise level using MAD.
    # Declare large deviations as outliers.
    # Replace outliers by interpolation.

    # odd window length
    win = int(round(win_ms * 1e-3 * fs))
    # use OR operator to force the least significant bit to be 1, which guarantees that win is odd.
    win = max(3, win | 1)

    if False:  # debug info
        print(
            f"  ⚠ Using window length of {win} samples ({win/fs*1000:.1f} ms) for outlier detection")
        print("Window in samples = ", win)

    # local median baseline
    med = medfilt(x, kernel_size=win)

    # local robust deviation estimate
    # estimate the local noise level in a way that is robust to outliers, but based on the
    # Median Absolute Deviation (MAD) instead of the standard deviation.
    r = x - med
    mad = medfilt(np.abs(r), kernel_size=win)
    # Convert MAD into an equivalent standard deviation
    # For Gaussian noise, MAD=0.6745 * sigma, so sigma = MAD / 0.6745 = 1.4826 * MAD
    # sigma is a robust estimate of the local standard deviation.
    sigma = 1.4826 * mad + 1e-12

    # detect impulsive outliers
    mask = np.abs(r) > k * sigma

    # replace outliers by linear interpolation
    # Only samples identified as outliers are modified. Therefore:
    # No outliers (mask all False) ==> y == x.
    y = x.copy()
    idx = np.arange(len(x))
    good = ~mask
    y[mask] = np.interp(idx[mask], idx[good], x[good])

    if should_plot:
        plot_signal_outliers(
            x, y, mask, title="Input / outuput signals and detected outliers")

    return y, mask


class BandpassButterworthFilter:
    def __init__(self, order, low, high, fs):
        self.order = order
        self.low = low
        self.high = high
        self.fs = fs

        self.Bz, self.Az = butter(
            order,
            [low, high],
            btype="band",
            fs=fs
        )


class LowpassButterworthFilter:
    def __init__(self, order, cutoff, fs):
        self.order = order
        self.cutoff = cutoff
        self.fs = fs

        self.Bz, self.Az = butter(
            order,
            [cutoff],
            btype="low",
            fs=fs
        )


def savitzky_golay_filtering(x, window=11, poly=3):
    """
    Applies Savitzky-Golay smoothing with safety checks.
    """
    n = len(x)

    # Window must be odd and <= n
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < 5:
        return x  # too short → skip

    if window % 2 == 0:
        window -= 1

    return savgol_filter(x, window_length=window, polyorder=poly)
