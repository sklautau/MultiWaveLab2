'''
Utility functions for plotting.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

import neurokit2 as nk
import pandas as pd


def plot_individual_beats(signals: pd.DataFrame, info: dict, title="Individual beats"):
    """
    Plot only the 'Individual beats' panel from NeuroKit2's ppg_plot().

    Parameters
    ----------
    signals : DataFrame
        Returned by nk.ppg_process().
    info : dict
        Returned by nk.ppg_process().
    """

    figsize = (4, 5)  # Adjust the figure size as needed
    fig, ax = plt.subplots(figsize=figsize)

    nk.ppg_segment(
        signals["PPG_Clean"].values,
        peaks=info["PPG_Peaks"],
        sampling_rate=info["sampling_rate"],
        show="return",
        ax=ax,
    )

    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_input_output_waveforms(input_wav, output_wav, fs, title, title1="Raw signal", title2="Processed signal"):
    t = np.arange(len(input_wav)) / fs

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

    axes[0].plot(t, input_wav, lw=0.8, color="#79B1CE")
    axes[0].set_title(f"Input: {title1}")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, output_wav, lw=0.8, color="#332288")
    axes[1].set_title(f"Output: {title2}")
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    axes[0].set_xlabel("Time (s)")
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


def plot_input_output_psds(
    input_wav,
    output_wav,
    fs,
    title,
    title1="Raw signal",
    title2="Processed signal",
    nperseg=None,
):
    """
    Plot the Power Spectral Density (PSD) of the input and output signals
    using Welch's method.

    Parameters
    ----------
    input_wav : ndarray
        Input signal.
    output_wav : ndarray
        Output signal.
    fs : float
        Sampling frequency (Hz).
    title : str
        Figure title.
    title1 : str, optional
        Title for the input PSD.
    title2 : str, optional
        Title for the output PSD.
    nperseg : int or None, optional
        Segment length passed to scipy.signal.welch().
        If None, scipy chooses an appropriate default.
    """

    f_in, psd_in = welch(input_wav, fs=fs, nperseg=nperseg)
    f_out, psd_out = welch(output_wav, fs=fs, nperseg=nperseg)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

    axes[0].semilogy(f_in, psd_in, lw=1.0, color="#79B1CE")
    axes[0].set_title(f"Input: {title1}")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD")
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(f_out, psd_out, lw=1.0, color="#332288")
    axes[1].set_title(f"Output: {title2}")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def plot_rmse_matrix(mse_dict, notes=""):
    """
    Convert a dict { (methodA, methodB): mse_value } into a symmetric
    RMSE matrix and plot it with annotations.
    The detector order follows the order of appearance in the dictionary.
    """

    # -------- Build detector list in order of appearance --------
    detectors = []
    for (a, b) in mse_dict.keys():
        if a not in detectors:
            detectors.append(a)
        if b not in detectors:
            detectors.append(b)

    n = len(detectors)

    # -------- Initialize RMSE matrix --------
    rmse_matrix = np.zeros((n, n))

    # -------- Fill the matrix --------
    for (a, b), mse in mse_dict.items():
        i = detectors.index(a)
        j = detectors.index(b)
        rmse = np.sqrt(mse)
        rmse_matrix[i, j] = rmse
        rmse_matrix[j, i] = rmse  # symmetrical

    # -------- Plot --------
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(rmse_matrix)

    # Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("RMSE", rotation=90)

    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(detectors, rotation=45, ha="right")
    ax.set_yticklabels(detectors)

    # -------- Annotate each cell --------
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{rmse_matrix[i, j]:.2f}",
                    ha="center", va="center",
                    color="white")

    ax.set_title("RMSE Between Detectors" + (" - " + notes if notes else ""))
    plt.tight_layout()
    plt.show()


def old_sorted_order_plot_rmse_matrix(mse_dict):
    """
    Convert a dict { (methodA, methodB): mse_value } into a symmetric
    RMSE matrix and plot it with annotations.
    """

    # -------- Extract the list of unique detector names --------
    detectors = sorted({k[0] for k in mse_dict} | {k[1] for k in mse_dict})
    n = len(detectors)

    # -------- Initialize square RMSE matrix --------
    rmse_matrix = np.zeros((n, n))

    # -------- Fill the symmetric RMSE matrix --------
    for (a, b), mse in mse_dict.items():
        i = detectors.index(a)
        j = detectors.index(b)
        rmse = np.sqrt(mse)
        rmse_matrix[i, j] = rmse
        rmse_matrix[j, i] = rmse  # symmetry

    # -------- Plot --------
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(rmse_matrix)

    # Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("RMSE", rotation=90)

    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(detectors, rotation=45, ha="right")
    ax.set_yticklabels(detectors)

    # -------- Annotate each cell --------
    for i in range(n):
        for j in range(n):
            value = rmse_matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center",
                    va="center", color="white")

    ax.set_title("RMSE Between Detectors")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# Plot comparison in subplots
# ---------------------------------------------------------


def plot_signal_with_estimated_points(signal, results, fs, keys_name="peaks", note=""):
    '''
    Plot the signal and the detected peaks from multiple detectors in subplots.
    synchronize zoom/pan across all subplots by sharing the x-axis among them.
    Matplotlib already has built-in support for this: all you need is to create the subplots with a shared x-axis (sharex=ax0), or use plt.subplots(..., sharex=True).
    '''
    t = np.arange(len(signal)) / fs
    num_detectors = len(results)

    fig, axes = plt.subplots(num_detectors, 1, figsize=(14, 10), sharex=True)

    # In case num_detectors == 1, wrap axes in a list
    if num_detectors == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        ax.plot(t, signal, label="PPG Signal")

        if len(res[keys_name]) > 0:
            ax.scatter(t[res[keys_name]], signal[res[keys_name]],
                       color='g', marker='o', s=30, label="Estimated values")

        ax.set_title(res["name"] + (" - " + note if note else ""))
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")  # only bottom plot has x-label
    fig.tight_layout()
    plt.show()


def plot_signal_with_sqi(
    signal,
    results,
    fs,
    note="",
    min_value=0,
    max_value=1
):
    """
    Plot SQI comparisons in two figures with three subplots each:
    1) Time-domain waveforms (full signal, 10 s best SQI window, 10 s worst SQI window)
    2) Frequency-domain Welch PSDs for the same three windows
    """

    # ----- scale PPG to [min_value, max_value] -----
    sig_min = np.min(signal)
    sig_max = np.max(signal)

    if sig_max == sig_min:
        scaled_signal = np.zeros_like(signal)
    else:
        scaled_signal = (signal - sig_min) / (sig_max - sig_min)
        scaled_signal = scaled_signal * (max_value - min_value) + min_value

    n_samples = len(signal)
    t = np.arange(n_samples) / fs

    # colors and styles for multiple SQI curves
    sqi_colors = ["red", "blue", "green",
                  "orange", "purple", "magenta", "brown"]
    sqi_styles = ["-", "--", "-.", ":",
                  (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]

    valid_sqi_waveforms = []
    sqi_plot_meta = []

    for idx, res in enumerate(results):

        sqi = np.asarray(res.get("sqi_for_each_signal_sample", []))

        if sqi.size == 0:
            print(f"[WARNING] {res['name']} has no SQI.")
            continue

        if len(sqi) != len(signal):
            print(
                f"[WARNING] {res['name']} SQI length mismatch: len(sqi)={len(sqi)} vs len(signal)={len(signal)}.")
            continue

        # pick unique color and style
        color = sqi_colors[idx % len(sqi_colors)]
        style = sqi_styles[idx % len(sqi_styles)]

        valid_sqi_waveforms.append((res["name"], sqi))
        sqi_plot_meta.append({
            "label": f"SQI ({res['name']})",
            "data": sqi,
            "color": color,
            "linestyle": style,
            "linewidth": 1.4,
        })

    # Sum valid SQIs sample-wise to create total_sqi.
    if valid_sqi_waveforms:
        total_sqi = np.sum([w[1] for w in valid_sqi_waveforms], axis=0)
    else:
        total_sqi = np.zeros_like(scaled_signal)
        print("[WARNING] No valid SQI vectors were found. total_sqi is all zeros.")

    # Build centered fixed 10 s windows around max/min total_sqi while
    # excluding first/last 6 s from the max/min search.
    window_sec = 10.0
    exclude_edge_sec = 6.0
    window_samples = max(2, int(round(window_sec * fs)))
    half_window_samples = window_samples // 2
    exclude_edge_samples = int(round(exclude_edge_sec * fs))

    def _fixed_window_bounds(center_idx):
        start = center_idx - half_window_samples
        end = start + window_samples
        return start, end

    if n_samples < window_samples:
        print(
            f"[WARNING] Signal shorter than 10 s (len={n_samples/fs:.2f} s). Using full signal for zoom panels.")
        idx_max = int(np.argmax(total_sqi))
        idx_min = int(np.argmin(total_sqi))
        best_bounds = (0, n_samples)
        worst_bounds = (0, n_samples)
    else:
        # Centers that produce valid fixed-size 10 s windows.
        center_min_valid = half_window_samples
        center_max_valid = n_samples - window_samples + half_window_samples

        # Search region that avoids endpoints by 6 s.
        search_start = max(exclude_edge_samples, center_min_valid)
        search_end = min(n_samples - exclude_edge_samples -
                         1, center_max_valid)

        # Fallback if exclusion is too restrictive for short signals.
        if search_start > search_end:
            search_start = center_min_valid
            search_end = center_max_valid

        search_slice = total_sqi[search_start:search_end + 1]
        idx_max = int(search_start + np.argmax(search_slice))
        idx_min = int(search_start + np.argmin(search_slice))

        best_bounds = _fixed_window_bounds(idx_max)
        worst_bounds = _fixed_window_bounds(idx_min)

    full_bounds = (0, n_samples)

    panels = [
        ("Complete duration", full_bounds),
        ("10 s around highest total_sqi", best_bounds),
        ("10 s around lowest total_sqi", worst_bounds),
    ]

    ppg_waveform = {
        "label": "PPG (scaled)",
        "data": scaled_signal,
        "color": "black",
        "linestyle": "-",
        "linewidth": 1.2,
    }

    waveforms_top_panel = [ppg_waveform, *sqi_plot_meta]

    # ---------------- Figure 1: time domain ----------------
    fig_time, axes_time = plt.subplots(3, 1, figsize=(14, 14), sharey=True)

    for panel_idx, (ax, (panel_title, bounds)) in enumerate(zip(axes_time, panels)):
        start, end = bounds

        if panel_idx == 0:
            # Top panel: full duration with PPG + SQI curves.
            for wf in waveforms_top_panel:
                ax.plot(
                    t[start:end],
                    wf["data"][start:end],
                    color=wf["color"],
                    linestyle=wf["linestyle"],
                    linewidth=wf["linewidth"],
                    label=wf["label"],
                )
        else:
            # Zoom panels: show only the scaled PPG waveform.
            ax.plot(
                t[start:end],
                ppg_waveform["data"][start:end],
                color=ppg_waveform["color"],
                linestyle=ppg_waveform["linestyle"],
                linewidth=ppg_waveform["linewidth"],
                label=ppg_waveform["label"],
            )

        ax.set_title(panel_title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude / SQI")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)

    time_title = "SQI comparison in time domain"
    if note:
        time_title += " - " + note
    fig_time.suptitle(time_title)
    fig_time.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()

    # ---------------- Figure 2: frequency domain (Welch PSD) ----------------
    fig_psd, axes_psd = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    plot_index = 0
    for ax, (panel_title, bounds) in zip(axes_psd, panels):
        start, end = bounds
        segment = ppg_waveform["data"][start:end]
        if len(segment) < 4:
            print(
                f"[WARNING] Segment too short for PSD ({ppg_waveform['label']}) in '{panel_title}'.")
            continue
        if plot_index == 0:
            nperseg = min(8192, len(segment))
        else:
            nperseg = min(128, len(segment))
        freqs, psd = welch(segment, fs=fs, nperseg=min(1024, len(segment)))
        ax.semilogy(
            freqs,
            psd,
            color=ppg_waveform["color"],
            linestyle=ppg_waveform["linestyle"],
            linewidth=max(1.0, ppg_waveform["linewidth"] - 0.2),
            label=ppg_waveform["label"],
        )

        ax.set_title(panel_title)
        ax.set_ylabel("PSD")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)
        plot_index += 1

    axes_psd[-1].set_xlabel("Frequency (Hz)")

    psd_title = "SQI comparison in frequency domain (Welch PSD)"
    if note:
        psd_title += " - " + note
    fig_psd.suptitle(psd_title)
    fig_psd.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


def plot_only_signal_with_sqi(
    signal,
    results,
    fs,
    note="",
    min_value=0,
    max_value=1
):
    """
    Plot the scaled PPG signal and superimpose SQI curves
    from multiple detectors on a SINGLE plot.
    """

    # ----- scale PPG to [min_value, max_value] -----
    sig_min = np.min(signal)
    sig_max = np.max(signal)

    if sig_max == sig_min:
        scaled_signal = np.zeros_like(signal)
    else:
        scaled_signal = (signal - sig_min) / (sig_max - sig_min)
        scaled_signal = scaled_signal * (max_value - min_value) + min_value

    t = np.arange(len(signal)) / fs

    # Create ONE plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # plot scaled PPG once
    ax.plot(t, scaled_signal, color="black",
            linewidth=1.2, label="PPG (scaled)")

    # colors and styles for multiple SQI curves
    sqi_colors = ["red", "blue", "green",
                  "orange", "purple", "magenta", "brown"]
    sqi_styles = ["-", "--", "-.", ":",
                  (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]

    for idx, res in enumerate(results):

        sqi = np.asarray(res.get("sqi_for_each_signal_sample", []))

        if sqi.size == 0:
            print(f"[WARNING] {res['name']} has no SQI.")
            continue

        if len(sqi) != len(signal):
            print(
                f"[WARNING] {res['name']} SQI length mismatch: len(sqi)={len(sqi)} vs len(signal)={len(signal)}.")
            continue

        # pick unique color and style
        color = sqi_colors[idx % len(sqi_colors)]
        style = sqi_styles[idx % len(sqi_styles)]

        ax.plot(
            t,
            sqi,
            color=color,
            linestyle=style,
            linewidth=1.4,
            label=f"SQI ({res['name']})"
        )

    # Title, labels, grid
    title = "SQI comparison"
    if note:
        title += " - " + note
    ax.set_title(title)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude / SQI")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_length_histogram(df, key_name, bins=30):
    """
    Plot histogram of segment durations stored in df[key_name]
    and print total value.
    """

    # Extract durations
    durations = df[key_name].values

    # --- Plot histogram ---
    plt.figure(figsize=(10, 5))
    plt.hist(durations, bins=bins, edgecolor="black")
    plt.xlabel(key_name)
    plt.ylabel("Count")
    plt.title("Histogram")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # --- Compute and print total duration ---
    try:
        durations = durations.astype(float)
        total_duration = durations.sum()
        print(
            f"Total duration across all segments: {total_duration:.2f} seconds")
    except Exception as e:
        print(f"Could not compute total sum of durations: {e}")
