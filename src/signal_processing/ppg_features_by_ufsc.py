import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from scipy.interpolate import interp1d

# import scipy as sc
from scipy.stats import skew
from scipy.signal import find_peaks, welch
from scipy.signal import savgol_filter
from scipy.signal import butter, lfilter, lfilter_zi, savgol_filter, find_peaks
'''
from helper.preprocessing_baseline_lib import preprocess, max_min_normalization
from helper.pulse_extraction import extract_pulses
from helper.signal_quality_index import SQI
from helper.pulse_features import pulseFeatures
from helper.resampling import resample_signal
from helper.baseline_removal import remove_baseline
'''
import neurokit2 as nk
from scipy.stats import kurtosis
from statistics import variance
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast


f_settings = {
    'lc': 0.5,
    'hc': 15,
    'o_b': 4,
    'frame': 19,
    'o_svg': 4,
    'use_butter': True,
    'use_svg': True,
    'use_minmax': True
}

# Debug flag to explain why UFSC extraction returns None.
# Keep False in production to avoid verbose logs.
DEBUG_UFSC_EXTRACTION = False


def _debug_ufsc(verbose: bool, message: str) -> None:
    """Print UFSC debug messages when verbose mode or debug flag is enabled."""
    if verbose or DEBUG_UFSC_EXTRACTION:
        print(message)


def extract_HRV_features_by_ufsc(ppg_frame: np.ndarray, fs: int, show: bool = False) -> pd.DataFrame:
    """Extract time-domain HRV features from a PPG frame."""
    # Get peaks
    peaks = nk.ppg_findpeaks(ppg_frame, sampling_rate=fs, show=False)
    peaks = list(peaks['PPG_Peaks'])
    # Compute HRV features
    hrv_results = nk.hrv_time(peaks, sampling_rate=fs, show=False)
    # ex = hrv.HRV_extractor(peaks, fs, None)
    # hrv_results = ex.compute(mode='all')
    # hrv_results = pd.DataFrame(hrv_results, index=[0])

    if (show):
        print(peaks)
        print(hrv_results)
        plt.plot(ppg_frame)
        plt.scatter(peaks, ppg_frame[peaks], c='r')
        plt.show()

    return hrv_results


def extract_psd_features_by_ufsc(ppg_frame: np.ndarray, fs: int, show: bool = False) -> pd.DataFrame:
    '''
    @brief: Function to extract PSD features
    @param ppg_frame: signal to be processed
    @param fs: sampling frequency
    @return: psd features
    '''
    # Extract PSD
    freqs, psd = welch(ppg_frame, fs=fs)

    mean_psd = np.mean(psd)
    std_psd = np.std(psd)
    var_psd = variance(psd)
    kurtosis_psd = kurtosis(psd)
    # Extract peaks from PSD to find first and second harmonic
    peaks_psd, _ = find_peaks(psd)

    # Get the first harmonic
    max_power_1st = np.max(psd)
    idx_max_power_1st = np.argmax(psd)
    f_max_1st = freqs[idx_max_power_1st]

    # Remove the peaks before first harmonic
    peaks_psd = peaks_psd[peaks_psd >= idx_max_power_1st]

    if (len(peaks_psd) > 1):
        f_max_2nd = freqs[peaks_psd[1]]
        max_power_2nd = psd[peaks_psd[1]]
        idx_max_power_2nd = peaks_psd[1]
    else:
        f_max_2nd = 0
        max_power_2nd = 0
        idx_max_power_2nd = 0

    if (show):
        fig, axs = plt.subplots(2, 1, figsize=(9, 7))
        axs[0].plot(ppg_frame, 'purple')
        axs[0].set_xlabel('Samples @60Hz', fontsize=14, fontweight='bold')
        axs[0].set_ylabel('Amp. (n.u.)', fontsize=14, fontweight='bold')

        axs[1].plot(freqs, psd)
        axs[1].plot(freqs[idx_max_power_1st], max_power_1st, 'ro')
        axs[1].plot(freqs[idx_max_power_2nd], max_power_2nd, 'ro')
        axs[1].set_xlim([0, 15])
        axs[1].set_xlabel('Frequency [Hz]', fontsize=14, fontweight='bold')
        axs[1].set_ylabel('PSD [V**2/Hz]', fontsize=14, fontweight='bold')
        plt.show()

    df_features = pd.DataFrame()
    df_features['MAX_POWER_1st'] = [max_power_1st]
    df_features['IDX_MAX_1st'] = [idx_max_power_1st]
    df_features['F_MAX_1st'] = [f_max_1st]
    df_features['MAX_POWER_2nd'] = [max_power_2nd]
    df_features['IDX_MAX_2nd'] = [idx_max_power_2nd]
    df_features['F_MAX_2nd'] = [f_max_2nd]
    df_features['MEAN_PSD'] = [mean_psd]
    df_features['STD_PSD'] = [std_psd]
    df_features['VAR_PSD'] = [var_psd]
    df_features['KUR_PSD'] = [kurtosis_psd]

    # return max_power_1st, f_max_1st, idx_max_power_1st, max_power_2nd, f_max_2nd, idx_max_power_2nd, mean_psd, std_psd, var_psd, kurtosis_psd
    return df_features


def remove_baseline(signal: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate and remove baseline using valley interpolation between peaks."""
    # Extract peaks using scipy
    peaks, _ = find_peaks(signal, height=0.5, distance=int(fs/2))

    # Flat or very noisy signals can produce too few peaks for valley interpolation.
    # In this case, keep the original signal and return empty valley metadata.
    if len(peaks) < 2:
        return signal, np.zeros_like(signal), peaks, np.asarray([], dtype=int)

    valley = []
    for e, peak in enumerate(peaks):
        if (e < len(peaks)-1):
            sig_wind = signal[peak:peaks[e+1]]
            valley.append(np.argmin(sig_wind)+peak)

    # Interpolate all valleys
    valley = np.asarray(valley, dtype=int)
    if valley.size == 0:
        return signal, np.zeros_like(signal), peaks, valley
    valley_int = np.interp(np.arange(len(signal)), valley, signal[valley])

    signal_baseline = signal - valley_int

    return signal_baseline, valley_int, peaks, valley


def obtain_isolated_pulses(ppg_frame: np.ndarray, fs_orig: int, fs_out: int):
    '''
    Obtain each pulse.
    '''
    # Resample signal frame
    ppg_frame_resampled = resample_signal(
        ppg_frame, fs_inp=fs_orig, fs_out=fs_out)
    # Extract the individual pulses
    pulses = extract_pulses(ppg_frame_resampled, show=False)
    return pulses


def extract_isolated_pulse_features_by_ufsc(pulses: List[np.ndarray], fs: int):
    '''
    Generate features set for each pulse.
    '''
    features = list()
    for e_pulse in range(len(pulses)-1):
        # print("#### Debug: e_pulse = ", e_pulse)
        # Get the pulse
        pulse = pulses[e_pulse]
        # Adjust pulse length
        pulse = adjust_signal(pulse, samples=fs)
        # Extract pulse features
        pf = pulseFeatures(pulse, fs)
        pf.compute(show=False)

        # Create dataframe
        fts = np.asarray(pf.features)
        fts = np.reshape(fts, (1, len(fts)))
        # df = pd.DataFrame(fts, columns=pf.fnames, index=[0])
        features.append(pf.features)
    # Create dataframe with all the features
    try:
        df_features = pd.DataFrame(features, columns=pf.fnames)
    except:
        df_features = pd.DataFrame()
    return df_features


def filter_based_on_shape(ppg_frame: np.ndarray, fs_orig: int, fs_out: int) -> Tuple[pd.DataFrame, bool, List[np.ndarray]]:
    """Keep pulses with good morphology/SQI and return per-pulse features."""

    # Resample signal frame
    ppg_frame_resampled = resample_signal(
        ppg_frame, fs_inp=fs_orig, fs_out=fs_out)
    # Extract the individual pulses
    pulses = extract_pulses(ppg_frame_resampled, show=False)
    # SQI for each one of the pules
    # print("INSIDE filter_based_on_shape()")
    # sqi_pulses, sqi_signal = SQI(pulses) # too slow version
    sqi_pulses, sqi_signal = SQI_fast(pulses)

    # Walk through each of the pulses
    features = []
    remove_list = []
    good_quality_pulse = []
    for e_pulse in range(len(pulses)-1):
        # print("#### Debug: e_pulse = ", e_pulse)
        # Get the pulse SQI
        pulse = pulses[e_pulse]
        # Get the pulse SQI
        sqi = sqi_pulses[e_pulse]
        # Check if the SQI is optimal
        if ((sqi > 0.80) and (len(pulse) < int(fs_out+(fs_out*0.2))) and (len(pulse) > int(fs_out-(fs_out*0.6)))):
            # Adjust pulse length
            pulse = adjust_signal(pulse, samples=fs_out)
            # try:
            if (True):
                # Extract pulse features
                pf = pulseFeatures(pulse, fs_out)
                pf.compute(show=False)

                # Create dataframe
                fts = np.asarray(pf.features)
                fts = np.reshape(fts, (1, len(fts)))
                df = pd.DataFrame(fts, columns=pf.fnames, index=[0])

                # Check if the pulse is valid
                peak_pos_v = cast(float, df.loc[0, 'peak_pos'])
                skewness_v = cast(float, df.loc[0, 'skewness'])
                ipa_v = cast(float, df.loc[0, 'IPA'])
                if ((peak_pos_v > 150) or (skewness_v <= 0) or (skewness_v >= 2) or (ipa_v >= 1.5)):
                    remove_list.append(True)
                else:
                    features.append(pf.features)
                    remove_list.append(False)
                    good_quality_pulse.append(pulse)
            # except:
            #    df = pd.DataFrame()

        else:
            remove_list.append(True)

    # Create dataframe with all the features
    try:
        df_features = pd.DataFrame(features, columns=pf.fnames)
    except:
        df_features = pd.DataFrame()

    # After evaluating all the pulses, remove the window if there is more than one pulse with a bad SQI
    if (len(remove_list) != 0):
        # Remove first pulse and not consider
        remove_list.pop(0)
        pulse_list = np.where(remove_list)[0]
    else:
        pulse_list = [0, 1, 2, 3]
        good_quality_pulse = []

    # Originally, only one bad pulse was allowed
    # if (len(pulse_list) > 1 or (len(pulses) < 5)):
    if (len(pulses) < 5):
        is_valid = False
    else:
        is_valid = True

    return df_features, is_valid, good_quality_pulse


def resample_signal(signal_inp: np.ndarray, fs_inp: int, fs_out: int) -> np.ndarray:
    """
    Resample the one-dim or multi-dim signal with axis=0 for time axis
    from sampling frequency fs_inp to frequency fs_out.

    :param signal_inp: one-dim or multi-dim input signal array with axis=0 for time axis
    :param fs_inp: sampling frequency of the input signal
    :param fs_out: sampling frequency of the output signal
    :return: resampled signal_out
    """
    num_inp = signal_inp.shape[0]
    ts_inp = np.arange(num_inp, dtype=np.float64)
    ts_inp /= fs_inp

    num_out = int(ts_inp[-1] * fs_out) + 1
    if (num_out-1) / fs_out > ts_inp[-1]:
        num_out -= 1

    ts_out = np.arange(num_out, dtype=np.float64)
    ts_out /= fs_out

    interpolate_cubic = interp1d(
        ts_inp, signal_inp, axis=0, kind='cubic')    # cubic spline
    signal_out = interpolate_cubic(ts_out)

    return signal_out


def max_min_normalization(data: Sequence[float]) -> List[float]:
    """Scale a 1-D signal to the [0, 1] range using min-max normalization."""
    # Min Max Normalization ---------------------------------------------------------------------
    # Normalize the Data
    normalized, mi, ma = [], min(data), max(data)
    for x in data:
        if (ma - mi) != 0:
            y = (x - mi) / (ma - mi)
        else:
            y = 1e10
        normalized.append(y)
    return normalized


def butter_bandpass_filter_zi(data: Sequence[float], lowcut: float, highcut: float, sRate: int, order: int = 5) -> np.ndarray:
    """Apply band-pass filtering while compensating the initial transient."""
    # This function will apply the filter considering the initial transient.
    b, a = butter_bandpass(lowcut, highcut, sRate, order=order)
    zi = lfilter_zi(b, a)
    y, zo = lfilter(b, a, data, zi=zi*data[0])
    return y


def butter_bandpass(lowcut: float, highcut: float, sRate: int, order: int = 5) -> Any:
    """Create Butterworth band-pass filter coefficients."""
    nyq = 0.5 * sRate
    low = lowcut / nyq
    high = highcut / nyq
    coeffs = butter(order, [low, high], btype='band')
    b, a = coeffs[0], coeffs[1]
    return b, a


def savitzky_golay_filter(data: Sequence[float], frame: int, order: int) -> np.ndarray:
    """Smooth a signal with a Savitzky-Golay filter."""
    # Savitzky-Golay filter ---------------------------------------------------------------------
    return savgol_filter(data, frame, order)  # frame lenght, polynomial order


class pulseFeatures:
    """Extract pulse-shape features from a single normalized pulse."""

    def __init__(self, pulse: Sequence[float], fps: int) -> None:
        """Store pulse samples and sampling rate used by feature extraction."""
        self.pulse = np.asarray(pulse)
        self.fps = fps

    def getFeatures(self) -> pd.DataFrame:
        """Return currently computed features as a DataFrame."""
        df = pd.DataFrame(self.features, columns=self.fnames)
        return df

    def compute(self, show: bool = False) -> None:
        """Compute all pulse morphology features and optionally plot diagnostics."""
        # Peak Value
        peak_value = np.max(self.pulse)
        # Peak position
        peak_pos = np.argmax(self.pulse)

        # Derivatives
        vpg = np.diff(self.pulse, n=1)
        vpg = list(vpg)
        vpg.append(vpg[-1])
        # apg = np.diff(self.pulse, n=2)
        apg = np.diff(vpg)
        apg = list(apg)
        apg.append(apg[-1])

        apg = savitzky_golay_filter(apg, 30, 4)

        # Rising time
        segment = apg[0: peak_pos]
        if segment.size > 0:
            rising_start = np.argmax(segment)

            rising_end = peak_pos
            rising_value = self.pulse[rising_end] - self.pulse[rising_start]
            rising_time = rising_end-rising_start
            rising_time_ms = (rising_time/self.fps)*1000
        else:
            if peak_pos == 0:
                rising_start = 0
                rising_end = 0
                rising_value = 0
                rising_time = 0
                rising_time_ms = 0
            else:
                raise Exception("This should not happen! peak_pos = " +
                                str(peak_pos) + " and apg=" + str(apg))

        # Skewness, Kurtosis, Variance
        skewness = skew(self.pulse)
        kurtosis_f = kurtosis(self.pulse, fisher=True)
        kurtosis_p = kurtosis(self.pulse, fisher=False)
        variance = np.var(self.pulse)

        # More robust code
        # Width: 10, 25, 50, 75
        delta_amplitude = np.max(self.pulse) - np.min(self.pulse)
        half_amplitude = (delta_amplitude/2) + np.min(self.pulse)

        # Check peak_pos validity
        if peak_pos <= 0 or peak_pos >= len(self.pulse):
            # Can't compute widths reliably
            # w10 = w25 = w50 = w75 = np.nan
            w10 = w25 = w50 = w75 = -1
            w50a_x = w50a_y = -1
            w50b_x = w50b_y = -1
        else:
            half_pulse_a = np.asarray(self.pulse[0:peak_pos])
            half_pulse_b = np.asarray(self.pulse[peak_pos:])

            def safe_argmin(arr: np.ndarray, target: float) -> int:
                if arr.size == 0:
                    return 0
                return int(np.abs(arr - target).argmin())

            # Helper to compute width safely
            def compute_width(amplitude_level: float) -> Tuple[int, int, float]:
                xa = safe_argmin(half_pulse_a, amplitude_level)
                xb = safe_argmin(half_pulse_b, amplitude_level)
                xb += peak_pos
                return int(xa), int(xb), float((xb - xa) / self.fps)

            # W50
            w50a_x, w50b_x, w50 = compute_width(half_amplitude)
            w50a_y = self.pulse[w50a_x]
            w50b_y = self.pulse[w50b_x]

            # W25
            quarter_amplitude = (delta_amplitude/4) + np.min(self.pulse)
            w25a_x, w25b_x, w25 = compute_width(quarter_amplitude)
            w25a_y = self.pulse[w25a_x]
            w25b_y = self.pulse[w25b_x]

            # W75
            amplitude_75 = (delta_amplitude*0.75) + np.min(self.pulse)
            w75a_x, w75b_x, w75 = compute_width(amplitude_75)
            w75a_y = self.pulse[w75a_x]
            w75b_y = self.pulse[w75b_x]

            # W10
            amplitude_10 = (delta_amplitude*0.1) + np.min(self.pulse)
            w10a_x, w10b_x, w10 = compute_width(amplitude_10)
            w10a_y = self.pulse[w10a_x]
            w10b_y = self.pulse[w10b_x]

        # print("More robust code finished")

        # FIND NOTCH

        min_apg = np.argmin(apg)
        ROI_apg = apg[min_apg:]
        peaks, _ = find_peaks(ROI_apg, height=0)
        if (len(peaks) == 0):
            notch_x = 0
            notch_y = 0
            apg_peak_x = 0
        else:
            apg_peak_x = peaks[0]+min_apg
            peaks, _ = find_peaks(vpg, height=-0.2)
            # vpg_peak_x = peaks[-1]
            # notch_x = int((vpg_peak_x+apg_peak_x)/2)
            notch_x = apg_peak_x
            notch_y = self.pulse[notch_x]

        # FIND DIASTOLIC PEAK
        ROI_apg = apg[apg_peak_x:]
        diastolic_peak_x = np.argmin(ROI_apg)+apg_peak_x
        # diastolic_peak_x = int((diastolic_peak_x+vpg_peak_x)/2)
        diastolic_peak_y = self.pulse[diastolic_peak_x]

        # Diastolic slope
        diastolic_slope = (diastolic_peak_x - peak_pos)/self.fps

        # Compute the area using the composite trapezoidal rule.
        area = np.trapezoid(self.pulse, dx=5)
        area = ((np.round(area, 2)))

        # Based on the notch, find A1 and A2
        area_1, area_2 = 0, 0

        period_before_notch = self.pulse[0:notch_x]
        period_after_notch = self.pulse[notch_x:]
        area_1 = np.trapezoid(period_before_notch, dx=5)
        area_2 = np.trapezoid(period_after_notch, dx=5)
        if area_1 == 0:
            ipa = np.nan
        else:
            ipa = area_2/area_1

        # APG Data Points
        peaks, _ = find_peaks(apg, height=0, distance=15)
        if (len(peaks) < 2):
            apg_a_x, apg_b_x, apg_c_x = 0, 0, 0
        else:
            apg_a_x = peaks[0]
            apg_a_y = apg[apg_a_x]
            apg_c_x = peaks[1]
            apg_c_y = apg[apg_c_x]
            apg_b_x = np.argmin(apg[peaks[0]:peaks[1]])+peaks[0]
            apg_b_y = apg[apg_b_x]
        if (len(peaks) < 3):
            apg_d_x, apg_e_x = 0, 0
        else:
            apg_e_x = peaks[2]
            apg_e_y = apg[apg_e_x]
            apg_d_x = np.argmin(apg[peaks[1]:peaks[2]])+peaks[1]
            apg_d_y = apg[apg_d_x]

        if apg_a_x == 0:
            apg_a_x = 1e-12

        b_a = apg_b_x/apg_a_x
        b_a_time = (b_a/self.fps)*1000

        c_a = apg_c_x/apg_a_x
        c_a_time = (c_a/self.fps)*1000

        d_a = apg_d_x/apg_a_x
        d_a_time = (d_a/self.fps)*1000

        e_a = apg_c_x/apg_a_x
        e_a_time = (e_a/self.fps)*1000

        # PPG Amplitudes
        x_ppg = peak_value - self.pulse[-1]
        y_ppg = diastolic_peak_y - self.pulse[-1]
        if x_ppg == 0:
            augmentation_index = np.nan
            reflection_index = np.nan
        else:
            augmentation_index = (x_ppg-y_ppg)/x_ppg
            reflection_index = y_ppg/x_ppg
        delta_T = np.abs(peak_pos - diastolic_peak_x)
        delta_Tt = (np.abs(peak_pos - diastolic_peak_x)/self.fps)*1000
        # Pulse interval

        self.features = [peak_value, peak_pos, rising_time, skewness,
                         kurtosis_f, kurtosis_p, variance, w75, w50,
                         w25, w10, notch_y, diastolic_peak_y, diastolic_slope, area,
                         area_1, area_2, ipa, apg_a_x, apg_b_x, apg_c_x, apg_d_x, apg_e_x,
                         b_a, b_a_time, c_a, c_a_time, d_a, d_a_time, e_a, e_a_time,
                         x_ppg, y_ppg, augmentation_index, reflection_index, delta_T, delta_Tt]

        self.fnames = ['peak_value', 'peak_pos', 'rising_time', 'skewness',
                       'kurtosis_f', 'kurtosis_p', 'variance', 'w75', 'w50',
                       'w25', 'w10', 'notch_y', 'diastolic_peak_y', 'diastolic_slope', 'area',
                       'area_1', 'area_2', 'IPA', 'A', 'B', 'C', 'D', 'E', 'B/A', 'B/A_t', 'C/A', 'C/A_t',
                       'D/A', 'D/A_t', 'E/A', 'E/A_t', 'X', 'Y', 'AI', 'RI', 'DT', 'DTt']

        if (show):

            print("### FEATURES ###")
            print(">> Peak Value    : %.2f" % (peak_value))
            print(">> Peak POS      : %d" % (peak_pos))
            print(">> Rising Time   : %.2f" % (rising_time))
            print(">> Rising Time ms: %.2f" % (rising_time_ms))
            print(">> Rising Value  : %.2f" % (rising_value))
            print(">> Skewness      : %.2f" % (skewness))
            print(">> Kurtosis F    : %.2f" % (kurtosis_f))
            print(">> Kurtosis P    : %.2f" % (kurtosis_p))
            print(">> Variance      : %.2f" % (variance))
            print(">> W75           : %.2f" % (w75))
            print(">> W50           : %.2f" % (w50))
            print(">> W25           : %.2f" % (w25))
            print(">> W10           : %.2f" % (w10))
            print(">> Notch         : %.2f" % (notch_y))
            print(">> Diastolic Peak X: %.2f" % (diastolic_peak_x))
            print(">> Diastolic Peak Y: %.2f" % (diastolic_peak_y))
            print(">> Diastolic Slope: %.2f" % (diastolic_slope))
            print(">> Area T: %.2f" % (area))
            print(">> Area 1: %.2f" % (area_1))
            print(">> Area 2: %.2f" % (area_2))
            print(">> APG A: %.2f" % (apg_a_x))
            print(">> APG B: %.2f" % (apg_b_x))
            print(">> APG C: %.2f" % (apg_c_x))
            print(">> APG D: %.2f" % (apg_d_x))
            print(">> APG E: %.2f" % (apg_e_x))
            print(">> B/A : %.2f" % (b_a))
            print(">> B/A Time: %.2f" % (b_a_time))
            print(">> C/A : %.2f" % (c_a))
            print(">> C/A Time: %.2f" % (c_a_time))
            print(">> D/A : %.2f" % (d_a))
            print(">> D/A Time: %.2f" % (d_a_time))
            print(">> E/A : %.2f" % (e_a))
            print(">> E/A Time: %.2f" % (e_a_time))
            print(">> X PPG: %.2f" % (x_ppg))
            print(">> Y PPG: %.2f" % (y_ppg))
            print(">> IPA: %.2f" % (ipa))
            print(">> AI: %.2f" % (augmentation_index))
            print(">> RI: %.2f" % (reflection_index))
            print(">> DT: %.2f" % (delta_T))
            print(">> DTt: %.2f" % (delta_Tt))
            print(rising_start, rising_end)

            fig, axes = plt.subplots(1, 3, figsize=(10, 4))
            t_ax = np.linspace(0, len(self.pulse)/self.fps, len(self.pulse))
            axes[0].plot(self.pulse, label='PPG')
            axes[0].plot(peak_pos, peak_value, 'go')
            axes[0].plot([rising_start, rising_end], [
                         self.pulse[rising_start], self.pulse[rising_end]])
            axes[0].plot(rising_start, self.pulse[rising_start], '^')
            axes[0].plot(w50a_x, w50a_y, 'go')
            axes[0].plot(w50b_x, w50b_y, 'go')
            axes[0].plot([w50a_x, w50b_x], [w50a_y, w50b_y],
                         '--', color='black')

            # If there is a notch, than we can plot the specific areas
            if (notch_x != 0):
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                axes[0].axvline(notch_x, ymax=1, ymin=0, ls='--', color='red')

                ax = np.linspace(0, len(period_before_notch), len(
                    period_before_notch), endpoint=True)
                axes[0].fill_between(ax, period_before_notch, min(
                    period_before_notch), hatch='|', alpha=0.30)
                ax = np.linspace(len(period_before_notch), len(
                    self.pulse), len(period_after_notch), endpoint=True)
                axes[0].fill_between(ax, period_after_notch, min(
                    period_after_notch), hatch='/', alpha=0.30)

                axes[0].text(len(period_before_notch)+10, 0.5, 'A2', fontsize=16,
                             verticalalignment='top', bbox=props)
                axes[0].text(int(len(period_before_notch)/2), 0.5, 'A1', fontsize=16,
                             verticalalignment='top', bbox=props)
                # axs[1].plot(notch_x, notch_y, "r^", color = 'red')

            axes[0].plot(w25a_x, w25a_y, 'go')
            axes[0].plot(w25b_x, w25b_y, 'go')
            axes[0].plot([w25a_x, w25b_x], [w25a_y, w25b_y],
                         '--', color='black')

            axes[0].plot(w75a_x, w75a_y, 'go')
            axes[0].plot(w75b_x, w75b_y, 'go')
            axes[0].plot([w75a_x, w75b_x], [w75a_y, w75b_y],
                         '--', color='black')

            axes[0].plot(w10a_x, w10a_y, 'go')
            axes[0].plot(w10b_x, w10b_y, 'go')
            axes[0].plot([w10a_x, w10b_x], [w10a_y, w10b_y],
                         '--', color='black')

            axes[0].plot(notch_x, notch_y, '^')
            axes[0].plot(diastolic_peak_x, diastolic_peak_y, '^')

            axes[0].plot([peak_pos, diastolic_peak_x], [
                         self.pulse[peak_pos], self.pulse[diastolic_peak_x]])
            axes[0].legend()

            axes[1].plot(vpg, color='red', label='VPG')
            axes[1].hlines(0, xmin=0, xmax=len(vpg), color='black')
            axes[1].legend()

            axes[2].plot(apg, color='green', label='APG')
            axes[2].hlines(0, xmin=0, xmax=len(apg), color='black')
            axes[2].plot(apg_a_x, apg_a_y, 'go')
            axes[2].plot(apg_b_x, apg_b_y, 'go')
            axes[2].plot(apg_c_x, apg_c_y, 'go')
            axes[2].plot(apg_d_x, apg_d_y, 'go')
            axes[2].plot(apg_e_x, apg_e_y, 'go')
            # axes[0].set_title("PPG Pulse with Features")
            axes[0].set_ylabel('Amp.', fontweight='bold', fontsize=12)
            fig.supxlabel('Samples @200Hz', fontweight='bold', fontsize=12)

            axes[2].legend()
            plt.show()


def adjust_signal(signal: Sequence[float], samples: int) -> Any:
    '''
    @brief: Function to adjust the pulses to a fixed size
    @param signal: pulse to be adjusted
    @param samples: number of samples to be adjusted to
    @return: pulse adjusted
    '''
    if (len(signal) < (samples)):
        pad = samples - len(signal)
        signal_pad = np.pad(signal, (0, pad), 'edge')
    else:
        signal_pad = signal[0:samples]
    return signal_pad


def adjust_signal_edges(signal: Sequence[float], samples: int) -> Any:
    """Pad a pulse symmetrically at both edges up to the target sample count."""
    if (len(signal) < (samples)):
        pad = samples - len(signal)
        pad = int(pad/2)
        print("PAD VALUE: ", pad)
        signal_pad = np.pad(signal, pad_width=(pad, pad), mode='constant')
    else:
        signal_pad = signal[0:samples]
    return signal_pad


def crop_signal(ppg_frame: np.ndarray, fs: int) -> np.ndarray:
    '''
    @brief: Remove beginning and end of the signal
    @param: ppg_frame: signal to be cropped
    fs: sampling frequency
    @return: ppg_frame_crop: cropped signal
    '''
    # Extract peaks using scipy
    peaks, _ = find_peaks(ppg_frame, height=0.5, distance=int(fs/2))

    # Get signal from first to seconds peak
    roi_signal = ppg_frame[peaks[0]:peaks[1]]
    # Find the minimum value
    start_point = np.argmin(roi_signal)
    # Get the index of the minimum value
    start_signal = start_point+peaks[0]

    # Get signal from last two peaks
    roi_signal = ppg_frame[peaks[-2]:peaks[-1]]
    # Find the minimum value
    end_point = np.argmin(roi_signal)
    # Get the index of the minimum value
    end_signal = end_point+peaks[-2]

    # Crop signal to avoid noise at the beginning and end
    ppg_frame_crop = ppg_frame[start_signal:end_signal]
    # Pad signal to avoid edge effects
    ppg_frame_crop = np.pad(ppg_frame_crop, (start_signal, end_point), 'edge')

    # Normalize after cropping
    ppg_frame_crop = np.asarray(max_min_normalization(ppg_frame_crop))

    return ppg_frame_crop


def old_filter_based_on_shape(ppg_frame: np.ndarray, fs_orig: int, fs_out: int) -> Tuple[pd.DataFrame, bool, List[np.ndarray]]:
    '''
    @brief: This function filters the signal based on the shape of the pulses
    @param: ppg_frame: PPG signal frame (10 seconds)
    @param: fs_orig: Original sampling frequency
    @param: fs_out: Output sampling frequency
    @return: features from window and bollean indicating if the window is reliable or not
    '''
    # Resample signal frame
    ppg_frame_resampled = resample_signal(
        ppg_frame, fs_inp=fs_orig, fs_out=fs_out)
    # Extract the individual pulses
    pulses = extract_pulses(ppg_frame_resampled, show=False)
    # SQI for each one of the pules
    sqi_pulses, sqi_signal = SQI(pulses)

    # Walk through each of the pulses
    features = []
    remove_list = []
    good_quality_pulse = []
    for e_pulse in range(len(pulses)-1):
        # Get the pulse SQI
        pulse = pulses[e_pulse]
        # Get the pulse SQI
        sqi = sqi_pulses[e_pulse]
        # Check if the SQI is optimal
        if ((sqi > 0.80) and (len(pulse) < int(fs_out+(fs_out*0.2))) and (len(pulse) > int(fs_out-(fs_out*0.6)))):
            # Adjust pulse length
            pulse = adjust_signal(pulse, samples=fs_out)
            # try:
            if (True):
                # Extract pulse features
                pf = pulseFeatures(pulse, fs_out)
                pf.compute(show=False)

                # Create dataframe
                fts = np.asarray(pf.features)
                fts = np.reshape(fts, (1, len(fts)))
                df = pd.DataFrame(fts, columns=pf.fnames, index=[0])

                # Check if the pulse is valid
                peak_pos_v = cast(float, df.loc[0, 'peak_pos'])
                skewness_v = cast(float, df.loc[0, 'skewness'])
                ipa_v = cast(float, df.loc[0, 'IPA'])
                if ((peak_pos_v > 150) or (skewness_v <= 0) or (skewness_v >= 2) or (ipa_v >= 1.5)):
                    remove_list.append(True)
                else:
                    features.append(pf.features)
                    remove_list.append(False)
                    good_quality_pulse.append(pulse)
            # except:
            #    df = pd.DataFrame()

        else:
            remove_list.append(True)

    # Create dataframe with all the features
    try:
        df_features = pd.DataFrame(features, columns=pf.fnames)
    except:
        df_features = pd.DataFrame()

    # After evaluating all the pulses, remove the window if there is more than one pulse with a bad SQI
    if (len(remove_list) != 0):
        # Remove first pulse and not consider
        remove_list.pop(0)
        pulse_list = np.where(remove_list)[0]
    else:
        pulse_list = [0, 1, 2, 3]
        good_quality_pulse = []

    # Only one bad pulse allowed
    if (len(pulse_list) > 1 or (len(pulses) < 5)):
        return df_features, False, good_quality_pulse
    else:
        return df_features, True, good_quality_pulse


def extract_pulses(signal: np.ndarray, show: bool = False) -> List[np.ndarray]:
    """Split a PPG signal into individual pulses using valley boundaries."""
    # Extract peaks using scipy
    peaks, _ = find_peaks(signal, height=0.6, distance=50)
    # Filter out any double peak
    peaks = filter_double_peaks(peaks)

    valley = []
    last_peak = 0
    # Walk through all the peaks positions
    for e, p in enumerate(peaks):
        if (e != 0):
            # Get the signal segment between two peaks
            segment = signal[last_peak:p]
            # Find lowest value in this segment
            min_value = np.min(segment)
            pos_low = np.where(segment == min_value)[0]
            pos_low = pos_low[0]
            # Add valley
            valley.append(pos_low+last_peak)
            # Update variables
            last_peak = p
        else:
            valley.append(0)
            last_peak = p
    # print(valley)
    pulses = []
    for i in range(len(valley)-1):
        ip = valley[i]
        if (i == len(valley)):
            ep = len(signal)
        else:
            ep = valley[i+1]

        pulse = signal[ip:ep]
        pulses.append(pulse)

    # Plot All the signal with negative and positive peaks
    if (show):
        print(">> Found %d peaks and %d pulses" % (len(peaks), len(pulses)))
        plt.title("PPG Signal with highlighted peaks",
                  fontsize='large', fontweight='bold')
        plt.ylabel("Amplitude", fontsize='large', fontweight='bold')
        plt.xlabel("time", fontsize='large', fontweight='bold')
        plt.plot(signal)
        plt.plot([p for p in peaks], [signal[pk] for pk in peaks], "go")
        plt.plot([pn for pn in valley], [signal[pn] for pn in valley], "r^")
        plt.legend(['Signal', 'Peak', 'Vale'])

        plt.show()

    return pulses


def filter_double_peaks(peaks: np.ndarray) -> np.ndarray:
    '''
    @brief: Filter any double peak in a set of peaks
    @param: peaks is an array with the peaks indices
    @return: the corrected peak array
    '''
    aux_peaks = []
    for e, p in enumerate(peaks):
        if (e > 0):
            if (np.abs(p - peaks[e-1]) > 5):
                aux_peaks.append(p)
            # else is a double peaks and must be removed
        else:
            aux_peaks.append(p)

    peaks = np.asarray(aux_peaks)
    return peaks


def extract_pulse_features(pulse: Sequence[float], fs: int, show: bool = False) -> pd.DataFrame:
    """Extract pulse-shape features for one pulse and return a single-row DataFrame."""
    # Compute features
    pf = pulseFeatures(pulse, fs)
    pf.compute(show=show)
    # Create dataframe
    fts = np.asarray(pf.features)
    fts = np.reshape(fts, (1, len(fts)))
    df = pd.DataFrame(fts, columns=pf.fnames, index=[0])
    return df


def extract_features_from_segment(
    ppg_segment: np.ndarray,
    fs: int,
    fs_out: Optional[int] = None,
    window_size_in_secs: int = 2,
    ep: int = 0,
) -> Optional[pd.DataFrame]:
    """Extract robust PPG features from one segment, returning None when invalid."""
    verbose = False
    if fs_out == None:
        fs_out = fs

    # Get signal window
    # ppg_frame_raw_normal = ppg_normal[ip:ep]
    # ppg_frame_raw_high = ppg_high[ip:ep]

    # Process signal - Process and invert
    ppg_frame_normal_proc = np.asarray(preprocess(
        ppg_segment.copy(), fs, f_settings))
    # preprocess already normalizes between 0 and 1
    # no need to do it again below:
    # ppg_frame_normal_norm = np.asarray(
    #    max_min_normalization(ppg_frame_normal_proc.copy()))
    # ppg_frame_normal_inv = ppg_frame_normal_norm[::-1]
    ppg_frame_normal_inv = ppg_frame_normal_proc[::-1]
    ppg_frame_normal_brem, _, _, _ = remove_baseline(
        ppg_frame_normal_inv, fs)

    # Filter based on shape
    _, isValid_normal, pulses_normal = filter_based_on_shape(
        ppg_frame_normal_brem, fs, fs_out)

    if (isValid_normal):
        # Calculate the new coordinates
        ip = ep  # Init point
        ep = int(ip+(fs * window_size_in_secs))  # End point
        # print("We are going to extract an image because this window is reliable")

        # Remove beginning and end of the signal
        ppg_frame_crop_normal = crop_signal(ppg_frame_normal_brem, fs)

        # Average Pulses
        pad_pulses_normal = []
        ax = np.linspace(0, 200/fs_out, 200)
        for p in pulses_normal:
            p_pad = adjust_signal(p, 200)
            pad_pulses_normal.append(p_pad)
            # plt.plot(ax, p_pad, color="blue")

        avg_pulse_normal = np.mean(pad_pulses_normal, axis=0)
        # print("LEN PULSE: ", avg_pulse_normal.shape)
        # avg_pulse_normal = adjust_signal_edges(avg_pulse_normal, 300)
        # plt.plot(ax, avg_pulse_normal, color="red")
        # plt.xlabel('Time (s)', fontweight='bold', fontsize=12)
        # plt.ylabel('Average Amp. (n.u.)', fontweight='bold', fontsize=12)
        # plt.show()

        df_pulse = extract_pulse_features(
            avg_pulse_normal, fs_out, show=False)

        # print(df_hrv)
        # print(df_pulse)
        df_features = pd.concat([df_pulse], axis=1)
        if verbose:
            print(">> FEATURES FOR THE WINDOW:")
            print(df_features)
        return df_features


def old_process_all_input_files(
    database_folder: str,
    input_cvs: str = 'all_pacients.csv',
    output_cvs: str = 'selected_segments.csv',
    verbose: bool = False,
    should_show: bool = False
) -> None:
    """
    Main entry. Reads structured CSV (subject list), processes each subject,
    and writes collected pulse feature rows to output_cvs.
    """

    # Sampling settings
    fs_orig = 60
    fs_out = 200
    window_seconds = 10
    small_step_seconds = 2

    # Read dataset (do not treat first col as index)
    df = pd.read_csv(input_cvs)
    if verbose:
        print("Original dataset shape:", df.shape)
    # remove NOTES == 'A' rows
    if 'NOTES' in df.columns:
        df = df[df['NOTES'] != 'A'].reset_index(drop=True)
        if verbose:
            print("Filtered dataset shape:", df.shape)

    df_all_features = pd.DataFrame()   # empty dataframe

    num_subjects = df.shape[0]
    for idx in range(num_subjects):
        subject_id = str(df.loc[idx, 'SUBJECT_ID'])
        glc = df.loc[idx, 'GLC']

        print(
            f"Processing subject {idx+1}/{num_subjects} (ID={subject_id}, GLC={glc})")

        # folder is named 001 and id is #001. Remove "#"
        subject_folder = os.path.join(database_folder, subject_id[1:])

        if not os.path.isdir(subject_folder):
            raise Exception(
                f"Subject folder not found: {subject_folder}, skipping subject {subject_id}")

        # find wave file
        wave_files = [fn for fn in os.listdir(
            subject_folder) if fn.endswith('wave.csv')]
        if not wave_files:
            if verbose:
                print(f"No wave.csv for subject {subject_id}, skipping")
            continue

        wave_path = os.path.join(subject_folder, wave_files[0])
        try:
            sig_df = pd.read_csv(wave_path)
            raw_signal = sig_df['Wave'].values
        except Exception as e:
            print(f"Error reading {wave_path}: {e}")
            raise e

        # Process the subject's signal and collect rows
        new_df = process_all_segments_of_a_subject(
            raw_signal,
            fs_orig=fs_orig,
            fs_out=fs_out,
            window_seconds=window_seconds,
            small_step_seconds=small_step_seconds,
            glc=glc,
            subject_id=subject_id,
            verbose=verbose,
            should_show=should_show
        )

        if new_df is not None:
            df_all_features = pd.concat(
                [df_all_features, new_df], ignore_index=True)
            print(">> Collected rows (segments) for subject",
                  subject_id, ":", new_df.shape[0])
            print(new_df)
        else:
            print(">> No valid segments for subject", subject_id)

        print(
            f"Collected rows so far: {df_all_features.shape[0]}")

    df_all_features.to_csv(output_cvs, index=False)
    if verbose:
        print("Wrote file:", output_cvs)


def old_process_all_segments_of_a_subject(
    signal: Any,
    fs_orig: int,
    fs_out: int,
    window_seconds: int,
    small_step_seconds: int,
    glc: float,
    subject_id: str,
    verbose: bool,
    should_show: bool
) -> pd.DataFrame:
    """
    This processes all windows (segments) for a single subject's raw_signal.
    Slide windows across the raw_signal and process each window, returning a list of rows.
    """
    window_length = int(window_seconds * fs_orig)
    small_window_shift = int(small_step_seconds * fs_orig)

    rows = []  # store feature rows here
    ip = 0
    ep = ip + window_length

    while ep <= len(signal):
        if verbose:
            print(f"IP: {ip} EP: {ep} (signal_len={len(signal)})")

        segment = signal[ip:ep]

        result = extract_features_from_segment(
            segment, fs_orig, fs_out, window_seconds, ep)
        if result is not None:
            # Add extra info columns
            result['SUBJECT_ID'] = subject_id
            result['GLC'] = glc
            result['WINDOW_START'] = ip
            result['WINDOW_END'] = ep

            rows.append(result)

            if should_show:
                print("Features for segment:")
                print(result)

            # Move window by full step
            ip += ep
            ep = ip + window_length
        else:
            # Move by small step and re-evaluate
            ip += small_window_shift
            ep = ip + window_length
    return result


def process_all_segments_of_a_subject(
    signal: Any,
    fs_orig: int,
    fs_out: int,
    window_seconds: int,
    small_step_seconds: int,
    glc: float,
    subject_id: str,
    verbose: bool,
    should_show: bool,
) -> pd.DataFrame:
    """Backward-compatible alias for the legacy per-subject segment processor."""
    return old_process_all_segments_of_a_subject(
        signal,
        fs_orig,
        fs_out,
        window_seconds,
        small_step_seconds,
        glc,
        subject_id,
        verbose,
        should_show,
    )


def SQI_fast(pulses: List[np.ndarray]) -> Tuple[List[float], List[float]]:
    """
    Fast SQI using cosine similarity between consecutive pulses.
    """

    sqi_pulses = []
    sqi_signal = []

    for i in range(len(pulses) - 1):
        x = np.asarray(pulses[i])
        y = np.asarray(pulses[i + 1])

        L = min(len(x), len(y))
        if L < 5:
            continue

        x = x[:L]
        y = y[:L]

        # Normalize (optional but recommended)
        x = x - np.mean(x)
        y = y - np.mean(y)

        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)

        if norm_x == 0 or norm_y == 0:
            c = 0
        else:
            c = np.dot(x, y) / (norm_x * norm_y)

        # same nonlinear mapping
        s1 = 50 * (c + 1) / 99
        s1 *= 8
        sqi = np.exp(s1) / np.exp(8)

        sqi_pulses.append(sqi)
        sqi_signal.extend([sqi] * len(x))

    return sqi_pulses, sqi_signal


def SQI(pulses: List[np.ndarray]) -> Tuple[List[float], List[float]]:
    '''
    This is the original SQI function based on cosine similarity. Its very slow implementation
    is a problem. Use its companion SQI_fast() faster version.

    @brief: Compute the SQI for a list of pulses
    @param: pulses is a list/array with the individual, already extracted PPG pulses
    @return: sqi_pulses is a list with the SQI for each individual pulse
    @return sqi_signal is a list with the same size as the original signal with the SQI values normalized
    '''
    sqi_signal = []
    sqi_pulses = []
    for e in range(len(pulses)-1):
        x = pulses[e]
        y = pulses[e+1]
        min_pulse = np.min([len(x), len(y)])

        c = 0
        for i in range(0, (min_pulse)):
            first = (x[i]*y[i])
            second_1, second_2 = 0, 0
            for j in range(0, (min_pulse)):
                second_1 += x[j]**2
                second_2 += y[j]**2

            second = np.sqrt((second_1*second_2))
            aux = first/second
            c += aux

        s1 = 50*(c+1)/99
        s1 = s1*8
        s1 = math.e**s1
        SQI = (s1 / (math.e**8))
        sqi_pulses.append(SQI)
        for r in range(0, len(x)):
            sqi_signal.append(SQI)
        # print(SQI)
        # plt.plot(x)
        # plt.show()

    return sqi_pulses, sqi_signal


def _run_single_test2(ppg: np.ndarray, fs: int, fs_out: int, sizeOfWindow: int, label: str, plot: bool = False) -> None:
    """Run a synthetic test case for the legacy segment extractor."""
    print(f"\n=== TEST: {label} ===")

    segment = ppg[: int(fs * sizeOfWindow)]

    try:
        df = extract_features_from_segment(
            segment,
            fs=fs,
            fs_out=fs_out,
            window_size_in_secs=sizeOfWindow,
            ep=0
        )

        if df is None:
            print("Result: FAILED (returned None)")
        elif isinstance(df, pd.DataFrame):
            print("Result: OK")
            print(df.head())
        else:
            print("Result: Unexpected output type:", type(df))

    except Exception as e:
        print("Result: EXCEPTION")
        print(str(e))

    if plot:
        plt.figure(figsize=(10, 3))
        plt.plot(ppg)
        plt.title(f"PPG Signal - {label}")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()


def main() -> None:
    """Generate synthetic signals and run legacy robustness smoke tests."""

    fs = 100
    fs_out = 100
    duration = 60
    sizeOfWindow = 10

    print("Generating synthetic PPG signals...")

    # --- Scenario A: Clean ---
    ppg_clean = nk.ppg_simulate(
        duration=duration,
        sampling_rate=fs,
        heart_rate=70,
        random_state=42
    )

    # --- Scenario B: Moderate noise ---
    ppg_noisy = nk.ppg_simulate(
        duration=duration,
        sampling_rate=fs,
        heart_rate=70,
        motion_amplitude=0.3,
        random_state=42
    )

    # --- Scenario C: Strong motion artifacts ---
    ppg_very_noisy = nk.ppg_simulate(
        duration=duration,
        sampling_rate=fs,
        heart_rate=70,
        motion_amplitude=0.8,
        random_state=42
    )

    # --- Scenario D: Irregular HR ---
    ppg_irregular = nk.ppg_simulate(
        duration=duration,
        sampling_rate=fs,
        heart_rate=90,
        frequency_modulation=0.2,
        random_state=42
    )

    # --- Scenario E: Short signal (edge case) ---
    ppg_short = nk.ppg_simulate(
        duration=3,
        sampling_rate=fs,
        heart_rate=70,
        random_state=42
    )

    # --- Scenario F: Flat / invalid signal ---
    ppg_flat = np.zeros(fs * duration)

    test_cases = {
        "clean": ppg_clean,
        "noisy": ppg_noisy,
        "very_noisy": ppg_very_noisy,
        "irregular": ppg_irregular,
        "short": ppg_short,
        "flat": ppg_flat,
    }

    print("\nRunning tests...")

    for label, signal in test_cases.items():
        _run_single_test2(
            signal,
            fs,
            fs_out,
            sizeOfWindow,
            label,
            plot=(label in ["clean", "very_noisy"])
        )

    print("\nDone.")


def preprocess(signal: Sequence[float], sps: int, f_settings: Optional[Dict[str, Any]] = None) -> Any:
    """Apply optional smoothing, band-pass filtering, and min-max normalization."""
    # Hyperparameters
    settings: Dict[str, Any] = f_settings if f_settings else dict(
        globals().get('f_settings', {}))

    lowcut = settings['lc']
    highcut = settings['hc']
    order_butter = settings['o_b']
    frame = settings['frame']
    order_svg = settings['o_svg']

    if (settings['use_svg']):
        # Apply smoothing
        signal = savitzky_golay_filter(signal, frame, order_svg)
    if (settings['use_butter']):
        # Apply band pass filter
        signal = butter_bandpass_filter_zi(
            signal, lowcut, highcut, sps, order_butter)
    if (settings['use_minmax']):
        # Normaliza the Data
        signal = max_min_normalization(signal)

    return signal


def list_feature_extraction_methods() -> List[str]:
    """
    Identify all top-level feature extraction methods in this module.
    """
    methods = []
    for name, obj in globals().items():
        if callable(obj) and name.startswith('extract_') and ('feature' in name):
            methods.append(name)
    return sorted(methods)


def _to_single_row_dataframe(result: Any) -> pd.DataFrame:
    """
    Normalize extractor output to a one-row DataFrame.
    """
    if result is None:
        return pd.DataFrame()

    if isinstance(result, pd.DataFrame):
        if result.empty:
            return pd.DataFrame()
        return result.iloc[[0]].reset_index(drop=True)

    if isinstance(result, dict):
        return pd.DataFrame([result])

    return pd.DataFrame()


def _prepare_feature_context(ppg_segment: np.ndarray, fs: int, fs_out: Optional[int] = None, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Build shared intermediate signals used by feature extractors.
    """
    if fs_out is None:
        fs_out = 2 * fs

    signal = np.asarray(preprocess(ppg_segment.copy(), fs, f_settings))
    if len(signal) < fs * 3:
        _debug_ufsc(
            verbose,
            f"### UFSC None reason: segment too short ({len(signal)} samples, need >= {fs * 3}).",
        )
        return None

    signal = signal[::-1]
    signal, _, _, _ = remove_baseline(signal, fs)

    _, is_valid, pulses = filter_based_on_shape(signal, fs, fs_out)

    # If strict pulse-quality filtering removes too many pulses, fallback to
    # raw pulse extraction on the resampled signal to keep UFSC extraction usable.
    if len(pulses) < 3:
        signal_resampled = resample_signal(signal, fs, fs_out)
        pulses = extract_pulses(signal_resampled, show=False)
        _debug_ufsc(
            verbose,
            "### Fallback to raw pulses. "
            f"is_valid={is_valid}, recovered_pulses={len(pulses)}",
        )

    if len(pulses) < 3:
        _debug_ufsc(
            verbose,
            f"### UFSC None reason: insufficient pulses after fallback ({len(pulses)}).",
        )
        return None

    clean_pulses = []
    for p in pulses:
        if len(p) < 0.4 * fs_out or len(p) > 1.8 * fs_out:
            continue
        if np.std(p) < 1e-4:
            continue
        clean_pulses.append(p)

    if len(clean_pulses) < 3:
        _debug_ufsc(
            verbose,
            f"### UFSC None reason: insufficient clean pulses ({len(clean_pulses)}).",
        )
        return None

    norm_pulses = []
    for p in clean_pulses:
        p = p - np.min(p)
        denom = np.max(p)
        if denom > 0:
            p = p / denom
        norm_pulses.append(p)

    target_len = 200
    resampled = []
    for p in norm_pulses:
        p_rs = resample_signal(p, fs, fs_out)
        p_rs = adjust_signal(p_rs, target_len)
        resampled.append(p_rs)

    if len(resampled) < 3:
        _debug_ufsc(
            verbose,
            f"### UFSC None reason: insufficient resampled pulses ({len(resampled)}).",
        )
        return None

    resampled = np.array(resampled)

    aligned = []
    peak_positions = [np.argmax(p) for p in resampled]
    ref_peak = int(np.median(peak_positions))
    for p, pk in zip(resampled, peak_positions):
        shift = ref_peak - pk
        p_shifted = np.roll(p, shift)
        if shift > 0:
            p_shifted[:shift] = p_shifted[shift]
        elif shift < 0:
            p_shifted[shift:] = p_shifted[shift - 1]
        aligned.append(p_shifted)

    avg_pulse = np.median(np.array(aligned), axis=0)
    ppg_frame_crop = crop_signal(signal, fs)

    return {
        'fs': fs,
        'fs_out': fs_out,
        'signal': signal,
        'ppg_frame_crop': ppg_frame_crop,
        'avg_pulse': avg_pulse,
        'ppg_segment': ppg_segment,
    }


def extract_all_UFSC_features_dataframe(
    ppg_segment: np.ndarray,
    fs: int,
    fs_out: Optional[int] = None,
    verbose: bool = False,
    output_csv: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Build a wide DataFrame by concatenating outputs of all feature extractors.

    Column names are prefixed with method IDs (m1_, m2_, ...) so the source
    extraction method is explicit. If output_csv is provided, the final
    DataFrame is written to disk.
    """
    if fs_out is None:
        fs_out = 2 * fs

    context = _prepare_feature_context(
        ppg_segment, fs, fs_out=fs_out, verbose=verbose)
    if context is None:
        _debug_ufsc(
            verbose, "### UFSC None reason: _prepare_feature_context returned None.")
        return None

    extractors = [
        ('m1', 'extract_HRV_features_by_ufsc', lambda: extract_HRV_features_by_ufsc(
            context['ppg_frame_crop'], fs)),
        ('m2', 'extract_psd_features_by_ufsc', lambda: extract_psd_features_by_ufsc(
            context['ppg_frame_crop'], fs)),
        ('m3', 'extract_pulse_features', lambda: extract_pulse_features(
            context['avg_pulse'], fs_out, show=False)),
        ('m4', 'extract_features_from_segment', lambda: extract_features_from_segment(
            context['ppg_segment'], fs, fs_out)),
    ]

    all_frames = []
    method_prefix_map = {}
    for method_id, extractor_name, extractor in extractors:
        method_prefix_map[method_id] = extractor_name
        try:
            df_extractor = _to_single_row_dataframe(extractor())
        except Exception as exc:
            if verbose:
                print(f"Warning: extractor {extractor_name} failed: {exc}")
            df_extractor = pd.DataFrame()

        if not df_extractor.empty:
            all_frames.append(df_extractor.add_prefix(f"{method_id}_"))

    if not all_frames:
        _debug_ufsc(
            verbose, "### UFSC None reason: all extractors returned empty outputs.")
        return None

    df_all_features = pd.concat(all_frames, axis=1)

    if output_csv:
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df_all_features.to_csv(output_csv, index=False)

    if verbose:
        methods = ', '.join(list_feature_extraction_methods())
        similar_methods = identify_similar_methods()
        print("Feature extraction methods identified:", methods)
        print("Similar methods:", similar_methods)
        print("Method prefix mapping:", method_prefix_map)
        print("Wide features shape:", df_all_features.shape)
        if output_csv:
            print("Saved features CSV:", output_csv)

    return df_all_features


def extract_features_from_average_pulse(ppg_segment: np.ndarray, fs: int, fs_out: Optional[int] = None, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Robust feature extraction from a PPG segment using:
    - robust pulse selection
    - alignment
    - median-based averaging

    Returns
    -------
    pd.DataFrame or None
    """
    df_features = extract_all_UFSC_features_dataframe(
        ppg_segment,
        fs,
        fs_out=fs_out,
        verbose=verbose
    )

    if df_features is None or df_features.empty:
        return None

    return {str(k): v for k, v in df_features.iloc[0].to_dict().items()}


# =============================================================================
# TESTING
# =============================================================================

def _run_single_test(
    signal: np.ndarray,
    fs: int,
    fs_out: int,
    label: str
) -> None:
    """Run robustness test on one signal."""
    print(f"\n=== {label} ===")

    try:
        df = extract_all_UFSC_features_dataframe(signal, fs, fs_out)

        if df is None:
            print("FAILED")
        elif isinstance(df, dict):
            print("OK")
            # Keep deterministic and compact output for dict-based return.
            preview_items = list(df.items())[:10]
            print(dict(preview_items))
        else:
            print("OK")
            print("Shape:", df.shape)
            print(df.head() if hasattr(df, 'head') else df)

    except Exception as e:
        print("EXCEPTION:", str(e))


# =============================================================================
# MAIN
# =============================================================================

def main2() -> None:
    """
    Test pipeline using synthetic signals from NeuroKit2.
    """

    fs = 100
    fs_out = 500
    duration = 30

    print("Generating signals...")

    signals = {
        "clean": nk.ppg_simulate(duration=duration, sampling_rate=fs),
        "noisy": nk.ppg_simulate(duration=duration, sampling_rate=fs, motion_amplitude=0.4),
        "irregular": nk.ppg_simulate(duration=duration, sampling_rate=fs, frequency_modulation=0.2),
        "short": nk.ppg_simulate(duration=2, sampling_rate=fs),
        "flat": np.zeros(fs * duration),
    }

    final_rows = []
    for name, sig in signals.items():
        _run_single_test(sig, fs, fs_out, name)

        df_signal = extract_all_UFSC_features_dataframe(sig, fs, fs_out)
        if df_signal is not None and not df_signal.empty:
            df_signal = df_signal.copy()
            df_signal['signal_name'] = name
            final_rows.append(df_signal)

    if final_rows:
        final_df = pd.concat(final_rows, ignore_index=True)
        output_csv = "ppg_features_final.csv"
        final_df.to_csv(output_csv, index=False)
        print(f"Final DataFrame saved to: {output_csv}")
        print("Final shape:", final_df.shape)
    else:
        print("No valid features were extracted. Final CSV was not generated.")


def identify_similar_methods() -> Dict[str, List[str]]:
    """Group methods with similar purpose based on naming conventions."""
    method_groups = {
        'shape_filtering': ['filter_based_on_shape', 'old_filter_based_on_shape'],
        'sqi_estimation': ['SQI_fast', 'SQI'],
        'single_segment_extraction': ['extract_features_from_segment', 'extract_features_from_average_pulse'],
        'test_entrypoints': ['main', 'main2'],
    }

    existing_names = {name for name, obj in globals().items() if callable(obj)}
    return {
        group: [name for name in names if name in existing_names]
        for group, names in method_groups.items()
    }


if __name__ == "__main__":
    main()
    main2()
