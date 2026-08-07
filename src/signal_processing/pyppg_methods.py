'''
Code from
https://pyppg.readthedocs.io/en/latest/tutorials/pyPPG_example.html
'''
import pyPPG
from pyPPG import PPG, Fiducials, Biomarkers
from pyPPG.datahandling import load_data, plot_fiducials, save_data, load_fiducials
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM
import pyPPG.ppg_sqi as SQI
from dotmap import DotMap
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import json
import pandas as pd
import scipy.io

pad_value = -0.15  # value to pad SQI arrays when needed due to incorrect lengths

###########################################################################
################################## EXAMPLE ################################
###########################################################################


def estimate_sqi_pyppg(ppg, fs):
    '''
    Our simplified SQI estimator using pyPPG. It returns the SQI per pulse and the SQI for each signal sample.
    '''

    sqi_per_pulse, sqi_for_each_signal_sample = get_sqi_estimates(ppg, fs=fs, plotfig=False,
                                                                  filtering=False, print_flag=False)

    # create dictionary to return
    # out = {"name": "pyPPG",
    #       "sqi_per_pulse": sqi_per_pulse,
    #       "sqi_for_each_signal_sample": sqi_for_each_signal_sample}
    return sqi_for_each_signal_sample


def compare_number_of_samples(sqi_for_each_signal_sample, signal):

    num_signal_samples = len(signal)
    sqi_for_each_signal_sample = np.asarray(sqi_for_each_signal_sample)
    num_sqi_samples = len(sqi_for_each_signal_sample)

    if num_signal_samples != num_sqi_samples:
        print(
            f"  WARNING: Number of samples in SQI ({num_sqi_samples}) does not match number of samples in signal ({num_signal_samples})")


def convert_raw_data_to_dotmap(signal, fs=60):
    s = DotMap()

    s.start_sig = 0
    s.end_sig = len(signal)
    s.v = signal
    s.fs = fs
    s.name = "unknown_PPG_signal"
    return s


def get_sqi_estimates(raw_signal, fs=60, start_sig=0, end_sig=-1, fiducials=pd.DataFrame(), process_type="both", channel="Pleth",
                      filtering=True, fL=0.5000001, fH=12, order=4, sm_wins={'ppg': 50, 'vpg': 10, 'apg': 10, 'jpg': 10}, correction=pd.DataFrame(),
                      plotfig=True, savingfolder="temp_dir", savefig=True, show_fig=True, savingformat="both", print_flag=True, use_tk=False,
                      check_ppg_len=True, saved_fiducials="", savedata=True):
    '''
    This is an example code for PPG analysis. The main parts:
        1) Loading a raw PPG signal: various file formats such as .mat, .csv, .txt, or .edf.
        2) Get Fiducial points: extract the fiducial points of PPG, PPG', PPG'' and PPG'" signals
        3) Plot Fiducial Points

    :param data_path: path of the PPG signal
    :type data_path: str
    :param fs: sampling_frequency
    :type fs: int
    :param start_sig: beginning the of signal in sample
    :type start_sig: int
    :param end_sig: end of the signal in sample
    :type end_sig: int
    :param fiducials: DataFrame of the fiducial points
    :type fiducials: pyPPG.Fiducials DataFrame
    :param process_type: the type of the process, which can be "fiducials", "biomarkers", "both", or "only_sig"
    :type process_type: str
    :param channel: channel of the .edf file
    :type channel: channel of the .edf file
    :param filtering: a bool for filtering
    :type filtering: bool
    :param fL: Lower cutoff frequency (Hz)
    :type fL: float
    :param fH: Upper cutoff frequency (Hz)
    :type fH: float
    :param order: Filter order
    :type order: int
    :param sm_wins: dictionary of smoothing windows in millisecond:
        - ppg: window for PPG signal
        - vpg: window for PPG' signal
        - apg: window for PPG" signal
        - jpg: window for PPG'" signal
    :type sm_wins: dict
    :param correction: DataFrame where the key is the name of the fiducial points and the value is bool
    :type correction: DataFrame
    :param plotfig: a bool for plot figure
    :type plotfig: bool
    :param savingfolder: location of the saved data
    :type savingfolder: str
    :param savefig: a bool for current figure saving
    :type savefig: bool
    :param show_fig: a bool for show figure
    :type show_fig: bool
    :param savingformat: file format of the saved date, the provided file formats ".mat", ".csv", "both", or "none"
    :type savingformat: str
    :param print_flag: a bool for print message
    :type print_flag: bool
    :param use_tk: a bool for using tkinter interface
    :type use_tk: bool
    :param check_ppg: a bool for checking ppg length and sampling frequency
    :type check_ppg: bool
    :param saved_fiducials: path of the file of the saved fiducial points
    :type saved_fiducials: str
    :param savedata: a bool for saving data
    :type savedata: bool

    :return: file_names: dictionary of the saved file names

    Example:

        .. code-block:: python

            from pyPPG.example import ppg_example

            # run example code
            ppg_example()

    '''
    # Convert to DotMap
    signal = convert_raw_data_to_dotmap(raw_signal, fs=fs)

    # Preprocessing
    # Initialise the filters
    prep = PP.Preprocess(fL=fL, fH=fH, order=order, sm_wins=sm_wins)

    # Filter and calculate the PPG, PPG', PPG", and PPG'" signals
    signal.filtering = filtering
    signal.fL = fL
    signal.fH = fH
    signal.order = order
    signal.sm_wins = sm_wins
    signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

    # Initialise the correction for fiducial points
    corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']

    # If correction does NOT exist or is empty, initialize it properly
    if 'correction' not in globals() or correction.empty:
        correction = pd.DataFrame({c: [False] for c in corr_on})
    else:
        # Ensure required columns exist
        for c in corr_on:
            if c not in correction.columns:
                correction[c] = False

        # Ensure index 0 exists
        if 0 not in correction.index:
            correction.loc[0] = False

    # Make columns boolean
    correction[corr_on] = correction[corr_on].astype(bool)

    # Now safe — no warnings:
    correction.loc[0, corr_on] = True

    signal.correction = correction

    # Create a PPG class
    s = PPG(s=signal, check_ppg_len=check_ppg_len)

    if True:
        # Get Fiducial points
        if process_type == 'fiducials' or process_type == 'both':
            # Initialise the fiducials package
            fpex = FP.FpCollection(s=s)

            # Extract fiducial points
            fiducials = fpex.get_fiducials(s=s)

            fiducials = fiducials.apply(lambda col: col.map(
                lambda x: np.nan if pd.isna(x) else x
            ))

            # fiducials = fiducials.applymap(
            #    lambda x: np.nan if pd.isna(x) else x)

            if print_flag:
                print("Fiducial points:\n", fiducials + s.start_sig)
            # fiducials is a DataFrame
            peaks = fiducials['sp'].values.astype(int)
            onsets = fiducials['on'].values.astype(int)
            offsets = fiducials['off'].values.astype(int)
            output_points = {"name": "pyPPG", "onsets": onsets,
                             "peaks": peaks, "offsets": offsets}
            # Create a fiducials class
            fp = Fiducials(fp=fiducials)
        # Plot fiducial points
            if plotfig:

                plot_fiducials(s=s, fp=fp, savefig=savefig, savingfolder=savingfolder,
                               show_fig=show_fig, print_flag=print_flag, use_tk=use_tk)

        # PPG SQI

            # Calculate SQI
            sqi_values_per_pulse = SQI.get_ppgSQI(
                ppg=s.ppg, fs=s.fs, annotation=fp.sp)
            ppgSQI = round(np.mean(sqi_values_per_pulse) * 100, 2)
            if print_flag:
                print('Mean PPG SQI: ', ppgSQI, '%')

    signal_length = len(s.ppg)
    sqi_samples = sqi_per_sample_from_pulses(
        onsets, offsets, sqi_values_per_pulse, signal_length=signal_length, fill_value=pad_value)

    return sqi_values_per_pulse, sqi_samples


def sqi_per_sample_from_pulses(onsets, offsets, sqi_values, signal_length=None, fill_value=np.nan):
    """
    Map pulse-level SQI values to a per-sample SQI array.

    Parameters
    ----------
    onsets : array-like of int
        Pulse onset sample indices (inclusive).
    offsets : array-like of int
        Pulse offset sample indices (inclusive). Must align with onsets.
        If an offset is NaN or < onset, that pulse is skipped.
    sqi_values : array-like of float
        SQI value for each pulse (expected in range [0,1]). Will be clipped.
    signal_length : int, optional
        Length of the output array in samples. If None it is inferred as
        max(max(offsets), max(onsets)) + 1.
    fill_value : scalar, optional
        Value to use for samples not covered by any pulse (default np.nan).

    Returns
    -------
    sqi_samples : np.ndarray, shape (signal_length,)
        Per-sample SQI values. For samples covered by multiple (overlapping)
        pulses the returned value is the mean of the overlapping pulses' SQIs.
        Samples not covered by any pulse contain `fill_value`.
    """
    onsets = np.asarray(onsets, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    sqi_values = np.asarray(sqi_values, dtype=float)

    if onsets.shape != offsets.shape:
        raise ValueError(
            "onsets and offsets must have the same shape" +
            " but got {}, {}".format(onsets.shape, offsets.shape))

    N = len(onsets)
    M = len(sqi_values)

    # Allow the N-1 case
    if M == N - 1:
        # OK -- interval-based SQI
        pass
    else:
        raise ValueError(f"Unexpected SQI length: onsets={N}, sqi_values={M}")

    # Clip SQI to [0,1]
    # sqi_values = np.clip(sqi_values, 0.0, 1.0)

    # Determine signal length if not given
    max_index = -1
    if signal_length is None:
        if onsets.size > 0:
            max_index = max(int(np.nanmax(onsets)), int(np.nanmax(offsets)))
            signal_length = int(max_index) + 1
        else:
            signal_length = 0

    if signal_length <= 0:
        return np.array([], dtype=float)

    # Arrays to accumulate sums and counts (for average on overlaps)
    sum_arr = np.zeros(signal_length, dtype=float)
    count_arr = np.zeros(signal_length, dtype=int)

    # Iterate pulses
    for i, (st, ed, v) in enumerate(zip(onsets, offsets, sqi_values)):
        # Skip NaN or invalid offsets
        if np.isnan(st) or np.isnan(ed):
            continue

        st = int(st)
        ed = int(ed)

        # Ensure valid interval (onset <= offset)
        if ed < st:
            # skip or optionally swap: continue (skip)
            continue

        # Clip to signal bounds
        if st >= signal_length:
            # completely outside to the right
            continue
        if ed < 0:
            # completely outside to the left
            continue

        st_clamped = max(0, st)
        ed_clamped = min(signal_length - 1, ed)

        # Add Sqi to sum and increment count
        # vectorized slice update
        sum_arr[st_clamped:ed_clamped + 1] += float(v)
        count_arr[st_clamped:ed_clamped + 1] += 1

    # Compute average where count>0
    sqi_samples = np.full(signal_length, fill_value, dtype=float)
    mask = count_arr > 0
    sqi_samples[mask] = sum_arr[mask] / count_arr[mask]

    return sqi_samples


def get_pyppg_features(raw_signal, fs=60, start_sig=0, end_sig=-1, fiducials=pd.DataFrame(), process_type="both", channel="Pleth",
                       filtering=True, fL=0.5000001, fH=12, order=4, sm_wins={'ppg': 50, 'vpg': 10, 'apg': 10, 'jpg': 10}, correction=pd.DataFrame(),
                       plotfig=True, savingfolder="temp_dir", savefig=True, show_fig=True, savingformat="both", print_flag=True, use_tk=False,
                       check_ppg_len=True, saved_fiducials="", savedata=True):
    '''
    This is an example code for PPG analysis. The main parts:
        1) Loading a raw PPG signal: various file formats such as .mat, .csv, .txt, or .edf.
        2) Get Fiducial points: extract the fiducial points of PPG, PPG', PPG'' and PPG'" signals
        3) Plot Fiducial Points

    :param data_path: path of the PPG signal
    :type data_path: str
    :param fs: sampling_frequency
    :type fs: int
    :param start_sig: beginning the of signal in sample
    :type start_sig: int
    :param end_sig: end of the signal in sample
    :type end_sig: int
    :param fiducials: DataFrame of the fiducial points
    :type fiducials: pyPPG.Fiducials DataFrame
    :param process_type: the type of the process, which can be "fiducials", "biomarkers", "both", or "only_sig"
    :type process_type: str
    :param channel: channel of the .edf file
    :type channel: channel of the .edf file
    :param filtering: a bool for filtering
    :type filtering: bool
    :param fL: Lower cutoff frequency (Hz)
    :type fL: float
    :param fH: Upper cutoff frequency (Hz)
    :type fH: float
    :param order: Filter order
    :type order: int
    :param sm_wins: dictionary of smoothing windows in millisecond:
        - ppg: window for PPG signal
        - vpg: window for PPG' signal
        - apg: window for PPG" signal
        - jpg: window for PPG'" signal
    :type sm_wins: dict
    :param correction: DataFrame where the key is the name of the fiducial points and the value is bool
    :type correction: DataFrame
    :param plotfig: a bool for plot figure
    :type plotfig: bool
    :param savingfolder: location of the saved data
    :type savingfolder: str
    :param savefig: a bool for current figure saving
    :type savefig: bool
    :param show_fig: a bool for show figure
    :type show_fig: bool
    :param savingformat: file format of the saved date, the provided file formats ".mat", ".csv", "both", or "none"
    :type savingformat: str
    :param print_flag: a bool for print message
    :type print_flag: bool
    :param use_tk: a bool for using tkinter interface
    :type use_tk: bool
    :param check_ppg: a bool for checking ppg length and sampling frequency
    :type check_ppg: bool
    :param saved_fiducials: path of the file of the saved fiducial points
    :type saved_fiducials: str
    :param savedata: a bool for saving data
    :type savedata: bool

    :return: file_names: dictionary of the saved file names

    Example:

        .. code-block:: python

            from pyPPG.example import ppg_example

            # run example code
            ppg_example()

    '''
    # Convert to DotMap
    signal = convert_raw_data_to_dotmap(raw_signal, fs=fs)

    # Preprocessing
    # Initialise the filters
    prep = PP.Preprocess(fL=fL, fH=fH, order=order, sm_wins=sm_wins)

    # Filter and calculate the PPG, PPG', PPG", and PPG'" signals
    signal.filtering = filtering
    signal.fL = fL
    signal.fH = fH
    signal.order = order
    signal.sm_wins = sm_wins
    signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

    # Initialise the correction for fiducial points
    corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']

    # If correction does NOT exist or is empty, initialize it properly
    if 'correction' not in globals() or correction.empty:
        correction = pd.DataFrame({c: [False] for c in corr_on})
    else:
        # Ensure required columns exist
        for c in corr_on:
            if c not in correction.columns:
                correction[c] = False

        # Ensure index 0 exists
        if 0 not in correction.index:
            correction.loc[0] = False

    # Make columns boolean
    correction[corr_on] = correction[corr_on].astype(bool)

    # Now safe — no warnings:
    correction.loc[0, corr_on] = True

    signal.correction = correction

    # Create a PPG class
    s = PPG(s=signal, check_ppg_len=check_ppg_len)

    if True:
        # Get Fiducial points
        if process_type == 'fiducials' or process_type == 'both':
            # Initialise the fiducials package
            fpex = FP.FpCollection(s=s)

            # Extract fiducial points
            fiducials = fpex.get_fiducials(s=s)

            fiducials = fiducials.apply(lambda col: col.map(
                lambda x: np.nan if pd.isna(x) else x
            ))

            # fiducials = fiducials.applymap(
            #    lambda x: np.nan if pd.isna(x) else x)

            if print_flag:
                print("Fiducial points:\n", fiducials + s.start_sig)
            # fiducials is a DataFrame
            peaks = fiducials['sp'].values.astype(int)
            onsets = fiducials['on'].values.astype(int)
            offsets = fiducials['off'].values.astype(int)
            output_points = {"name": "pyPPG", "onsets": onsets,
                             "peaks": peaks, "offsets": offsets}
        # Plot fiducial points
            if plotfig:
                # Create a fiducials class
                fp = Fiducials(fp=fiducials)
                plot_fiducials(s=s, fp=fp, savefig=savefig, savingfolder=savingfolder,
                               show_fig=show_fig, print_flag=print_flag, use_tk=use_tk)

    return output_points


def get_pyppg_features_and_biomarkers(signal, fs=0, start_sig=0, end_sig=-1, fiducials=pd.DataFrame(), process_type="both", channel="Pleth",
                                      filtering=True, fL=0.5000001, fH=12, order=4, sm_wins={'ppg': 50, 'vpg': 10, 'apg': 10, 'jpg': 10}, correction=pd.DataFrame(),
                                      plotfig=True, savingfolder="temp_dir", savefig=True, show_fig=True, savingformat="both", print_flag=True, use_tk=False,
                                      check_ppg_len=True, saved_fiducials="", savedata=True):
    '''
    This is an example code for PPG analysis. The main parts:
        1) Loading a raw PPG signal: various file formats such as .mat, .csv, .txt, or .edf.
        2) Get Fiducial points: extract the fiducial points of PPG, PPG', PPG'' and PPG'" signals
        3) Plot Fiducial Points
        4) Get Biomarkers: extract 74 PPG biomarkers in four categories:
            - PPG signal
            - Signal ratios
            - PPG derivatives
            - Derivatives ratios
        5) Get Statistics: summary of the 74 PPG biomarkers
        6) SQI calculation: calculates the PPG Signal Quality Index
        7) Save data: save the extracted Fiducial points, Biomarkers, and Statistics into .csv file

    :param data_path: path of the PPG signal
    :type data_path: str
    :param fs: sampling_frequency
    :type fs: int
    :param start_sig: beginning the of signal in sample
    :type start_sig: int
    :param end_sig: end of the signal in sample
    :type end_sig: int
    :param fiducials: DataFrame of the fiducial points
    :type fiducials: pyPPG.Fiducials DataFrame
    :param process_type: the type of the process, which can be "fiducials", "biomarkers", "both", or "only_sig"
    :type process_type: str
    :param channel: channel of the .edf file
    :type channel: channel of the .edf file
    :param filtering: a bool for filtering
    :type filtering: bool
    :param fL: Lower cutoff frequency (Hz)
    :type fL: float
    :param fH: Upper cutoff frequency (Hz)
    :type fH: float
    :param order: Filter order
    :type order: int
    :param sm_wins: dictionary of smoothing windows in millisecond:
        - ppg: window for PPG signal
        - vpg: window for PPG' signal
        - apg: window for PPG" signal
        - jpg: window for PPG'" signal
    :type sm_wins: dict
    :param correction: DataFrame where the key is the name of the fiducial points and the value is bool
    :type correction: DataFrame
    :param plotfig: a bool for plot figure
    :type plotfig: bool
    :param savingfolder: location of the saved data
    :type savingfolder: str
    :param savefig: a bool for current figure saving
    :type savefig: bool
    :param show_fig: a bool for show figure
    :type show_fig: bool
    :param savingformat: file format of the saved date, the provided file formats ".mat", ".csv", "both", or "none"
    :type savingformat: str
    :param print_flag: a bool for print message
    :type print_flag: bool
    :param use_tk: a bool for using tkinter interface
    :type use_tk: bool
    :param check_ppg: a bool for checking ppg length and sampling frequency
    :type check_ppg: bool
    :param saved_fiducials: path of the file of the saved fiducial points
    :type saved_fiducials: str
    :param savedata: a bool for saving data
    :type savedata: bool

    :return: file_names: dictionary of the saved file names

    Example:

        .. code-block:: python

            from pyPPG.example import ppg_example

            # run example code
            ppg_example()

    '''

    # Preprocessing
    # Initialise the filters
    prep = PP.Preprocess(fL=fL, fH=fH, order=order, sm_wins=sm_wins)

    # Filter and calculate the PPG, PPG', PPG", and PPG'" signals
    signal.filtering = filtering
    signal.fL = fL
    signal.fH = fH
    signal.order = order
    signal.sm_wins = sm_wins
    signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

    # Initialise the correction for fiducial points
    corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
    correction.loc[0, corr_on] = True
    signal.correction = correction

    # Create a PPG class
    s = PPG(s=signal, check_ppg_len=check_ppg_len)

    # Save signal
    if process_type == "only_sig":
        file_names = save_data(
            savingformat=savingformat, savingfolder=savingfolder, print_flag=print_flag, s=s)
    else:
        # Get Fiducial points
        if process_type == 'fiducials' or process_type == 'both':
            # Initialise the fiducials package
            fpex = FP.FpCollection(s=s)

            # Extract fiducial points
            fiducials = fpex.get_fiducials(s=s)
            fiducials = fiducials.applymap(
                lambda x: np.nan if pd.isna(x) else x)
            if print_flag:
                print("Fiducial points:\n", fiducials + s.start_sig)

            # Create a fiducials class
            fp = Fiducials(fp=fiducials)

            # Save data
            if savedata:
                fp_new = Fiducials(fp=fp.get_fp() + s.start_sig)
                file_names = save_data(
                    savingformat=savingformat, savingfolder=savingfolder, print_flag=print_flag, s=s, fp=fp_new)

        # PPG SQI

            # Calculate SQI
            ppgSQI = round(np.mean(SQI.get_ppgSQI(
                ppg=s.ppg, fs=s.fs, annotation=fp.sp)) * 100, 2)
            if print_flag:
                print('Mean PPG SQI: ', ppgSQI, '%')

        # Plot fiducial points
            if plotfig:
                plot_fiducials(s=s, fp=fp, savefig=savefig, savingfolder=savingfolder,
                               show_fig=show_fig, print_flag=print_flag, use_tk=use_tk)

        # Get Biomarkers and Statistics
        if (process_type == 'biomarkers' or process_type == 'both') and len(fiducials) > 0:
            # Initialise the biomarkers package
            fp = Fiducials(fp=fiducials)

            bmex = BM.BmCollection(s=s, fp=fp)

            # Extract biomarkers
            bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()

            if print_flag:
                tmp_keys = bm_stats.keys()
                print('Statistics of the biomarkers:')
                for i in tmp_keys:
                    print(i, '\n', bm_stats[i])

            # Create a biomarkers class
            bm = Biomarkers(bm_defs=bm_defs, bm_vals=bm_vals,
                            bm_stats=bm_stats)

            # Save data
            if savedata:
                fp_new = Fiducials(fp=fp.get_fp() + s.start_sig)
                file_names = save_data(
                    savingformat=savingformat, savingfolder=savingfolder, print_flag=print_flag, s=s, fp=fp_new, bm=bm)

    if print_flag:
        print('Program finished')

    return file_names


###########################################################################
############################## RUN EXAMPLE CODE ###########################
###########################################################################
if __name__ == "__main__":

    # Example from CSV file
    data_path = "C:\\github\\GODA_pyPPG\\sample_data\\Sample_PPG_CSV_125Hz.csv"
    fs = 125
    use_tk = True
    print_flag = True

    print("Running example code for pyPPG preprocessing using file", data_path)
    # Loading a raw PPG signal as storing as a DotMap class
    signalDotMap = load_data(data_path=data_path, fs=fs,
                             use_tk=use_tk, print_flag=print_flag)
    signal = signalDotMap.v
    # print(signal)
    points = get_pyppg_features(signal, savefig=True, fs=fs)
    print(points)

    sqis = estimate_sqi_pyppg(signal, fs)
    compare_number_of_samples(sqis, signal)
    print("sqis", sqis[:10])

    sqi_values_per_pulse, sqi_samples = get_sqi_estimates(signal, fs=fs)

    print("sqi_samples", sqi_samples[:10])
