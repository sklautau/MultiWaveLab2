'''Simple script to inspect files'''


from datasets_util.waveform_files import read_sigmf_file
from datasets_util.util_visualize_plots import plot_input_output_waveforms

FILE_NAME = "../output_ieb3/waveforms/ppg_inversion/ieb_04/S2_2025-08-04_07-57/ppg/ppg_141mgdL_ppg_inversion.sigmf-data"

# Load SigMF file

# Read full signal
data, input_metadata = read_sigmf_file(FILE_NAME)
fs = input_metadata["global"]["core:sample_rate"]

# plot the signal
plot_input_output_waveforms(data, data, fs, "")
