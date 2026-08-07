'''
This script analyzes the frequency of features across different datasets.
Very simple script just to check results.
'''


from collections import Counter
from itertools import chain

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

# collect from log.txt files. Search for
# Features selected by mRMR:
ieb3f = ['ppg_u5_m3_diastolic_peak_y', 'ecg_hrv_csi_modified', 'ppg_b2_Tr_norm_cv', 'ppg_u5_m3_D', 'ppg_u5_m3_area_2', 'ecg_hrv_maxnn', 'ppg_u5_m3_rising_time', 'ecg_hrv_prc80nn', 'ecg_s2_KUR_PSD',
         'ppg_u5_m3_area', 'ecg_hrv_cvi', 'ecg_hrv_mediannn', 'ppg_u5_m3_peak_value', 'ppg_u5_m3_E', 'ppg_u5_m3_variance', 'ppg_p1_amp_mean', 'ecg_rr_mean', 'ecg_hrv_pnn50', 'ppg_u5_m3_DT', 'ecg_hrv_tinn']

ieb2f = ['ppg_s3_spectral_entropy', 'ecg_hrv_hti', 'ppg_s3_MAX_POWER_2nd', 'ppg_u5_m3_B', 'ppg_s3_E1_norm', 'ecg_hrv_c2d', 'ecg_hr_std', 'ppg_b2_AUC_cv', 'ecg_hrv_lzc', 'ppg_b2_width_median',
         'ecg_hrv_symbolic_equalprob4_2uv', 'ppg_s3_KUR_PSD', 'ppg_u5_m3_IPA', 'ecg_hrv_sd1sd2', 'ecg_s2_E2_norm', 'ppg_b2_amplitude_cv', 'ppg_u5_m3_D/A', 'ecg_s2_MAX_POWER_1st', 'ecg_hrv_kfd', 'ppg_i6_IMF_1_kurtosis']

ieb1f = ['ppg_b2_width_median', 'ppg_h4_PRV_HRV_HTI', 'ppg_i6_IMF_1_if_mean', 'ppg_u5_m4_D', 'ppg_b2_Tr_norm_median', 'ppg_b2_T_median', 'ppg_u5_m4_peak_pos', 'ppg_i6_IMF_1_zcr', 'ppg_u5_m1_HRV_HTI', 'ppg_p1_d2_max_mean',
         'ppg_i6_IMF_1_sc', 'ppg_u5_m4_C', 'ppg_i6_IMF_1_extrema', 'ppg_h4_PRV_HRV_pNN20', 'ppg_i6_IMF_1_dom_freq', 'ppg_b2_AUC_median', 'ppg_u5_m3_peak_pos', 'ppg_h4_PRV_HRV_MadNN', 'ppg_p1_d1_max_mean', 'ppg_s3_spectral_centroid']

datasets = {
    "IEB1": ieb1f,
    "IEB2": ieb2f,
    "IEB3": ieb3f,
}

# ---------------------------------------------------------------------
# Exact feature frequency
# ---------------------------------------------------------------------

print("=" * 70)
print("EXACT FEATURE FREQUENCY")
print("=" * 70)

counter = Counter(chain.from_iterable(datasets.values()))

for feature, n in counter.most_common():
    if n > 1:
        present = [k for k, v in datasets.items() if feature in v]
        print(f"{feature:40s}  {n} datasets   {present}")

# ---------------------------------------------------------------------
# Common features
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("COMMON FEATURES")
print("=" * 70)

s1 = set(ieb1f)
s2 = set(ieb2f)
s3 = set(ieb3f)

print("\nPresent in ALL THREE:")
common3 = sorted(s1 & s2 & s3)
if common3:
    for f in common3:
        print("  ", f)
else:
    print("  None")

print("\nPresent in exactly TWO datasets:")
pairs = [
    ("IEB1", "IEB2", (s1 & s2) - s3),
    ("IEB1", "IEB3", (s1 & s3) - s2),
    ("IEB2", "IEB3", (s2 & s3) - s1),
]

for a, b, feats in pairs:
    print(f"\n{a} & {b}")
    if feats:
        for f in sorted(feats):
            print("  ", f)
    else:
        print("  None")

# ---------------------------------------------------------------------
# Feature family statistics
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE FAMILY FREQUENCY")
print("=" * 70)


def family(feature):
    """
    Keep the first two underscore-separated fields.
    Example:
        ppg_i6_IMF_1_var      -> ppg_i6
        ppg_s3_MAX_POWER_2nd  -> ppg_s3
        ecg_hrv_sd1sd2        -> ecg_hrv
    """
    parts = feature.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else feature


fam_counter = Counter(family(f)
                      for f in chain.from_iterable(datasets.values()))

for fam, n in fam_counter.most_common():
    print(f"{fam:25s} {n}")
