'''
For this type of exploratory analysis, a good workflow is:

Min-Max normalize all features (mainly useful for visualization and some distance measures).
Compute the correlation matrix among features to detect redundancy.
Cluster highly correlated features.
Compute correlation of each feature with GLC.
Optionally aggregate results by the feature groups (ppg_p1_*, ppg_p2_*, etc.).

Note that Pearson correlation is scale-invariant, so Min-Max normalization does not change the correlation coefficients themselves. It mainly helps plots and distance-based methods.

Using:
https://pypi.org/project/mrmr-selection/
'''
import argparse
from ctypes import cast
import os
from pathlib import Path

from scipy.spatial.distance import squareform
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns

from mrmr import mrmr_regression

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression

from datasets_util.naming_conventions import DatasetConfig
from features.clean_features_dataframe import diagnose_problematic_values

# threshold for discarding columns due to high correlation
CORRELATION_THRESHOLD = 0.95
MUTUAL_INFORMATION_THRESHOLD = 0.4
SHOULD_PLOT = False
USE_MIN_MAX_NORMALIZATION = False

# UFSC feature extraction:
# INPUT_FILE_NAME = r"..\output_ieb1\ml\ufsc_dissertation_ppg_features.csv"
# INPUT_FILE_NAME = r"..\output_ieb1\ml\ufsc_dissertation_ppg_features_aggregated.csv"
# INPUT_FILE_NAME = r"..\output_ieb1\ml\ufsc_tcc_ppg_features.csv"
# INPUT_FILE_NAME = r"..\output_ieb1\ml\ufsc_tcc_ppg_features_aggregated.csv"
# INPUT_FILE_NAME = r"C:\git_sofis\tcc_guilherme\files\dataRecord_spectrogram_v5.csv"
# INPUT_FILE_NAME = r"C:\git_sofis\output_luis_ieb1\all_ppg_features.csv"


def plot_hierarchical_clustering(X_norm: pd.DataFrame) -> None:
    # ======================================================
    # Hierarchical clustering
    # ======================================================
    corr_features = X_norm.corr(
    )

    # Protect against numerical errors
    corr_features = corr_features.clip(lower=0.0, upper=1.0)

    # Replace possible NaN correlations
    corr_features = corr_features.fillna(0.0)

    distance_matrix = 1 - np.abs(corr_features)

    # Force exact zeros on diagonal
    np.fill_diagonal(distance_matrix.values, 0.0)

    # Convert square distance matrix to condensed form
    condensed_distance = squareform(distance_matrix.values, checks=False)

    Z = linkage(condensed_distance, method="average")

    plt.figure(figsize=(14, 6))
    dendrogram(
        Z,
        labels=X_norm.columns.tolist(),
        leaf_rotation=90,
        leaf_font_size=8
    )
    plt.title("Hierarchical clustering of features")
    plt.tight_layout()
    plt.show()


def plot_some_feature_vs_glucose(corr_glc_mi_df, df, X_norm, corr_glc, mi_glc_series):
    # it takes too long to plot all features, so we will plot only the top features by correlation with GLC
    # 1) first plot the ones with maximum correlation:
    max_num_features_to_plot = 3
    # among the features in top_features, select those with the highest absolute correlation with GLC
    top_features = corr_glc_mi_df.sort_values(
        by="AbsCorrelation",
        ascending=False,
    )["Feature"].tolist()
    smaller_list = top_features[0:max_num_features_to_plot]
    # show each feature and GLC in a scatter plot, with a linear regression line, and color by participant_id
    plot_feature_vs_glucose(df, X_norm, corr_glc,
                            mi_glc_series, smaller_list)

    # 2) now plot the ones with maximum mutual information:
    max_num_features_to_plot = 3
    # among the features in top_features, select those with the highest absolute correlation with GLC
    top_features = corr_glc_mi_df.sort_values(
        by="MutualInformation_with_GLC",
        ascending=False,
    )["Feature"].tolist()
    smaller_list = top_features[0:max_num_features_to_plot]
    # show each feature and GLC in a scatter plot, with a linear regression line, and color by participant_id
    plot_feature_vs_glucose(df, X_norm, corr_glc,
                            mi_glc_series, smaller_list)


def plot_correlation_heatmap(X_norm, correlation_largest_pairs_df):
    # Plot correlation heatmap for all features
    if not correlation_largest_pairs_df.empty:
        top_features_from_pairs = (
            pd.concat([
                correlation_largest_pairs_df.head(30)["Feature1"],
                correlation_largest_pairs_df.head(30)["Feature2"],
            ])
            .drop_duplicates()
            .tolist()
        )

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            X_norm[top_features_from_pairs].corr(),
            cmap="coolwarm",
            center=0,
        )
        plt.title("Correlation among features from top correlated pairs")
        plt.tight_layout()
        plt.show()


def plot_mutual_information_heatmap(X_norm, pairs_df_mutual_information):
    # Plot mutual information heatmap for all features
    # use both Feature1 and Feature2 from the top 30 pairs to create a list of unique features for plotting
    top_features = (
        pd.concat([pairs_df_mutual_information.head(30)["Feature1"],
                  pairs_df_mutual_information.head(30)["Feature2"]])
        .drop_duplicates()
        .tolist()
    )
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        mutual_information_matrix(X_norm[top_features]),
        cmap="coolwarm",
        center=0
    )
    plt.title("Mutual Information among top 30 features")
    plt.tight_layout()
    plt.show()


def remove_feature_if_affine(X_norm: pd.DataFrame, pairs_df: pd.DataFrame, statistics_output_folder: str, tolerance: float = 1e-10) -> list[str]:
    affine_df = check_affine_equivalence(X_norm, pairs_df)

    # if affine_equivalent is True, then the two features are linearly related (one is an affine transformation of the other)
    # remove one element of each pair of affine equivalent features
    affine_equivalent_pairs = affine_df[affine_df["affine_equivalent"]]
    # pick one feature from each pair to drop
    features_to_drop_because_affine = set(
        affine_equivalent_pairs["Feature2"].tolist())
    # X_final = df.drop(columns=features_to_drop_because_affine)

    print("\nAffine equivalence check for highly correlated pairs:")
    print(affine_df.head(50))

    affine_df.to_csv(
        os.path.join(
            statistics_output_folder, "highly_correlated_pairs_affine_check.csv"),
        index=False
    )
    print(
        f"Saved highly correlated pairs affine check to '{os.path.join(statistics_output_folder, 'highly_correlated_pairs_affine_check.csv')}'")

    return list(features_to_drop_because_affine)


def find_largest_values(values_matrix_df: pd.DataFrame,
                        threshold: float,
                        title: str) -> pd.DataFrame:
    '''
    values_matrix_df is a pd.DataFrame containing the pairwise mutual information values
    organized as
    | Feature1 | Feature2 | MutualInformation |
    and was created with
    pd.DataFrame(mi, index=cols, columns=cols)
    # ------------------------------------------------------
    # Find pairs of highly correlated features
    # ------------------------------------------------------
    '''
    # threshold = MUTUAL_INFORMATION_THRESHOLD  # threshold for high MI

    features = values_matrix_df.columns.to_list()

    pairs = []

    for i, feature_i in enumerate(features):
        for j in range(i + 1, len(features)):
            feature_j = features[j]
            corr_ij = values_matrix_df.loc[feature_i, feature_j]

            if abs(corr_ij) >= threshold:
                pairs.append({
                    "Feature1": feature_i,
                    "Feature2": feature_j,
                    title: corr_ij,
                    "Abs" + title: abs(corr_ij)
                })

    pairs_df = pd.DataFrame(pairs)

    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(
            by="Abs" + title,
            ascending=False,
        )

    return pairs_df


def mutual_information_matrix(X, random_state=42):
    """
    Compute the pairwise mutual information matrix between columns of X.

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame containing only numerical features.

    Returns
    -------
    pandas.DataFrame
        Symmetric mutual information matrix.
    """
    cols = X.columns
    n = len(cols)

    mi = np.zeros((n, n))

    for i in range(n):
        mi[i, i] = np.nan  # or 0 if you prefer

        for j in range(i + 1, n):
            value = mutual_info_regression(
                X[[cols[i]]],
                X[cols[j]],
                random_state=random_state,
            )[0]

            mi[i, j] = value
            mi[j, i] = value

    return pd.DataFrame(mi, index=cols, columns=cols)


def plot_feature_vs_glucose(
    df: pd.DataFrame,
    X_norm: pd.DataFrame,
    corr_glc: pd.Series,
    mi_glc: pd.Series,
    top_features: list[str],
) -> None:
    # in a loop over all top_features, plot the series of each feature against GLC, with a linear regression line
    # recover the subject_id and use a different color for each subject_id

    # the name can be SUBJECT_ID or participant_id, depending on the dataset
    if "SUBJECT_ID" in df.columns:
        subject_ids = df["SUBJECT_ID"]
    else:
        subject_ids = df["participant_id"]

    for feature in top_features:
        corr_value = corr_glc.get(feature, np.nan)
        mi_value = mi_glc.get(feature, np.nan)

        plt.figure(figsize=(8, 6))

        sns.scatterplot(
            x=X_norm[feature],
            y=df["GLC"],
            hue=subject_ids,
            palette="tab20",   # or "husl", "Set2", etc.
            s=60,
        )

        # Overall regression line (optional)
        sns.regplot(
            x=X_norm[feature],
            y=df["GLC"],
            scatter=False,
            line_kws={"color": "black", "linewidth": 2},
        )

        plt.title(
            f"{feature} vs GLC (corr = {corr_value:.3f}, MI = {mi_value:.3f})"
        )
        plt.xlabel(feature)
        plt.ylabel("GLC")
        plt.legend(title="Participant", bbox_to_anchor=(
            1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()


def check_affine_equivalence(
    X: pd.DataFrame,
    pairs_df: pd.DataFrame,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    rows = []

    for _, row in pairs_df.iterrows():

        f1 = row["Feature1"]
        f2 = row["Feature2"]

        x = X[f1].to_numpy(dtype=float).reshape(-1, 1)
        y = X[f2].to_numpy(dtype=float)

        model = LinearRegression()
        model.fit(x, y)

        y_pred = model.predict(x)
        residual = y - y_pred

        rows.append({
            "Feature1": f1,
            "Feature2": f2,
            "Correlation": row["Correlation"],
            "slope": model.coef_[0],
            "intercept": model.intercept_,
            "max_abs_residual": np.max(np.abs(residual)),
            "mean_abs_residual": np.mean(np.abs(residual)),
            "affine_equivalent": np.max(np.abs(residual)) < tolerance,
        })

    return pd.DataFrame(rows)


def compare_highly_correlated_pairs(
    X: pd.DataFrame,
    pairs_df: pd.DataFrame,
    atol: float = 1e-12,
    rtol: float = 1e-9,
) -> pd.DataFrame:
    rows = []

    # print("XXXA", pairs_df.head())

    for _, row in pairs_df.iterrows():
        f1 = row["Feature1"]
        f2 = row["Feature2"]
        corr = row["Correlation"]

        x1 = X[f1].to_numpy(dtype=float)
        x2 = X[f2].to_numpy(dtype=float)

        diff = x1 - x2
        abs_diff = np.abs(diff)

        identical_exact = np.array_equal(x1, x2)
        identical_close = np.allclose(x1, x2, atol=atol, rtol=rtol)

        neg_identical_exact = np.array_equal(x1, -x2)
        neg_identical_close = np.allclose(x1, -x2, atol=atol, rtol=rtol)

        rows.append({
            "Feature1": f1,
            "Feature2": f2,
            "Correlation": corr,

            "identical_exact": identical_exact,
            "identical_close": identical_close,

            "negative_identical_exact": neg_identical_exact,
            "negative_identical_close": neg_identical_close,

            "mean_abs_diff": abs_diff.mean(),
            "max_abs_diff": abs_diff.max(),
            "std_abs_diff": abs_diff.std(),

            "n_equal_exact": np.sum(x1 == x2),
            "pct_equal_exact": 100 * np.mean(x1 == x2),

            "n_samples": len(x1),
        })

    return pd.DataFrame(rows)


def main(datasetConfig: DatasetConfig) -> None:
    """
    Run feature statistics and correlation analysis for a given CSV file.
    """

    # dataset_name = datasetConfig.get_value("DATASET_NAME")
    input_file_name = Path(datasetConfig.features_path) / \
        datasetConfig.get_features_file_name()
    # features_file_prefix = datasetConfig.get_value("FEATURES_FILE_PREFIX")

    statistics_output_folder = datasetConfig.get_statistics_output_folder()

    # input_path = datasetConfig.get_value("SIMULATIONS_INPUT_PATH")
    # input_path = Path(input_path) / "metadata" / "metadata_columns.txt"
    # read metadata_columns.txt, skip first row (header) and store the column names in a list
    # metadata_columns = [line.strip()
    #                    for line in open(input_path).read().splitlines()[1:]]

    metadata_columns = datasetConfig.get_chosen_metadata_columns()

    target = "GLC"

    # If the target is part of the list,
    # exclude the target column from the list metadata_columns
    if target in metadata_columns:
        metadata_columns.remove(target)

    excluded_meta_info_cols = metadata_columns

    # create output folder if it does not exist
    os.makedirs(statistics_output_folder, exist_ok=True)

    # ======================================================
    # Load CSV
    # ======================================================
    print("Opening", input_file_name)
    df = pd.read_csv(input_file_name)

    # Guilherme's TCC dataset has a column called "ID" instead of "SUBJECT_ID" or "participant_id"
    # if df does not have a column called participant_id nor a
    # column called SUBJECT_ID, then replace ID by SUBJECT_ID
    if "participant_id" not in df.columns and "SUBJECT_ID" not in df.columns:
        df = df.rename(columns={"ID": "SUBJECT_ID"})

    # ======================================================
    # Select feature columns, excluding the glucose target and any meta-information columns
    # ======================================================
    feature_cols = [
        c for c in df.columns
        if c not in (excluded_meta_info_cols + [target])
    ]
    original_number_of_features = len(feature_cols)

    # subtract 1 for the target column
    num_dropped_meta_info_cols = len(df.columns) - len(feature_cols) - 1
    print(f"Dropped {num_dropped_meta_info_cols} meta-information columns:")

    print(f"Initial number of features = {len(feature_cols)}")

    if USE_MIN_MAX_NORMALIZATION:
        # Min-Max normalization
        scaler = MinMaxScaler()
    else:
        # Z-score normalization
        scaler = StandardScaler()

    X_norm = pd.DataFrame(
        scaler.fit_transform(df[feature_cols]),
        columns=feature_cols,
        index=df.index,
    )

    # ======================================================
    # Mutual information among features
    # ======================================================
    if True:
        print("\nComputing mutual information among features...")
        mi_features = mutual_information_matrix(
            X_norm)

    pairs_df_mutual_information = find_largest_values(mi_features,
                                                      MUTUAL_INFORMATION_THRESHOLD,
                                                      "MutualInformation")

    print("\nHighly similar features according to Mutual Information:")
    print(pairs_df_mutual_information.head(50))

    pairs_df_mutual_information.to_csv(os.path.join(
        statistics_output_folder, "high_mutual_information_pairs.csv"), index=False)
    print(
        f"Saved pairs with high mutual information to '{os.path.join(statistics_output_folder, 'high_mutual_information_pairs.csv')}'")

    if SHOULD_PLOT:
        plot_mutual_information_heatmap(X_norm, pairs_df_mutual_information)

    # ======================================================
    # Correlation among features
    # ======================================================
    corr_features = X_norm.corr()  # calculate the correlation matrix among features

    # ------------------------------------------------------
    # Find pairs of highly correlated features
    # ------------------------------------------------------
    correlation_largest_pairs_df = find_largest_values(corr_features,
                                                       CORRELATION_THRESHOLD,
                                                       "Correlation")

    print("\nHighly similar features according to Correlation:")
    print(correlation_largest_pairs_df.head(50)
          if not correlation_largest_pairs_df.empty else "No highly correlated pairs found.")

    output_file = os.path.join(
        statistics_output_folder, "highly_correlated_pairs.csv")
    correlation_largest_pairs_df.to_csv(output_file, index=False)

    print(f"Saved highly correlated pairs to '{output_file}'")

    # ------------------------------------------------------
    # Plot correlation heatmap
    # ------------------------------------------------------
    if SHOULD_PLOT:
        plot_correlation_heatmap(X_norm, correlation_largest_pairs_df)

    # Check if the highly correlated features are identical or nearly identical
    comparison_df = compare_highly_correlated_pairs(
        X_norm,
        correlation_largest_pairs_df,
        atol=1e-12,
        rtol=1e-9,
    )

    print("\nComparison of highly correlated pairs:")
    print(comparison_df.head(50))

    comparison_df.to_csv(
        os.path.join(
            statistics_output_folder, "highly_correlated_pairs_identity_check.csv"),
        index=False
    )
    print(
        f"Saved highly correlated pairs identity check to '{os.path.join(statistics_output_folder, 'highly_correlated_pairs_identity_check.csv')}'")

    # ======================================================
    # Check for affine equivalence among highly correlated features
    # ======================================================
    features_to_drop_because_affine = remove_feature_if_affine(
        X_norm, correlation_largest_pairs_df, statistics_output_folder, tolerance=1e-10)
    X_norm = X_norm.drop(columns=features_to_drop_because_affine)

    # ======================================================
    # Correlation and Mutual Information with the target GLC
    # ======================================================

    # Pearson correlation
    corr_glc = X_norm.join(df["GLC"]).corr()["GLC"].drop("GLC")

    # Mutual Information using sklearn's mutual_info_regression
    mi_glc = mutual_info_regression(
        X_norm,
        df["GLC"],
        n_neighbors=3,      # can try 3, 5 or 10 because MI are noisy
        random_state=42,
    )
    mi_glc_series = pd.Series(mi_glc, index=X_norm.columns)

    # Build summary DataFrame
    corr_glc_mi_df = pd.DataFrame({
        "Feature": X_norm.columns,
        "Correlation_with_GLC": corr_glc.loc[X_norm.columns].values,
        "AbsCorrelation": np.abs(corr_glc.loc[X_norm.columns].values),
        "MutualInformation_with_GLC": mi_glc,
    })

    # sort by both corr and MI, first by MI then by abs(corr)
    corr_glc_mi_df = corr_glc_mi_df.sort_values(
        by=["MutualInformation_with_GLC", "AbsCorrelation"],
        ascending=False,
    )

    print("\nTop features according to MI and correlation with GLC:")
    print(corr_glc_mi_df.head(30))

    print("\nTop features by Mutual Information with GLC:")
    print(
        corr_glc_mi_df.sort_values(
            by="MutualInformation_with_GLC",
            ascending=False,
        ).head(30)
    )

    print("\nTop features by correlation with GLC:")
    print(
        corr_glc_mi_df.sort_values(
            by="AbsCorrelation",
            ascending=False,
        ).head(30)
    )

    corr_glc_mi_df.to_csv(os.path.join(
        statistics_output_folder, "correlation_and_mi_with_GLC.csv"), index=False)
    print(
        f"Saved correlation and mutual information with GLC to '{os.path.join(statistics_output_folder, 'correlation_and_mi_with_GLC.csv')}'")

    # ======================================================
    # Heatmap of top 30 correlated features
    # ======================================================

    if SHOULD_PLOT:
        plot_some_feature_vs_glucose(
            corr_glc_mi_df, df, X_norm, corr_glc, mi_glc_series)
        # plot_feature_vs_glucose(df, X_norm, corr_glc, mi_glc_series, features_to_plot)

    if SHOULD_PLOT:
        plot_hierarchical_clustering(X_norm)

    # read FINAL_NUMBER_OF_FEATURES
    final_number_of_features = datasetConfig.get_value(
        "FINAL_NUMBER_OF_FEATURES")

    #  Minimum Redundancy Maximum Relevance (mRMR) feature selection for regression tasks
    print("\nPerforming mRMR feature selection...")
    selected = mrmr_regression(
        X=X_norm,
        y=df[target],
        K=final_number_of_features
    )
    print("Features selected by mRMR:\n", selected)
    # keep only the selected features in X_final
    X_final = X_norm[selected]

    # save the cleaned and normalized features to a CSV file
    # create an output file name based on the input file name
    # remove the extension from basename

    # before saving, need to incorporate the target column GLC back into the DataFrame
    # and make sure they are aligned correctly by index
    X_final = X_final.copy()  # Create a copy to avoid SettingWithCopyWarning

    # the name can be SUBJECT_ID or participant_id, depending on the dataset
    if "participant_id" in df.columns:
        # also add back columns ['participant_id', 'session_id', 'datetime']
        # X_final = X_final.join(
        #    df[['participant_id', 'session_id', 'datetime']], how='right')

        # find the columns in df that are in metadata_columns
        metadata_columns_in_df = [
            col for col in metadata_columns if col in df.columns]

        X_final = X_final.join(
            df[metadata_columns_in_df], how='right')
        X_and_y_final = X_final.join(df[target])
        if "session_id" not in df.columns:
            # create a new column called session_id with all values equal to 0
            df["session_id"] = 1
        if "datetime" not in df.columns:
            # create a new column called datetime
            df["datetime"] = -1
    elif "SUBJECT_ID" in df.columns:
        # add back columns ['SUBJECT_ID']
        X_final = X_final.join(
            df[['SUBJECT_ID']], how='right')
        X_and_y_final = X_final.join(df[target])
    else:
        # if neither participant_id nor SUBJECT_ID is present, just add back the target column
        raise ValueError(
            "Neither 'participant_id' nor 'SUBJECT_ID' found in the DataFrame. Cannot join back the target column.")

    # Save new DataFrame with cleaned features and target GLC to CSV
    output_file_name = datasetConfig.get_selected_features_file_name()
    X_and_y_final.to_csv(output_file_name, index=False)
    print(
        f"Saved cleaned (non-normalized) features and target GLC to '{output_file_name}'")

    print("\n#### Final result ####")
    print("Input file name = ", input_file_name)
    print("Original # of features = ", original_number_of_features)
    print("Original # of examples = ", df.shape[0])

    print("# of dropped features because had meta information = ",
          num_dropped_meta_info_cols)
    print("# of dropped features due to high correlation = ",
          len(features_to_drop_because_affine))
    # subtract 1 for the target column
    print("Final # of features = ", len(X_and_y_final.columns)-1)
    print("Final # of examples = ", X_and_y_final.shape[0])
    print("\nFinal used columns = ", X_final.columns.tolist())


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

    datasetConfig = DatasetConfig(args.json_file)
    main(datasetConfig)
