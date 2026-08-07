'''
The “supervised” methods are learning embeddings
using glucose (GLC) to visualize glucose structure,
and later the plots use patient_id to show
subject separability.
'''
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import os
import umap

from datasets_util.naming_conventions import DatasetConfig

INPUT_FEATURES_FILE = "dataset_selected_features_loso.csv"
METHOD_TO_ANNOTATE = "UMAP"  # "PLS (GLC)"  # or "Supervised UMAP (GLC)"


def run_dimensionality_reduction(datasetConfig,
                                 method_to_annotate: str = METHOD_TO_ANNOTATE,
                                 ) -> None:
    output_dir = datasetConfig.get_dataset_machine_learning_path()

    input_file_name = os.path.join(
        datasetConfig.get_dataset_machine_learning_path(), INPUT_FEATURES_FILE)

    df = pd.read_csv(input_file_name)

    # remove participants
    # df = df.loc[df["participant_id"] != "ieb_02"]

    y_glucose = df['GLC'].values              # supervision signal
    patient_ids = df['participant_id'].values  # ONLY for coloring

    # remove columns:
    metadata_cols = ['participant_id', 'session_id', 'datetime', 'GLC']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    X = df[feature_cols].values

    # Normalize
    X = StandardScaler().fit_transform(X)

    # Create numeric labels ONLY for plotting
    unique_patients = np.unique(patient_ids)
    y_plot = np.array([np.where(unique_patients == pid)[0][0]
                      for pid in patient_ids])

    print(f"X shape: {X.shape}")
    print(f"Num patients: {len(unique_patients)}")

    # ==============================
    # 2. Define methods
    # ==============================

    methods = {}

    # Unsupervised
    methods["PCA"] = PCA(n_components=2)

    methods["t-SNE"] = TSNE(
        n_components=2,
        perplexity=10,
        learning_rate='auto',
        init='pca',
        random_state=0
    )

    methods["UMAP"] = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=0,
        n_jobs=1
    )

    # Supervised (using GLC)
    methods["Supervised UMAP (GLC)"] = umap.UMAP(
        n_components=2,
        n_neighbors=10,  # 5 to 10, smaller n_neighbors => more local structure => tighter clusters grouping
        min_dist=0.1,  # 0.0 to 0.1, smaller min_dist => points can collapse into clusters
        random_state=0,
        metric="cosine",
        init="pca",
        n_jobs=1
    )

    methods["PLS (GLC)"] = PLSRegression(n_components=2, max_iter=1000)

    # ==============================
    # 3. Fit + transform
    # ==============================

    embeddings = {}

    for name, model in methods.items():
        print(f"Running {name}...")

        if "Supervised UMAP" in name:
            Z = model.fit_transform(X, y_glucose)

        elif "PLS" in name:
            Z = model.fit_transform(X, y_glucose.reshape(-1, 1))[0]

        else:
            Z = model.fit_transform(X)

        embeddings[name] = Z

    # ==============================
    # 4. Plot (colored by patient_id)
    # ==============================

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Embeddings (Supervised on GLC, Colored by Patient)',
                 fontsize=16, fontweight='bold')

    for i, (name, Z) in enumerate(embeddings.items(), 1):
        plt.subplot(2, 3, i)

        for label_idx, patient in enumerate(unique_patients):
            mask = y_plot == label_idx
            plt.scatter(
                Z[mask, 0],
                Z[mask, 1],
                label=patient,
                s=20,
                alpha=0.7
            )

        plt.title(name)
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')

        if i == 1:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_methods.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # ==============================
    # 5. Detailed plot with GLC labels
    # ==============================

    # Choose which embedding to annotate
    # method_to_annotate = "Supervised UMAP (GLC)"  # or "PLS (GLC)"

    Z = embeddings[method_to_annotate]

    plt.figure(figsize=(8, 6))
    plt.title(f"{method_to_annotate} with GLC labels",
              fontsize=14, fontweight='bold')

    # Scatter (colored by patient as before)
    for label_idx, patient in enumerate(unique_patients):
        mask = y_plot == label_idx
        plt.scatter(
            Z[mask, 0],
            Z[mask, 1],
            label=patient,
            s=25,
            alpha=0.7
        )

    # ---- Add GLC annotations ----
    for i in range(Z.shape[0]):
        plt.text(
            Z[i, 0],
            Z[i, 1],
            f"{y_glucose[i]:.0f}",   # integer display (cleaner)
            fontsize=8,              # small font to reduce clutter
            alpha=0.6
        )

    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_with_glc_labels.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def main() -> None:

    # ==============================
    # 1. Load data
    # ==============================
    # Create a DatasetConfig instance
    # dataset_config_file = "multimodal_dataset_folders.json"

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

    # Create a DatasetConfig instance
    dataset_config_file = args.json_file

    datasetConfig = DatasetConfig(dataset_config_file)

    run_dimensionality_reduction(datasetConfig)


if __name__ == "__main__":
    main()
