'''Plot probability density functions (PDFs) of glucose values from different datasets.'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

use_histograms = False  # Set to True to use histograms instead of PDFs

# ---------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------
files = {
    "IEB1": "glc_ieb1.csv",
    "IEB2": "glc_ieb2.csv",
    "IEB3": "glc_ieb3.csv",
}

# Colors (ColorBrewer-friendly)
colors = {
    "IEB1": "#007ea7",  # blue
    "IEB2": "#7f055f",  # orange
    "IEB3": "#558E41",  # green
}

# ---------------------------------------------------------------------
# Read data
# ---------------------------------------------------------------------
datasets = {}

for name, filename in files.items():
    df = pd.read_csv(filename)
    datasets[name] = df["GLC"].dropna().to_numpy()

# ---------------------------------------------------------------------
# Common bins for fair comparison
# ---------------------------------------------------------------------
all_values = np.concatenate(list(datasets.values()))
bins = np.histogram_bin_edges(all_values, bins="auto")

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))

for name in ["IEB1", "IEB2", "IEB3"]:
    if use_histograms:
        plt.hist(
            datasets[name],
            bins=bins,
            alpha=0.45,
            color=colors[name],
            edgecolor="black",
            linewidth=0.8,
            label=f"{name} (n={len(datasets[name])})",
        )
    else:
        # use pdfs
        if True:
            from scipy.stats import gaussian_kde

            x = np.linspace(all_values.min(), all_values.max(), 400)
            kde = gaussian_kde(datasets[name])
            plt.plot(x, kde(x), lw=2.5, color=colors[name])

            plt.plot(
                datasets[name],
                np.zeros_like(datasets[name]),
                "|",
                color=colors[name],
                markersize=10,
                alpha=0.5,
            )
        else:
            plt.hist(
                datasets[name],
                bins=bins,
                density=True,
                alpha=0.45,
                color=colors[name],
                edgecolor="black",
                linewidth=0.8,
                label=name,
            )

plt.xlabel("Glucose (mg/dL)", fontsize=12)
plt.ylabel("Estimated PDFs", fontsize=12)
plt.title("Glucose distribution across datasets", fontsize=12)


legend_elements = [
    Line2D([0], [0], color=colors["IEB1"], lw=2.5, label="IEB1"),
    Line2D([0], [0], color=colors["IEB2"], lw=2.5, label="IEB2"),
    Line2D([0], [0], color=colors["IEB3"], lw=2.5, label="IEB3"),
]

plt.legend(handles=legend_elements, loc="best", frameon=True)

# plt.legend(frameon=True)

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.show()
