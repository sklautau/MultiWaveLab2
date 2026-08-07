import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression


np.random.seed(42)
n = 1000


# ============================================================
# Example 1: low correlation, high mutual information
# y = x1^2
# Pearson correlation is exactly zero (because positive and negative values cancel),
# but knowing x1 almost completely determines y.
# ============================================================

x1 = np.random.uniform(-2, 2, n)
y1 = x1**2 + 0.05 * np.random.randn(n)

corr1, _ = pearsonr(x1, y1)
mi1 = mutual_info_regression(x1.reshape(-1, 1), y1, random_state=42)[0]

plt.figure(figsize=(7, 5))
plt.scatter(x1, y1, alpha=0.5)
plt.xlabel("x1")
plt.ylabel("y")
plt.title(
    "Example 1: Low correlation, high mutual information\n"
    f"Pearson r = {corr1:.3f}, MI = {mi1:.3f}"
)
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Example 2: high correlation, high mutual information
# y = 5*x2 + noise
# ============================================================

x2 = np.random.uniform(-2, 2, n)
y2 = 5 * x2 + np.random.randn(n)

corr2, _ = pearsonr(x2, y2)
mi2 = mutual_info_regression(x2.reshape(-1, 1), y2, random_state=42)[0]

plt.figure(figsize=(7, 5))
plt.scatter(x2, y2, alpha=0.5)
plt.xlabel("x2")
plt.ylabel("y")
plt.title(
    "Example 2: High correlation, high mutual information\n"
    f"Pearson r = {corr2:.3f}, MI = {mi2:.3f}"
)
plt.grid(True)
plt.tight_layout()
plt.show()
