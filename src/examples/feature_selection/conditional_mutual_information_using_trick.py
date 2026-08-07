'''
Illustrate conditional mutual information and redundancy in feature selection,
using the trick of removing the linear effect of a selected feature from both the target and the candidate feature.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression


np.random.seed(42)
n = 3000

# ============================================================
# Generate redundant features
# ============================================================
# Latent variable that determines y
z = np.random.randn(n)

# Target
y = z + 0.25 * np.random.randn(n)

# x2 is strongly informative about y
x2 = z + 0.25 * np.random.randn(n)

# x3 is almost a copy of x2, so it is also strongly informative about y,
# but mostly redundant once x2 is already known.
x3 = x2 + 0.05 * np.random.randn(n)


# ============================================================
# Basic correlation and mutual information
# ============================================================

corr_x2_y = pearsonr(x2, y)[0]
corr_x3_y = pearsonr(x3, y)[0]

mi_x2_y = mutual_info_regression(x2.reshape(-1, 1), y, random_state=42)[0]
mi_x3_y = mutual_info_regression(x3.reshape(-1, 1), y, random_state=42)[0]

print("Individual relevance")
print("--------------------")
print(f"corr(x2, y) = {corr_x2_y:.3f}")
print(f"corr(x3, y) = {corr_x3_y:.3f}")
print(f"MI(x2; y)   = {mi_x2_y:.3f}")
print(f"MI(x3; y)   = {mi_x3_y:.3f}")


# ============================================================
# Approximate conditional mutual information I(x3; y | x2)
# by removing the linear effect of x2 from both x3 and y.
#
# If x3 adds no new information after x2, then:
#     residual(x3 | x2) should have little MI with residual(y | x2)
# ============================================================

model_y_given_x2 = LinearRegression()
model_y_given_x2.fit(x2.reshape(-1, 1), y)
y_residual = y - model_y_given_x2.predict(x2.reshape(-1, 1))

model_x3_given_x2 = LinearRegression()
model_x3_given_x2.fit(x2.reshape(-1, 1), x3)
x3_residual = x3 - model_x3_given_x2.predict(x2.reshape(-1, 1))

partial_corr = pearsonr(x3_residual, y_residual)[0]

conditional_mi_approx = mutual_info_regression(
    x3_residual.reshape(-1, 1),
    y_residual,
    random_state=42,
)[0]

print("\nRedundancy after selecting x2")
print("----------------------------")
print(f"partial corr(x3, y | x2) ≈ {partial_corr:.3f}")
print(f"CMI approximation I(x3; y | x2) ≈ {conditional_mi_approx:.3f}")


# ============================================================
# Plot 1: x2 vs y
# ============================================================

plt.figure(figsize=(7, 5))
plt.scatter(x2, y, alpha=0.35)
plt.xlabel("x2")
plt.ylabel("y")
plt.title(
    "x2 is useful for predicting y\n"
    f"corr(x2, y) = {corr_x2_y:.3f}, MI(x2; y) = {mi_x2_y:.3f}"
)
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Plot 2: x3 vs y
# ============================================================

plt.figure(figsize=(7, 5))
plt.scatter(x3, y, alpha=0.35)
plt.xlabel("x3")
plt.ylabel("y")
plt.title(
    "x3 also appears useful by itself\n"
    f"corr(x3, y) = {corr_x3_y:.3f}, MI(x3; y) = {mi_x3_y:.3f}"
)
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Plot 3: residual x3 vs residual y after removing x2
# ============================================================

plt.figure(figsize=(7, 5))
plt.scatter(x3_residual, y_residual, alpha=0.35)
plt.xlabel("residual of x3 after predicting x3 from x2")
plt.ylabel("residual of y after predicting y from x2")
plt.title(
    "After selecting x2, x3 adds little new information\n"
    f"partial corr ≈ {partial_corr:.3f}, CMI approx ≈ {conditional_mi_approx:.3f}"
)
plt.grid(True)
plt.tight_layout()
plt.show()
