'''
Illustrate conditional mutual information and redundancy in feature selection,
now using the discrete CMI equation instead of the linear regression trick.

Sklearn `mutual_info_regression` estimates ordinary MI, not conditional MI.

Main differences between this simple code and the one that sklearn
MI implements:
`sklearn` is less arbitrary than binning, more adaptive, and usually
more appropriate for continuous variables. More specifically:

1. **No binning**

   My simple CMI code discretizes continuous variables into bins. `sklearn`’s `mutual_info_regression` does **not** discretize continuous variables by default. It uses a **k-nearest-neighbor entropy estimator**, which is usually better for continuous data. ([Scikit-Learn][1])

2. **Adaptive local resolution**

   Binning uses fixed intervals or quantiles. kNN MI estimation adapts to the local data density: dense regions get small neighborhoods; sparse regions get larger neighborhoods.

3. **Bias–variance control via `n_neighbors`**

   In `sklearn`, `n_neighbors` controls the estimator smoothness. Larger values reduce variance but may increase bias. ([Scikit-Learn][1])

4. **Separate handling of continuous and discrete variables**

   `sklearn` distinguishes continuous and discrete features using `discrete_features`. This matters because the estimator changes depending on the variable type. ([Scikit-Learn][1])

5. **Noise injection to break ties**

   For continuous variables, `sklearn` adds very small random noise to remove repeated values, controlled by `random_state`. ([Scikit-Learn][1])

6. **Nonnegative clipping**

   True MI cannot be negative, but numerical estimates can be. `sklearn` replaces negative estimates by zero. ([Scikit-Learn][1])

[1]: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html"
'''
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression


def conditional_mutual_information(x2, x3, y):
    # ============================================================
    # Individual correlation and MI
    # ============================================================

    corr_x2_y = pearsonr(x2, y)[0]
    corr_x3_y = pearsonr(x3, y)[0]

    mi_x2_y = mutual_info_regression(
        x2.reshape(-1, 1), y, random_state=42
    )[0]

    mi_x3_y = mutual_info_regression(
        x3.reshape(-1, 1), y, random_state=42
    )[0]

    # ============================================================
    # Actual discrete CMI estimate
    # ============================================================

    cmi_x3_y_given_x2 = conditional_mutual_information_discrete(
        x=x3,
        y=y,
        z=x2,
        n_bins=10,
    )

    cmi_x2_y_given_x3 = conditional_mutual_information_discrete(
        x=x2,
        y=y,
        z=x3,
        n_bins=10,
    )

    print("Individual relevance")
    print("--------------------")
    print(f"corr(x2, y) = {corr_x2_y:.3f}")
    print(f"corr(x3, y) = {corr_x3_y:.3f}")
    print(f"MI(x2; y)   = {mi_x2_y:.3f}")
    print(f"MI(x3; y)   = {mi_x3_y:.3f}")

    print("\nConditional mutual information")
    print("------------------------------")
    print(f"I(x3; y | x2) ≈ {cmi_x3_y_given_x2:.5f}")
    print(f"I(x2; y | x3) ≈ {cmi_x2_y_given_x3:.5f}")


def discretize_equal_frequency(x, n_bins=10):
    """
    Discretize a continuous variable into approximately equal-frequency bins.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(x, quantiles)

    # Avoid duplicate bin edges when data has repeated values
    bin_edges = np.unique(bin_edges)

    # np.digitize gives bins 1, 2, ..., len(edges)
    # subtract 1 to get 0-based bin indices
    return np.digitize(x, bin_edges[1:-1], right=False)


def conditional_mutual_information_discrete(x, y, z, n_bins=10, eps=1e-12):
    """
    Estimate I(X; Y | Z) using the discrete CMI equation.

    This computes:

        I(X;Y|Z) = sum p(x,y,z) log( p(x,y|z) / (p(x|z)p(y|z)) )

    after discretizing x, y, and z.
    """

    x_disc = discretize_equal_frequency(x, n_bins)
    y_disc = discretize_equal_frequency(y, n_bins)
    z_disc = discretize_equal_frequency(z, n_bins)

    x_vals = np.unique(x_disc)
    y_vals = np.unique(y_disc)
    z_vals = np.unique(z_disc)

    n = len(x_disc)
    cmi = 0.0

    for xv in x_vals:
        for yv in y_vals:
            for zv in z_vals:
                p_xyz = np.mean(
                    (x_disc == xv) & (y_disc == yv) & (z_disc == zv)
                )

                if p_xyz == 0:
                    continue

                p_z = np.mean(z_disc == zv)

                p_xz = np.mean((x_disc == xv) & (z_disc == zv))
                p_yz = np.mean((y_disc == yv) & (z_disc == zv))

                p_xy_given_z = p_xyz / (p_z + eps)
                p_x_given_z = p_xz / (p_z + eps)
                p_y_given_z = p_yz / (p_z + eps)

                cmi += p_xyz * np.log(
                    (p_xy_given_z + eps)
                    / ((p_x_given_z + eps) * (p_y_given_z + eps))
                )

    return cmi


def gen_example1():
    # ============================================================
    # Generate example 1
    # ============================================================

    np.random.seed(42)
    n = 5000

    z = np.random.randn(n)

    y = z + 0.25 * np.random.randn(n)

    x2 = z + 0.25 * np.random.randn(n)

    # x3 is almost a duplicate of x2
    x3 = x2 + 0.05 * np.random.randn(n)
    return x2, x3, y


def gen_example2():
    # ============================================================
    # Generate example 2
    # ============================================================

    np.random.seed(42)
    n = 5000

    z2 = np.random.randn(n)
    z3 = np.random.randn(n)

    x2 = z2 + 0.2 * np.random.randn(n)
    x3 = z3 + 0.2 * np.random.randn(n)

    y = 1*x2 - 2*x3 + 0.01 * np.random.randn(n)

    return x2, x3, y


print("Example 1: x3 is redundant given x2")
x2, x3, y = gen_example1()
conditional_mutual_information(x2, x3, y)

print("\n\nExample 2: x3 is not redundant given x2")
x2, x3, y = gen_example2()
conditional_mutual_information(x2, x3, y)
