'''
Define models and hyperparameter search spaces for evaluation.
'''

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from datasets_util.naming_conventions import DatasetConfig

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, Matern, WhiteKernel
from sklearn.linear_model import ARDRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import HuberRegressor

# Models
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

# from machinelearning.regression_evaluation import ModelConfigDict
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import BayesianRidge

INCLUDE_MODELS_THAT_TAKE_LONG_TIME = False

ModelConfigDict = dict[str, Any]


def get_model_configs():
    if True:
        return get_model_configs_low_overfit()
        # return get_few_models_for_debugging()
    else:
        return get_model_configs_more_complex_models()


def get_few_models_for_debugging(
    random_state: int | None = 42,
) -> dict[str, ModelConfigDict]:
    """Build conservative model configs for small-n regression.

    Designed for small datasets, e.g. n ≈ 80, where overfitting risk is high.
    """
    configs: dict[str, ModelConfigDict] = {}

    configs["dummy"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DummyRegressor()),
        ]),
        "params": {
            "model__strategy": ["mean", "median"],
        },
    }

    configs["ridge"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ]),
        "params": {
            "model__alpha": np.logspace(0, 4, 9),  # 1 ... 10000
        },
    }
    return configs


def get_model_configs_low_overfit(
    random_state: int | None = 42,
) -> dict[str, ModelConfigDict]:
    """Build conservative model configs for small-n regression.

    Designed for small datasets, e.g. n ≈ 80, where overfitting risk is high.
    """
    configs: dict[str, ModelConfigDict] = {}

    configs["dummy"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DummyRegressor()),
        ]),
        "params": {
            "model__strategy": ["mean", "median"],
        },
    }
    '''
    configs["ridge"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ]),
        "params": {
            "model__alpha": np.logspace(0, 4, 9),  # 1 ... 10000
        },
    }
    '''

    configs["lasso"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(
                max_iter=100000,
                tol=1e-4,
                selection="cyclic",
                random_state=random_state,
            )),
        ]),
        "params": {
            "model__alpha": np.logspace(-2, 2, 13),  # 0.01 ... 100
        },
    }

    '''
    configs["elasticnet"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                max_iter=100000,
                tol=1e-4,
                selection="cyclic",
                random_state=random_state,
            )),
        ]),
        "params": {
            "model__alpha": np.logspace(-2, 3, 16),
            "model__l1_ratio": [0.05, 0.1, 0.2, 0.5, 0.8],
        },
    }
    '''

    '''
    configs["bayesian_ridge"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", BayesianRidge()),
        ]),
        "params": {
            "model__alpha_1": [1e-6, 1e-5, 1e-4],
            "model__alpha_2": [1e-6, 1e-5, 1e-4],
            "model__lambda_1": [1e-6, 1e-5, 1e-4],
            "model__lambda_2": [1e-6, 1e-5, 1e-4],
        },
    }
    '''

    '''
    configs["pls"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", PLSRegression()),
        ]),
        "params": {
            "model__n_components": [1, 2, 3, 4, 5],
        },
    }
    '''

    configs["svr"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR()),
        ]),
        "params": [
            # Linear kernel
            {
                "model__kernel": ["linear"],
                "model__C": [0.03, 0.1, 0.3, 1.0, 3.0],
                "model__epsilon": [1.0, 2.0, 5.0, 10.0],
            },

            # Polynomial kernel (degree 4)
            {
                "model__kernel": ["poly"],
                "model__degree": [4],
                "model__C": [0.03, 0.1, 1.0, 3.0],
                "model__epsilon": [1.0, 5.0],
                "model__gamma": ["scale", 0.003, 0.01, 0.03],
                "model__coef0": [0.0, 1.0],
            },

            # RBF kernel
            {
                "model__kernel": ["rbf"],
                "model__C": [0.03, 0.1, 0.3, 1.0, 3.0],
                "model__epsilon": [1.0, 2.0, 5.0, 10.0],
                "model__gamma": ["scale", 0.001, 0.003, 0.01, 0.03],
            },
        ],
    }
    configs["knn"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor()),
        ]),
        "params": {
            "model__n_neighbors": [1, 3, 5, 7],
            "model__weights": ["uniform"],
            "model__p": [1, 2],
        },
    }

    configs["gpr"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                normalize_y=True,
                optimizer=None,   # disables kernel hyperparameter optimization
                random_state=random_state,
            )),
        ]),
        "params": {
            "model__alpha": [0.1, 0.3, 1.0, 3.0, 10.0],
            "model__kernel": [
                C(1.0, constant_value_bounds="fixed")
                * RBF(length_scale=5.0, length_scale_bounds="fixed"),

                C(1.0, constant_value_bounds="fixed")
                * RBF(length_scale=10.0, length_scale_bounds="fixed"),

                C(1.0, constant_value_bounds="fixed")
                * Matern(
                    length_scale=5.0,
                    length_scale_bounds="fixed",
                    nu=1.5,
                ),

                C(1.0, constant_value_bounds="fixed")
                * Matern(
                    length_scale=10.0,
                    length_scale_bounds="fixed",
                    nu=2.5,
                ),
            ],
        },
    }

    configs["rf"] = {
        "pipeline": RandomForestRegressor(
            random_state=random_state,
            bootstrap=True,
        ),
        "params": {
            "n_estimators": [10, 30, 50],
            "max_depth": [2, 3, 4],
            "min_samples_leaf": [5, 10, 15],
            "max_features": [0.3, 0.5, "sqrt"],
        },
    }

    configs["gbr"] = {
        "pipeline": GradientBoostingRegressor(
            random_state=random_state,
        ),
        "params": {
            "n_estimators": [10, 30, 50],
            "learning_rate": [0.01, 0.03, 0.05],
            "max_depth": [1, 2, 3],
            "min_samples_leaf": [5, 10],
            "subsample": [0.5, 0.7],
        },
    }

    if INCLUDE_MODELS_THAT_TAKE_LONG_TIME:
        configs["mlp"] = {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(
                    solver="lbfgs",
                    activation="tanh",
                    max_iter=5000,
                    random_state=random_state,
                )),
            ]),
            "params": {
                "model__hidden_layer_sizes": [(3,), (5,), (8,), (10,)],
                "model__alpha": np.logspace(-2, 3, 10),
            },
        }

    return configs

# -----------------------------
# Model + Hyperparameter Space
# -----------------------------


def get_model_configs_more_complex_models(random_state: int | None = 42) -> dict[str, ModelConfigDict]:
    """Build model pipelines and hyperparameter grids used in evaluation.

    Parameters
    ----------
    random_state : int | None, optional
        Random seed used by stochastic models.

    Returns
    -------
    dict[str, ModelConfigDict]
        Mapping from model name to pipeline and parameter grid.
    """
    configs: dict[str, ModelConfigDict] = {}

    # Dummy Regressor (baseline)
    configs["dummy"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DummyRegressor(strategy="mean"))
        ]),
        "params": {"model__strategy": ["mean", "median"]}
    }

    # Ridge
    configs["ridge"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge())
        ]),
        "params": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    }

    # Lasso
    configs["lasso"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(
                max_iter=100000,
                tol=1e-3,
                selection="random",
                random_state=42,
            ))
        ]),
        "params": {
            "model__alpha": np.logspace(-3, 1, 20)
        }
    }

    configs["elasticnet"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                max_iter=100000,
                tol=1e-3,
                selection="random",
                random_state=42,
            ))
        ]),
        "params": {
            "model__alpha": np.logspace(-2, 2, 25),
            "model__l1_ratio": [0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0],
        }
    }

    # SVR
    configs["svr"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR())
        ]),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__epsilon": [0.01, 0.1, 0.5],
            "model__kernel": ["rbf"],
            "model__gamma": ["scale", "auto"]
        }
    }

    # GPR
    configs["gpr"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42,
            ))
        ]),
        "params": {
            "model__alpha": [1e-3, 1e-2, 1e-1, 1.0],
            "model__kernel": [
                C(1.0, (1e-2, 1e2)) * RBF(
                    length_scale=1.0,
                    length_scale_bounds=(1e-1, 1e2)
                ),
                C(1.0, (1e-2, 1e2)) * Matern(
                    length_scale=1.0,
                    length_scale_bounds=(1e-1, 1e2),
                    nu=1.5
                ),
                C(1.0, (1e-2, 1e2)) * Matern(
                    length_scale=1.0,
                    length_scale_bounds=(1e-1, 1e2),
                    nu=2.5
                ),
            ],
        }
    }

    # KNN
    configs["knn"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor())
        ]),
        "params": {
            "model__n_neighbors": [1, 3, 5, 7],
            "model__weights": ["uniform", "distance"]
        }
    }

    # Random Forest
    configs["rf"] = {
        "pipeline": RandomForestRegressor(random_state=random_state),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 3, 5, 10],
            "min_samples_leaf": [1, 3, 5]
        }
    }

    # Gradient Boosting
    configs["gbr"] = {
        "pipeline": GradientBoostingRegressor(random_state=random_state),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4]
        }
    }

    configs["huber"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", HuberRegressor(
                max_iter=1000,
            ))
        ]),
        "params": {
            "model__epsilon": [1.1, 1.2, 1.35, 1.5, 2.0],
            "model__alpha": np.logspace(-5, 1, 7),
        }
    }
    configs["quantile"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", QuantileRegressor(
                quantile=0.5,
                solver="highs",
            ))
        ]),
        "params": {
            "model__alpha": np.logspace(-5, 1, 10),
        }
    }

    configs["ransac"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RANSACRegressor(
                estimator=Ridge(alpha=1.0),
                random_state=random_state,
            ))
        ]),
        "params": {
            "model__min_samples": [0.5, 0.7, 0.9],
            "model__max_trials": [100, 300],
            "model__residual_threshold": [5, 10, 15, 20],
        }
    }
    configs["ard"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ARDRegression())
        ]),
        "params": {
            "model__threshold_lambda": [1e3, 1e4, 1e5],
        }
    }

    if INCLUDE_MODELS_THAT_TAKE_LONG_TIME:
        # MLP takes long time
        configs["mlp"] = {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(
                    solver="adam",
                    activation="relu",
                    early_stopping=True,
                    validation_fraction=0.10,
                    n_iter_no_change=30,
                    max_iter=5000,
                    random_state=42,
                ))
            ]),
            "params": {
                "model__hidden_layer_sizes": [
                    (16,),
                    (32,),
                    (32, 16),
                    (64, 32),
                ],
                "model__alpha": np.logspace(-6, -1, 6),
                "model__learning_rate_init": [5e-4, 1e-3],
            }
        }

    return configs
