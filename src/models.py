"""Forecast models, error metrics and simple cost estimates.

Two models are compared:

* **Persistence**, ``T_hat(t + H) = T(t)``. It assumes the temperature will not
  change. It has no fitted parameters and is hard to beat over short horizons.
* **Lagged linear regression**, ``T_hat(t + H) = b + sum(a_i * T(t - i * step))``,
  fitted by ordinary least squares.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .features import lag_columns


class PersistenceForecaster:
    """Predict that the future value equals the latest observed value."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceForecaster":
        """Do nothing and return ``self``: persistence has no parameters to fit."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the ``lag_0`` column, which is T(t), as the forecast."""
        return X["lag_0"].to_numpy()


class LaggedLinearRegression:
    """Ordinary least squares on the first ``n_lags`` lag columns.

    No feature scaling is used: every input is in °C and the coefficients stay
    directly interpretable, and scaling does not change an OLS prediction.

    Args:
        n_lags: Number of lag columns to use, at least 1.
    """

    def __init__(self, n_lags: int):
        if n_lags < 1:
            raise ValueError("n_lags must be at least 1")
        self.n_lags = n_lags
        self.columns = lag_columns(n_lags)
        self.model = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LaggedLinearRegression":
        """Fit the coefficients and return ``self``."""
        self.model.fit(X[self.columns].to_numpy(), np.asarray(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return forecasts in °C for each row of ``X``."""
        return self.model.predict(X[self.columns].to_numpy())

    def coefficients(self) -> dict[str, float]:
        """Return the fitted intercept (°C) and lag coefficients (dimensionless)."""
        out = {"intercept": float(self.model.intercept_)}
        out.update({c: float(a) for c, a in zip(self.columns, self.model.coef_)})
        return out


def mae(y_true, y_pred) -> float:
    """Mean absolute error, in the unit of ``y_true``."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, in the unit of ``y_true``."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def bias(y_true, y_pred) -> float:
    """Mean of ``prediction - observed``; positive means forecasts run high."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def mae_skill(model_mae: float, baseline_mae: float) -> float:
    """Fractional MAE improvement over the baseline; above 0 beats the baseline.

    Raises:
        ValueError: If ``baseline_mae`` is not positive, so skill is undefined.
    """
    if baseline_mae <= 0:
        raise ValueError("baseline_mae must be positive")
    return 1.0 - model_mae / baseline_mae


def condition_numbers(X: np.ndarray) -> dict[str, float]:
    """Measure how numerically sensitive the least-squares fit is.

    Neighbouring temperatures are almost identical, so the lag columns are
    strongly collinear. A large condition number means small input changes can
    move individual coefficients a lot, even if predictions stay stable.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.

    Returns:
        ``centred`` is the condition number after removing column means, which
        is the matrix LinearRegression factorises. ``standardised`` also divides
        by column standard deviations, and is NaN if any column is constant.
    """
    centred = X - X.mean(axis=0)
    std = X.std(axis=0)
    standardised = np.linalg.cond(centred / std) if np.all(std > 0) else float("nan")
    return {"centred": float(np.linalg.cond(centred)),
            "standardised": float(standardised)}


def flop_estimate(n_samples: int, n_features: int) -> dict[str, float]:
    """Estimate floating-point operations for a model of this size.

    These are textbook order-of-magnitude counts, not measurements. Fitting uses
    the Householder QR count ``2 n p^2 - (2/3) p^3``; scikit-learn's SVD-based
    solver has a larger constant but scales the same way. Each prediction needs
    ``p`` multiplications and ``p`` additions. Persistence needs no arithmetic.

    Args:
        n_samples: Training rows, ``n``.
        n_features: Features, ``p``.

    Returns:
        ``fit_flops`` and ``predict_flops_per_sample``.
    """
    n, p = float(n_samples), float(n_features)
    return {"fit_flops": 2 * n * p**2 - (2 / 3) * p**3,
            "predict_flops_per_sample": 2 * p}
