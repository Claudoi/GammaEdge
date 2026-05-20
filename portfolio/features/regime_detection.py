"""
Regime Detection using Hidden Markov Models

This module provides tools for detecting market regimes (Bull, Bear, Crisis)
using Hidden Markov Models (HMM) with Gaussian emissions.

Based on:
- Ang & Bekaert (2002): "Regime Switches in Interest Rates"
- Kritzman et al. (2012): "Regime Shifts: Implications for Dynamic Strategies"

Author: GammaEdge TIER 1 Enhancement
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# Regime labels mapping
REGIME_LABELS = {
    0: "Bull",  # Low vol, positive drift
    1: "Bear",  # High vol, negative drift
    2: "Crisis",  # Extreme vol, sharp drawdowns
}


class RegimeDetector:
    """
    Hidden Markov Model for market regime detection.

    Features used:
    - Returns (momentum)
    - Realized volatility (risk level)
    - Drawdown depth (stress indicator)

    States:
    - Bull: Low vol, positive returns
    - Bear: High vol, negative returns
    - Crisis: Extreme vol, large drawdowns

    Example:
        >>> detector = RegimeDetector(n_regimes=3)
        >>> detector.fit(returns_df)
        >>> regimes = detector.predict(returns_df)
        >>> print(regimes['regime'].value_counts())
    """

    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: Literal["full", "diag", "spherical", "tied"] = "full",
        n_iter: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize regime detector.

        Args:
            n_regimes: Number of hidden states (typically 2-3)
            covariance_type: Covariance structure of emissions
            n_iter: Maximum EM iterations
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        # HMM model (will be fitted)
        self.model: hmm.GaussianHMM | None = None
        self.scaler = StandardScaler()

        # Regime interpretation (fitted from data)
        self.regime_stats: pd.DataFrame | None = None

    def _prepare_features(
        self,
        df: pl.DataFrame,
        returns_col: str = "returns",
        vol_window: int = 20,
        dd_window: int = 20,
    ) -> np.ndarray:
        """
        Extract HMM features from returns time series.

        Features:
        1. Returns (momentum signal)
        2. Realized volatility (rolling std)
        3. Drawdown depth (cumulative loss from peak)

        Args:
            df: Polars DataFrame with date and returns columns
            returns_col: Column name for returns
            vol_window: Window for volatility calculation
            dd_window: Window for drawdown calculation

        Returns:
            (T, 3) array of standardized features
        """
        # Convert to pandas for easier rolling calculations
        df_pd = df.to_pandas()

        # Feature 1: Returns
        ret = df_pd[returns_col].values

        # Feature 2: Realized volatility (rolling std)
        vol = df_pd[returns_col].rolling(vol_window, min_periods=1).std().fillna(0).values

        # Feature 3: Drawdown depth
        cumret = (1 + df_pd[returns_col]).cumprod()
        running_max = cumret.expanding().max()
        dd = (cumret / running_max - 1).fillna(0).values

        # Stack features
        X = np.column_stack([ret, vol, dd])

        return X

    def fit(
        self,
        df: pl.DataFrame,
        returns_col: str = "returns",
        vol_window: int = 20,
        dd_window: int = 20,
    ) -> RegimeDetector:
        """
        Fit HMM to historical data.

        Args:
            df: Polars DataFrame with date and returns
            returns_col: Column name for returns
            vol_window: Window for volatility
            dd_window: Window for drawdown

        Returns:
            self (fitted)
        """
        # Validate minimum observations before fitting HMM
        min_obs = max(100, self.n_regimes * 30)
        if df.height < min_obs:
            raise ValueError(
                f"RegimeDetector.fit() requires at least {min_obs} observations "
                f"(n_regimes={self.n_regimes}); got {df.height}. "
                "Load more historical data."
            )

        # Prepare features
        X = self._prepare_features(df, returns_col, vol_window, dd_window)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )

        self.model.fit(X_scaled)

        # Compute regime statistics for interpretation
        states = self.model.predict(X_scaled)
        df_pd = df.to_pandas()

        regime_stats = []
        for regime in range(self.n_regimes):
            mask = states == regime
            if mask.sum() > 0:
                regime_stats.append(
                    {
                        "regime": regime,
                        "mean_return": df_pd.loc[mask, returns_col].mean(),
                        "volatility": df_pd.loc[mask, returns_col].std(),
                        "frequency": mask.mean(),
                        "avg_duration": self._compute_avg_duration(states, regime),
                    }
                )

        self.regime_stats = pd.DataFrame(regime_stats)

        # Sort regimes by mean return (descending) for labeling
        # Highest return → Bull, Lowest → Crisis
        self.regime_stats = self.regime_stats.sort_values(
            "mean_return", ascending=False
        ).reset_index(drop=True)

        return self

    def _compute_avg_duration(self, states: np.ndarray, regime: int) -> float:
        """Compute average duration of regime stays."""
        durations = []
        current_duration = 0

        for s in states:
            if s == regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return np.mean(durations) if durations else 0.0

    def predict(
        self,
        df: pl.DataFrame,
        returns_col: str = "returns",
        vol_window: int = 20,
        dd_window: int = 20,
    ) -> pl.DataFrame:
        """
        Predict regime states for new data.

        Args:
            df: Polars DataFrame with returns
            returns_col: Column name for returns
            vol_window: Window for volatility
            dd_window: Window for drawdown

        Returns:
            Polars DataFrame with additional columns:
            - regime: Most likely state (0, 1, 2)
            - regime_label: Human-readable label (Bull, Bear, Crisis)
            - prob_0, prob_1, prob_2: State probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare features
        X = self._prepare_features(df, returns_col, vol_window, dd_window)
        X_scaled = self.scaler.transform(X)

        # Predict states and probabilities
        states = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)

        # Map states to labels (sorted by mean return)
        state_mapping = dict(
            zip(
                self.regime_stats["regime"].values,
                ["Bull", "Bear", "Crisis"][: self.n_regimes],
                strict=False,
            )
        )

        labels = [state_mapping.get(s, f"Regime_{s}") for s in states]

        # Build result DataFrame
        result = df.clone()
        result = result.with_columns(
            [
                pl.Series("regime", states),
                pl.Series("regime_label", labels),
            ]
        )

        # Add probability columns
        for i in range(self.n_regimes):
            result = result.with_columns([pl.Series(f"prob_{i}", probs[:, i])])

        return result

    def predict_proba(
        self,
        df: pl.DataFrame,
        returns_col: str = "returns",
        vol_window: int = 20,
        dd_window: int = 20,
    ) -> np.ndarray:
        """
        Predict regime probabilities (online inference).

        Returns:
            (T, n_regimes) array of probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._prepare_features(df, returns_col, vol_window, dd_window)
        X_scaled = self.scaler.transform(X)

        return self.model.predict_proba(X_scaled)

    def get_regime_stats(self) -> pd.DataFrame:
        """
        Get regime statistics (mean return, vol, frequency).

        Returns:
            DataFrame with regime characteristics
        """
        if self.regime_stats is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.regime_stats.copy()

    def get_transition_matrix(self) -> np.ndarray:
        """
        Get regime transition probability matrix.

        Returns:
            (n_regimes, n_regimes) array where T[i, j] = P(regime_j | regime_i)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.transmat_


def detect_regimes(
    df: pl.DataFrame,
    returns_col: str = "returns",
    n_regimes: int = 3,
    vol_window: int = 20,
    dd_window: int = 20,
    **hmm_kwargs,
) -> pl.DataFrame:
    """
    Convenience function for one-shot regime detection.

    Args:
        df: Polars DataFrame with returns
        returns_col: Column name for returns
        n_regimes: Number of hidden states
        vol_window: Volatility window
        dd_window: Drawdown window
        **hmm_kwargs: Additional arguments for RegimeDetector

    Returns:
        DataFrame with regime predictions

    Example:
        >>> df_regimes = detect_regimes(returns_df, n_regimes=3)
        >>> print(df_regimes.select(["date", "returns", "regime_label"]))
    """
    detector = RegimeDetector(n_regimes=n_regimes, **hmm_kwargs)
    detector.fit(df, returns_col, vol_window, dd_window)

    return detector.predict(df, returns_col, vol_window, dd_window)


def compute_regime_performance(
    df: pl.DataFrame,
    regime_col: str = "regime_label",
    returns_col: str = "returns",
) -> pd.DataFrame:
    """
    Compute performance metrics by regime.

    Args:
        df: DataFrame with regime labels and returns
        regime_col: Column name for regime labels
        returns_col: Column name for returns

    Returns:
        DataFrame with metrics per regime:
        - mean_return: Average daily return
        - volatility: Std dev of returns
        - sharpe: Sharpe ratio (annualized)
        - frequency: % of days in regime

    Example:
        >>> perf = compute_regime_performance(df_regimes)
        >>> print(perf)
    """
    df_pd = df.to_pandas()

    metrics = []
    for regime in df_pd[regime_col].unique():
        mask = df_pd[regime_col] == regime
        ret = df_pd.loc[mask, returns_col]

        mean_ret = ret.mean()
        vol = ret.std()
        sharpe = (mean_ret / vol) * np.sqrt(252) if vol > 0 else 0.0

        metrics.append(
            {
                "regime": regime,
                "mean_return": mean_ret,
                "volatility": vol,
                "sharpe": sharpe,
                "frequency": mask.mean(),
                "n_days": mask.sum(),
            }
        )

    return pd.DataFrame(metrics).sort_values("mean_return", ascending=False)
