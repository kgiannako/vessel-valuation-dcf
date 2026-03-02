"""
calibration.py
--------------
Fits log-OU parameters (κ, μ, σ) to historical rate data via AR(1) MLE.

Supports two data modes:
  - BDI_INDEX : raw Baltic index units (BPI, BDI etc.)
                κ and σ are unit-free and directly usable in the OU process.
                μ is in log(index) units; a separate tce_multiplier converts
                index → $/day for display purposes only.
  - TCE_USD   : native $/day TCE rates (when real route data is available).
                All parameters directly in $/day.

The model is identical in both modes — only axis labels and the conversion
step differ. This makes it easy to upgrade to TCE_USD mode when real data
arrives, without touching any of the simulation or valuation logic.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal


DataMode = Literal["BDI_INDEX", "TCE_USD"]


@dataclass
class CalibrationResult:
    # ── OU parameters ─────────────────────────────────────────────────────────
    kappa: float          # mean reversion speed (annualised)
    mu_log: float         # long-run mean of log(rate)
    sigma: float          # annualised volatility of log(rate)
    half_life_years: float

    # ── AR(1) intermediates ───────────────────────────────────────────────────
    ar1_alpha: float
    ar1_beta: float
    ar1_sigma_eps: float
    ar1_r_squared: float
    n_obs: int
    dt: float             # time step in years

    # ── Data mode ─────────────────────────────────────────────────────────────
    data_mode: DataMode
    tce_multiplier: float  # only used in BDI_INDEX mode for display

    # ── Date range ────────────────────────────────────────────────────────────
    date_from: pd.Timestamp
    date_to: pd.Timestamp

    # ── Convenience properties ────────────────────────────────────────────────
    @property
    def mu_level(self) -> float:
        """Long-run mean in native units (index or $/day)."""
        return np.exp(self.mu_log)

    @property
    def mu_tce(self) -> float:
        """Long-run mean in $/day (applies multiplier in BDI_INDEX mode)."""
        if self.data_mode == "TCE_USD":
            return self.mu_level
        return self.mu_level * self.tce_multiplier

    @property
    def current_level_tce(self) -> float:
        """Placeholder — set externally from latest data point."""
        return self.mu_tce  # fallback to LR mean

    def summary(self) -> str:
        mode_str = f"BDI Index (×{self.tce_multiplier:.1f} → $/day)" \
                   if self.data_mode == "BDI_INDEX" else "TCE $/day"
        return (
            f"Data mode  : {mode_str}\n"
            f"Period     : {self.date_from.date()} → {self.date_to.date()} "
            f"({self.n_obs} obs, Δt={self.dt*52:.0f}wk)\n"
            f"κ          : {self.kappa:.3f}  (half-life {self.half_life_years:.2f} yrs)\n"
            f"μ          : {self.mu_level:,.0f} native units  →  ${self.mu_tce:,.0f}/day\n"
            f"σ          : {self.sigma:.3f}  ({self.sigma:.0%} annualised)\n"
            f"AR(1) R²   : {self.ar1_r_squared:.4f}"
        )


def load_bpi_csv(path: str) -> pd.Series:
    """
    Load and clean an Investing.com BPI CSV.
    Returns a daily pd.Series indexed by date, values = BPI index level.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"').strip() for c in df.columns]
    df["Date"]  = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["Price"] = df["Price"].astype(str).str.replace(",", "").astype(float)
    df = df[["Date", "Price"]].dropna().sort_values("Date")
    return df.set_index("Date")["Price"]


def calibrate(
    series: pd.Series,
    data_mode: DataMode = "BDI_INDEX",
    tce_multiplier: float = 6.5,
    resample_freq: str = "W",
    date_from: str = None,
    date_to: str = None,
) -> CalibrationResult:
    """
    Fit log-OU parameters to a price series via closed-form AR(1) MLE.

    Parameters
    ----------
    series        : pd.Series with DatetimeIndex, values = index or TCE rate
    data_mode     : 'BDI_INDEX' or 'TCE_USD'
    tce_multiplier: conversion factor index → $/day (BDI_INDEX mode only)
    resample_freq : 'W' (weekly) or 'M' (monthly) — reduces microstructure noise
    date_from/to  : optional window slicing (strings like '2015-01-01')
    """
    s = series.copy()

    # Window slice
    if date_from:
        s = s[s.index >= date_from]
    if date_to:
        s = s[s.index <= date_to]

    # Resample to reduce gaps and noise
    s = s.resample(resample_freq).mean().dropna()

    if len(s) < 52:
        raise ValueError(f"Too few observations ({len(s)}) after resampling. "
                         "Use a longer history or lower resample frequency.")

    log_s = np.log(s.values)
    n = len(log_s)

    # Time step in years
    freq_to_dt = {"W": 1/52, "M": 1/12, "D": 1/252}
    dt = freq_to_dt.get(resample_freq, 1/52)

    # ── Closed-form AR(1) MLE ─────────────────────────────────────────────────
    x  = log_s[:-1]
    xn = log_s[1:]
    m  = len(x)

    sx  = np.sum(x)
    sxn = np.sum(xn)
    sxx = np.sum(x * x)
    sxxn = np.sum(x * xn)

    beta  = (m * sxxn - sx * sxn) / (m * sxx - sx ** 2)
    alpha = (sxn - beta * sx) / m
    eps   = xn - (alpha + beta * x)
    sigma_eps = np.std(eps, ddof=2)
    r_sq = float(np.corrcoef(x, xn)[0, 1] ** 2)

    # Guard against beta ≤ 0 (explosive or random walk)
    beta = max(beta, 1e-6)

    # ── Convert AR(1) → OU ────────────────────────────────────────────────────
    kappa    = -np.log(beta) / dt
    mu_log   = alpha / (1 - beta)
    # OU volatility from discrete residuals
    denom = max(1 - np.exp(-2 * kappa * dt), 1e-10)
    sigma = sigma_eps * np.sqrt(2 * kappa / denom)
    half_life = np.log(2) / kappa

    return CalibrationResult(
        kappa          = kappa,
        mu_log         = mu_log,
        sigma          = sigma,
        half_life_years= half_life,
        ar1_alpha      = alpha,
        ar1_beta       = beta,
        ar1_sigma_eps  = sigma_eps,
        ar1_r_squared  = r_sq,
        n_obs          = m,
        dt             = dt,
        data_mode      = data_mode,
        tce_multiplier = tce_multiplier,
        date_from      = s.index[0],
        date_to        = s.index[-1],
    )


def rolling_calibration(
    series: pd.Series,
    data_mode: DataMode = "BDI_INDEX",
    tce_multiplier: float = 6.5,
    window_years: int = 5,
    step_months: int = 6,
    resample_freq: str = "W",
) -> pd.DataFrame:
    """
    Run calibration on rolling windows.
    Returns DataFrame with columns: date, kappa, mu_tce, sigma, half_life, r_squared.
    Useful for visualising parameter instability over time.
    """
    freq_to_dt = {"W": 1/52, "M": 1/12}
    dt = freq_to_dt.get(resample_freq, 1/52)
    s = series.resample(resample_freq).mean().dropna()

    window_obs = int(window_years / dt)
    step_obs   = int((step_months / 12) / dt)

    rows = []
    for i in range(0, len(s) - window_obs, step_obs):
        window = s.iloc[i: i + window_obs]
        try:
            cal = calibrate(
                window,
                data_mode      = data_mode,
                tce_multiplier = tce_multiplier,
                resample_freq  = resample_freq,
            )
            mid_date = window.index[len(window) // 2]
            rows.append({
                "date":      mid_date,
                "kappa":     cal.kappa,
                "mu_tce":    cal.mu_tce,
                "sigma":     cal.sigma,
                "half_life": cal.half_life_years,
                "r_squared": cal.ar1_r_squared,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def suggest_app_params(
    cal_full: CalibrationResult,
    rolling_df: pd.DataFrame,
    current_bpi: float,
    tce_multiplier: float,
    recent_years: int = 5,
) -> dict:
    """
    Produce a recommended parameter dict for the app.

    Strategy:
      - κ and σ: full-history estimates (describe structural behaviour)
      - μ: forward-looking — user's judgment, informed by recent rolling window
      - spot: derived from latest BPI × multiplier
    """
    recent = rolling_df[
        rolling_df["date"] > rolling_df["date"].max() - pd.DateOffset(years=recent_years)
    ]
    mu_recent = recent["mu_tce"].mean() if len(recent) else cal_full.mu_tce

    return {
        "kappa":           round(cal_full.kappa, 2),
        "sigma":           round(cal_full.sigma, 2),
        "mu_tce_full":     round(cal_full.mu_tce, 0),
        "mu_tce_recent":   round(mu_recent, 0),
        "spot_tce":        round(current_bpi * tce_multiplier, 0),
        "data_mode":       cal_full.data_mode,
        "tce_multiplier":  tce_multiplier,
        "calibration_date": cal_full.date_to.date(),
    }
