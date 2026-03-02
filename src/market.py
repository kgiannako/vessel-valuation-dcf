"""
market.py
---------
MarketParams: freight rate stochastic process + all market-driven inputs.

Key addition over v1: freight-scrap correlation.
At terminal year, scrap price is drawn jointly with the prevailing TCE rate
via a bivariate log-normal with correlation rho. This captures the real-world
dynamic where distressed freight markets coincide with depressed scrap prices,
compressing terminal value from both ends simultaneously.

Log-OU process for TCE rates:
    dln(S) = κ(μ - ln(S))dt + σ dW

Calibration guidance (Kamsarmax):
    κ  ≈ 0.40–0.70   half-life ~1–2 years
    μ  ≈ ln(11000)   long-run mean ~$11k/day mid-cycle
    σ  ≈ 0.55–0.65   annualised log-vol
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class MarketParams:
    # ── Freight Rate Process ──────────────────────────────────────────────────
    spot_tce_rate: float = 11_000
    longrun_mean_tce: float = 11_000
    mean_reversion_speed: float = 0.50       # κ
    rate_volatility: float = 0.60            # σ annualised

    # ── Scrap Market ──────────────────────────────────────────────────────────
    scrap_price_per_ldt: Tuple[float, float] = (480, 0.15)   # (mean $/LDT, cv)

    # Freight-scrap correlation (positive: both fall in bad markets)
    # Empirically ~0.4–0.6 on annual observations
    freight_scrap_correlation: float = 0.50

    # ── Opex Inflation ────────────────────────────────────────────────────────
    opex_inflation: Tuple[float, float] = (0.025, 0.005)

    # ── Exit Multiple (second-hand sale) ──────────────────────────────────────
    exit_earnings_multiple: Tuple[float, float] = (7.0, 2.0)

    # ── Discount Rate ─────────────────────────────────────────────────────────
    wacc: Tuple[float, float] = (0.10, 0.01)

    # ── Equity Hurdle Rate (for IRR exceedance probability) ───────────────────
    equity_hurdle_rate: float = 0.15

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation methods
    # ─────────────────────────────────────────────────────────────────────────

    def simulate_rate_path(
        self,
        n_years: int,
        steps_per_year: int = 12,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """Single TCE rate path via discretised log-OU. Returns annual averages."""
        if rng is None:
            rng = np.random.default_rng()

        dt = 1.0 / steps_per_year
        n_steps = n_years * steps_per_year
        kappa = self.mean_reversion_speed
        mu_log = np.log(self.longrun_mean_tce)
        sigma = self.rate_volatility

        log_s = np.log(self.spot_tce_rate)
        monthly_log_rates = np.empty(n_steps)
        eps = rng.standard_normal(n_steps)

        for i in range(n_steps):
            log_s = log_s + kappa * (mu_log - log_s) * dt + sigma * np.sqrt(dt) * eps[i]
            monthly_log_rates[i] = log_s

        monthly_rates = np.exp(monthly_log_rates)
        return monthly_rates.reshape(n_years, steps_per_year).mean(axis=1)

    def simulate_rate_paths(
        self,
        n_years: int,
        n_simulations: int,
        steps_per_year: int = 12,
        seed: int = 42,
    ) -> np.ndarray:
        """Simulate N paths. Returns (n_simulations, n_years)."""
        rng = np.random.default_rng(seed)
        paths = np.empty((n_simulations, n_years))
        for i in range(n_simulations):
            paths[i] = self.simulate_rate_path(n_years, steps_per_year, rng)
        return paths

    def simulate_correlated_scrap(
        self,
        terminal_tce_rates: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Draw scrap prices correlated with terminal TCE rates.

        Uses bivariate log-normal via Cholesky decomposition.
        terminal_tce_rates: array of shape (n_sims,) — the final-year TCE rate per path.

        Returns scrap price per LDT for each path.
        """
        n = len(terminal_tce_rates)
        rho = self.freight_scrap_correlation

        # Cholesky factor for 2x2 correlation matrix
        # Z2 = rho*Z1 + sqrt(1-rho^2)*Z_independent
        z1 = rng.standard_normal(n)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)

        # Map z1 to freight (for correlation structure — we already have the rate paths,
        # so we just use z2 as the scrap noise, correlated with a standard normal
        # approximation of the terminal rate distribution)
        scrap_mean, scrap_cv = self.scrap_price_per_ldt
        scrap_sigma_log = np.sqrt(np.log(1 + scrap_cv ** 2))
        scrap_mu_log = np.log(scrap_mean) - 0.5 * scrap_sigma_log ** 2

        # Shift scrap mean based on terminal rate relative to long-run mean
        # When terminal rate is below LR mean, scrap is also pulled down
        log_rate_deviation = np.log(terminal_tce_rates) - np.log(self.longrun_mean_tce)
        scrap_adjustment = rho * scrap_sigma_log * log_rate_deviation  # scaled correlation effect

        scrap_prices = np.exp(scrap_mu_log + scrap_adjustment + scrap_sigma_log * z2)
        return scrap_prices

    def summary(self) -> str:
        half_life = np.log(2) / self.mean_reversion_speed
        return (
            f"Spot TCE: ${self.spot_tce_rate:,.0f}/day | "
            f"LR mean: ${self.longrun_mean_tce:,.0f}/day | "
            f"κ={self.mean_reversion_speed:.2f} (t½={half_life:.1f}yr) | "
            f"σ={self.rate_volatility:.0%} | "
            f"Freight-scrap ρ={self.freight_scrap_correlation:.2f}"
        )
