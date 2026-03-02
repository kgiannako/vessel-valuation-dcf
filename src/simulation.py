"""
simulation.py
-------------
Monte Carlo orchestrator — now with:
  1. IRR distribution per path (scipy brentq solver)
  2. Correlated freight-scrap terminal value draws (via market.py Cholesky method)
  3. Full results matrix for downstream analytics
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import brentq

from src.vessel import VesselSpec
from src.market import MarketParams
from src.debt import DebtSchedule
from src.cashflows import compute_annual_cashflows, compute_terminal_value


# ── IRR solver ───────────────────────────────────────────────────────────────

def compute_irr(
    equity_outflow: float,
    annual_fcfs: np.ndarray,
    terminal_value_to_equity: float,
) -> float:
    """
    Solve for IRR: the rate r that makes equity NPV = 0.

        0 = -equity_outflow + Σ FCF_t/(1+r)^t + TV/(1+r)^T

    Returns IRR as a decimal, or np.nan if no solution found.
    """
    if equity_outflow <= 0:
        return np.nan

    n = len(annual_fcfs)
    years = np.arange(1, n + 1)

    def npv_at_rate(r):
        dfs = 1 / (1 + r) ** years
        return -equity_outflow + np.sum(annual_fcfs * dfs) + terminal_value_to_equity * dfs[-1]

    # Check that a sign change exists (necessary for brentq)
    try:
        low, high = -0.99, 10.0   # bracket: -99% to +1000% IRR
        if npv_at_rate(low) * npv_at_rate(high) > 0:
            return np.nan
        irr = brentq(npv_at_rate, low, high, xtol=1e-6, maxiter=200)
        return irr
    except (ValueError, RuntimeError):
        return np.nan


# ── Results container ─────────────────────────────────────────────────────────

@dataclass
class SimulationResults:
    n_simulations: int
    holding_years: int
    purchase_price: float
    equity_invested: float
    vessel_name: str

    # Per-path arrays (length n_simulations)
    equity_npvs: np.ndarray
    asset_npvs: np.ndarray
    irrs: np.ndarray
    terminal_values: np.ndarray
    wacc_samples: np.ndarray

    # Per-path × per-year matrices (n_sims × holding_years)
    annual_fcf_matrix: np.ndarray
    annual_ebitda_matrix: np.ndarray
    annual_rate_matrix: np.ndarray
    annual_revenue_matrix: np.ndarray
    annual_opex_matrix: np.ndarray

    # ── Convenience stats ─────────────────────────────────────────────────────

    def stats(self, arr_name: str = "equity_npvs") -> dict:
        arr = getattr(self, arr_name)
        valid = arr[~np.isnan(arr)]
        return {
            "mean": np.mean(valid),
            "median": np.median(valid),
            "std": np.std(valid),
            "p5":  np.percentile(valid, 5),
            "p10": np.percentile(valid, 10),
            "p25": np.percentile(valid, 25),
            "p75": np.percentile(valid, 75),
            "p90": np.percentile(valid, 90),
            "p95": np.percentile(valid, 95),
        }

    def prob_exceeds(self, arr_name: str, threshold: float) -> float:
        arr = getattr(self, arr_name)
        valid = arr[~np.isnan(arr)]
        return float(np.mean(valid > threshold))

    def moic_distribution(self) -> np.ndarray:
        """Return on equity as multiple of invested capital."""
        if self.equity_invested <= 0:
            return np.full(self.n_simulations, np.nan)
        return (self.equity_npvs + self.equity_invested) / self.equity_invested


# ── Main simulation function ──────────────────────────────────────────────────

def run_simulation(
    vessel: VesselSpec,
    market: MarketParams,
    debt: DebtSchedule,
    n_simulations: int = 5_000,
    seed: int = 42,
    exit_type: str = "second_hand",
    steps_per_year: int = 12,
) -> SimulationResults:
    """
    Run Monte Carlo simulation.

    Each path:
      1. Simulate TCE rate path (log-OU, monthly steps → annual averages)
      2. Sample uncertain scalar params from their distributions
      3. Draw correlated scrap price using terminal-year TCE rate
      4. Compute annual cash flow waterfall
      5. Compute terminal value
      6. Discount to NPV (asset and equity)
      7. Solve for IRR via brentq
    """
    rng = np.random.default_rng(seed)
    n_years = vessel.effective_holding_years
    equity_invested = vessel.purchase_price_usd - debt.loan_amount

    # ── Pre-simulate all rate paths (vectorised) ──────────────────────────────
    rate_paths = market.simulate_rate_paths(
        n_years, n_simulations, steps_per_year=steps_per_year, seed=seed
    )
    terminal_rates = rate_paths[:, -1]   # final year rate per path

    # ── Sample all scalar uncertain params up-front ───────────────────────────
    def sample_lognormal(mean, cv, size):
        sl = np.sqrt(np.log(1 + cv ** 2))
        ml = np.log(mean) - 0.5 * sl ** 2
        return rng.lognormal(ml, sl, size)

    def sample_normal(mean, std, size, floor=None):
        s = rng.normal(mean, std, size)
        return s if floor is None else np.maximum(s, floor)

    opex_samples        = sample_lognormal(*vessel.daily_opex, n_simulations)
    opex_inf_samples    = sample_normal(*market.opex_inflation, n_simulations, floor=0)
    wacc_samples        = sample_normal(*market.wacc, n_simulations, floor=0.03)
    exit_mult_samples   = sample_normal(*market.exit_earnings_multiple, n_simulations, floor=1.0)

    # Correlated scrap prices — drawn jointly with terminal TCE rates
    scrap_samples = market.simulate_correlated_scrap(terminal_rates, rng)

    # Drydock samples (one draw per event per simulation)
    dd_years = vessel.drydock_years
    dd_cost_samps   = {y: sample_lognormal(*vessel.drydock_cost, n_simulations) for y in dd_years}
    dd_offhire_samps = {y: sample_lognormal(*vessel.drydock_offhire_days, n_simulations) for y in dd_years}

    # ── Storage ───────────────────────────────────────────────────────────────
    equity_npvs    = np.empty(n_simulations)
    asset_npvs     = np.empty(n_simulations)
    irrs           = np.empty(n_simulations)
    terminal_vals  = np.empty(n_simulations)
    fcf_matrix     = np.empty((n_simulations, n_years))
    ebitda_matrix  = np.empty((n_simulations, n_years))
    revenue_matrix = np.empty((n_simulations, n_years))
    opex_matrix    = np.empty((n_simulations, n_years))

    years = np.arange(1, n_years + 1)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for i in range(n_simulations):
        dd_costs   = {y: dd_cost_samps[y][i]    for y in dd_years}
        dd_offhire = {y: dd_offhire_samps[y][i] for y in dd_years}

        cf = compute_annual_cashflows(
            vessel, debt,
            tce_rates       = rate_paths[i],
            daily_opex_base = opex_samples[i],
            opex_inflation  = opex_inf_samples[i],
            drydock_costs   = dd_costs,
            drydock_offhire = dd_offhire,
        )

        tv = compute_terminal_value(
            vessel, debt, cf,
            scrap_price_per_ldt    = scrap_samples[i],
            exit_earnings_multiple = exit_mult_samples[i],
            exit_type              = exit_type,
        )

        # Store matrices
        fcf_matrix[i]     = cf["fcf_levered"].values
        ebitda_matrix[i]  = cf["ebitda"].values
        revenue_matrix[i] = cf["total_revenue"].values
        opex_matrix[i]    = cf["total_opex"].values
        terminal_vals[i]  = tv["gross_terminal_value"]

        # Discount factors
        wacc = wacc_samples[i]
        dfs = 1 / (1 + wacc) ** years

        # Asset NPV (unlevered)
        asset_npvs[i] = (
            np.sum(cf["fcf_unlevered"].values * dfs)
            + tv["gross_terminal_value"] * dfs[-1]
        )

        # Equity NPV (levered)
        net_tv = tv["net_terminal_value_to_equity"]
        equity_npvs[i] = np.sum(cf["fcf_levered"].values * dfs) + net_tv * dfs[-1]

        # IRR
        irrs[i] = compute_irr(
            equity_outflow              = equity_invested,
            annual_fcfs                 = cf["fcf_levered"].values,
            terminal_value_to_equity    = net_tv,
        )

    return SimulationResults(
        n_simulations       = n_simulations,
        holding_years       = n_years,
        purchase_price      = vessel.purchase_price_usd,
        equity_invested     = equity_invested,
        vessel_name         = vessel.name,
        equity_npvs         = equity_npvs,
        asset_npvs          = asset_npvs,
        irrs                = irrs,
        terminal_values     = terminal_vals,
        wacc_samples        = wacc_samples,
        annual_fcf_matrix   = fcf_matrix,
        annual_ebitda_matrix= ebitda_matrix,
        annual_rate_matrix  = rate_paths,
        annual_revenue_matrix = revenue_matrix,
        annual_opex_matrix  = opex_matrix,
    )
