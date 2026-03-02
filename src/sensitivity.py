"""
sensitivity.py
--------------
One-at-a-time sensitivity analysis producing a tornado-ready DataFrame.

Each parameter is shocked by ±1σ (or ±20% if no σ defined).
The model is re-run at reduced n_sims for speed; median equity NPV is
recorded for each shocked variant and compared to the base case.
"""

import copy
import numpy as np
import pandas as pd

from src.vessel import VesselSpec
from src.market import MarketParams
from src.debt import DebtSchedule
from src.simulation import run_simulation


def run_sensitivity(
    vessel: VesselSpec,
    market: MarketParams,
    debt: DebtSchedule,
    n_sims: int = 2_000,
    seed: int = 99,
) -> tuple[pd.DataFrame, float]:
    """
    Returns (tornado_df, base_median_equity_npv).
    tornado_df has columns: low, high, low_delta, high_delta, range
    indexed by parameter name, sorted ascending by range (for tornado display).
    """

    def median_npv(v, m, d):
        res = run_simulation(v, m, d, n_simulations=n_sims, seed=seed)
        return float(np.median(res.equity_npvs))

    base = median_npv(vessel, market, debt)
    rows = []

    # ── WACC ±1σ ─────────────────────────────────────────────────────────────
    for direction, delta in [("low", -market.wacc[1]), ("high", +market.wacc[1])]:
        m = copy.deepcopy(market)
        m.wacc = (max(0.03, market.wacc[0] + delta), market.wacc[1])
        rows.append({"parameter": "WACC", "direction": direction, "value": median_npv(vessel, m, debt)})

    # ── Long-run TCE ±20% ─────────────────────────────────────────────────────
    for direction, pct in [("low", -0.20), ("high", +0.20)]:
        m = copy.deepcopy(market)
        m.longrun_mean_tce = market.longrun_mean_tce * (1 + pct)
        rows.append({"parameter": "LR TCE Rate", "direction": direction, "value": median_npv(vessel, m, debt)})

    # ── Rate volatility ±30% ──────────────────────────────────────────────────
    for direction, pct in [("low", -0.30), ("high", +0.30)]:
        m = copy.deepcopy(market)
        m.rate_volatility = max(0.10, market.rate_volatility * (1 + pct))
        rows.append({"parameter": "Rate Volatility", "direction": direction, "value": median_npv(vessel, m, debt)})

    # ── Daily opex ±1σ ────────────────────────────────────────────────────────
    for direction, pct in [("low", -vessel.daily_opex[1]), ("high", +vessel.daily_opex[1])]:
        v = copy.deepcopy(vessel)
        v.daily_opex = (vessel.daily_opex[0] * (1 + pct), vessel.daily_opex[1])
        rows.append({"parameter": "Daily Opex", "direction": direction, "value": median_npv(v, market, debt)})

    # ── Exit multiple ±1σ ─────────────────────────────────────────────────────
    for direction, delta in [("low", -market.exit_earnings_multiple[1]),
                              ("high", +market.exit_earnings_multiple[1])]:
        m = copy.deepcopy(market)
        m.exit_earnings_multiple = (
            max(1.0, market.exit_earnings_multiple[0] + delta),
            market.exit_earnings_multiple[1],
        )
        rows.append({"parameter": "Exit Multiple", "direction": direction, "value": median_npv(vessel, m, debt)})

    # ── Freight-scrap correlation ±0.3 ────────────────────────────────────────
    for direction, delta in [("low", -0.30), ("high", +0.30)]:
        m = copy.deepcopy(market)
        m.freight_scrap_correlation = float(np.clip(market.freight_scrap_correlation + delta, -0.99, 0.99))
        rows.append({"parameter": "Freight–Scrap ρ", "direction": direction, "value": median_npv(vessel, m, debt)})

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="parameter", columns="direction", values="value")
    pivot["range"]      = (pivot["high"] - pivot["low"]).abs()
    pivot["low_delta"]  = pivot["low"]  - base
    pivot["high_delta"] = pivot["high"] - base
    pivot = pivot.sort_values("range", ascending=True)

    return pivot, base
