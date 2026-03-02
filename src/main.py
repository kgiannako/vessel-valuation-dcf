"""
main.py
-------
Example valuation: Kamsarmax bulk carrier, 5-year old vessel,
partially time-chartered, moderate leverage.

Run with: python main.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from vessel import VesselSpec
from market import MarketParams
from debt import DebtSchedule
from simulation import run_simulation
from valuation import plot_valuation_dashboard, compute_valuation_metrics

# ── Vessel Specification ──────────────────────────────────────────────────────
vessel = VesselSpec(
    name="MV Pacific Horizon",
    vessel_type="Kamsarmax Bulk Carrier",
    dwt=82_000,
    ldt=14_000,
    vessel_age_years=5,
    useful_life_years=25,
    holding_period_years=10,

    # 30% on TC at $13,500/day for remaining 2 years, rest spot
    tc_coverage=0.0,           # after TC expires
    tc_rate_usd_day=13_500,
    tc_remaining_years=2,

    operating_days=350,

    # Opex: $8,200/day, 8% uncertainty (cv)
    daily_opex=(8_200, 0.08),

    # Drydock every 5 years, ~$1.8M, 25 days off-hire
    drydock_interval_years=5,
    drydock_cost=(1_800_000, 0.20),
    drydock_offhire_days=(25, 0.20),

    tax_rate=0.00,             # tonnage tax regime
    purchase_price_usd=30_000_000,
)

# ── Market Parameters ─────────────────────────────────────────────────────────
market = MarketParams(
    spot_tce_rate=13_000,          # current spot market
    longrun_mean_tce=11_000,       # mid-cycle equilibrium
    mean_reversion_speed=0.50,     # ~1.4 year half-life
    rate_volatility=0.60,          # 60% annualised log-vol
    scrap_price_per_ldt=(480, 0.15),
    opex_inflation=(0.025, 0.005),
    exit_earnings_multiple=(7.0, 2.0),
    wacc=(0.10, 0.01),
)

# ── Debt Structure ────────────────────────────────────────────────────────────
# $18M loan, 12-year tenor, SOFR + ~250bps ≈ 7.5%
debt = DebtSchedule(
    loan_amount=18_000_000,
    loan_tenor_years=12,
    interest_rate=0.075,
    amortisation_type="straight",
    years_elapsed=5,    # 5 years already repaid
)

# ── Print Setup Summary ───────────────────────────────────────────────────────
print("=" * 65)
print(vessel.summary())
print()
print(market.summary())
print()
print(debt.summary())
print("=" * 65)
print()

# ── Run Simulation ────────────────────────────────────────────────────────────
print("Running Monte Carlo simulation (5,000 paths)...")
results = run_simulation(
    vessel, market, debt,
    n_simulations=5_000,
    seed=42,
    exit_type="second_hand",
)
print("Done.\n")

# ── Print Key Results ─────────────────────────────────────────────────────────
metrics = compute_valuation_metrics(results, vessel, debt)

print("VALUATION RESULTS")
print("-" * 65)
print(f"  Asset NPV (median):        ${metrics['asset_npv_median']/1e6:>8.2f}M")
print(f"  Asset NPV P10 – P90:       ${metrics['asset_npv_p10']/1e6:.2f}M  –  ${metrics['asset_npv_p90']/1e6:.2f}M")
print()
print(f"  Equity NPV (median):       ${metrics['equity_npv_median']/1e6:>8.2f}M")
print(f"  Equity NPV P10 – P90:      ${metrics['equity_npv_p10']/1e6:.2f}M  –  ${metrics['equity_npv_p90']/1e6:.2f}M")
print()
print(f"  Equity invested:           ${metrics['equity_invested']/1e6:>8.2f}M")
print(f"  MOIC (median):             {metrics['moic_median']:>8.2f}×")
print(f"  P(positive equity NPV):    {metrics['prob_positive_equity_npv']:>8.0%}")
print(f"  P(loss > 50%):             {metrics['prob_loss_gt_50pct']:>8.0%}")
print("-" * 65)

# ── Plot Dashboard ────────────────────────────────────────────────────────────
print("\nGenerating valuation dashboard...")
fig, _ = plot_valuation_dashboard(
    results, vessel, market, debt,
    save_path="/mnt/user-data/outputs/ship_valuation_dashboard.png"
)
print("Complete.")
