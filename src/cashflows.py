"""
cashflows.py
------------
Pure cash flow engine. No randomness — all stochastic inputs are resolved
upstream by the simulation layer before calling these functions.

This separation makes the financial logic independently testable and
allows swapping the simulation engine without touching the model.
"""

import numpy as np
import pandas as pd
from src.vessel import VesselSpec
from src.debt import DebtSchedule


def compute_annual_cashflows(
    vessel: VesselSpec,
    debt: DebtSchedule,
    tce_rates: np.ndarray,
    daily_opex_base: float,
    opex_inflation: float,
    drydock_costs: dict,       # {year: cost}
    drydock_offhire: dict,     # {year: offhire_days}
) -> pd.DataFrame:
    """
    Year-by-year cash flow waterfall for a single simulation path.
    Returns DataFrame with one row per year.
    """
    n_years = vessel.effective_holding_years
    debt_schedule = debt.build_schedule(n_years)
    rows = []

    for y in range(1, n_years + 1):
        idx = y - 1
        tce_rate = tce_rates[idx]

        # Operating days
        offhire = drydock_offhire.get(y, 0)
        op_days = max(0, vessel.operating_days - offhire)

        # Revenue: TC vs spot split
        if vessel.tc_remaining_years >= y:
            tc_days, spot_days = op_days, 0.0
        elif vessel.tc_remaining_years >= (y - 1):
            tc_frac = vessel.tc_remaining_years - (y - 1)
            tc_days = op_days * tc_frac
            spot_days = op_days * (1 - tc_frac)
        else:
            tc_days = op_days * vessel.tc_coverage
            spot_days = op_days * (1 - vessel.tc_coverage)

        total_revenue = tc_days * vessel.tc_rate_usd_day + spot_days * tce_rate

        # Costs
        daily_opex = daily_opex_base * ((1 + opex_inflation) ** idx)
        total_opex = daily_opex * 365         # opex runs regardless of off-hire
        dd_cost = drydock_costs.get(y, 0)

        ebitda = total_revenue - total_opex - dd_cost

        # Debt service
        ds = debt_schedule[debt_schedule["year"] == y].iloc[0]
        interest = ds["interest"]
        principal = ds["principal"]

        # Tax (on EBITDA - interest, floored at 0)
        tax = max(0, ebitda - interest) * vessel.tax_rate

        # FCF
        fcf_levered = ebitda - ds["total_debt_service"] - tax
        fcf_unlevered = ebitda * (1 - vessel.tax_rate)

        rows.append({
            "year": y,
            "tce_rate": tce_rate,
            "operating_days": op_days,
            "offhire_days": offhire,
            "tc_revenue": tc_days * vessel.tc_rate_usd_day,
            "spot_revenue": spot_days * tce_rate,
            "total_revenue": total_revenue,
            "daily_opex": daily_opex,
            "total_opex": total_opex,
            "drydock_cost": dd_cost,
            "ebitda": ebitda,
            "interest": interest,
            "principal": principal,
            "debt_service": ds["total_debt_service"],
            "tax": tax,
            "fcf_levered": fcf_levered,
            "fcf_unlevered": fcf_unlevered,
            "debt_balance_eoy": ds["closing_balance"],
        })

    return pd.DataFrame(rows)


def compute_terminal_value(
    vessel: VesselSpec,
    debt: DebtSchedule,
    cashflows: pd.DataFrame,
    scrap_price_per_ldt: float,
    exit_earnings_multiple: float,
    exit_type: str = "second_hand",
) -> dict:
    """
    Terminal value at end of holding period.

    exit_type: 'second_hand' | 'scrap' | 'end_of_life'
    """
    n = vessel.effective_holding_years
    remaining_debt = debt.outstanding_at_year(n)
    scrap_floor = vessel.ldt * scrap_price_per_ldt

    if exit_type in ("scrap", "end_of_life"):
        gross_tv = scrap_floor
    else:
        last_ebitda = cashflows.iloc[-1]["ebitda"]
        gross_tv = max(last_ebitda * exit_earnings_multiple, scrap_floor)

    return {
        "gross_terminal_value": gross_tv,
        "remaining_debt_at_exit": remaining_debt,
        "net_terminal_value_to_equity": gross_tv - remaining_debt,
        "exit_type": exit_type,
    }
