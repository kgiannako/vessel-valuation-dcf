"""
vessel.py
---------
VesselSpec: all physical and commercial characteristics of the ship.

Uncertain inputs are expressed as (mean, cv) tuples where:
    cv = coefficient of variation = std / mean

This is more intuitive for strictly positive quantities than raw std,
and maps cleanly to log-normal sampling in the simulation layer.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VesselSpec:
    # ── Identity ─────────────────────────────────────────────────────────────
    name: str = "Unnamed Vessel"
    vessel_type: str = "Kamsarmax Bulk Carrier"
    dwt: float = 82_000
    ldt: float = 14_000                      # light displacement tonnes (scrap calc)

    # ── Life parameters ───────────────────────────────────────────────────────
    vessel_age_years: float = 0
    useful_life_years: int = 25
    holding_period_years: int = 10

    # ── Commercial structure ──────────────────────────────────────────────────
    tc_coverage: float = 0.0                 # fraction of spot days on TC after TC expires
    tc_rate_usd_day: float = 12_000
    tc_remaining_years: float = 0

    # ── Operating parameters ──────────────────────────────────────────────────
    operating_days: float = 350
    daily_opex: Tuple[float, float] = (8_000, 0.08)   # (mean $/day, cv)

    # ── Drydock ───────────────────────────────────────────────────────────────
    drydock_interval_years: int = 5
    drydock_cost: Tuple[float, float] = (1_800_000, 0.20)
    drydock_offhire_days: Tuple[float, float] = (25, 0.20)

    # ── Financials ────────────────────────────────────────────────────────────
    tax_rate: float = 0.0
    purchase_price_usd: float = 30_000_000

    # ── Derived ───────────────────────────────────────────────────────────────
    @property
    def remaining_life_years(self) -> int:
        return max(0, self.useful_life_years - int(self.vessel_age_years))

    @property
    def effective_holding_years(self) -> int:
        return min(self.holding_period_years, self.remaining_life_years)

    @property
    def drydock_years(self) -> list:
        age = int(self.vessel_age_years)
        years_since_last = age % self.drydock_interval_years
        next_dd = self.drydock_interval_years - years_since_last
        dds = []
        while next_dd <= self.effective_holding_years:
            dds.append(next_dd)
            next_dd += self.drydock_interval_years
        return dds

    def summary(self) -> str:
        return (
            f"{self.name} | {self.vessel_type} | {self.dwt:,.0f} DWT\n"
            f"Age: {self.vessel_age_years:.0f} yrs | Holding: {self.effective_holding_years} yrs | "
            f"Drydocks at years: {self.drydock_years}"
        )
