"""
debt.py
-------
DebtSchedule: loan amortisation, interest, and outstanding balance.

Ship finance conventions:
  - Senior secured against vessel
  - 50–65% LTV at origination
  - Straight-line or mortgage amortisation
  - Floating rate (SOFR + spread), modelled as fixed for simplicity
  - Balloon at loan maturity (often before vessel end-of-life)

Set loan_amount=0 for unlevered / all-equity analysis.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class DebtSchedule:
    loan_amount: float = 18_000_000
    loan_tenor_years: int = 12
    interest_rate: float = 0.075
    amortisation_type: str = "straight"      # 'straight' or 'mortgage'
    balloon_year: int = None
    years_elapsed: float = 0                 # years already repaid at valuation date

    def __post_init__(self):
        if self.balloon_year is None:
            self.balloon_year = self.loan_tenor_years

    def build_schedule(self, n_years: int) -> pd.DataFrame:
        """Annual debt service for n_years from valuation date."""
        rows = []
        balance = self.loan_amount
        annual_principal = self.loan_amount / self.loan_tenor_years

        for y in range(1, n_years + 1):
            if balance <= 0 or self.loan_amount == 0:
                rows.append(dict(year=y, opening_balance=0, interest=0,
                                 principal=0, closing_balance=0, total_debt_service=0))
                continue

            opening = balance
            interest = opening * self.interest_rate
            years_into_loan = self.years_elapsed + y

            if years_into_loan >= self.balloon_year:
                principal = opening
            elif self.amortisation_type == "straight":
                principal = min(annual_principal, opening)
            else:
                n_rem = max(1, self.loan_tenor_years - years_into_loan + 1)
                payment = opening * self.interest_rate / (1 - (1 + self.interest_rate) ** -n_rem)
                principal = payment - interest

            principal = max(0, min(principal, opening))
            closing = opening - principal
            rows.append(dict(year=y, opening_balance=opening, interest=interest,
                             principal=principal, closing_balance=closing,
                             total_debt_service=interest + principal))
            balance = closing

        return pd.DataFrame(rows)

    def outstanding_at_year(self, year: float) -> float:
        if self.loan_amount == 0:
            return 0.0
        if self.amortisation_type == "straight":
            annual_repayment = self.loan_amount / self.loan_tenor_years
            repaid = min(annual_repayment * (self.years_elapsed + year), self.loan_amount)
            return max(0.0, self.loan_amount - repaid)
        sched = self.build_schedule(int(np.ceil(year)))
        return float(sched.iloc[-1]["closing_balance"]) if len(sched) else 0.0

    def summary(self) -> str:
        remaining = self.outstanding_at_year(0)
        return (
            f"Loan: ${self.loan_amount:,.0f} | Rate: {self.interest_rate:.2%} | "
            f"Tenor: {self.loan_tenor_years}yr | Outstanding: ${remaining:,.0f}"
        )
