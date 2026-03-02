# Ship Valuation Tool 🚢

Probabilistic DCF valuation engine for dry bulk cargo vessels.

**Method:** Monte Carlo simulation (5,000 paths by default) over a log-normal Ornstein-Uhlenbeck freight rate process. Every uncertain input is a distribution — no point estimates.

**Features:**
- Log-OU TCE rate process with calibrable κ, μ, σ
- Full cash flow waterfall: TCE revenue, opex, drydocking, debt service
- IRR distribution solved per path via Brentq
- Freight–scrap terminal value correlation via Cholesky decomposition
- One-at-a-time sensitivity tornado
- MOIC, NPV (equity + asset), IRR dashboards

---

## Quick Start (local)

```bash
git clone https://github.com/YOUR_USERNAME/ship-valuation-tool
cd ship-valuation-tool
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud (free, shareable link)

1. Push this repo to GitHub (can be public or private with Streamlit connected)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → `app.py` as the entry point
4. Click **Deploy** — you'll get a `yourapp.streamlit.app` URL within ~2 minutes

No other configuration needed. `requirements.txt` handles all dependencies.

---

## Repository Structure

```
ship-valuation-tool/
├── app.py                  # Streamlit UI
├── requirements.txt        # Dependencies for Streamlit Cloud
├── README.md
└── src/
    ├── vessel.py           # VesselSpec dataclass
    ├── market.py           # MarketParams + log-OU simulation + freight-scrap correlation
    ├── debt.py             # DebtSchedule — amortisation and interest
    ├── cashflows.py        # Pure cash flow engine (no randomness)
    ├── simulation.py       # Monte Carlo orchestrator + IRR solver
    ├── sensitivity.py      # One-at-a-time sensitivity analysis
    └── charts.py           # All matplotlib figures
```

---

## Key Parameters

| Parameter | Description | Typical range (Kamsarmax) |
|-----------|-------------|--------------------------|
| κ (kappa) | Mean reversion speed | 0.4–0.7 (half-life 1–2 years) |
| μ (LR mean TCE) | Long-run equilibrium rate | $9k–$13k/day |
| σ (sigma) | Annualised log-vol of TCE | 55–65% |
| ρ (rho) | Freight–scrap correlation | 0.4–0.6 |
| Daily opex | Vessel running costs | $7k–$10k/day |
| WACC | Unlevered cost of capital | 9–12% (spot), 7–9% (TC'd) |

---

## Roadmap

- [ ] Calibration from historical Baltic Exchange data (AR(1) MLE)
- [ ] Multi-vessel portfolio with correlated rate paths
- [ ] IRR sensitivity / waterfall attribution
- [ ] Tanker and LNG vessel type support
- [ ] CSV export of simulation results
