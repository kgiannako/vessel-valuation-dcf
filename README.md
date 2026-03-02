# Ship Valuation Tool 🚢

Probabilistic DCF valuation engine for dry bulk cargo vessels, built as a data product and capability demonstrator.

**Core idea:** every uncertain input is a distribution, not a point estimate. The output is not a single valuation but a full distribution of outcomes across 5,000 simulated freight rate environments — giving a structured, quantified view of how vessel value depends on the cycle.

---

## Method

Freight rates are modelled as a **log-normal Ornstein-Uhlenbeck process** — mean-reverting (rates don't wander indefinitely) and strictly positive (vessels always have scrap option value). On each simulated rate path, a full annual P&L is built: TCE revenue, operating costs, drydocking, and (optionally) debt service. Terminal value uses either a second-hand sale multiple or scrap demolition, with freight–scrap correlation modelled via Cholesky decomposition.

**Rate process parameters are calibrated from data** — not assumed. κ and σ are fitted to the Baltic Panamax Index (BPI) 2012–2025 via AR(1) MLE on weekly log-rates. The long-run mean μ is flagged as a judgement input, anchored to the calibration range.

---

## Features

- Log-OU TCE rate simulation — κ, σ calibrated from BPI data; μ user-adjustable
- Unlevered asset valuation as default — financing optional via toggle
- Full cash flow waterfall: spot + TC revenue, opex, drydocking, debt service
- IRR distribution solved per path (Brentq root-finding)
- Freight–scrap terminal value correlation (Cholesky)
- One-at-a-time sensitivity tornado
- MOIC, NPV (asset + equity), IRR, P(positive NPV) dashboards

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/ship-valuation-tool
cd ship-valuation-tool
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. **New app** → select repo → `app.py` as entry point → **Deploy**

You'll have a shareable `yourapp.streamlit.app` link within ~2 minutes. No other configuration needed — `requirements.txt` handles all dependencies.

---

## Repository Structure

```
ship-valuation-tool/
├── app.py                      # Streamlit UI
├── requirements.txt
├── README.md
├── data/
│   └── baltic_panamax_historical_data.csv   # BPI daily data, 2012–2025
├── notebooks/
│   └── bpi_calibration.ipynb   # OU parameter estimation from BPI data
└── src/
    ├── vessel.py               # VesselSpec dataclass
    ├── market.py               # MarketParams + log-OU simulation + freight-scrap correlation
    ├── debt.py                 # DebtSchedule — amortisation and interest
    ├── cashflows.py            # Cash flow engine
    ├── simulation.py           # Monte Carlo orchestrator + IRR solver
    ├── sensitivity.py          # One-at-a-time sensitivity analysis
    ├── calibration.py          # AR(1) MLE calibration utilities
    └── charts.py               # All matplotlib figures
```

---

## Calibrated Parameters (Panamax, BPI 2012–2025)

| Parameter | Value | Source | Note |
|-----------|-------|--------|------|
| κ (mean reversion speed) | 1.01 | AR(1) MLE on weekly log(BPI) | Half-life ~0.68 years. Stable across rolling windows — safe to calibrate. |
| σ (annualised log-vol) | 0.71 | AR(1) MLE on weekly log(BPI) | Stable across rolling windows — safe to calibrate. |
| μ (long-run mean TCE) | $9,500/day | Anchored to calibration range | Full-history $8.1k, post-2015 $10.2k. Treat as judgement — adjust for cycle view. |
| Spot TCE | $9,756/day | Latest BPI × 6.5 | Update from Baltic Exchange or broker reports. |

See `calibration/bpi_calibration.ipynb` for the full derivation, diagnostic plots, and model validation.

---

## Parameter Guide

| Parameter | Description | Typical range |
|-----------|-------------|---------------|
| κ | Mean reversion speed | 0.8–1.2 (Panamax) |
| μ | Long-run equilibrium TCE | $8k–$11k/day (Panamax) |
| σ | Annualised log-vol of TCE | 65–75% (Panamax) |
| ρ | Freight–scrap correlation | 0.4–0.6 |
| Daily opex | Vessel running costs | $7k–$9k/day (Panamax) |
| WACC | Unlevered cost of capital | 9–12% (spot), 7–9% (TC'd) |

---

## Roadmap

- [x] Log-OU rate process with mean reversion
- [x] Full cash flow waterfall
- [x] IRR distribution (Brentq per path)
- [x] Freight–scrap terminal value correlation
- [x] Sensitivity tornado
- [x] Unlevered / levered toggle (asset valuation vs deal analysis)
- [x] OU parameter calibration from Baltic Panamax Index data
- [ ] Route-level TCE data to replace BPI index approximation
- [ ] Two-factor rate model (OU + slow orderbook cycle component)
- [ ] Multi-vessel portfolio with correlated rate paths
- [ ] CSV export of simulation results
