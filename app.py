"""
app.py
------
Streamlit dashboard for ship DCF valuation.

Run locally:  streamlit run app.py
Deploy:       push to GitHub → connect to Streamlit Community Cloud
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import streamlit as st

from src.vessel import VesselSpec
from src.market import MarketParams
from src.debt import DebtSchedule
from src.simulation import run_simulation
from src.sensitivity import run_sensitivity
from src.charts import (
    plot_npv_distribution,
    plot_irr_distribution,
    plot_rate_fan,
    plot_cashflow_evolution,
    plot_tornado,
    plot_moic_distribution,
    plot_freight_scrap_scatter,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ship Valuation Tool",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1a1d27;
        border-radius: 8px;
        padding: 16px 20px;
        border: 1px solid #2a2d3a;
    }
    .stMetric label { color: #9da5b4 !important; font-size: 0.78rem !important; }
    .stMetric div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🚢 Ship Valuation")
    st.caption("Probabilistic DCF · Monte Carlo · Log-OU Rates")
    st.divider()

    # ── Vessel ────────────────────────────────────────────────────────────────
    st.subheader("Vessel")
    vessel_name   = st.text_input("Name", value="MV Pacific Horizon")
    vessel_type   = st.selectbox("Type", ["Kamsarmax Bulk Carrier", "Capesize Bulk Carrier",
                                           "Panamax Bulk Carrier", "Handysize Bulk Carrier"])
    dwt           = st.number_input("DWT (tonnes)", value=82_000, step=1_000)
    ldt           = st.number_input("LDT (tonnes)", value=14_000, step=500,
                                     help="Light displacement tonnage — used for scrap value")
    vessel_age    = st.slider("Current age (years)", 0, 20, 5)
    holding_yrs   = st.slider("Holding period (years)", 3, 20, 10)
    purchase_price = st.number_input("Purchase price ($)", value=30_000_000, step=500_000,
                                      format="%d")

    st.divider()

    # ── Commercial structure ──────────────────────────────────────────────────
    st.subheader("Commercial Structure")
    tc_remaining  = st.slider("TC remaining (years)", 0.0, 5.0, 2.0, step=0.5)
    tc_rate       = st.number_input("TC rate ($/day)", value=13_500, step=500) if tc_remaining > 0 else 12_000
    operating_days = st.slider("Operating days/year", 330, 365, 350)

    st.divider()

    # ── Operating costs ───────────────────────────────────────────────────────
    st.subheader("Operating Costs")
    opex_mean     = st.number_input("Daily opex ($/day)", value=8_200, step=100)
    opex_cv       = st.slider("Opex uncertainty (cv)", 0.03, 0.25, 0.08,
                               help="Coefficient of variation = std/mean")
    dd_cost_mean  = st.number_input("Drydock cost ($)", value=1_800_000, step=100_000, format="%d")
    dd_cost_cv    = st.slider("Drydock cost uncertainty (cv)", 0.05, 0.40, 0.20)

    st.divider()

    # ── Debt ──────────────────────────────────────────────────────────────────
    st.subheader("Financing")
    loan_amount   = st.number_input("Loan amount ($)", value=18_000_000, step=500_000, format="%d")
    loan_tenor    = st.slider("Loan tenor (years)", 5, 18, 12)
    interest_rate = st.slider("Interest rate (%)", 4.0, 12.0, 7.5, step=0.25) / 100
    years_elapsed = st.slider("Loan years already elapsed", 0, 10, 5)

    st.divider()

    # ── Market ────────────────────────────────────────────────────────────────
    st.subheader("Market Parameters")
    spot_tce      = st.number_input("Current spot TCE ($/day)", value=13_000, step=500)
    lr_tce        = st.number_input("Long-run mean TCE ($/day)", value=11_000, step=500,
                                     help="Mid-cycle equilibrium rate — the OU process reverts here")
    kappa         = st.slider("Mean reversion speed κ", 0.20, 1.20, 0.50, step=0.05,
                               help="Higher = faster reversion. Half-life = ln(2)/κ")
    sigma         = st.slider("Rate volatility σ (annualised)", 0.30, 0.90, 0.60, step=0.05)
    rho           = st.slider("Freight–scrap correlation ρ", 0.0, 0.90, 0.50, step=0.05,
                               help="How much scrap prices co-move with freight at exit")

    st.divider()

    # ── Exit ──────────────────────────────────────────────────────────────────
    st.subheader("Exit Assumptions")
    exit_type     = st.selectbox("Exit type", ["second_hand", "scrap"])
    exit_mult_mean = st.slider("Exit EV/EBITDA multiple (mean)", 3.0, 12.0, 7.0, step=0.5)
    exit_mult_std  = st.slider("Exit multiple uncertainty (std)", 0.5, 4.0, 2.0, step=0.5)
    scrap_mean    = st.number_input("Scrap price ($/LDT)", value=480, step=10)
    scrap_cv      = st.slider("Scrap price uncertainty (cv)", 0.05, 0.30, 0.15)

    st.divider()

    # ── Discount / hurdle ─────────────────────────────────────────────────────
    st.subheader("Returns")
    wacc_mean     = st.slider("WACC — mean (%)", 6.0, 16.0, 10.0, step=0.5) / 100
    wacc_std      = st.slider("WACC — std (%)", 0.25, 2.0, 1.0, step=0.25) / 100
    hurdle_rate   = st.slider("Equity hurdle rate (%)", 8.0, 25.0, 15.0, step=1.0) / 100

    st.divider()

    # ── Simulation ────────────────────────────────────────────────────────────
    st.subheader("Simulation")
    n_sims        = st.select_slider("Simulations", options=[1_000, 2_000, 5_000, 10_000], value=5_000)
    run_sens      = st.checkbox("Run sensitivity analysis", value=True,
                                 help="Adds ~30s. Runs OAT sensitivity and tornado chart.")
    run_btn       = st.button("▶ Run Valuation", type="primary", use_container_width=True)


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Probabilistic Ship Valuation")
st.caption("Kamsarmax bulk carrier · Monte Carlo DCF · Log-OU freight rate process")

if not run_btn:
    st.info("Configure vessel and market parameters in the sidebar, then click **▶ Run Valuation**.")

    # Explainer
    with st.expander("ℹ️  Methodology overview", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
**Rate process**
TCE rates follow a log-normal Ornstein-Uhlenbeck process — capturing
mean reversion while keeping rates strictly positive. Parameters: κ (speed),
μ (long-run mean), σ (volatility).
""")
        with col2:
            st.markdown("""
**Uncertainty**
Every input is a distribution, not a point estimate. Opex, drydock costs,
WACC, scrap prices, and exit multiples are all sampled independently
per simulation path (5,000 by default).
""")
        with col3:
            st.markdown("""
**Freight–scrap correlation**
Terminal scrap prices are drawn jointly with the exit-year freight rate
via Cholesky decomposition. Bad freight markets → lower scrap prices,
compressing terminal value from both ends.
""")
    st.stop()


# ── Build objects from sidebar inputs ─────────────────────────────────────────

vessel = VesselSpec(
    name                 = vessel_name,
    vessel_type          = vessel_type,
    dwt                  = dwt,
    ldt                  = ldt,
    vessel_age_years     = vessel_age,
    useful_life_years    = 25,
    holding_period_years = holding_yrs,
    tc_coverage          = 0.0,
    tc_rate_usd_day      = tc_rate,
    tc_remaining_years   = tc_remaining,
    operating_days       = operating_days,
    daily_opex           = (opex_mean, opex_cv),
    drydock_interval_years = 5,
    drydock_cost         = (dd_cost_mean, dd_cost_cv),
    drydock_offhire_days = (25, 0.20),
    tax_rate             = 0.0,
    purchase_price_usd   = purchase_price,
)

market = MarketParams(
    spot_tce_rate               = spot_tce,
    longrun_mean_tce            = lr_tce,
    mean_reversion_speed        = kappa,
    rate_volatility             = sigma,
    scrap_price_per_ldt         = (scrap_mean, scrap_cv),
    freight_scrap_correlation   = rho,
    opex_inflation              = (0.025, 0.005),
    exit_earnings_multiple      = (exit_mult_mean, exit_mult_std),
    wacc                        = (wacc_mean, wacc_std),
    equity_hurdle_rate          = hurdle_rate,
)

debt = DebtSchedule(
    loan_amount         = loan_amount,
    loan_tenor_years    = loan_tenor,
    interest_rate       = interest_rate,
    amortisation_type   = "straight",
    years_elapsed       = years_elapsed,
)

equity_invested = purchase_price - loan_amount

# ── Run simulation ────────────────────────────────────────────────────────────

with st.spinner(f"Running {n_sims:,} Monte Carlo paths..."):
    results = run_simulation(
        vessel, market, debt,
        n_simulations = n_sims,
        seed          = 42,
        exit_type     = exit_type,
    )

# ── Metrics row ───────────────────────────────────────────────────────────────

eq_stats   = results.stats("equity_npvs")
ast_stats  = results.stats("asset_npvs")
irr_valid  = results.irrs[~np.isnan(results.irrs)]
moic       = results.moic_distribution()
moic_valid = moic[~np.isnan(moic)]

prob_pos_npv   = results.prob_exceeds("equity_npvs", 0)
prob_hurdle    = float(np.mean(irr_valid > hurdle_rate)) if len(irr_valid) > 0 else 0.0
median_irr     = float(np.median(irr_valid)) if len(irr_valid) > 0 else float("nan")
median_moic    = float(np.median(moic_valid)) if len(moic_valid) > 0 else float("nan")

st.subheader("Summary Metrics")
c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Asset NPV (median)",   f"${ast_stats['median']/1e6:.1f}M",
           delta=f"P10–P90: ${ast_stats['p10']/1e6:.1f}M / ${ast_stats['p90']/1e6:.1f}M")
c2.metric("Equity NPV (median)",  f"${eq_stats['median']/1e6:.1f}M",
           delta=f"P10–P90: ${eq_stats['p10']/1e6:.1f}M / ${eq_stats['p90']/1e6:.1f}M")
c3.metric("Equity invested",      f"${equity_invested/1e6:.1f}M")
c4.metric("MOIC (median)",        f"{median_moic:.2f}×",
           delta=f"P(>1×): {np.mean(moic_valid>1):.0%}")
c5.metric("IRR (median)",         f"{median_irr:.1%}",
           delta=f"P(>hurdle {hurdle_rate:.0%}): {prob_hurdle:.0%}")
c6.metric("P(positive equity NPV)", f"{prob_pos_npv:.0%}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 NPV & IRR",
    "📈 Rate Paths",
    "💵 Cash Flows",
    "📉 MOIC",
    "🔗 Freight–Scrap",
    "🌪️ Sensitivity",
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Equity NPV Distribution")
        st.pyplot(plot_npv_distribution(results, "equity"), use_container_width=True)
    with col2:
        st.subheader("IRR Distribution")
        st.pyplot(plot_irr_distribution(results, hurdle_rate), use_container_width=True)

    with st.expander("Asset-level (unlevered) NPV"):
        st.pyplot(plot_npv_distribution(results, "asset"), use_container_width=True)

with tab2:
    st.subheader("Simulated TCE Rate Paths")
    st.pyplot(plot_rate_fan(results, longrun_mean=lr_tce), use_container_width=True)

    half_life = np.log(2) / kappa
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current spot", f"${spot_tce:,.0f}/day")
    c2.metric("Long-run mean", f"${lr_tce:,.0f}/day")
    c3.metric("Half-life", f"{half_life:.1f} years")
    c4.metric("Annualised vol", f"{sigma:.0%}")

with tab3:
    st.subheader("Cash Flow Evolution")
    st.pyplot(
        plot_cashflow_evolution(results, drydock_years=vessel.drydock_years),
        use_container_width=True,
    )
    st.caption("Vertical blue dotted lines mark scheduled drydock years. "
               "Shaded band = P10–P90. Solid line = median.")

with tab4:
    st.subheader("MOIC Distribution")
    st.pyplot(plot_moic_distribution(results), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Median MOIC", f"{median_moic:.2f}×")
    c2.metric("P10 MOIC", f"{np.percentile(moic_valid, 10):.2f}×")
    c3.metric("P90 MOIC", f"{np.percentile(moic_valid, 90):.2f}×")

with tab5:
    st.subheader("Freight–Scrap Correlation")
    terminal_rates = results.annual_rate_matrix[:, -1]
    # Reconstruct scrap samples for scatter (re-draw with same seed for consistency)
    import numpy as _np
    _rng = _np.random.default_rng(42)
    terminal_scrap = market.simulate_correlated_scrap(terminal_rates, _rng)

    st.pyplot(
        plot_freight_scrap_scatter(terminal_rates, terminal_scrap),
        use_container_width=True,
    )
    st.caption(
        f"Configured ρ = {rho:.2f}. Scatter shows terminal-year TCE rate vs scrap price "
        f"across all simulation paths. In bad freight markets, scrap prices fall simultaneously, "
        f"compressing terminal value from both ends."
    )

with tab6:
    if not run_sens:
        st.info("Enable **Run sensitivity analysis** in the sidebar to see this chart.")
    else:
        with st.spinner("Running sensitivity analysis (~30s)..."):
            tornado_df, base_median = run_sensitivity(vessel, market, debt, n_sims=1_500)

        st.subheader("Sensitivity Tornado")
        st.pyplot(plot_tornado(tornado_df, base_median), use_container_width=True)

        st.caption(
            "One-at-a-time sensitivity. Each parameter shocked ±1σ (or ±20–30% where σ not defined). "
            "Orange = downside shock. Blue = upside shock. Sorted by total range of impact."
        )

        with st.expander("Raw sensitivity data"):
            st.dataframe(
                tornado_df[["low", "high", "range", "low_delta", "high_delta"]]
                .style.format("${:,.0f}"),
                use_container_width=True,
            )

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"**{vessel_name}** · {vessel_type} · {dwt:,.0f} DWT · "
    f"{n_sims:,} simulations · "
    f"Log-OU rate process (κ={kappa:.2f}, σ={sigma:.0%}) · "
    f"Freight–scrap ρ={rho:.2f}"
)
