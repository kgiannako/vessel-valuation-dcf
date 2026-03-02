"""
app.py
------
Streamlit dashboard for probabilistic Panamax bulk carrier valuation.
Rate process parameters calibrated from Baltic Panamax Index (BPI), 2012–2025.

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

# ── Calibrated OU parameters (Baltic Panamax Index, 2012–2025) ────────────────
# Derived via AR(1) MLE on weekly log(BPI). See calibration/bpi_calibration.ipynb.
CALIB_KAPPA = 1.01   # mean reversion speed — half-life ~0.68 years
CALIB_SIGMA = 0.71   # annualised log-volatility
CALIB_MU    = 9_500  # long-run mean TCE $/day (mid-range of full-history $8.1k and recent $10.2k)
CALIB_SPOT  = 9_756  # current spot estimate (latest BPI × 6.5)
CALIB_DATE  = "2025-03"  # calibration vintage

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ship Valuation Tool",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Page background ── */
    .stApp, .main { background-color: #200F07; }
    .block-container { padding-top: 1.5rem; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #2A1F1A;
        border-right: 1px solid #3A2A22;
    }

    /* ── Metrics ── */
    .stMetric {
        background: #2A1F1A;
        border: 1px solid #3A2A22;
        border-radius: 8px;
        padding: 10px 14px;
    }
    .stMetric label {
        color: #C2D8C4 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stMetric div[data-testid="stMetricValue"] {
        color: #F8F5F2 !important;
        font-size: 1.4rem !important;
        font-weight: 700;
    }
    .stMetric div[data-testid="stMetricDelta"] {
        color: #C5E384 !important;
        font-size: 0.72rem !important;
    }

    /* ── Dividers ── */
    hr { border-color: #3A2A22 !important; }

    /* ── Parameter tags ── */
    .tag-calibratable {
        background: #385144;
        color: #C5E384;
        font-size: 0.68rem;
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .tag-judgement {
        background: #3A2A0A;
        color: #E8A736;
        font-size: 0.68rem;
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    /* ── Info box ── */
    .stAlert {
        background-color: #2A1F1A !important;
        border: 1px solid #385144 !important;
        color: #C2D8C4 !important;
    }

    /* ── Tab bar ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2A1F1A;
        border-radius: 6px;
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #C2D8C4;
        border-radius: 4px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #385144 !important;
        color: #F8F5F2 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🚢 Ship Valuation")
    st.caption("Probabilistic DCF · Monte Carlo · Log-OU Rates")

    # ── RUN BUTTON — pinned to top ────────────────────────────────────────────
    st.divider()
    n_sims   = st.select_slider("Simulations", options=[1_000, 2_000, 5_000, 10_000], value=5_000,
                                 help="More = tighter distributions, slower runtime. 5,000 is a good balance.")
    run_sens = st.checkbox("Include sensitivity analysis", value=True,
                            help="One-at-a-time parameter shocks. Adds ~30s.")
    run_btn  = st.button("▶  Run Valuation", type="primary", use_container_width=True)
    st.divider()

    # ── Parameter legend ──────────────────────────────────────────────────────
    st.markdown(
        '📡 <span class="tag-calibratable">CALIBRATABLE</span> &nbsp;'
        'Estimable from historical market data<br><br>'
        '🎯 <span class="tag-judgement">JUDGEMENT</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        'Set based on deal terms or market view',
        unsafe_allow_html=True,
    )
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — VESSEL & DEAL
    # Known facts from the deal sheet. Not uncertain in the modelling sense.
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🚢  Vessel & Deal")
    st.caption("Fixed facts about the asset and transaction. Read from the deal sheet.")

    vessel_name = st.text_input("Vessel name", value="MV Pacific Horizon")
    vessel_type = st.selectbox("Type", [
        "Panamax Bulk Carrier", "Kamsarmax Bulk Carrier", "Capesize Bulk Carrier", "Handysize Bulk Carrier",
    ])

    col1, col2 = st.columns(2)
    with col1:
        dwt = st.number_input("DWT", value=76_000, step=1_000,
                               help="Deadweight tonnes — cargo capacity")
    with col2:
        ldt = st.number_input("LDT", value=13_000, step=500,
                               help="Light displacement tonnes — determines scrap steel value")

    col1, col2 = st.columns(2)
    with col1:
        vessel_age  = st.slider("Age (yrs)", 0, 20, 5)
    with col2:
        holding_yrs = st.slider("Hold (yrs)", 3, 20, 10,
                                 help="How long you intend to own before exit")

    purchase_price = st.number_input("Purchase price ($)", value=30_000_000,
                                      step=500_000, format="%d")

    st.markdown("**Financing**")
    include_financing = st.toggle(
        "Include financing / leverage",
        value=False,
        help="Off = unlevered asset valuation (default for vessel valuation as a data product). "
             "On = levered equity analysis for a specific deal structure.",
    )

    if include_financing:
        col1, col2 = st.columns(2)
        with col1:
            loan_amount   = st.number_input("Loan ($)", value=18_000_000, step=500_000, format="%d")
            interest_rate = st.slider("Rate (%)", 4.0, 12.0, 7.5, step=0.25) / 100
        with col2:
            loan_tenor    = st.slider("Tenor (yrs)", 5, 18, 12)
            years_elapsed = st.slider("Elapsed (yrs)", 0, 10, 5,
                                       help="Years of loan already repaid at valuation date")
    else:
        loan_amount   = 0
        interest_rate = 0.075
        loan_tenor    = 12
        years_elapsed = 0
        st.caption("Showing unlevered asset valuation. Toggle on to model a specific financing structure.")

    st.markdown("**Commercial**")
    tc_remaining   = st.slider("TC remaining (yrs)", 0.0, 5.0, 2.0, step=0.5,
                                help="Years left on any fixed time charter contract")
    tc_rate        = st.number_input("TC rate ($/day)", value=13_500, step=500,
                                      help="Contracted daily hire rate") if tc_remaining > 0 else 12_000
    operating_days = st.slider("Operating days/yr", 330, 365, 350,
                                help="Days available for trading — excludes planned off-hire")

    st.markdown("**Operating costs**")
    col1, col2 = st.columns(2)
    with col1:
        opex_mean    = st.number_input("Daily opex ($)", value=8_200, step=100,
                                        help="Crew, insurance, stores, maintenance")
        dd_cost_mean = st.number_input("Drydock cost ($)", value=1_800_000,
                                        step=100_000, format="%d",
                                        help="Cost per drydock event (every 5 years)")
    with col2:
        opex_cv    = st.slider("Opex unc. (cv)", 0.03, 0.25, 0.08,
                                help="Uncertainty as coefficient of variation. 0.08 = ±8% (1σ)")
        dd_cost_cv = st.slider("Drydock unc. (cv)", 0.05, 0.40, 0.20)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — MARKET PARAMETERS
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("📊  Market Parameters")
    st.caption("Drive the freight rate simulation and terminal value calculation.")

    # ── Spot rate — user-facing, changes daily ────────────────────────────────
    spot_tce = st.number_input(
        "Current spot TCE ($/day)", value=CALIB_SPOT, step=500,
        help="Today's market rate — starting point for all simulated paths. "
             "Update from Baltic Exchange or broker reports.",
    )

    # ── Calibrated parameters — shown as info, expandable to override ─────────
    st.markdown(
        '📡 <span class="tag-calibratable">CALIBRATED FROM DATA</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"κ, σ, and long-run mean calibrated from Baltic Panamax Index (BPI), "
        f"2012–2025 via AR(1) MLE. Vintage: {CALIB_DATE}."
    )

    # Calibrated values as a compact inline summary — metric cards are too wide for sidebar
    st.markdown(
        f"κ = **{CALIB_KAPPA}** &nbsp;·&nbsp; "
        f"σ = **{CALIB_SIGMA}** &nbsp;·&nbsp; "
        f"μ = **${CALIB_MU:,.0f}/day**",
        unsafe_allow_html=True,
    )

    # Collapsible override section
    with st.expander("Override calibrated parameters"):
        st.caption(
            "Default values are data-derived. Override here to stress-test assumptions "
            "or explore alternative cycle views."
        )
        lr_tce = st.number_input(
            "Long-run mean TCE ($/day)", value=CALIB_MU, step=500,
            help="Mid-cycle equilibrium. Full-history calibration gives $8,143/day; "
                 "post-2015 average gives $10,217/day. Current default splits the difference.",
        )
        kappa = st.slider(
            "Mean reversion speed κ", 0.20, 2.00, CALIB_KAPPA, step=0.05,
            help=f"Calibrated: {CALIB_KAPPA}. Half-life = ln(2)/κ. "
                 "Higher = faster reversion after shocks.",
        )
        sigma = st.slider(
            "Rate volatility σ", 0.30, 1.20, CALIB_SIGMA, step=0.05,
            help=f"Calibrated: {CALIB_SIGMA}. Annualised log-volatility of TCE rates. "
                 "Panamax historically 65–75%.",
        )
    # Note: st.expander widgets always execute in Streamlit (collapsed or not),
    # so lr_tce, kappa, sigma are always set from widgets with CALIB_* defaults.

    # ── Judgement-based ───────────────────────────────────────────────────────
    st.markdown(
        '🎯 <span class="tag-judgement">JUDGEMENT</span>',
        unsafe_allow_html=True,
    )
    st.caption("Set based on your market view or deal terms.")

    rho = st.slider(
        "Freight–scrap correlation ρ", 0.0, 0.90, 0.50, step=0.05,
        help="🎯 How much scrap prices co-move with freight at exit. "
             "Positive = both fall in bad markets. Empirically ~0.4–0.6.",
    )
    exit_type = st.selectbox("Exit type", ["second_hand", "scrap"],
                              help="🎯 How the vessel is disposed of at end of holding period")

    col1, col2 = st.columns(2)
    with col1:
        exit_mult_mean = st.slider("Exit EV/EBITDA (mean)", 3.0, 12.0, 7.0, step=0.5,
                                    help="🎯 Expected sale price as multiple of EBITDA at exit.")
        scrap_mean     = st.number_input("Scrap ($/LDT)", value=480, step=10,
                                          help="🎯 Demolition price per light tonne. Currently $400–550/LDT.")
    with col2:
        exit_mult_std = st.slider("Exit multiple unc.", 0.5, 4.0, 2.0, step=0.5,
                                   help="🎯 Uncertainty in exit multiple — reflects cycle timing risk.")
        scrap_cv      = st.slider("Scrap unc. (cv)", 0.05, 0.30, 0.15)

    wacc_mean = st.slider(
        "WACC (%)", 6.0, 16.0, 10.0, step=0.5,
        help="🎯 Unlevered cost of capital. Spot vessel: 9–12%. TC'd vessel: 7–9%.",
    ) / 100
    wacc_std = st.slider(
        "WACC uncertainty (%)", 0.25, 2.0, 1.0, step=0.25,
        help="🎯 Propagated through all discount calculations.",
    ) / 100
    hurdle_rate = st.slider(
        "Equity hurdle rate (%)", 8.0, 25.0, 15.0, step=1.0,
        help="🎯 Minimum acceptable IRR. Shipping PE typically targets 12–18%.",
    ) / 100

    st.divider()




st.title("Probabilistic Ship Valuation")
st.caption("Monte Carlo DCF · Log-OU freight rate process · All outputs are distributions, not point estimates")

if not run_btn:
    st.markdown("### How this works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**Step 1 — Simulate freight markets**

TCE rates are simulated thousands of times using a mean-reverting stochastic process.
Each path is a plausible future for what the vessel could earn day-to-day.
The fan chart shows the distribution of outcomes — not a forecast, but a structured view of uncertainty.
""")
    with col2:
        st.markdown("""
**Step 2 — Build cash flows on each path**

On each simulated rate path, the model builds a full annual P&L:
revenue (spot + any TC contracts), operating costs, drydocking, and debt service.
Cost inputs also vary randomly — no single assumption is held fixed.
""")
    with col3:
        st.markdown("""
**Step 3 — Value the equity**

Each path produces cash flows and a terminal sale value, discounted to today
at a risk-adjusted rate (WACC). The result is NPV, IRR, and return multiple
— one answer per scenario, thousands of scenarios in total.
""")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Reading the outputs**

- **Asset NPV** — does the vessel earn above its cost of capital, ignoring financing?
- **Equity NPV** — after debt service, does the equity make money in the median case?
- **IRR** — annualised equity return. Compare to your hurdle rate.
- **MOIC** — how many times do you get your equity back? 1× = break-even.
- **P(positive NPV)** — in what fraction of scenarios does equity come out ahead?
""")
    with col2:
        st.markdown("""
**Parameter types**

📡 **CALIBRATABLE** parameters describe *how rates behave* and can be estimated
directly from historical Baltic Exchange freight data using statistical methods.
Once the calibration module is complete, these will be auto-populated from data.

🎯 **JUDGEMENT** parameters reflect your market view or deal terms —
exit price, cost of capital, scrap assumptions. These are assumptions you own.
""")

    st.info("👈  Configure parameters in the sidebar, then click **▶ Run Valuation**.")
    st.stop()


# ── Build model objects ────────────────────────────────────────────────────────

vessel = VesselSpec(
    name                   = vessel_name,
    vessel_type            = vessel_type,
    dwt                    = dwt,
    ldt                    = ldt,
    vessel_age_years       = vessel_age,
    useful_life_years      = 25,
    holding_period_years   = holding_yrs,
    tc_coverage            = 0.0,
    tc_rate_usd_day        = tc_rate,
    tc_remaining_years     = tc_remaining,
    operating_days         = operating_days,
    daily_opex             = (opex_mean, opex_cv),
    drydock_interval_years = 5,
    drydock_cost           = (dd_cost_mean, dd_cost_cv),
    drydock_offhire_days   = (25, 0.20),
    tax_rate               = 0.0,
    purchase_price_usd     = purchase_price,
)

market = MarketParams(
    spot_tce_rate             = spot_tce,
    longrun_mean_tce          = lr_tce,
    mean_reversion_speed      = kappa,
    rate_volatility           = sigma,
    scrap_price_per_ldt       = (scrap_mean, scrap_cv),
    freight_scrap_correlation = rho,
    opex_inflation            = (0.025, 0.005),
    exit_earnings_multiple    = (exit_mult_mean, exit_mult_std),
    wacc                      = (wacc_mean, wacc_std),
    equity_hurdle_rate        = hurdle_rate,
)

debt = DebtSchedule(
    loan_amount        = loan_amount,
    loan_tenor_years   = loan_tenor,
    interest_rate      = interest_rate,
    amortisation_type  = "straight",
    years_elapsed      = years_elapsed,
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

# ── Summary metrics ───────────────────────────────────────────────────────────

eq_stats   = results.stats("equity_npvs")
ast_stats  = results.stats("asset_npvs")
irr_valid  = results.irrs[~np.isnan(results.irrs)]
moic       = results.moic_distribution()
moic_valid = moic[~np.isnan(moic)]

prob_pos_asset = results.prob_exceeds("asset_npvs", 0)
prob_pos_npv   = results.prob_exceeds("equity_npvs", 0)
prob_hurdle    = float(np.mean(irr_valid > hurdle_rate)) if len(irr_valid) > 0 else 0.0
median_irr     = float(np.median(irr_valid)) if len(irr_valid) > 0 else float("nan")
median_moic    = float(np.median(moic_valid)) if len(moic_valid) > 0 else float("nan")

st.subheader("Summary Metrics")

if not include_financing:
    # ── Unlevered / asset-level view — the core data product ─────────────────
    st.caption("Asset-level valuation — unlevered, independent of any financing structure.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Asset NPV (median)",
        f"${ast_stats['median']/1e6:.1f}M",
        help=f"Vessel earns this much above its cost of capital in the median scenario.\n"
             f"P10: ${ast_stats['p10']/1e6:.1f}M  |  P90: ${ast_stats['p90']/1e6:.1f}M",
    )
    c2.metric(
        "Asset value (median)",
        f"${(ast_stats['median'] + purchase_price)/1e6:.1f}M",
        help=f"Model's estimate of fair vessel value (purchase price + NPV).\n"
             f"P10: ${(ast_stats['p10'] + purchase_price)/1e6:.1f}M  |  "
             f"P90: ${(ast_stats['p90'] + purchase_price)/1e6:.1f}M",
    )
    c3.metric(
        "P(NPV > 0)", f"{prob_pos_asset:.0%}",
        help="Fraction of scenarios where the vessel earns above its cost of capital.",
    )
    c4.metric(
        "WACC applied", f"{wacc_mean:.1%}",
        help="Discount rate used. Reflects asset-level risk, not financing.",
    )
    st.caption(
        f"Simulated range — "
        f"Asset NPV: P10 {ast_stats['p10']/1e6:.1f}M, P90 {ast_stats['p90']/1e6:.1f}M  ·  "
        f"Asset value: P10 {(ast_stats['p10']+purchase_price)/1e6:.1f}M, P90 {(ast_stats['p90']+purchase_price)/1e6:.1f}M  (USD)"
    )
else:
    # ── Levered / equity view — deal-specific ─────────────────────────────────
    st.caption("Levered equity analysis — results reflect the specific financing structure entered.")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Asset NPV", f"${ast_stats['median']/1e6:.1f}M",
               help=f"Vessel value independent of financing.\n"
                    f"P10: ${ast_stats['p10']/1e6:.1f}M  |  P90: ${ast_stats['p90']/1e6:.1f}M")
    c2.metric("Equity NPV", f"${eq_stats['median']/1e6:.1f}M",
               help=f"After debt service. Negative = debt consumes more than the vessel earns.\n"
                    f"P10: ${eq_stats['p10']/1e6:.1f}M  |  P90: ${eq_stats['p90']/1e6:.1f}M")
    c3.metric("Equity invested", f"${equity_invested/1e6:.1f}M",
               help="Purchase price minus loan — your cash outlay.")
    c4.metric("MOIC", f"{median_moic:.2f}×",
               delta=f"P(>1×): {np.mean(moic_valid>1):.0%}",
               help="Multiple of invested capital. 1× = return of capital.")
    c5.metric("IRR", f"{median_irr:.1%}",
               delta=f"P(>hurdle {hurdle_rate:.0%}): {prob_hurdle:.0%}",
               help="Annualised equity return. Compare to your hurdle rate.")
    c6.metric("P(positive NPV)", f"{prob_pos_npv:.0%}",
               help="Fraction of scenarios where equity comes out ahead.")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

if not include_financing:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  Asset Valuation",
        "📈  Rate Paths",
        "💵  Cash Flows",
        "🌪️  Sensitivity",
    ])
    tab5 = tab6 = None
else:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊  NPV & IRR",
        "📈  Rate Paths",
        "💵  Cash Flows",
        "📉  MOIC",
        "🔗  Freight–Scrap",
        "🌪️  Sensitivity",
    ])

with tab1:
    if not include_financing:
        # ── Unlevered: asset NPV is the headline ─────────────────────────────
        st.caption(
            "Asset-level valuation — each bar is a simulated scenario. "
            "The distribution shows the range of plausible vessel values under different freight rate environments."
        )
        st.pyplot(plot_npv_distribution(results, "asset"), use_container_width=True)
    else:
        # ── Levered: equity NPV + IRR side by side ────────────────────────────
        st.caption(
            "Each bar is a simulated scenario. The width of the distribution reflects "
            "sensitivity to freight rates — a wide spread means the outcome is highly rate-dependent."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_npv_distribution(results, "equity"), use_container_width=True)
        with col2:
            st.pyplot(plot_irr_distribution(results, hurdle_rate), use_container_width=True)
        with st.expander("Asset-level (unlevered) NPV — vessel value independent of financing"):
            st.caption("Strips out the debt structure. Useful for comparing vessels on a like-for-like basis.")
            st.pyplot(plot_npv_distribution(results, "asset"), use_container_width=True)

with tab2:
    half_life = np.log(2) / kappa
    st.caption(
        f"Rates start at the current spot of ${spot_tce:,.0f}/day and gradually revert toward "
        f"the long-run equilibrium of ${lr_tce:,.0f}/day — a process that takes roughly {half_life:.1f} years on average. "
        f"Faint lines are individual simulated paths. Shaded bands show the P10-P90 and P25-P75 ranges across all scenarios."
    )
    st.pyplot(plot_rate_fan(results, longrun_mean=lr_tce), use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current spot", f"${spot_tce:,.0f}/day")
    c2.metric("Long-run mean", f"${lr_tce:,.0f}/day")
    c3.metric("Rate half-life", f"{half_life:.1f} years",
               help="Time for the gap between spot and long-run mean to halve")
    c4.metric("Annualised vol", f"{sigma:.0%}")

with tab3:
    st.caption(
        "EBITDA and free cash flow across the holding period. "
        "Drydock years (dotted lines) show as dips — both from the cost itself and lost revenue during off-hire."
    )
    st.pyplot(
        plot_cashflow_evolution(results, drydock_years=vessel.drydock_years),
        use_container_width=True,
    )

# ── Financing-only tabs ───────────────────────────────────────────────────────

if include_financing and tab5 is not None:
    with tab4:
        st.caption("How many times do you get your equity back? 1x = break-even. Spread reflects rate and exit timing uncertainty.")
        st.pyplot(plot_moic_distribution(results), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Median MOIC", f"{median_moic:.2f}x")
        c2.metric("P10 MOIC", f"{np.percentile(moic_valid, 10):.2f}x", help="Downside scenario")
        c3.metric("P90 MOIC", f"{np.percentile(moic_valid, 90):.2f}x", help="Upside scenario")

    with tab5:
        st.caption(
            f"In distressed freight markets, owners scrap aggressively, flooding the demolition market "
            f"and pushing scrap prices down simultaneously. Correlation set to {rho:.2f} — "
            f"terminal value is compressed from both ends in downside scenarios."
        )
        terminal_rates = results.annual_rate_matrix[:, -1]
        import numpy as _np
        _rng = _np.random.default_rng(42)
        terminal_scrap = market.simulate_correlated_scrap(terminal_rates, _rng)
        st.pyplot(plot_freight_scrap_scatter(terminal_rates, terminal_scrap), use_container_width=True)

    with tab6:
        if not run_sens:
            st.info("Enable **Include sensitivity analysis** in the sidebar to see this chart.")
        else:
            with st.spinner("Running sensitivity analysis (~30s)..."):
                tornado_df, base_median = run_sensitivity(vessel, market, debt, n_sims=1_500)
            st.caption(
                "Each parameter is shocked up and down by one standard deviation (or 20-30% where no std is defined). "
                "Bar length = impact on median equity NPV. Longest bars = where to focus diligence."
            )
            st.pyplot(plot_tornado(tornado_df, base_median), use_container_width=True)
            with st.expander("Raw numbers"):
                st.dataframe(
                    tornado_df[["low", "high", "range", "low_delta", "high_delta"]]
                    .style.format("${:,.0f}"),
                    use_container_width=True,
                )

elif tab4 is not None:
    # Unlevered sensitivity tab
    with tab4:
        if not run_sens:
            st.info("Enable **Include sensitivity analysis** in the sidebar to see this chart.")
        else:
            with st.spinner("Running sensitivity analysis (~30s)..."):
                tornado_df, base_median = run_sensitivity(vessel, market, debt, n_sims=1_500)
            st.caption(
                "Each parameter is shocked up and down. "
                "Bar length = impact on median asset NPV. Longest bars = dominant value drivers."
            )
            st.pyplot(plot_tornado(tornado_df, base_median), use_container_width=True)
            with st.expander("Raw numbers"):
                st.dataframe(
                    tornado_df[["low", "high", "range", "low_delta", "high_delta"]]
                    .style.format("${:,.0f}"),
                    use_container_width=True,
                )

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"**{vessel_name}** · {vessel_type} · {dwt:,.0f} DWT · {n_sims:,} simulations · "
    f"Log-OU (κ={kappa:.2f}, σ={sigma:.0%}, LR=${lr_tce/1e3:.0f}k) · "
    f"Freight–scrap ρ={rho:.2f} · WACC {wacc_mean:.1%}"
)
