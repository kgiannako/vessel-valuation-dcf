"""
valuation.py / analysis.py (combined)
--------------------------------------
NPV engine, NAV calculation, sensitivity analysis, and visualisation.

Produces a comprehensive output dashboard showing:
  1. NPV distribution (equity and asset level)
  2. TCE rate path fan chart
  3. Annual FCF distribution over time
  4. Sensitivity tornado
  5. Summary statistics table
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings("ignore")

from vessel import VesselSpec
from market import MarketParams
from debt import DebtSchedule
from simulation import SimulationResults, run_simulation


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_millions(x, pos=None):
    return f"${x/1e6:.1f}M"

def fmt_thousands(x, pos=None):
    return f"${x/1e3:.0f}k"


# ── Core valuation metrics ────────────────────────────────────────────────────

def compute_valuation_metrics(
    results: SimulationResults,
    vessel: VesselSpec,
    debt: DebtSchedule,
) -> dict:
    """
    Summarise valuation results into key metrics.
    """
    eq = results.equity_npvs
    asset = results.asset_npvs
    equity_invested = vessel.purchase_price_usd - debt.loan_amount

    metrics = {
        # Asset-level (unlevered)
        "asset_npv_mean": np.mean(asset),
        "asset_npv_median": np.median(asset),
        "asset_npv_p10": np.percentile(asset, 10),
        "asset_npv_p90": np.percentile(asset, 90),

        # Equity-level (levered)
        "equity_npv_mean": np.mean(eq),
        "equity_npv_median": np.median(eq),
        "equity_npv_p10": np.percentile(eq, 10),
        "equity_npv_p25": np.percentile(eq, 25),
        "equity_npv_p75": np.percentile(eq, 75),
        "equity_npv_p90": np.percentile(eq, 90),

        # Return metrics
        "equity_invested": equity_invested,
        "moic_mean": (np.mean(eq) + equity_invested) / equity_invested if equity_invested > 0 else np.nan,
        "moic_median": (np.median(eq) + equity_invested) / equity_invested if equity_invested > 0 else np.nan,
        "prob_positive_equity_npv": np.mean(eq > 0),
        "prob_loss_gt_50pct": np.mean(eq < -0.5 * equity_invested) if equity_invested > 0 else np.nan,
    }
    return metrics


# ── Sensitivity analysis ──────────────────────────────────────────────────────

def sensitivity_tornado(
    vessel: VesselSpec,
    market: MarketParams,
    debt: DebtSchedule,
    n_sims: int = 2000,
    seed: int = 99,
) -> pd.DataFrame:
    """
    One-at-a-time sensitivity analysis.
    Each parameter is shocked ±1 std (or ±20% if no std available).
    Returns DataFrame sorted by impact range for tornado chart.
    """
    base_results = run_simulation(vessel, market, debt, n_simulations=n_sims, seed=seed)
    base_median = np.median(base_results.equity_npvs)

    sensitivities = []

    def run_variant(label, mod_vessel=None, mod_market=None, mod_debt=None, direction="high"):
        v = mod_vessel or vessel
        m = mod_market or market
        d = mod_debt or debt
        res = run_simulation(v, m, d, n_simulations=n_sims, seed=seed)
        return np.median(res.equity_npvs)

    # WACC shock
    import copy
    for label, wacc_delta, direction in [("WACC -1σ", -market.wacc[1], "low"),
                                          ("WACC +1σ", +market.wacc[1], "high")]:
        m = copy.deepcopy(market)
        m.wacc = (max(0.03, market.wacc[0] + wacc_delta), market.wacc[1])
        val = run_variant(label, mod_market=m)
        sensitivities.append({"parameter": "WACC", "direction": direction, "median_npv": val})

    # Long-run TCE rate shock (±20%)
    for label, tce_delta, direction in [("LR TCE -20%", -0.20, "low"),
                                         ("LR TCE +20%", +0.20, "high")]:
        m = copy.deepcopy(market)
        m.longrun_mean_tce = market.longrun_mean_tce * (1 + tce_delta)
        val = run_variant(label, mod_market=m)
        sensitivities.append({"parameter": "LR TCE Rate", "direction": direction, "median_npv": val})

    # Rate volatility shock
    for label, vol_delta, direction in [("Vol -30%", -0.30, "low"),
                                         ("Vol +30%", +0.30, "high")]:
        m = copy.deepcopy(market)
        m.rate_volatility = max(0.1, market.rate_volatility * (1 + vol_delta))
        val = run_variant(label, mod_market=m)
        sensitivities.append({"parameter": "Rate Volatility", "direction": direction, "median_npv": val})

    # Opex shock
    for label, opex_delta, direction in [("Opex -1σ", -vessel.daily_opex[1], "low"),
                                          ("Opex +1σ", +vessel.daily_opex[1], "high")]:
        v = copy.deepcopy(vessel)
        v.daily_opex = (vessel.daily_opex[0] * (1 + opex_delta), vessel.daily_opex[1])
        val = run_variant(label, mod_vessel=v)
        sensitivities.append({"parameter": "Daily Opex", "direction": direction, "median_npv": val})

    # Exit multiple shock
    for label, mult_delta, direction in [("Exit Multiple -1σ", -market.exit_earnings_multiple[1], "low"),
                                          ("Exit Multiple +1σ", +market.exit_earnings_multiple[1], "high")]:
        m = copy.deepcopy(market)
        m.exit_earnings_multiple = (max(1, market.exit_earnings_multiple[0] + mult_delta),
                                     market.exit_earnings_multiple[1])
        val = run_variant(label, mod_market=m)
        sensitivities.append({"parameter": "Exit Multiple", "direction": direction, "median_npv": val})

    df = pd.DataFrame(sensitivities)
    pivot = df.pivot(index="parameter", columns="direction", values="median_npv")
    pivot["range"] = (pivot["high"] - pivot["low"]).abs()
    pivot["low_delta"] = pivot["low"] - base_median
    pivot["high_delta"] = pivot["high"] - base_median
    pivot = pivot.sort_values("range", ascending=True)
    return pivot, base_median


# ── Master visualisation ──────────────────────────────────────────────────────

def plot_valuation_dashboard(
    results: SimulationResults,
    vessel: VesselSpec,
    market: MarketParams,
    debt: DebtSchedule,
    show_sensitivity: bool = True,
    save_path: str = None,
):
    """
    Generate comprehensive 6-panel valuation dashboard.
    """
    metrics = compute_valuation_metrics(results, vessel, debt)
    n_years = vessel.effective_holding_years

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    DARK_BG = "#0f1117"
    PANEL_BG = "#1a1d27"
    ACCENT1 = "#4f9cf9"   # blue
    ACCENT2 = "#f97b4f"   # orange
    ACCENT3 = "#4ff9a0"   # green
    TEXT = "#e0e4f0"
    GRID = "#2a2d3a"

    def style_ax(ax, title=""):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
        if title:
            ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)

    # ── Panel 1: Equity NPV Distribution ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    eq = results.equity_npvs / 1e6
    ax1.hist(eq, bins=80, color=ACCENT1, alpha=0.8, edgecolor="none")
    ax1.axvline(np.percentile(eq, 10), color=ACCENT2, lw=1.5, ls="--", label="P10")
    ax1.axvline(np.median(eq), color=ACCENT3, lw=2, ls="-", label="Median")
    ax1.axvline(np.percentile(eq, 90), color=ACCENT2, lw=1.5, ls="--", label="P90")
    ax1.axvline(0, color="white", lw=1, ls=":", alpha=0.5, label="Break-even")
    ax1.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:.1f}M"))
    ax1.set_xlabel("Equity NPV ($M)")
    ax1.set_ylabel("Frequency")
    style_ax(ax1, f"Equity NPV Distribution  |  Median: ${np.median(eq):.1f}M  |  P(>0): {metrics['prob_positive_equity_npv']:.0%}")

    # ── Panel 2: Summary Stats ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PANEL_BG)
    ax2.axis("off")
    style_ax(ax2, "Key Metrics")

    stat_lines = [
        ("Asset NPV (median)", f"${metrics['asset_npv_median']/1e6:.1f}M"),
        ("Asset NPV P10–P90", f"${metrics['asset_npv_p10']/1e6:.1f}M – ${metrics['asset_npv_p90']/1e6:.1f}M"),
        ("", ""),
        ("Equity NPV (median)", f"${metrics['equity_npv_median']/1e6:.1f}M"),
        ("Equity NPV P10–P90", f"${metrics['equity_npv_p10']/1e6:.1f}M – ${metrics['equity_npv_p90']/1e6:.1f}M"),
        ("", ""),
        ("Equity invested", f"${metrics['equity_invested']/1e6:.1f}M"),
        ("MOIC (median)", f"{metrics['moic_median']:.2f}×"),
        ("P(positive NPV)", f"{metrics['prob_positive_equity_npv']:.0%}"),
        ("P(loss >50%)", f"{metrics['prob_loss_gt_50pct']:.0%}"),
    ]

    y_pos = 0.95
    for label, value in stat_lines:
        if label == "":
            y_pos -= 0.05
            continue
        color = ACCENT3 if "NPV" in label or "MOIC" in label else TEXT
        ax2.text(0.02, y_pos, label, transform=ax2.transAxes,
                 color=TEXT, fontsize=8.5, va="top")
        ax2.text(0.98, y_pos, value, transform=ax2.transAxes,
                 color=color, fontsize=8.5, va="top", ha="right", fontweight="bold")
        y_pos -= 0.09

    # ── Panel 3: TCE Rate Fan Chart ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    years = np.arange(1, n_years + 1)
    rate_matrix = results.annual_rate_matrix / 1e3  # to $/day in thousands

    p10 = np.percentile(rate_matrix, 10, axis=0)
    p25 = np.percentile(rate_matrix, 25, axis=0)
    p50 = np.percentile(rate_matrix, 50, axis=0)
    p75 = np.percentile(rate_matrix, 75, axis=0)
    p90 = np.percentile(rate_matrix, 90, axis=0)

    ax3.fill_between(years, p10, p90, alpha=0.2, color=ACCENT1, label="P10–P90")
    ax3.fill_between(years, p25, p75, alpha=0.35, color=ACCENT1, label="P25–P75")
    ax3.plot(years, p50, color=ACCENT1, lw=2, label="Median")
    ax3.axhline(market.longrun_mean_tce / 1e3, color=ACCENT3, lw=1, ls="--", alpha=0.7, label="LR Mean")
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:.0f}k"))
    ax3.set_xlabel("Year")
    ax3.set_ylabel("TCE Rate ($/day)")
    ax3.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    style_ax(ax3, "TCE Rate Path Distribution")

    # ── Panel 4: Annual EBITDA Distribution ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ebitda_matrix = results.annual_ebitda_matrix / 1e6

    ebitda_p10 = np.percentile(ebitda_matrix, 10, axis=0)
    ebitda_p50 = np.percentile(ebitda_matrix, 50, axis=0)
    ebitda_p90 = np.percentile(ebitda_matrix, 90, axis=0)

    ax4.fill_between(years, ebitda_p10, ebitda_p90, alpha=0.25, color=ACCENT3)
    ax4.plot(years, ebitda_p50, color=ACCENT3, lw=2, label="Median EBITDA")
    ax4.plot(years, ebitda_p10, color=ACCENT2, lw=1, ls="--", label="P10")
    ax4.plot(years, ebitda_p90, color=ACCENT1, lw=1, ls="--", label="P90")
    ax4.axhline(0, color="white", lw=0.8, ls=":", alpha=0.5)

    # Mark drydock years
    for ddy in vessel.drydock_years:
        if ddy <= n_years:
            ax4.axvline(ddy, color=ACCENT2, lw=1, ls=":", alpha=0.5)
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:.1f}M"))
    ax4.set_xlabel("Year")
    ax4.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    style_ax(ax4, "Annual EBITDA (orange dashes = drydock years)")

    # ── Panel 5: Annual FCF Distribution ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    fcf_matrix = results.annual_fcf_matrix / 1e6
    fcf_p10 = np.percentile(fcf_matrix, 10, axis=0)
    fcf_p50 = np.percentile(fcf_matrix, 50, axis=0)
    fcf_p90 = np.percentile(fcf_matrix, 90, axis=0)

    ax5.fill_between(years, fcf_p10, fcf_p90, alpha=0.25, color=ACCENT2)
    ax5.plot(years, fcf_p50, color=ACCENT2, lw=2, label="Median levered FCF")
    ax5.plot(years, fcf_p10, color=ACCENT2, lw=1, ls="--", alpha=0.6)
    ax5.plot(years, fcf_p90, color=ACCENT1, lw=1, ls="--", alpha=0.6)
    ax5.axhline(0, color="white", lw=0.8, ls=":", alpha=0.5)
    ax5.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:.1f}M"))
    ax5.set_xlabel("Year")
    ax5.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    style_ax(ax5, "Annual Levered FCF")

    # ── Panel 6: Tornado Chart ────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    style_ax(ax6, "Sensitivity Analysis (impact on median equity NPV)")

    try:
        tornado_df, base_median = sensitivity_tornado(vessel, market, debt, n_sims=1500)
        params = tornado_df.index.tolist()
        low_deltas = tornado_df["low_delta"].values / 1e6
        high_deltas = tornado_df["high_delta"].values / 1e6
        y_pos_tornado = np.arange(len(params))

        for i, (param, lo, hi) in enumerate(zip(params, low_deltas, high_deltas)):
            ax6.barh(i, lo, left=0, color=ACCENT2, alpha=0.8, height=0.5)
            ax6.barh(i, hi, left=0, color=ACCENT1, alpha=0.8, height=0.5)

        ax6.set_yticks(y_pos_tornado)
        ax6.set_yticklabels(params, color=TEXT, fontsize=9)
        ax6.axvline(0, color="white", lw=1, ls="-", alpha=0.6)
        ax6.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:+.1f}M"))
        ax6.set_xlabel("Delta from base median equity NPV ($M)")
        ax6.text(0.99, 0.95, f"Base median: ${base_median/1e6:.1f}M",
                 transform=ax6.transAxes, ha="right", va="top",
                 color=ACCENT3, fontsize=9, fontweight="bold")
    except Exception as e:
        ax6.text(0.5, 0.5, f"Sensitivity analysis failed:\n{e}",
                 ha="center", va="center", transform=ax6.transAxes, color=TEXT)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Ship Valuation Dashboard  |  {vessel.name}  |  {vessel.vessel_type}  |  "
        f"{vessel.dwt:,.0f} DWT  |  {results.n_simulations:,} simulations",
        color=TEXT, fontsize=13, fontweight="bold", y=0.98
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"Dashboard saved to: {save_path}")

    return fig, metrics
