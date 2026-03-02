"""
charts.py
---------
All visualisation logic, decoupled from the Streamlit UI layer.
Each function returns a matplotlib Figure that Streamlit renders via st.pyplot().

Design: dark theme throughout, consistent colour palette.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings("ignore")

from src.simulation import SimulationResults

# ── Palette ───────────────────────────────────────────────────────────────────
# Warm dark theme derived from the provided colour pairs:
#   #C5E384 / #200F07  — lime-sage on near-black
#   #C2D8C4 / #222222  — muted sage on charcoal
#   #385144 / #F8F5F2  — forest green on warm off-white
#   #E8A736 / #FAEFD9  — amber on parchment

DARK_BG  = "#200F07"    # near-black, warm brown undertone — page background
PANEL_BG = "#2A1F1A"    # slightly lifted warm dark — chart panel
LIME     = "#C5E384"    # lime-sage — primary accent: histograms, fan median, main series
AMBER    = "#E8A736"    # amber — secondary accent: downside bars, warnings, P10 lines
SAGE     = "#C2D8C4"    # muted sage — tertiary: upside bars, P90 lines
FOREST   = "#385144"    # forest green — deep accent: IRR/MOIC fills
PARCHMENT= "#FAEFD9"    # warm parchment — highlight text on dark
TEXT     = "#F8F5F2"    # warm off-white — all axis labels and titles
GRID     = "#3A2A22"    # warm dark grid lines


def _style(ax, title=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)


def _fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    return fig, ax


def _fmt_M(x, p=None):   return f"${x/1e6:.1f}M"
def _fmt_k(x, p=None):   return f"${x/1e3:.0f}k"
def _fmt_pct(x, p=None): return f"{x:.0%}"


# ── 1. NPV distribution ───────────────────────────────────────────────────────

def plot_npv_distribution(results: SimulationResults, series: str = "equity") -> plt.Figure:
    """Histogram of NPV with P10/median/P90 marked."""
    arr   = results.equity_npvs if series == "equity" else results.asset_npvs
    label = "Equity NPV" if series == "equity" else "Asset NPV (Unlevered)"
    fill  = LIME if series == "equity" else SAGE

    fig, ax = _fig(10, 4.5)
    ax.hist(arr / 1e6, bins=80, color=fill, alpha=0.85, edgecolor="none")

    # Vertical markers: P10 amber, median parchment, P90 sage
    for pct, col, ls, lbl in [
        (10,  AMBER,     "--", "P10"),
        (50,  PARCHMENT, "-",  "Median"),
        (90,  SAGE,      "--", "P90"),
    ]:
        v = np.percentile(arr, pct) / 1e6
        ax.axvline(v, color=col, lw=1.8, ls=ls, label=f"{lbl}: ${v:.1f}M")

    ax.axvline(0, color=TEXT, lw=1, ls=":", alpha=0.35, label="Break-even")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:.1f}M"))
    ax.set_xlabel(f"{label} ($M)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    prob_pos = np.mean(arr > 0)
    _style(ax, f"{label} Distribution  |  P(>0): {prob_pos:.0%}  |  n={results.n_simulations:,}")
    fig.tight_layout()
    return fig


# ── 2. IRR distribution ───────────────────────────────────────────────────────

def plot_irr_distribution(results: SimulationResults, hurdle_rate: float = 0.15) -> plt.Figure:
    """Histogram of equity IRR with hurdle rate marked."""
    valid = results.irrs[~np.isnan(results.irrs)]

    fig, ax = _fig(10, 4.5)
    ax.hist(valid * 100, bins=80, color=FOREST, alpha=0.90, edgecolor="none")

    median_irr = np.median(valid)
    ax.axvline(median_irr * 100, color=LIME,  lw=2.0, ls="-",
               label=f"Median IRR: {median_irr:.1%}")
    ax.axvline(hurdle_rate * 100, color=AMBER, lw=2.0, ls="--",
               label=f"Hurdle: {hurdle_rate:.0%}")
    ax.axvline(0, color=TEXT, lw=1, ls=":", alpha=0.35)

    prob_hurdle = np.mean(valid > hurdle_rate)
    ax.set_xlabel("Equity IRR (%)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    _style(ax, f"IRR Distribution  |  P(>hurdle {hurdle_rate:.0%}): {prob_hurdle:.0%}  |  "
               f"P10: {np.percentile(valid,10):.1%}  P90: {np.percentile(valid,90):.1%}")
    fig.tight_layout()
    return fig


# ── 3. TCE rate fan chart ─────────────────────────────────────────────────────

def plot_rate_fan(
    results: SimulationResults,
    longrun_mean: float = None,
    n_sample_paths: int = 40,
    seed: int = 0,
) -> plt.Figure:
    """
    Fan chart: faint individual paths as background texture,
    distributional envelope and median clearly on top.
    """
    rate_matrix = results.annual_rate_matrix
    n_years = results.holding_years
    years = np.arange(1, n_years + 1)

    fig, ax = _fig(10, 4.5)

    # Faint paths — texture only, sits behind everything
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rate_matrix), size=min(n_sample_paths, len(rate_matrix)), replace=False)
    for path_idx in idx:
        ax.plot(years, rate_matrix[path_idx] / 1e3,
                lw=0.6, alpha=0.08, color=LIME, zorder=1)

    # Distributional envelope
    p = {pct: np.percentile(rate_matrix, pct, axis=0) / 1e3 for pct in [10, 25, 50, 75, 90]}
    ax.fill_between(years, p[10], p[90], alpha=0.12, color=LIME, label="P10–P90", zorder=2)
    ax.fill_between(years, p[25], p[75], alpha=0.30, color=LIME, label="P25–P75", zorder=3)
    ax.plot(years, p[50], color=LIME, lw=2.5, label="Median", zorder=4)

    if longrun_mean:
        ax.axhline(longrun_mean / 1e3, color=AMBER, lw=1.3, ls="--",
                   alpha=0.85, label=f"LR mean ${longrun_mean/1e3:.0f}k", zorder=5)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:.0f}k"))
    ax.set_xlabel("Year")
    ax.set_ylabel("TCE Rate ($/day)")
    ax.set_xticks(years)

    # Clip y-axis to P2–P98 so a handful of extreme paths don't compress the
    # median/fan area — the part the viewer actually needs to read
    all_rates_k = rate_matrix / 1e3
    y_lo = max(0, np.percentile(all_rates_k, 2)  * 0.85)
    y_hi =        np.percentile(all_rates_k, 98) * 1.10
    ax.set_ylim(y_lo, y_hi)

    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    _style(ax, "Simulated TCE Rate Paths")
    fig.tight_layout()
    return fig


# ── 4. Annual cash flow evolution ─────────────────────────────────────────────

def plot_cashflow_evolution(results: SimulationResults, drydock_years: list = None) -> plt.Figure:
    """Side-by-side: EBITDA and levered FCF fan charts over holding period."""
    n_years = results.holding_years
    years = np.arange(1, n_years + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.patch.set_facecolor(DARK_BG)

    for ax, matrix, colour, title in [
        (axes[0], results.annual_ebitda_matrix, SAGE,  "Annual EBITDA"),
        (axes[1], results.annual_fcf_matrix,    LIME,  "Annual Levered FCF"),
    ]:
        data = matrix / 1e6
        p10  = np.percentile(data, 10, axis=0)
        p50  = np.percentile(data, 50, axis=0)
        p90  = np.percentile(data, 90, axis=0)

        ax.fill_between(years, p10, p90, alpha=0.20, color=colour)
        ax.plot(years, p50, color=colour, lw=2.5, label="Median")
        ax.plot(years, p10, color=AMBER,  lw=1.0, ls="--", alpha=0.7, label="P10")
        ax.plot(years, p90, color=colour, lw=1.0, ls="--", alpha=0.5, label="P90")
        ax.axhline(0, color=TEXT, lw=0.8, ls=":", alpha=0.25)

        if drydock_years:
            for ddy in drydock_years:
                if ddy <= n_years:
                    ax.axvline(ddy, color=AMBER, lw=1, ls=":", alpha=0.55,
                               label="Drydock" if ddy == drydock_years[0] else "")

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:.1f}M"))
        ax.set_xlabel("Year")
        ax.set_ylabel("$M")
        ax.set_xticks(years)
        ax.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
        _style(ax, title)

    fig.tight_layout()
    return fig


# ── 5. Tornado / sensitivity ──────────────────────────────────────────────────

def plot_tornado(tornado_df: pd.DataFrame, base_median: float) -> plt.Figure:
    """Horizontal bar tornado chart of sensitivity deltas."""
    fig, ax = _fig(12, max(3.5, len(tornado_df) * 0.85))

    params = tornado_df.index.tolist()
    low_d  = tornado_df["low_delta"].values / 1e6
    high_d = tornado_df["high_delta"].values / 1e6

    for i, (lo, hi) in enumerate(zip(low_d, high_d)):
        ax.barh(i, lo, color=AMBER,  alpha=0.88, height=0.52)
        ax.barh(i, hi, color=FOREST, alpha=0.88, height=0.52)
        ax.text(-0.15, i, f"${lo:+.1f}M", ha="right", va="center", color=TEXT, fontsize=7.5)
        ax.text( 0.15, i, f"${hi:+.1f}M", ha="left",  va="center", color=TEXT, fontsize=7.5)

    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params, color=TEXT, fontsize=9)
    ax.axvline(0, color=PARCHMENT, lw=1.2, alpha=0.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:+.1f}M"))
    ax.set_xlabel("Δ from base median equity NPV ($M)")
    ax.text(0.98, 0.97, f"Base median: ${base_median/1e6:.1f}M",
            transform=ax.transAxes, ha="right", va="top",
            color=LIME, fontsize=9, fontweight="bold")
    _style(ax, "Sensitivity — Impact on Median Equity NPV  (amber = downside  ·  green = upside)")
    fig.tight_layout()
    return fig


# ── 6. MOIC distribution ─────────────────────────────────────────────────────

def plot_moic_distribution(results: SimulationResults) -> plt.Figure:
    """Distribution of equity multiple of invested capital."""
    moic  = results.moic_distribution()
    valid = moic[~np.isnan(moic)]

    fig, ax = _fig(9, 4)
    ax.hist(valid, bins=80, color=FOREST, alpha=0.88, edgecolor="none")
    ax.axvline(1.0, color=TEXT,  lw=1.5, ls=":", alpha=0.5, label="1× (return of capital)")
    ax.axvline(np.median(valid), color=LIME, lw=2.2, ls="-",
               label=f"Median: {np.median(valid):.2f}×")

    prob_above_1 = np.mean(valid > 1.0)
    ax.set_xlabel("MOIC (×)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    _style(ax, f"MOIC Distribution  |  P(>1×): {prob_above_1:.0%}")
    fig.tight_layout()
    return fig


# ── 7. Freight vs scrap scatter ───────────────────────────────────────────────

def plot_freight_scrap_scatter(
    terminal_rates: np.ndarray,
    terminal_scrap: np.ndarray,
    sample_size: int = 1000,
) -> plt.Figure:
    """Scatter of terminal-year TCE rate vs scrap price with OLS fit."""
    idx = np.random.choice(len(terminal_rates), min(sample_size, len(terminal_rates)), replace=False)
    x = terminal_rates[idx] / 1e3
    y = terminal_scrap[idx]

    fig, ax = _fig(7, 4)
    ax.scatter(x, y, alpha=0.25, s=12, color=SAGE, edgecolors="none")

    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), color=AMBER, lw=2.2, label="OLS fit")

    corr = np.corrcoef(x, y)[0, 1]
    ax.set_xlabel("Terminal TCE Rate ($/day, $k)")
    ax.set_ylabel("Scrap Price ($/LDT)")
    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    _style(ax, f"Freight–Scrap Correlation  |  ρ={corr:.2f} (realised in simulation)")
    fig.tight_layout()
    return fig


# ── 8. Calibration dashboard ──────────────────────────────────────────────────

def plot_calibration_dashboard(
    series: "pd.Series",
    cal,
    rolling_df: "pd.DataFrame",
) -> plt.Figure:
    """
    Four-panel calibration summary:
      1. Price history with long-run mean
      2. AR(1) scatter with fit line
      3. Rolling long-run mean (the window problem)
      4. Rolling κ and σ
    """
    import pandas as pd

    log_s = np.log(series.resample("W").mean().dropna().values)
    x, xn = log_s[:-1], log_s[1:]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.patch.set_facecolor(DARK_BG)

    # 1. History
    ax = axes[0, 0]
    s_weekly = series.resample("W").mean().dropna()
    ax.plot(s_weekly.index, s_weekly.values, color=LIME, lw=0.9, alpha=0.9)
    ax.axhline(cal.mu_level, color=AMBER, lw=1.5, ls="--",
               label=f"LR mean {cal.mu_level:,.0f}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:,.0f}"))
    unit = "$/day" if cal.data_mode == "TCE_USD" else "index pts"
    ax.set_ylabel(f"BPI ({unit})")
    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    _style(ax, f"Historical Series  |  {cal.date_from.date()} → {cal.date_to.date()}")

    # 2. AR(1) scatter
    ax = axes[0, 1]
    step = max(1, len(x) // 500)
    ax.scatter(x[::step], xn[::step], alpha=0.25, s=8, color=FOREST, edgecolors="none")
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, cal.ar1_alpha + cal.ar1_beta * xl, color=AMBER, lw=2,
            label=f"β={cal.ar1_beta:.4f}  R²={cal.ar1_r_squared:.3f}")
    ax.set_xlabel("log(rate) at t")
    ax.set_ylabel("log(rate) at t+1")
    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    _style(ax, f"AR(1) Fit  |  κ={cal.kappa:.2f}  half-life={cal.half_life_years:.2f} yrs")

    # 3. Rolling mu
    ax = axes[1, 0]
    ax.plot(rolling_df["date"], rolling_df["mu_tce"], color=LIME, lw=2,
            label="Rolling LR mean (5-yr window)")
    ax.axhline(cal.mu_tce, color=AMBER, lw=1.5, ls="--",
               label=f"Full-history: ${cal.mu_tce:,.0f}/day")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"${v/1e3:.0f}k"))
    ax.set_ylabel("Long-run mean ($/day equiv.)")
    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    _style(ax, "Rolling Long-Run Mean — The Window Problem")

    # 4. Rolling kappa + sigma (dual axis)
    ax = axes[1, 1]
    ax2 = ax.twinx()
    l1, = ax.plot(rolling_df["date"], rolling_df["kappa"], color=SAGE, lw=2, label="κ (left)")
    l2, = ax2.plot(rolling_df["date"], rolling_df["sigma"], color=AMBER, lw=2,
                   ls="--", label="σ (right)")
    ax.set_ylabel("κ", color=SAGE)
    ax2.set_ylabel("σ", color=AMBER)
    ax.tick_params(axis="y", colors=SAGE)
    ax2.tick_params(axis="y", colors=AMBER)
    ax2.set_facecolor(PANEL_BG)
    lines = [l1, l2]
    ax.legend(lines, [l.get_label() for l in lines],
              fontsize=8, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    _style(ax, f"Rolling κ and σ  |  Full-history: κ={cal.kappa:.2f}, σ={cal.sigma:.2f}")

    fig.suptitle("Rate Process Calibration — Baltic Panamax Index",
                 color=TEXT, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig
