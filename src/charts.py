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
DARK_BG   = "#0f1117"
PANEL_BG  = "#1a1d27"
BLUE      = "#4f9cf9"
ORANGE    = "#f97b4f"
GREEN     = "#4ff9a0"
PURPLE    = "#b06cf9"
TEXT      = "#e0e4f0"
GRID      = "#2a2d3a"


def _style(ax, title=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)


def _fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    return fig, ax


def _fmt_M(x, p=None):  return f"${x/1e6:.1f}M"
def _fmt_k(x, p=None):  return f"${x/1e3:.0f}k"
def _fmt_pct(x, p=None): return f"{x:.0%}"


# ── 1. NPV distribution ───────────────────────────────────────────────────────

def plot_npv_distribution(results: SimulationResults, series: str = "equity") -> plt.Figure:
    """Histogram of NPV with P10/median/P90 marked."""
    arr = results.equity_npvs if series == "equity" else results.asset_npvs
    label = "Equity NPV" if series == "equity" else "Asset NPV (Unlevered)"
    colour = BLUE if series == "equity" else GREEN

    fig, ax = _fig(10, 4.5)
    ax.hist(arr / 1e6, bins=80, color=colour, alpha=0.8, edgecolor="none")

    for pct, ls, lbl in [(10, "--", "P10"), (50, "-", "Median"), (90, "--", "P90")]:
        v = np.percentile(arr, pct) / 1e6
        ax.axvline(v, color=ORANGE, lw=1.5, ls=ls, label=f"{lbl}: ${v:.1f}M")

    ax.axvline(0, color="white", lw=1, ls=":", alpha=0.5, label="Break-even")
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
    irrs = results.irrs
    valid = irrs[~np.isnan(irrs)]

    fig, ax = _fig(10, 4.5)
    ax.hist(valid * 100, bins=80, color=PURPLE, alpha=0.8, edgecolor="none")

    median_irr = np.median(valid)
    ax.axvline(median_irr * 100, color=GREEN, lw=2, ls="-",
               label=f"Median IRR: {median_irr:.1%}")
    ax.axvline(hurdle_rate * 100, color=ORANGE, lw=2, ls="--",
               label=f"Hurdle rate: {hurdle_rate:.0%}")
    ax.axvline(0, color="white", lw=1, ls=":", alpha=0.5)

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
    Fan chart with faint individual paths drawn first as background texture,
    then the distributional envelope and median rendered clearly on top.
    The paths imply the simulation ran; the fan is what the viewer reads.
    """
    rate_matrix = results.annual_rate_matrix
    n_years = results.holding_years
    years = np.arange(1, n_years + 1)

    fig, ax = _fig(10, 4.5)

    # ── Faint individual paths (drawn first, sit behind everything) ───────────
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rate_matrix), size=min(n_sample_paths, len(rate_matrix)), replace=False)
    for path_idx in idx:
        ax.plot(years, rate_matrix[path_idx] / 1e3,
                lw=0.6, alpha=0.10, color=BLUE, zorder=1)

    # ── Distributional envelope (on top) ─────────────────────────────────────
    p = {pct: np.percentile(rate_matrix, pct, axis=0) / 1e3 for pct in [10, 25, 50, 75, 90]}
    ax.fill_between(years, p[10], p[90], alpha=0.15, color=BLUE, label="P10–P90", zorder=2)
    ax.fill_between(years, p[25], p[75], alpha=0.35, color=BLUE, label="P25–P75", zorder=3)
    ax.plot(years, p[50], color=BLUE, lw=2.5, label="Median", zorder=4)

    if longrun_mean:
        ax.axhline(longrun_mean / 1e3, color=GREEN, lw=1.2, ls="--",
                   alpha=0.8, label=f"LR mean ${longrun_mean/1e3:.0f}k", zorder=5)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:.0f}k"))
    ax.set_xlabel("Year")
    ax.set_ylabel("TCE Rate ($/day)")
    ax.set_xticks(years)
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
        (axes[0], results.annual_ebitda_matrix,  GREEN,  "Annual EBITDA"),
        (axes[1], results.annual_fcf_matrix,     ORANGE, "Annual Levered FCF"),
    ]:
        data = matrix / 1e6
        p10 = np.percentile(data, 10, axis=0)
        p50 = np.percentile(data, 50, axis=0)
        p90 = np.percentile(data, 90, axis=0)

        ax.fill_between(years, p10, p90, alpha=0.2, color=colour)
        ax.plot(years, p50, color=colour, lw=2.5, label="Median")
        ax.plot(years, p10, color=colour, lw=1, ls="--", alpha=0.6, label="P10/P90")
        ax.plot(years, p90, color=colour, lw=1, ls="--", alpha=0.6)
        ax.axhline(0, color="white", lw=0.8, ls=":", alpha=0.4)

        if drydock_years:
            for ddy in drydock_years:
                if ddy <= n_years:
                    ax.axvline(ddy, color=BLUE, lw=1, ls=":", alpha=0.5,
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
    fig, ax = _fig(12, max(3.5, len(tornado_df) * 0.8))

    params = tornado_df.index.tolist()
    low_d  = tornado_df["low_delta"].values  / 1e6
    high_d = tornado_df["high_delta"].values / 1e6

    for i, (lo, hi) in enumerate(zip(low_d, high_d)):
        ax.barh(i, lo,  color=ORANGE, alpha=0.85, height=0.5)
        ax.barh(i, hi,  color=BLUE,   alpha=0.85, height=0.5)
        ax.text(-0.2, i, f"${lo:+.1f}M", ha="right", va="center",
                color=TEXT, fontsize=7.5)
        ax.text(0.2,  i, f"${hi:+.1f}M", ha="left",  va="center",
                color=TEXT, fontsize=7.5)

    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params, color=TEXT, fontsize=9)
    ax.axvline(0, color="white", lw=1.2, alpha=0.7)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:+.1f}M"))
    ax.set_xlabel("Δ from base median equity NPV ($M)")
    ax.text(0.98, 0.97, f"Base median: ${base_median/1e6:.1f}M",
            transform=ax.transAxes, ha="right", va="top",
            color=GREEN, fontsize=9, fontweight="bold")
    _style(ax, "Sensitivity — Impact on Median Equity NPV  (orange=downside, blue=upside)")
    fig.tight_layout()
    return fig


# ── 6. MOIC distribution ─────────────────────────────────────────────────────

def plot_moic_distribution(results: SimulationResults) -> plt.Figure:
    """Distribution of equity multiple of invested capital."""
    moic = results.moic_distribution()
    valid = moic[~np.isnan(moic)]

    fig, ax = _fig(9, 4)
    ax.hist(valid, bins=80, color=GREEN, alpha=0.8, edgecolor="none")
    ax.axvline(1.0, color="white", lw=1.5, ls=":", alpha=0.6, label="1× (return of capital)")
    ax.axvline(np.median(valid), color=ORANGE, lw=2, ls="-",
               label=f"Median: {np.median(valid):.2f}×")

    prob_above_1 = np.mean(valid > 1.0)
    ax.set_xlabel("MOIC (×)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    _style(ax, f"MOIC Distribution  |  P(>1×): {prob_above_1:.0%}")
    fig.tight_layout()
    return fig


# ── 7. Freight vs scrap scatter (correlation check) ───────────────────────────

def plot_freight_scrap_scatter(
    terminal_rates: np.ndarray,
    terminal_scrap: np.ndarray,
    sample_size: int = 1000,
) -> plt.Figure:
    """Scatter of terminal-year TCE rate vs implied vessel scrap value."""
    idx = np.random.choice(len(terminal_rates), min(sample_size, len(terminal_rates)), replace=False)
    x = terminal_rates[idx] / 1e3
    y = terminal_scrap[idx]

    fig, ax = _fig(7, 4)
    ax.scatter(x, y, alpha=0.3, s=12, color=PURPLE, edgecolors="none")

    # Regression line
    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), color=ORANGE, lw=2, label="OLS fit")

    corr = np.corrcoef(x, y)[0, 1]
    ax.set_xlabel("Terminal TCE Rate ($/day, $k)")
    ax.set_ylabel("Scrap Price ($/LDT)")
    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    _style(ax, f"Freight–Scrap Correlation  |  ρ={corr:.2f} (realised in simulation)")
    fig.tight_layout()
    return fig
