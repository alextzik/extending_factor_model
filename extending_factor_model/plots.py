"""Plotting utilities for alpha_mod.

Provides functions to visualize single or multiple portfolio backtest results
including NAV, cumulative returns, and EWMA volatility overlays.

Plotly is used if available; otherwise falls back to matplotlib.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Sequence
import pandas as pd

try:  # Prefer plotly
    import plotly.graph_objects as go
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:  # pragma: no cover
    _HAS_PLOTLY = False

try:  # Fallback
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False


def _color_cycle(n: int) -> list[str]:
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    if n <= len(base):
        return base[:n]
    # repeat with slight transparency if many
    out = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out


def plot_multi_nav(results: Dict[str, Dict[str, Any]]):
    """Plot NAV curves of multiple backtests on one figure.

    Parameters
    ----------
    results : dict
        Mapping portfolio name -> backtest result dict (output of run_backtest())
    """
    nav_df = pd.DataFrame({k: v['portfolio_values'] for k, v in results.items()})
    if _HAS_PLOTLY:  # Interactive
        fig = go.Figure()
        for col in nav_df.columns:
            fig.add_trace(go.Scatter(x=nav_df.index, y=nav_df[col], mode='lines', name=col))
        fig.update_layout(title="Portfolio NAV Comparison", xaxis_title="Date", yaxis_title="Value")
        return fig
    elif _HAS_MPL:  # pragma: no cover
        colors = _color_cycle(len(nav_df.columns))
        for c, col in zip(colors, nav_df.columns):
            plt.plot(nav_df.index, nav_df[col], label=col, color=c)
        plt.title("Portfolio NAV Comparison")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        return plt.gcf()
    else:  # pragma: no cover
        raise RuntimeError("No plotting backend available (plotly or matplotlib).")


def plot_multi_cumulative(results: Dict[str, Dict[str, Any]]):
    cum_df = pd.DataFrame({k: (1 + v['returns']).cumprod() for k, v in results.items()})
    if _HAS_PLOTLY:
        fig = go.Figure()
        for col in cum_df.columns:
            fig.add_trace(go.Scatter(x=cum_df.index, y=cum_df[col], mode='lines', name=col))
        fig.update_layout(title="Cumulative Growth of $1", xaxis_title="Date", yaxis_title="Growth")
        return fig
    elif _HAS_MPL:  # pragma: no cover
        colors = _color_cycle(len(cum_df.columns))
        for c, col in zip(colors, cum_df.columns):
            plt.plot(cum_df.index, cum_df[col], label=col, color=c)
        plt.title("Cumulative Growth of $1")
        plt.xlabel("Date")
        plt.ylabel("Growth")
        plt.legend()
        plt.tight_layout()
        return plt.gcf()
    else:  # pragma: no cover
        raise RuntimeError("No plotting backend available.")


def plot_ewma_vol(results: Dict[str, Dict[str, Any]], half_life: int = 42):
    """Plot EWMA volatility for each portfolio's realized returns.

    If ex_ante_vol exists in each result, it will also be plotted (dashed).
    """
    vol_series = {}
    for name, res in results.items():
        vol_series[name] = res["ewma_volatility"]
    vol_df = pd.DataFrame(vol_series)
    if _HAS_PLOTLY:
        fig = go.Figure()
        for col in vol_df.columns:
            fig.add_trace(go.Scatter(x=vol_df.index, y=vol_df[col], mode='lines', name=f"{col} EWMA Vol"))
        fig.update_layout(title=f"EWMA Volatility (halflife={half_life})", xaxis_title="Date", yaxis_title="Ann. Volatility")
        return fig
    elif _HAS_MPL:  # pragma: no cover
        colors = _color_cycle(len(vol_df.columns))
        for c, col in zip(colors, vol_df.columns):
            plt.plot(vol_df.index, vol_df[col], label=f"{col} EWMA Vol", color=c)
        plt.title(f"EWMA Volatility (halflife={half_life})")
        plt.xlabel("Date")
        plt.ylabel("Ann. Volatility")
        plt.legend()
        plt.tight_layout()
        return plt.gcf()
    else:  # pragma: no cover
        raise RuntimeError("No plotting backend available.")

__all__ = [
    "plot_multi_nav",
    "plot_multi_cumulative",
    "plot_ewma_vol",
]
