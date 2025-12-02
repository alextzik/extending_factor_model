"""
Marimo Notebook: 500-Asset Backtest (Last 4 Years)

Downloads historical data for up to 500 US large-cap equities (S&P 500 constituents),
computes simple momentum alphas, runs the backtester from alpha_mod, and visualizes
portfolio NAV, cumulative returns, top average weights, and performance stats using Plotly.

To launch (after installing dependencies with poetry, incl. viz group):
    poetry run marimo edit notebooks/market_backtest.py
or headless run:
    poetry run marimo run notebooks/market_backtest.py
"""

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import math
    import datetime as dt
    import yfinance as yf
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    import pickle

    from extending_factor_model.markowitz.backtester import run_multiple_backtests
    from extending_factor_model.plots import (
        plot_multi_nav,
        plot_multi_cumulative,
        plot_ewma_vol,
    )

    # Constants (defined once; not re-defined in later cells)
    UNIVERSE_SIZE = 500
    YEARS = 10
    TODAY = dt.date.today()
    START_DATE = TODAY - dt.timedelta(days=252 * YEARS + 30)  # buffer
    INITIAL_CAPITAL = 1_000_000.0

    with open("data/processed/assets/returns_df.pkl", "rb") as _f:
        returns_df = pickle.load(_f)

    # Optimizer parameter defaults
    OPT_PARAMS = {
            "target_vol": 0.08,
            "leverage": 1.3, 
            "w_min": 0.001,
            "w_max": 0.0015,
        }

    np.random.seed(42)

    assets = pd.Index(['NEE', 'DPZ', 'WEC', 'INCY', 'MCK', 'HRL', 'FSLR', 'PFE', 'AKAM',
           'CTRA', 'HAS', 'MKC', 'LUMN', 'BMY', 'CMG', 'HUM', 'NWL', 'NI', 'WMT',
           'CMS', 'ETR', 'BDX', 'ABBV', 'MKTX', 'TSN', 'ED', 'DUK', 'PNW', 'HSY',
           'MO', 'AWK', 'ENPH', 'STX', 'GILD', 'XRAY', 'PSA', 'TGT', 'XEL', 'SO',
           'CLX', 'EIX', 'ATO', 'KMB', 'SJM', 'CAG', 'NRG', 'ORLY', 'KR', 'LNT',
           'AAP', 'DGX', 'DG', 'K', 'DLR', 'REGN', 'CNC', 'LLY', 'EA', 'FE', 'CPB',
           'CHTR', 'EVRG', 'ROL', 'VZ', 'NEM', 'DLTR', 'AEE', 'BF-B', 'BAX', 'LHX',
           'VTRS', 'PODD', 'CAH', 'KDP', 'CHRW', 'DVA', 'PGR', 'CHD', 'AEP', 'LMT',
           'TSCO', 'EXR', 'DHR', 'MRK', 'WBD'])
    return (
        INITIAL_CAPITAL,
        OPT_PARAMS,
        START_DATE,
        assets,
        mo,
        pd,
        pickle,
        plot_ewma_vol,
        plot_multi_cumulative,
        plot_multi_nav,
        px,
        returns_df,
        run_multiple_backtests,
    )


@app.cell
def _(pickle):
    Sigmas = {}
    # Sigmas["EIG"] = covariances_by_EIG(returns_df)

    with open("basic_risk_model.pkl", "rb") as _f:
        Sigmas["basic"] = pickle.load(_f)

    with open("extended_risk_model.pkl", "rb") as _f:
        Sigmas["extend"] = pickle.load(_f)
    return (Sigmas,)


@app.cell
def _(
    INITIAL_CAPITAL,
    OPT_PARAMS,
    START_DATE,
    Sigmas,
    assets,
    mo,
    pd,
    returns_df,
    run_multiple_backtests,
):
    # Configure multiple portfolios with different alphas / risk targets
    configs = {}
    for _name, _Sigmas in Sigmas.items():
        configs[_name] = dict(
            returns=returns_df[assets],
            alphas=None,
            risk_models={_key: _Sigmas[_key] for _key in _Sigmas.keys() if _key <pd.to_datetime("2019-01-01")}, 
            initial_capital=INITIAL_CAPITAL,
            start_date=pd.to_datetime(START_DATE),
            markowitz_pars=OPT_PARAMS,
        )

    with mo.status.spinner(f"Running {len(configs)} backtests in parallel …"):
        # Use all available cores; fall back to sequential if only one or issues arise
        try:
            results = run_multiple_backtests(configs, n_jobs=5)
        except Exception as e:
            mo.md(f"Parallel execution failed ({e}); falling back to sequential.")
            results = run_multiple_backtests(configs, n_jobs=1)
    return (results,)


@app.cell
def _(pd, plot_ewma_vol, plot_multi_cumulative, plot_multi_nav, results):
    # Build figures
    nav_fig = plot_multi_nav(results)
    cum_fig = plot_multi_cumulative(results)
    vol_fig = plot_ewma_vol(results, half_life=42)

    # Stats table across portfolios
    rows = []
    for name, res in results.items():
        rows.append(
            {
                'Portfolio': name,
                'Ann Return (%)': 100*res['total_return_ann'],
                'Sharpe': res['sharpe_ratio'],
                'Ann Vol (%)': 100*res['volatility'],
                'Max Drawdown (%)': 100*res['max_drawdown'],
                'Final Value': res['portfolio_values'].iloc[-1],
                "Turn-over (%)": res["turnover"].mean()
            }
        )
    stats_table = pd.DataFrame(rows).set_index('Portfolio')
    return cum_fig, nav_fig, stats_table, vol_fig


@app.cell
def _(mo, px, results):
    # Robust selector for different marimo versions (no mo.ui.select in some).
    portfolio_names = sorted(results.keys())

    def top_weights(port_name: str):
        if port_name not in results:
            return px.bar(x=[], y=[], title="Portfolio not found")
        w = results[port_name]['weights']
        avg_w = w.abs().mean(axis=0).sort_values(ascending=False).head(10)
        fig = px.bar(
            x=avg_w.index,
            y=avg_w.values,
            labels={'x': 'Ticker', 'y': 'Avg Abs Holding'},
            title=f'Top 10 Average Holdings ({port_name})',
        )
        return fig

    label = "Portfolio for Top Weights"
    selector_widget = mo.ui.dropdown(options=portfolio_names, value=portfolio_names[0], label=label)
    return selector_widget, top_weights


@app.cell
def _(
    cum_fig,
    mo,
    nav_fig,
    selector_widget,
    stats_table,
    top_weights,
    vol_fig,
):
    top_weights_reactive = mo.ui.plotly(top_weights(selector_widget.value))

    # Format stats table: keep raw numeric values, define a display formatting function
    def _format_cell(col: str, val):
        if isinstance(val, (int, float)):
            if 0 <= val < 1 and col not in {"Final Value"}:
                return f"{val:.1f}"
            if col == "Final Value":
                return f"{val:,.0f}"
            return f"{val:.1f}"
        return val

    formatted_stats = stats_table.copy()
    for c in formatted_stats.columns:
        formatted_stats[c] = formatted_stats[c].map(lambda v, col=c: _format_cell(col, v))

    stats_table_component = getattr(mo.ui, 'table', None)
    stats_view = stats_table_component(formatted_stats.reset_index())


    tabs = mo.ui.tabs(
        {
            'NAV': mo.ui.plotly(nav_fig),
            'Cumulative': mo.ui.plotly(cum_fig),
            'EWMA Vol': mo.ui.plotly(vol_fig),
            'Stats': stats_view,
            'Top Weights': mo.vstack([selector_widget, top_weights_reactive]),
        }
    )
    mo.vstack([
        mo.md("## Multi-Portfolio Backtest Overview"),
        tabs,
    ])
    return


@app.cell
def _():
    # import pickle
    # with open("results.pkl", "wb") as _f:
    #     pickle.dump(results, _f)
    return


if __name__ == "__main__":
    app.run()
