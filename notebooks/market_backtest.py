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

    from extending_factor_model.backtester import run_multiple_backtests, covariances_by_EIG, covariances_by_KL
    from extending_factor_model.plots import (
        plot_multi_nav,
        plot_multi_cumulative,
        plot_ewma_vol,
    )

    # Constants (defined once; not re-defined in later cells)
    UNIVERSE_SIZE = 500
    YEARS = 2
    TODAY = dt.date.today()
    START_DATE = TODAY - dt.timedelta(days=252 * YEARS + 30)  # buffer
    INITIAL_CAPITAL = 1_000_000.0

    # Optimizer parameter defaults
    OPT_PARAMS = dict(
        long_only=True,
        leverage_limit=2.0,
        target_vol=0.07,
        turnover_ann=2000.
    )

    np.random.seed(42)
    return (
        INITIAL_CAPITAL,
        OPT_PARAMS,
        START_DATE,
        TODAY,
        UNIVERSE_SIZE,
        covariances_by_EIG,
        covariances_by_KL,
        mo,
        pd,
        plot_ewma_vol,
        plot_multi_cumulative,
        plot_multi_nav,
        px,
        run_multiple_backtests,
        yf,
    )


@app.cell
def _(UNIVERSE_SIZE, mo, pd):
    # Fetch S&P 500 ticker list (dynamic). On failure, use fallback list subset.
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp500_df = tables[0]
        raw_symbols = sp500_df['Symbol'].tolist()
        universe = [s.replace('.', '-') for s in raw_symbols][:UNIVERSE_SIZE]
        source_note = "(Fetched dynamically from Wikipedia)"
    except Exception:
        # Expanded static fallback list (approximate S&P 500 constituents; periods replaced by '-')
        fallback = [
            # Top Mega Caps
            "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","BRK-B","LLY","JPM","XOM","AVGO","UNH","V","HD","PG","MA","CVX","ABBV","COST","MRK","PEP","KO","BAC","PFE","ADBE","CRM","NFLX","CSCO","WMT","ACN","TMO","ABT","DIS","INTC","CMCSA","MCD","DHR","LIN","WFC","TXN","NEE","BMY","PM","AMD","VZ","HON","AMGN","UNP","UPS","MS","RTX","IBM","QCOM","SCHW","LOW","SBUX","GS","ORCL","CAT","SPGI","GE","NOW","SYK","CB","BLK","ISRG","PLD","MDT","INTU","LMT","ADI","AMAT","BKNG","AXP","C","BA","DE","TGT","MMC","REGN","PYPL","GILD","ELV","PLTR","SO","CI","ZTS","WM","CL","MO","FDX","MU","PGR","ADP","USB","EQIX","ETN","SHW","EOG","AIG","APD","BDX","ICE","ITW","PH","GM","F","FISV","PANW","LRCX","FICO","MCK","AJG","KLAC","ADSK","MAR","HCA","AON","CDNS","MPC","CME","ORLY","MNST","CTAS","DHI","NKE","EL","ROP","CMG","MSI","AFL","FTNT","NXPI","PSX","PCAR","TEL","ODFL","TRV","SPG","SNPS","KMB","CTSH","GWW","AEP","PXD","AMP","KMI","VLO","DG","TFC","DLR","EXC","MCHP","ADM","PRU","APH","AIG","A","EA","MSCI","ALL","CNC","CSX","ED","EMR","YUM","AEE","STZ","WELL","MTB","HES","PPG","OKE","GLW","COF","DVN","KVUE","FDX","HSY","ENPH","PAYX","DUK","PSA","XEL","OTIS","AVB","SRE","BK","TT","FTV","WBD","NEM","ANET","KR","PEG","ROST","MLM","DTE","HAL","VICI","LVS","ECL","LEN","FAST","URI","AIZ","FTI","VTR","WMB","WEC","DLTR","FSLR","CTRA","NUE","GPN","FDS","IFF","STT","TSCO","ACGL","KDP","MKC","CAH","HIG","CBRE","BLL","PPL","EIX","AWK","EW","ZBH","EXR","GPC","VMC","AVY","CDW","CHD","CMS","RCL","DOW","LDOS","EXPD","WY","DRI","ETSY","HPE","KHC","KEYS","BKR","LUV","APA","ALB","TRGP","NDAQ","BR","PKI","MTD","RMD","CCL","IR","HWM","CTLT","DPZ","ANSS","CINF","HUM","IDXX","BRO","K","RF","CRL","BF-B","SYY","WRB","CARR","LHX","MAA","ZBRA","HOLX","SWK","PFG","JKHY","TSN","HII","JKHY","WHR","NRG","GL","AKAM","FANG","LYB","GRMN","CHTR","CLX","MAS","NVR","PWR","MKTX","HBAN","OMC","GNRC","PTC","WH","RJF","LNT","INCY","VTRS","AKR","AES","ALLE","AMD","APA","BAX","BXP","CAG","CPB","CE","CF","CHRW","CNP","COO","DVA","EQR","ESS","FMC","HAS","HRL","IP","JCI","J","KIM","LEG","LH","LUMN","LW","MGM","MNST","MPWR","MOS","NCLH","NWS","NWSA","NTRS","OGN","O","PARA","PBCT","PNW","PODD","POOL","RHI","ROL","RSG","SEE","SLB","SJM","SNA","SWKS","TXT","UHS","VFC","WRK","XRAY","XYL","ZION",
            # Additional to reach ~500 (some ETFs / placeholders if necessary to fill count)
            "AGR","ATO","BKH","CPT","EVRG","FE","HST","NI","PNR","POM","RRC","TAP","VAR","WAT","WU","BEN","IVZ","LNC","TROW","AMP","DFS","ETR","FRT","IT","JEF","MTG","NWL","PNC","RE","RGA","SNV","STX","TFX","WAT","WDC","XRX","XRAY","XYL","ZBRA","ZION","AAP","BBY","BBWI","BURL","KSS","M","ROST","TJX","DGX","AAL","DAL","UAL","LUV","AER","ALGT","ALK","BA","SPR","TXT","LUV2","EXPE","RCL2"
        ]
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for t in fallback:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        fallback = deduped
        universe = fallback[:UNIVERSE_SIZE]
        source_note = f"(Fallback static subset {len(universe)} tickers)"
    mo.md(f"### Universe Size: {len(universe)} {source_note}")
    return (universe,)


@app.cell
def _(START_DATE, TODAY, mo, pd, universe, yf):
    with mo.status.spinner(f"Downloading OHLCV data for {len(universe)} tickers …"):
        px_data = yf.download(
            universe,
            start=START_DATE,
            end=TODAY,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    # Extract Close prices (yfinance multi-index aware)
    if isinstance(px_data.columns, pd.MultiIndex):
        close = px_data.xs("Close", axis=1, level=1)
    else:  # Single symbol edge case
        close = px_data.to_frame(name=universe[0])
    # Drop assets with any missing data
    close = close.dropna(axis=1, how="any")
    returns_df = close.pct_change().dropna()
    returns_df = returns_df.replace(0.0, 1e-4)
    mo.md(f"Data range: **{returns_df.index.min().date()}** → **{returns_df.index.max().date()}**, final assets: **{returns_df.shape[1]}**")
    return (returns_df,)


@app.cell
def _(covariances_by_EIG, covariances_by_KL, returns_df):
    Sigmas = {}
    Sigmas["EIG"] = covariances_by_EIG(returns_df)
    Sigmas["KL"]  = covariances_by_KL(returns_df, Sigmas["EIG"], 66)
    return (Sigmas,)


@app.cell
def _(returns_df):
    # Build multiple alpha specifications
    # Align indices
    alphas_df = returns_df.rolling(10).mean()
    start_date = returns_df.index.sort_values()[252]
    start_date
    return alphas_df, start_date


@app.cell
def _(
    INITIAL_CAPITAL,
    OPT_PARAMS,
    Sigmas,
    alphas_df,
    mo,
    returns_df,
    run_multiple_backtests,
    start_date,
):
    # Configure multiple portfolios with different alphas / risk targets
    configs = {}
    for _name, _Sigmas in Sigmas.items():
        configs[_name] = dict(
            returns=returns_df,
            alphas=alphas_df,
            risk_models=_Sigmas, 
            initial_capital=INITIAL_CAPITAL,
            start_date=start_date,
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
    selector_widget
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
