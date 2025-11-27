"""
Backtester Module

Implements portfolio backtesting functionality to evaluate performance over time.
"""

####################################################
# Imports
import numpy as np
import pandas as pd
from typing import Callable, Optional, Dict, Any, Iterable
from .portfolio_optimizer import markowitz_problem
from extending_factor_model.risk_models.compute_risk_model_statistical import _ewma_series, _beta_from_half_life
from copy import deepcopy

####################################################
# Backtest Implementation
def run_backtest(
    returns: pd.DataFrame,
    alphas: pd.DataFrame,
    risk_models: dict, 
    initial_capital: float = 1_000_000.0,
    start_date: pd.Timestamp=None,
    markowitz_pars: dict = {
        "target_vol": 0.07,
        "leverage": 1.6, 
        "w_min": 0.01,
        "w_max": 0.015,
    }
) -> Dict[str, Any]:
    """
    Run a backtest of a portfolio strategy using the Markowitz optimizer with a
    two-step IEWMA (innovations EWMA) covariance estimator.

    The backtester iteratively:
      1. Uses historical data to estimate a IEWMA covariance matrix.
      2. Optionally modifies alpha signals via `alpha_modifier`.
      3. Solves the Markowitz optimization problem.
      4. Executes trades and tracks portfolio performance.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns data with datetime index and assets as columns.
    alphas : pd.DataFrame
        Alpha signals with datetime index and assets as columns.
        alpha[t] determines h_{t+1} which is acted upon by r_{t+1}
    risk_models : dict
        Precomputed covariance matrices for each date.
    initial_capital : float, optional
        Initial portfolio value (default: 1,000,000).
    start_date : datetime
        Start date for the backtest 
    markowitz_pars : dict
        Additional arguments passed to markowitz_problem().

    Assumption:
        Any quantity indexed by t is assumed to be known at time t.

    Returns
    -------
    dict
        - 'portfolio_values': time series of portfolio values (pd.Series)
        - 'returns': time series of portfolio returns (pd.Series)
        - 'weights': time series of portfolio weights per asset (pd.DataFrame)
        - 'turnover': time series of portfolio turnover (pd.Series)
        - 'total_return': cumulative return (float)
        - 'sharpe_ratio': annualized Sharpe ratio (float)
        - 'max_drawdown': maximum drawdown (float)
        - 'volatility': annualized volatility (float)
        - 'transactions': number of rebalancing events (int)
    """
    # Validate inputs
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a pandas DataFrame")
    # Ensure alphas index equals returns index
    if alphas is not None:
        if not all(alphas.index.isin(returns.index)):
            raise ValueError("Alphas index must be in returns index")
        # Ensure assets match
        if not (returns.columns == alphas.columns).all():
            raise ValueError("returns and alphas must have the same columns (assets)")
    
    # Constants
    asset_names = returns.columns

    # Tracking containers
    portfolio_values = []
    portfolio_holdings_history = []
    portfolio_returns = []
    turnover_history = []

    # Set initial portfolio
    cash_value = 0.0
    current_holdings = pd.Series(initial_capital / len(asset_names), index=asset_names)
    portfolio_value = initial_capital
    portfolio_values.append(portfolio_value)
    portfolio_holdings_history.append(current_holdings.to_numpy().copy())
    turnover_history.append(0.0)

    # Get dates to backtest
    # Ensure chronological order and apply burn-in
    dates_to_backtest = list(risk_models.keys())

    for date, next_date in zip(dates_to_backtest[:-1], dates_to_backtest[1:]):

        Sigma_t = risk_models[date]  # dict with risk model per date

        ############################################################
        # Solve Markowitz
        markowitz_output = markowitz_problem(
            alpha=alphas.loc[date] if alphas is not None else None,
            covariance_matrix=Sigma_t,
            prev_cash=cash_value,
            prev_holdings=current_holdings,
            params=markowitz_pars,
        )

        ############################################################
        current_holdings = markowitz_output['holdings']
        trades = markowitz_output['trades']
        cash_value = markowitz_output["cash"]

        # Update portfolio value and weights
        asset_returns = returns.loc[next_date]
        current_holdings = (1 + asset_returns) * current_holdings
        portfolio_value = cash_value + current_holdings.sum()

        # Record
        portfolio_values.append(portfolio_value)
        denom = portfolio_value
        denom = denom if denom != 0 else 1.0
        portfolio_holdings_history.append(current_holdings.to_numpy().copy())
        turnover_history.append(np.abs(trades).sum() / denom * 252 * 100)

    # Metrics
    portfolio_values_series = pd.Series(portfolio_values, index=dates_to_backtest)
    portfolio_returns_series = portfolio_values_series.pct_change().dropna()
    holdings_df = pd.DataFrame(portfolio_holdings_history, index=dates_to_backtest, columns=asset_names)

    total_return = (portfolio_value - initial_capital) / initial_capital
    total_ret_ann = (portfolio_value / initial_capital) ** (252 / len(portfolio_values_series)) - 1
    std_return = portfolio_returns_series.std() * np.sqrt(252)
    sharpe_ratio = (total_ret_ann / std_return)  if std_return > 0 else 0.0

    cumulative = (1 + portfolio_returns_series).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.abs(drawdown.min())

    ewma_portfolio_vol = np.sqrt(252) * np.sqrt(
        pd.Series(_ewma_series(portfolio_returns_series**2, _beta_from_half_life(42)), 
                  index=portfolio_returns_series.index)
        )
    ewma_portfolio_vol = ewma_portfolio_vol[42:]

    return {
        'portfolio_values': portfolio_values_series,
        'returns': portfolio_returns_series,
        'weights': holdings_df,
        'turnover': pd.Series(turnover_history, index=dates_to_backtest),
        'total_return': total_return,
        'total_return_ann': total_ret_ann,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'volatility': std_return,
        "ewma_volatility": ewma_portfolio_vol,
        'transactions': len(dates_to_backtest) - 1,
    }


def run_multiple_backtests(
    configs: Dict[str, Dict[str, Any]],
    shared_returns: Optional[pd.DataFrame] = None,
    shared_alphas: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    n_jobs: int = 1,
    backend: str = "loky",
    verbose: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Run several backtests and collect results.

    Parameters
    ----------
    configs : dict
        Mapping portfolio name -> configuration dictionary. Each config can contain any
        keyword arguments accepted by run_backtest. If 'returns' or 'alphas' are missing
        they will fall back to shared_returns / shared_alphas if provided.
    shared_returns : pd.DataFrame, optional
        Returns DataFrame used when individual config does not provide its own.
    shared_alphas : pd.DataFrame, optional
        Alphas DataFrame used when individual config does not provide its own.
    progress_callback : callable, optional
        Function called with portfolio name before each run (for UI progress bars, etc.).

    Additional Parameters
    ---------------------
    n_jobs : int, default 1
        If >1, run backtests in parallel using joblib. Use -1 to use all CPUs.
    backend : str, default 'loky'
        joblib backend ('loky','threading','multiprocessing'). CPU-bound code should prefer 'loky'.
    verbose : int, default 0
        Verbosity level passed to joblib. Higher prints progress.

    Returns
    -------
    dict
        Mapping portfolio name -> backtest result dictionary.
    """
    # Fast path: sequential execution
    if n_jobs == 1:
        results: Dict[str, Dict[str, Any]] = {}
        for name, cfg in configs.items():
            cfg = dict(cfg)  # shallow copy
            r = cfg.pop('returns', shared_returns)
            a = cfg.pop('alphas', shared_alphas)
            if r is None:
                raise ValueError(f"Missing returns for portfolio '{name}'.")
            if progress_callback:
                progress_callback(name)
            results[name] = run_backtest(returns=r, alphas=a, **cfg)
        return results

    # Parallel path
    from joblib import Parallel, delayed

    # Materialize argument list to avoid closure capture issues in joblib
    tasks = []
    for name, cfg in configs.items():
        cfg_copy = dict(cfg)
        r = cfg_copy.pop('returns', shared_returns)
        a = cfg_copy.pop('alphas', shared_alphas)
        if r is None:
            raise ValueError(f"Missing returns for portfolio '{name}'.")
        tasks.append((name, r, a, cfg_copy))

    def _run_one(name, r, a, cfg_local):
        if progress_callback:
            progress_callback(name)
        return name, run_backtest(returns=r, alphas=a, **cfg_local)

    parallel_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_run_one)(name, r, a, cfg_local) for (name, r, a, cfg_local) in tasks
    )
    return {name: res for name, res in parallel_results}