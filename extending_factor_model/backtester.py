"""
Backtester Module

Implements portfolio backtesting functionality to evaluate performance over time.
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional, Dict, Any, Iterable
from .portfolio_optimizer import markowitz_problem
from scipy.sparse.linalg import svds
from copy import deepcopy


def _beta_from_half_life(H: int) -> float:
    """Convert half-life H to EWMA decay beta.
    beta = exp(log(0.5)/H)
    """
    if H <= 0:
        raise ValueError("Half-life H must be positive")
    return float(np.exp(np.log(0.5) / H))

def _ewma_series(series, beta, start_val=None) -> np.ndarray:
    """Compute causal EWMA estimates for a 1D timeseries.
    """

    var_ewma = np.zeros_like(series)
    var_ewma[0] = start_val if start_val is not None else series[0]

    for t in range(1, len(series)):
        alpha_t = (1 - beta) / (1 - beta**(t + 1))
        alpha_t_minus = (1 - beta) / (1 - beta**t)
        var_ewma[t] = beta * (alpha_t / alpha_t_minus) * var_ewma[t - 1] + alpha_t * series[t]

    return var_ewma

def _ewma_cov(matrix, beta) -> list[np.ndarray]:
    """Compute causal EWMA covariance estimates for X (rows=time, cols=assets).
    """
    vec_0 = matrix[0].reshape(-1,1)
    cov = vec_0 @ vec_0.T
    cov_list = [cov]

    for t in range(1, matrix.shape[0]):
        vec_t = matrix[t].reshape(-1,1)
        outer = vec_t @ vec_t.T

        alpha_t = (1 - beta) / (1 - beta**(t + 1))
        alpha_t_minus = (1 - beta) / (1 - beta**t)


        cov = beta * (alpha_t / alpha_t_minus) * cov + alpha_t * outer
        cov_list.append(cov.copy())

    return cov_list

def covariances_by_EIG(df_returns, Hvol=42, Hcor=126, rank=10) -> dict:
    """Iterated EWMA covariance estimator + eigenvalue decomposition.

    Parameters
    ----------
    df_returns : pd.DataFrame
        DataFrame of asset returns with datetime index and assets as columns.
    Hvol : int, optional
        Half-life for volatility estimation (default: 42 days).
    Hcor : int, optional
        Half-life for correlation estimation (default: 126 days).
    rank : int, optional
        Number of top eigenvalues to retain in low-rank approximation (default: 10).

    Returns
    -------
    dict
        Dictionary containing the estimated covariance matrix and its low-rank approximation.
        keys are dates from df_returns.index.
    """
    # Validate input
    if not isinstance(df_returns, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    # Compute EWMA decay factors
    beta_vol = _beta_from_half_life(Hvol)
    beta_cor = _beta_from_half_life(Hcor)

    returns = df_returns.values
    T, n = returns.shape

    # 1. Compute EWMA volatilities for each asset across time
    vol_ewma = np.zeros_like(returns)
    for j in range(n):
        vol_ewma[:, j] = np.sqrt(_ewma_series(returns[:, j]**2, beta_vol))

    # 2. Standardize returns
    standardized = returns / vol_ewma

    # 3. EWMA covariance of standardized returns (i.e., correlation step)
    stand_covs = _ewma_cov(standardized, beta_cor)

    # 4. Get covariances
    covariances = {}
    for t in range(T):
        D_t = np.diag(vol_ewma[t])
        R_t = stand_covs[t]
        cov_t = D_t @ R_t @ D_t

        D = np.sqrt(np.diag(cov_t))
        corr_t = np.diag(1/D) @ cov_t @ np.diag(1/D)

        covariances[df_returns.index[t]] = {}
        covariances[df_returns.index[t]]["Sigma"] = cov_t

        if rank is not None:
            # Low-rank approximation
            U, S, V = svds(corr_t, k=min(rank, n-1))
            F = U @ np.diag(np.sqrt(S))
            D_t = np.maximum(np.diag(np.eye(F.shape[0]) - F @ F.T), 0)
            F_cov = np.diag(D) @ F
            D_cov = np.diag( np.diag(D) @ np.diag(D_t) @ np.diag(D) )
            covariances[df_returns.index[t]]["F"] = F_cov
            covariances[df_returns.index[t]]["D"] = D_cov

    return covariances

def covariances_by_KL(df_returns: pd.DataFrame, Sigma_dict: dict, burnin:int, H:int=126) -> dict:
    """
    Runs the expecation-maximization algorithm to estimate the time-varying covariance matrices for a 
    dataframe of returns.

    @ t uses returns up to and including t to estimate Sigma_t.

    Args:
    - df_returns (pd.DataFrame): DataFrame of asset returns with datetime index and assets as columns.
    - Sigma_dict (dict): Initial covariance matrices for each time point.
    - burnin (int): Number of initial periods to skip for covariance estimation.
    - H (int): Half-life for EWMA decay.

    Returns:
    - dict: Dictionary of estimated covariance matrices using EM
    
    """

    ### Checks
    if set(Sigma_dict.keys()) != set(df_returns.index.unique()):
        raise ValueError("Sigma_dict keys must match df_returns index")
    
    ### Initializations
    Sigma_em_dict = deepcopy(Sigma_dict)
    beta = _beta_from_half_life(H)

    ### Get IEWMA covariances as starting point
    IEWMA_returns = _ewma_cov(df_returns.values, beta)
    IEWMA_returns = dict(zip(df_returns.index.unique(), IEWMA_returns))

    ######################################################################
    ### Initialize EM for the first date
    _init_date = df_returns.index.unique()[burnin]

    F_prev = Sigma_dict[_init_date]['F']
    D_prev = Sigma_dict[_init_date]['D']
    inv_D_prev = 1 / D_prev
    G_prev = np.linalg.inv( F_prev.T * inv_D_prev[None, :] @ F_prev + np.eye(F_prev.shape[1])  )

    ######################################################################
    # Perform EM for dates
    for date in df_returns.index.unique()[burnin:]:

        if date == _init_date:
            num_iters = 1_000
        else:
            num_iters = 20

        C_rr = IEWMA_returns[date]

        for _ in range(num_iters):
            C_rs = (C_rr * inv_D_prev[None, :]) @ F_prev @ G_prev

            tmp_prev = (F_prev.T * inv_D_prev[None, :]) @ C_rr @ (F_prev * inv_D_prev[:, None])
            C_ss = G_prev + G_prev @ tmp_prev @ G_prev

            # Now solve
            F_prev = np.linalg.solve( C_ss.T, C_rs.T ).T

            D_prev = np.diag(C_rr) - 2 * np.sum(C_rs * F_prev, axis=1) + np.sum(F_prev * (F_prev @ C_ss), axis=1)
            inv_D_prev = 1 / D_prev

            G_prev = np.linalg.inv( F_prev.T * inv_D_prev[None, :] @ F_prev + np.eye(F_prev.shape[1]) )

        Sigma_em_dict[date]["F"] = F_prev
        Sigma_em_dict[date]["D"] = D_prev
        Sigma_em_dict[date]["Sigma"] = F_prev @ F_prev.T + np.diag(D_prev)
        Sigma_em_dict[date]["C_rr"] = C_rr

    return Sigma_em_dict



def run_backtest(
    returns: pd.DataFrame,
    alphas: pd.DataFrame,
    risk_models: dict, 
    initial_capital: float = 1_000_000.0,
    start_date: pd.Timestamp=None,
    markowitz_pars: dict = {
        "long_only": True,
        "leverage_limit": 2.0,
        "target_vol": 0.07,
        "turnover_ann": 2000.
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
    cash_value = initial_capital
    portfolio_value = initial_capital
    current_holdings = pd.Series(0.0, index=asset_names)
    portfolio_values.append(portfolio_value)
    portfolio_returns.append(0.0)
    portfolio_holdings_history.append(current_holdings.to_numpy().copy())
    turnover_history.append(0.0)

    # Get dates to backtest
    # Ensure chronological order and apply burn-in
    if start_date is not None:
        dates_to_backtest = returns.index[returns.index >= start_date].sort_values()
    else:
        dates_to_backtest = returns.index.sort_values()[128:]

    for date, next_date in zip(dates_to_backtest[:-1], dates_to_backtest[1:]):

        Sigma_t = risk_models[date]  # dict with Sigma,F,D

        ############################################################
        # Solve Markowitz
        markowitz_output = markowitz_problem(
            alpha=alphas.loc[date],
            covariance_matrix=Sigma_t,
            portfolio_value=portfolio_value,
            prev_holdings=current_holdings,
            params=markowitz_pars,
            grad_bool=True
        )

        ############################################################
        current_holdings = markowitz_output['holdings']
        trades = markowitz_output['trades']

        # Update portfolio value and weights
        asset_returns = returns.loc[next_date]
        current_holdings = (1 + asset_returns) * current_holdings
        cash_value -= trades.sum()
        portfolio_value = cash_value + current_holdings.sum()

        # Record
        portfolio_values.append(portfolio_value)
        denom = cash_value + current_holdings.sum()
        denom = denom if denom != 0 else 1.0
        portfolio_return = (current_holdings * asset_returns).sum() / denom
        portfolio_returns.append(portfolio_return)
        portfolio_holdings_history.append(current_holdings.to_numpy().copy())
        turnover_history.append(np.abs(trades).sum() / denom * 252 * 100)

    # Metrics
    portfolio_values_series = pd.Series(portfolio_values, index=dates_to_backtest)
    portfolio_returns_series = pd.Series(portfolio_returns, index=dates_to_backtest)
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
            if r is None or a is None:
                raise ValueError(f"Missing returns or alphas for portfolio '{name}'.")
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
        if r is None or a is None:
            raise ValueError(f"Missing returns or alphas for portfolio '{name}'.")
        tasks.append((name, r, a, cfg_copy))

    def _run_one(name, r, a, cfg_local):
        if progress_callback:
            progress_callback(name)
        return name, run_backtest(returns=r, alphas=a, **cfg_local)

    parallel_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_run_one)(name, r, a, cfg_local) for (name, r, a, cfg_local) in tasks
    )
    return {name: res for name, res in parallel_results}


def zscore(x, eps=1e-8):
    """
        Differentiable z-score normalization of 1D tensors.
    
    """
    mean = x.mean()
    std = x.std() + eps
    return (x - mean) / std