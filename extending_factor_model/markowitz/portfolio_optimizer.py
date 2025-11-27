"""
Portfolio Optimizer Module

Implements the convex Markowitz portfolio optimization problem.
"""

import numpy as np
import cvxpy as cp
from typing import Optional, Dict, Any
import pandas as pd

def markowitz_problem(
    alpha: pd.Series,
    covariance_matrix: dict,
    prev_cash: float,
    prev_holdings: pd.Series,
    params:dict={
        "target_vol": 0.07,
        "leverage": 1.6, 
        "w_min": 0.001,
        "w_max": 0.0015,
    },
) -> Dict[str, Any]:
    """
    Set up and solve the convex Markowitz portfolio optimization problem.
    
    The objective is to maximize: expected_return - (risk_aversion / 2) * portfolio_variance
    subject to various constraints.
    
    Parameters
    ----------
    alpha : pd.Series
        Alpha signal for each asset (n_assets,)
    covariance_matrix : dict
        Covariance matrix of asset returns 
        Low rank factor F
        Diagonal vector D
        total covariance Sigma
    prev_cash : float
        Previous cash position
    prev_holdings : pd.Series
        Previous portfolio holdings (n_assets,)
    params : dict
        Dictionary of optimization parameters:

    Assumptions
        The optimization variables are in weight space, i.e., fractions of cash value.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'holdings': optimal portfolio holdings (pd.Series)
        - "trades": optimal trades (pd.Series)
        - 'variance': portfolio variance (float)
        - 'volatility': portfolio standard deviation (float)
        - 'objective_value': optimal objective value (float)
        - 'status': solver status (str)

    """
    n_assets = len(prev_holdings.index)
    portfolio_value = prev_cash + prev_holdings.sum()
    
    # Validate inputs
    # Expect a dict with keys 'F_Omega_sqrt','D'
    if not all(k in covariance_matrix for k in ("F_Omega_sqrt","D")):
        raise ValueError("covariance_matrix must contain keys 'F_Omega_sqrt','D'")

    F_Omega_sqrt = covariance_matrix['F_Omega_sqrt']
    if F_Omega_sqrt.shape[0] != n_assets:
        raise ValueError(
            f"Covariance matrix shape {F_Omega_sqrt.shape} does not match number of assets {n_assets}"
        )

    D = covariance_matrix['D']
    if D.shape != (n_assets,):
        raise ValueError(
            f"Diagonal vector shape {D.shape} does not match number of assets {n_assets}"
        )
    
    F_Omega_sqrt = F_Omega_sqrt.reindex(prev_holdings.index).to_numpy()
    D = D.reindex(prev_holdings.index).to_numpy()

    # Define optimization variables for holdings and trades in weight space
    w = cp.Variable(n_assets) # next holdings
    z = cp.Variable(n_assets) # trades
    w_cash = cp.Variable()  # cash position
    z_cash = cp.Variable()  # cash trade
    
    # variance decomposition: F F' + diag(D)
    portfolio_variance = 252*(    cp.sum_squares(F_Omega_sqrt.T @ w) + cp.sum_squares(cp.multiply(np.sqrt(D), w))   )
    
    # Objective
    objective = portfolio_variance 
    
    # Constraints
    constraints = []
    constraints += [cp.sum(w) + w_cash== 1.]
    constraints += [w == prev_holdings.values / portfolio_value + z,
                    w_cash == prev_cash / portfolio_value + z_cash]  # Holdings update
    constraints += [cp.sum(cp.abs(w)) <= params["leverage"]]  # Leverage constraint
    constraints += [w >= params["w_min"]]  # Minimum weight constraint
    constraints += [w <= params["w_max"]]  # Maximum weight constraint
    # constraints += [0.5*cp.norm(z, 1) <= params["turnover_ann"] / 252 / 100] # Turnover constraint (annualized, in %)

    # Define and solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver_path=["CLARABEL", "SCS"])
        
    # Set optimal next holdings and trades
    optimal_holdings = w.value
    optimal_trades = z.value
    optimal_cash = w_cash.value 

    # Calculate portfolio statistics
    portfolio_var = portfolio_variance.value
    portfolio_vol = np.sqrt(portfolio_var)

    # Compute Jacobian of optimal weights w.r.t. alpha
    assert problem.is_dpp()

    return {
        'holdings': pd.Series(optimal_holdings * portfolio_value, index=prev_holdings.index),
        'trades': pd.Series(optimal_trades * portfolio_value, index=prev_holdings.index),
        'cash' : optimal_cash * portfolio_value,
        'variance': portfolio_var,
        'volatility': portfolio_vol,
        'objective_value': problem.value,
        'status': problem.status,
    }
