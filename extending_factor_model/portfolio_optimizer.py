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
    portfolio_value: float,
    prev_holdings: pd.Series,
    params:dict={
        "long_only": True,
        "leverage_limit": 2.0,
        "target_vol": 0.07,
        "turnover_ann": 2000.0,  # annualized turnover in percentage
    },
    grad_bool: bool = False,
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
    portfolio_value : float
        Total portfolio value 
    prev_holdings : pd.Series
        Previous portfolio holdings (n_assets,)
    params : dict
        Dictionary of optimization parameters:
    
        long_only : bool
            If True, constrain weights to be non-negative (default: True)
        leverage_limit : float
            Maximum sum of absolute weights
        target_vol : float
            Target portfolio volatility (standard deviation) annualized
        turnover_ann : float
            Maximum annualized turnover in percentage (e.g., 2000 for 2000%)
    grad_bool : bool
        If True, set up the problem for gradient computation using cvxpylayers.

    Assumptions
        The optimization variables are in weight space, i.e., fractions of cash value.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'holdings': optimal portfolio holdings (pd.Series)
        - 'expected_return': expected portfolio return (float)
        - 'variance': portfolio variance (float)
        - 'volatility': portfolio standard deviation (float)
        - 'objective_value': optimal objective value (float)
        - 'status': solver status (str)
        - 'problem': the cvxpy Problem object for advanced usage

    """
    n_assets = len(alpha.index)
    
    # Validate inputs
    # Expect a dict with keys 'Sigma','F','D'
    if not all(k in covariance_matrix for k in ("Sigma","F","D")):
        raise ValueError("covariance_matrix must contain keys 'Sigma','F','D'")
    Sigma = covariance_matrix['Sigma']
    if Sigma.shape != (n_assets, n_assets):
        raise ValueError(
            f"Covariance matrix shape {Sigma.shape} does not match number of assets {n_assets}"
        )

    # Define optimization variables for holdings and trades in weight space
    w = cp.Variable(n_assets) # next holdings
    z = cp.Variable(n_assets) # trades
    
    # Portfolio return-like term
    alpha_cp = cp.Parameter(n_assets)
    alpha_cp.value = alpha.values
    portfolio_return = alpha_cp @ w
    
    # Portfolio variance (risk)
    F = covariance_matrix['F']
    D = covariance_matrix['D']
    # variance decomposition: F F' + diag(D)
    portfolio_variance = cp.sum_squares(F.T @ w) + cp.sum_squares(cp.multiply(np.sqrt(D), w))
    
    # Objective
    objective = portfolio_return 
    
    # Constraints
    constraints = []
    # Long-only constraint
    if params["long_only"]:
        constraints += [w >= 0.]
    else: # long-short
        constraints += [cp.sum(w) == 0.]
    constraints += [w == prev_holdings.values / portfolio_value + z]  # Holdings update
    constraints += [cp.sum(cp.abs(w)) <= params["leverage_limit"]]  # Leverage constraint
    constraints += [252*portfolio_variance <= params["target_vol"]**2]  # Target volatility
    # constraints += [0.5*cp.norm(z, 1) <= params["turnover_ann"] / 252 / 100] # Turnover constraint (annualized, in %)

    # Define and solve the problem
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver_path=["CLARABEL", "SCS"])
        
    # Set optimal next holdings and trades
    optimal_holdings = w.value
    optimal_trades = z.value

    # Calculate portfolio statistics
    portfolio_var = 252*portfolio_variance.value
    portfolio_vol = np.sqrt(portfolio_var)

    # Compute Jacobian of optimal weights w.r.t. alpha
    assert problem.is_dpp()

    return {
        'holdings': pd.Series(optimal_holdings * portfolio_value, index=alpha.index),
        'trades': pd.Series(optimal_trades * portfolio_value, index=alpha.index),
        'variance': portfolio_var,
        'volatility': portfolio_vol,
        'objective_value': problem.value,
        'status': problem.status,
    }
