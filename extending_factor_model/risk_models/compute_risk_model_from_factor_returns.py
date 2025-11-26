"""
Module for computing a basic risk model from factor returns.

We compute a risk model F Omega F^T + D, where F is the factor loadings matrix, Omega is the factor covariance matrix,
and D is the specific risk diagonal matrix, using regression techniques.


"""

####################################################
# Imports
from curses import window
import numpy as np
import pandas as pd
import cvxpy as cp

####################################################
# Risk Model from Factor Returns
def compute_risk_model_given_factor_returns(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    halflife: int = 126,
) -> pd.DataFrame:
    """
    We compute a risk model given factor returns:

                    F Omega F^T + D

    Steps:
    1. Compute factor exposures F via linear regression of asset returns on factor returns.
            minimize_F Σ_t w_t ||r_t - F rf_t||^2,
         where w_t are exponential weights with given halflife (EWMA)
    2. Compute covariance of factor returns G using the same EWMA half-life
    3. Compute specific risk D as the diagonal matrix of residual variances from the regression, using the same EWMA half-life.

    Args:
        asset_returns (pd.DataFrame): Asset returns.
                        Assset 1  ...  Asset n  
            Index
            Date 1
            ...
            Date k
        factor_returns (pd.DataFrame): Factor returns.
                        Factor 1  ...  Factor m  
            Index
            Date 1
            ...
            Date k

        halflife (int, optional): Halflife for the exponential weights. Defaults to 126.

    Returns:
        pd.DataFrame: Factor exposure (F).
                        Factor 1  ...  Factor m
        Index   
        Asset 1
        ...
        Asset n

        pd.DataFrame: Factor covariance matrix (Omega).
                        Factor 1  ...  Factor m
        Factor 1
        ...
        Factor m    

        pd.Series: Specific risk diagonal matrix (D).
                        Asset 1  ...  Asset n

        pd.DataFrame: residual returns
                        Asset 1  ...  Asset n
        Index   
        Date 1
        ...
        Date k
    """

    ########################################################
    # Raise Value error if dates do not match
    if not asset_returns.index.equals(factor_returns.index):
        raise ValueError("Asset returns and factor returns must have the same dates.")
    
    # Sort both dataframes by date
    asset_returns = asset_returns.sort_index()
    factor_returns = factor_returns.sort_index()

    ########################################################
    n = asset_returns.shape[1]
    m = factor_returns.shape[1]
    factor_exposures = cp.Variable((n, m))

    decay_factor = np.exp(-np.log(2) / halflife)
    weights = np.array([decay_factor**i for i in range(len(asset_returns))])
    weights = weights / weights.sum()
    weights = weights[::-1].reshape(-1, 1)

    # Compute factor exposures via weighted least squares
    residuals = asset_returns.to_numpy() - factor_returns.to_numpy() @ factor_exposures.T
    objective = cp.sum(cp.multiply(weights, cp.square(residuals)))

    problem = cp.Problem(cp.Minimize(objective))
    problem.solve()

    F = pd.DataFrame(
        factor_exposures.value, index=asset_returns.columns, columns=factor_returns.columns
    )

    residuals_df = pd.DataFrame(
        residuals.value, index=asset_returns.index, columns=asset_returns.columns
    )

    # Compute factor covariance G using EWMA
    factor_returns_np = factor_returns.to_numpy()
    weighted_factor_returns = factor_returns_np * np.sqrt(weights)
    Omega = pd.DataFrame((weighted_factor_returns.T @ weighted_factor_returns), index=factor_returns.columns, columns=factor_returns.columns)
    # Compute specific risk D using EWMA
    residuals_np = residuals.value
    weighted_residuals = residuals_np * np.sqrt(weights)
    D = pd.Series(np.sum(weighted_residuals**2, axis=0), index=asset_returns.columns)

    return F, Omega, D, residuals_df


####################################################
# Compute risk models over time
def compute_risk_models_over_time_given_factor_returns(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    halflife: int = 63,
    burnin: int = 126,
) -> dict:
    """
    Compute causal risk models over time using a rolling window approach.

    Args:
        asset_returns (pd.DataFrame): Asset returns.
                        Assset 1  ...  Asset n  
            Index
            Date 1
            ...
            Date k
        factor_returns (pd.DataFrame): Factor returns.
                        Factor 1  ...  Factor m  
            Index
            Date 1
            ...
            Date k

        halflife (int, optional): Halflife for the exponential weights. Defaults to 126.
    
    Returns:
        dict: Dictionary of risk models with keys as end dates of the windows.
            For each key, the risk model can be used at any date >=key.
    """
    cov_dict = {}
    dates = asset_returns.index

    for end_date in dates[burnin:]:
        print(f"Computing risk model for date {end_date}")
        start_date = max(dates[0], end_date - pd.Timedelta(days=3*halflife))
        window_asset_returns = asset_returns.loc[start_date:end_date]
        window_factor_returns = factor_returns.loc[start_date:end_date]

        F, Omega, D, _ = compute_risk_model_given_factor_returns(
            window_asset_returns, window_factor_returns, halflife
        )

        cov_dict[end_date] = {
            "F": F,
            "Omega": Omega,
            "D": D,
            "Sigma": pd.DataFrame(F.values @ Omega.values @ F.values.T + np.diag(D.values), 
                                  index=asset_returns.columns, 
                                  columns=asset_returns.columns),
        }

    return cov_dict