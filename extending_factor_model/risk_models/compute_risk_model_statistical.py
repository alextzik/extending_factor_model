"""

We compute a risk model F G F^T + D, where F is the factor loadings matrix, G is the factor covariance matrix,
and D is the specific risk diagonal matrix, using statistical techniques.

"""

####################################################
# Imports
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.linalg import block_diag, eigh
from copy import deepcopy

####################################################
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


####################################################
# Statistical Risk Model Estimation

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
        D_t = np.diag(np.nan_to_num(vol_ewma[t], nan=0.0))
        R_t = np.nan_to_num(stand_covs[t], nan=0.0)
        cov_t = D_t @ R_t @ D_t

        D = np.sqrt(np.diag(cov_t))
        corr_t = np.diag(1/D) @ cov_t @ np.diag(1/D)

        covariances[df_returns.index[t]] = {}
        covariances[df_returns.index[t]]["Sigma"] = pd.DataFrame(cov_t, index=df_returns.columns, columns=df_returns.columns)

        if rank is not None:
            # Low-rank approximation
            U, S, V = svds(corr_t, k=min(rank, n-1))
            F = U @ np.diag(np.sqrt(S))
            D_t = np.maximum(np.diag(np.eye(F.shape[0]) - F @ F.T), 0)
            F_cov = np.diag(D) @ F
            D_cov = np.diag( np.diag(D) @ np.diag(D_t) @ np.diag(D) )
            covariances[df_returns.index[t]]["F"] = pd.DataFrame(F_cov, index=df_returns.columns, columns=pd.Index([f"Factor {i+1}" for i in range(F_cov.shape[1])]))
            covariances[df_returns.index[t]]["F_Omega_sqrt"] = pd.DataFrame(F_cov, index=df_returns.columns, columns=pd.Index([f"Factor {i+1}" for i in range(F_cov.shape[1])]))
            covariances[df_returns.index[t]]["Omega"] = pd.DataFrame(np.eye(F_cov.shape[1]), 
                                                            index=pd.Index([f"Factor {i+1}" for i in range(F_cov.shape[1])]), 
                                                            columns=pd.Index([f"Factor {i+1}" for i in range(F_cov.shape[1])]))
        
            covariances[df_returns.index[t]]["D"] = pd.Series(D_cov, index=df_returns.columns)
            covariances[df_returns.index[t]]["Sigma"] = pd.DataFrame(F_cov @ F_cov.T + np.diag(D_cov),
                                                    index=df_returns.columns,
                                                    columns=df_returns.columns)
    return covariances

def covariances_by_KL(
        df_returns: pd.DataFrame, 
        Sigma_dict: dict, 
        burnin:int, 
        H:int=126) -> dict:
    """
    Runs the expecation-maximization algorithm to estimate the time-varying covariance matrices for a 
    dataframe of returns. It fits the factor model
        F Ω F' + D

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
    if set(df_returns.index.unique()).issubset(set(Sigma_dict.keys())) is False:
        raise ValueError("Sigma_dict keys must be a superset of df_returns index")
    
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

        neg_log_likes = []
        frobs = []

        for _ in range(num_iters):
            C_rs = (C_rr * inv_D_prev[None, :]) @ F_prev @ G_prev

            tmp_prev = (F_prev.T * inv_D_prev[None, :]) @ C_rr @ (F_prev * inv_D_prev[:, None])
            C_ss = G_prev + G_prev @ tmp_prev @ G_prev

            # Now solve
            F_prev = np.linalg.solve( C_ss.T, C_rs.T ).T

            D_prev = np.diag(C_rr) - 2 * np.sum(C_rs * F_prev, axis=1) + np.sum(F_prev * (F_prev @ C_ss), axis=1)
            inv_D_prev = 1 / D_prev

            G_prev = np.linalg.inv( F_prev.T * inv_D_prev[None, :] @ F_prev + np.eye(F_prev.shape[1]) )

            #############################
            # Sigma_t = F_prev @ F_prev.T + np.diag(D_prev)

            # neg_log_likes.append(
            #     np.linalg.slogdet(Sigma_t)[1] + np.linalg.trace( np.linalg.inv(Sigma_t) @ C_rr )
            # )

            # frobs.append(
            #     np.linalg.norm(np.diag(Sigma_t) - np.diag(C_rr))**2 / np.linalg.norm(np.diag(C_rr))**2
            # )
            # print(_)
        print(date)

        Sigma_em_dict[date]["F"] = pd.DataFrame(F_prev, 
                                                index=df_returns.columns, 
                                                columns=pd.Index([f"Factor {i+1}" for i in range(F_prev.shape[1])]))
        Sigma_em_dict[date]["F_Omega_sqrt"] = pd.DataFrame(F_prev, 
                                                           index=df_returns.columns, 
                                                           columns=pd.Index([f"Factor {i+1}" for i in range(F_prev.shape[1])]))
        Sigma_em_dict[date]["Omega"] = pd.DataFrame(np.eye(F_prev.shape[1]), 
                                                            index=pd.Index([f"Factor {i+1}" for i in range(F_prev.shape[1])]), 
                                                            columns=pd.Index([f"Factor {i+1}" for i in range(F_prev.shape[1])]))
        
        Sigma_em_dict[date]["D"] = pd.Series(D_prev, index=df_returns.columns)
        Sigma_em_dict[date]["Sigma"] = pd.DataFrame(F_prev @ F_prev.T + np.diag(D_prev), 
                                                    index=df_returns.columns, 
                                                    columns=df_returns.columns)
        Sigma_em_dict[date]["C_rr"] = pd.DataFrame(C_rr, index=df_returns.columns, columns=df_returns.columns)
        Sigma_em_dict[date]["neg_log_likes"] = neg_log_likes
        Sigma_em_dict[date]["frobs"] = frobs

    return Sigma_em_dict


def extending_covariances_by_KL(
        df_returns: pd.DataFrame, 
        Sigma_dict: dict, 
        burnin:int=252, 
        H:int=126, 
        num_additional_factors:int=5) -> dict:
    """
    Extends the covariance matrices by adding new factors using a KL divergence approach.

    It fits the factor model
        [F_factors; F_new] [Omega     0] [F_factors; F_new]^T + D
                           [0,        I]

        where F_factors is the matrix of factor exposures from Sigma_dict,
        F_new is the matrix of new factor exposures to be estimated,
        Omega is the covariance matrix of the original factors to be estimated,
        D is the diagonal specific risk matrix to be estimated.

        @ t uses returns up to and including t to estimate the covariance matrix at time t.
        Args:
        - df_returns (pd.DataFrame): DataFrame of asset returns with datetime index and assets as columns.
        - Sigma_dict (dict): Initial covariance matrices for each time point. Index is date and each date is a dict with keys 
            "F", "Omega", "D", "Sigma". These are dataframes (everything except D) /series (for D).
        - burnin (int): Number of initial periods to skip for covariance estimation.
        - H (int): Half-life for EWMA decay.
        - num_additional_factors (int): Number of new factors to add.

    Returns:
        - dict: Dictionary of estimated covariance matrices using extended KL approach

    
    """

    ### Checks
    if set(Sigma_dict.keys()).issubset(set(df_returns.index.unique())) is False:
        raise ValueError("Sigma_dict keys must be a superset of df_returns index")
    if not all("F" in Sigma_dict[date] and "Omega" in Sigma_dict[date] and "D" in Sigma_dict[date] and "Sigma" in Sigma_dict[date] for date in Sigma_dict):
        raise ValueError("Each entry in Sigma_dict must contain keys 'F', 'Omega', 'D', and 'Sigma'")
    
    ### Initializations
    Sigma_em_dict = {}
    beta = _beta_from_half_life(H)

    ### Get IEWMA covariances as starting point
    IEWMA_returns = _ewma_cov(df_returns.values, beta)
    IEWMA_returns = dict(zip(df_returns.index.unique(), IEWMA_returns))

    ### Initialize factor covariance for additional factors
    Omega_added_factors = np.eye(num_additional_factors)
    Omega_added_factors_inv = np.eye(num_additional_factors)

    ######################################################################
    ### Initialize EM for the first date
    _init_date = df_returns.index.unique()[burnin]
    num_factors = Sigma_dict[_init_date]['F'].shape[1]
    num_assets = Sigma_dict[_init_date]['F'].shape[0]
    num_iters = 300

    for date in df_returns.index.unique()[burnin:]:

        C_rr = IEWMA_returns[date]

        F_factors = Sigma_dict[date]['F'].to_numpy()

        # Find initial F_added_factors
        R = df_returns.loc[:date].values.T
        S_factors = np.linalg.lstsq(F_factors, R, rcond=None)[0]
        residuals = R - F_factors @ S_factors
        U, S, Vt = svds(residuals, k=num_additional_factors)
        F_added_factors_prev = U @ np.diag(np.sqrt(S))

        # Set total F_prev
        F_prev = np.hstack([F_factors, F_added_factors_prev])

        # Set Omega_factors_prev
        Omega_factors_prev = Sigma_dict[date]['Omega'].to_numpy()
        Omega_inv_prev = block_diag( np.linalg.inv(Omega_factors_prev), Omega_added_factors_inv )

        # Set initial D_prev
        D_prev = Sigma_dict[date]['Sigma'].to_numpy() - (F_factors @ Omega_factors_prev @ F_factors.T + F_added_factors_prev @ F_added_factors_prev.T)
        D_prev = np.diag(D_prev)
        D_prev = np.maximum(D_prev, 1e-4*np.max(np.diag(Sigma_dict[date]["D"].to_numpy())))
        inv_D_prev = 1 / D_prev

        G_prev = np.linalg.inv( F_prev.T * inv_D_prev[None, :] @ F_prev + Omega_inv_prev )

        # Store log likelihoods and frobenius norms
        neg_log_likes = []
        frobs = []

        for _ in range(num_iters):
            C_rs = (C_rr * inv_D_prev[None, :]) @ F_prev @ G_prev

            tmp_prev = (F_prev.T * inv_D_prev[None, :]) @ C_rr @ (F_prev * inv_D_prev[:, None])
            C_ss = G_prev + G_prev @ tmp_prev @ G_prev

            C_ss_factors = C_ss[:num_factors, :num_factors]
            C_ss_added = C_ss[num_factors:, num_factors:]
            C_ss_cross = C_ss[:num_factors, num_factors:]

            # Compute Omega_factors update
            Omega_factors_prev = C_ss_factors

            # Compute F_added_factors update
            F_added_factors_prev = np.linalg.solve( C_ss_added.T, (C_rs[:, num_factors:] - F_factors @ C_ss_cross).T ).T
            F_prev = np.hstack([F_factors, F_added_factors_prev])

            # Compute D_prev update
            D_prev = np.diag(C_rr) - 2 * np.sum(C_rs * F_prev, axis=1) + np.sum(F_prev * (F_prev @ C_ss), axis=1)
            inv_D_prev = 1 / D_prev

            # Compute Omega_inv_prev
            Omega_inv_prev = block_diag( np.linalg.inv(Omega_factors_prev), Omega_added_factors_inv )

            G_prev = np.linalg.inv( F_prev.T * inv_D_prev[None, :] @ F_prev + Omega_inv_prev )

            # Store log likelihood and frobenius norm
            # Sigma_t = F_prev @ block_diag(Omega_factors_prev, Omega_added_factors) @ F_prev.T + np.diag(D_prev)


            # neg_log_likes.append(
            #     np.linalg.slogdet(Sigma_t)[1] + np.linalg.trace( np.linalg.inv(Sigma_t) @ C_rr )
            # )

            # frobs.append(
            #     np.linalg.norm(np.diag(Sigma_t) - np.diag(C_rr))**2 / np.linalg.norm(np.diag(C_rr))**2
            # )
        print(date)

        # Store results
        L, U = eigh(Omega_factors_prev)
        Omega_factors_prev_sqrt = U @ np.diag(np.sqrt(np.maximum(L, 0.)))
        Sigma_em_dict[date] = {}
        Sigma_em_dict[date]["F"] = pd.DataFrame(np.hstack([F_factors, F_added_factors_prev]), 
                                                index=df_returns.columns, 
                                                columns=Sigma_dict[date]['F'].columns.tolist() + [f"Added Factor {i+1}" for i in range(F_added_factors_prev.shape[1])])
        Sigma_em_dict[date]["Omega"] = pd.DataFrame(block_diag(Omega_factors_prev, Omega_added_factors), 
                                                    index=Sigma_em_dict[date]["F"].columns, 
                                                    columns=Sigma_em_dict[date]["F"].columns)
        Sigma_em_dict[date]["D"] = pd.Series(D_prev, index=df_returns.columns)
        Sigma_em_dict[date]["Sigma"] = pd.DataFrame(Sigma_em_dict[date]["F"].to_numpy() @ Sigma_em_dict[date]["Omega"].to_numpy() @ Sigma_em_dict[date]["F"].to_numpy().T + np.diag(D_prev),
                                                    index=df_returns.columns,
                                                    columns=df_returns.columns)
        
        Sigma_em_dict[date]["F_Omega_sqrt"] = pd.DataFrame(np.hstack([F_factors @ Omega_factors_prev_sqrt, F_added_factors_prev]), 
                                                        index=df_returns.columns, 
                                                        columns=Sigma_dict[date]['F'].columns.tolist() + [f"Added Factor {i+1}" for i in range(F_added_factors_prev.shape[1])])
        
        Sigma_em_dict[date]["F_original"] = Sigma_dict[date]['F']
        Sigma_em_dict[date]["Omega_original"] = pd.DataFrame(Omega_factors_prev, 
                                                            index=Sigma_dict[date]['F'].columns, 
                                                            columns=Sigma_dict[date]['F'].columns)
        Sigma_em_dict[date]["F_added"] = pd.DataFrame(F_added_factors_prev,
                                                    index=df_returns.columns,
                                                    columns=[f"Added Factor {i+1}" for i in range(F_added_factors_prev.shape[1])])
        Sigma_em_dict[date]["Omega_added"] = pd.DataFrame(np.eye(num_additional_factors),
                                                    index=[f"Added Factor {i+1}" for i in range(F_added_factors_prev.shape[1])],
                                                    columns=[f"Added Factor {i+1}" for i in range(F_added_factors_prev.shape[1])])
        
        Sigma_em_dict[date]["C_rr"] = pd.DataFrame(C_rr, index=df_returns.columns, columns=df_returns.columns)
        Sigma_em_dict[date]["neg_log_likes"] = neg_log_likes
        Sigma_em_dict[date]["frobs"] = frobs

    return Sigma_em_dict
