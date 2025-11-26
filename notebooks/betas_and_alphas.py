import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import cvxpy as cp
    import matplotlib.pyplot as plt
    from copy import deepcopy

    from extending_factor_model.data_loader.load_data import load_factor_data

    from scipy.linalg import lstsq
    import statsmodels as sm
    return load_factor_data, lstsq, np, pd, plt, sm


@app.cell
def _(load_factor_data, pd):
    factor_returns = load_factor_data()
    asset_returns  = pd.read_pickle("data/processed/assets/returns_df.pkl")

    common_dates = factor_returns.index.intersection(asset_returns.index)

    factor_returns = factor_returns.reindex(common_dates)
    asset_returns  = asset_returns.reindex(common_dates)
    return asset_returns, factor_returns


@app.cell
def _(lstsq, np, pd, sm):
    def ew_regression_with_r2(asset_returns: pd.Series, factor_returns: pd.DataFrame, halflife: float):
        """
        Exponentially weighted causal regression using scipy.linalg.lstsq
        Returns betas, alphas, and R² at each date.

        Parameters
        ----------
        asset_returns : pd.Series
        factor_returns : pd.DataFrame
        halflife : float
            Halflife for exponential weighting (in units of the index, e.g., days)

        Returns
        -------
        betas : pd.DataFrame
        alphas : pd.Series
        r2 : pd.Series
        """
        asset_returns = asset_returns.loc[factor_returns.index]
        X_full = factor_returns.copy()
        X_full = sm.tools.tools.add_constant(X_full)  # add intercept

        alphas_list = []
        betas_list = []
        r2_list = []
        residuals_list = []

        for t in range(1, len(X_full)):
            print(f"{t} / {len(X_full)}")
            X_t = X_full.iloc[:t+1, :].values
            y_t = asset_returns.iloc[:t+1].values

            # Exponentially weighted
            decay_factor = np.exp(-np.log(2) / halflife)
            weights = np.array([decay_factor**i for i in range(t+1)])
            weights = weights / weights.sum()
            weights = weights[::-1]
            W_sqrt = np.sqrt(np.diag(weights))  # sqrt for weighting in least squares

            Xw = W_sqrt @ X_t
            yw = W_sqrt @ y_t

            beta, residuals, rank, s = lstsq(Xw, yw)  # solve weighted least squares

            alphas_list.append(beta[0])
            betas_list.append(beta[1:])

            # R² = 1 - SS_res / SS_tot
            y_pred = X_t @ beta
            ss_res = np.sum(weights * (y_t - y_pred)**2)
            ss_tot = np.sum(weights * (y_t - np.average(y_t, weights=weights))**2)
            r2_list.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)

            residuals_list.append(y_t[-1] - y_pred[-1])

        betas = pd.DataFrame(betas_list, index=asset_returns.index[1:], columns=factor_returns.columns)
        alphas = pd.Series(alphas_list, index=asset_returns.index[1:], name="alpha")
        r2 = pd.Series(r2_list, index=asset_returns.index[1:], name="r2")
        residuals = pd.Series(residuals_list, index=asset_returns.index[1:], name="residuals")

        return betas, alphas, r2, residuals
    return (ew_regression_with_r2,)


@app.cell
def _(asset_returns, ew_regression_with_r2, factor_returns):
    betas, alphas, r2, residuals = ew_regression_with_r2(asset_returns.iloc[:, 0], factor_returns, 126)
    return alphas, betas, r2, residuals


@app.cell
def _(betas):
    betas.iloc[126:].plot()
    return


@app.cell
def _(r2):
    r2[126:].plot()
    return


@app.cell
def _(alphas):
    alphas.iloc[126:].plot()
    return


@app.cell
def _(asset_returns, np, pd, plt, residuals):
    from extending_factor_model.risk_models.compute_risk_model_statistical import _ewma_series as ewma_series
    from extending_factor_model.risk_models.compute_risk_model_statistical import _beta_from_half_life as beta_from_half_life

    ewma_returns = np.sqrt(
            pd.Series(ewma_series(asset_returns.iloc[:, 0]**2, beta_from_half_life(126)), 
                      index=asset_returns.iloc[:, 0].index)
            )

    ewma_residuals = np.sqrt(
            pd.Series(ewma_series(residuals**2, beta_from_half_life(126)), 
                      index=residuals.index)
            )

    ewma_returns[126:].plot(label="returns")
    ewma_residuals[126:].plot(label="residuals")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
