import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from extending_factor_model.risk_models.compute_risk_model_from_factor_returns import compute_risk_model_given_factor_returns, compute_risk_models_over_time_given_factor_returns

    from extending_factor_model.risk_models.compute_risk_model_statistical import extending_covariances_by_KL

    from extending_factor_model.data_loader.load_data import load_factor_data

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    return (
        compute_risk_models_over_time_given_factor_returns,
        extending_covariances_by_KL,
        load_factor_data,
        np,
        pd,
    )


@app.cell
def _():
    horizon = 1500
    start = 300
    halflife = 63
    return halflife, horizon, start


@app.cell
def _(
    compute_risk_models_over_time_given_factor_returns,
    halflife,
    horizon,
    load_factor_data,
    pd,
):
    factor_returns = load_factor_data()
    asset_returns  = pd.read_pickle("data/processed/assets/returns_df.pkl")

    common_dates = factor_returns.index.intersection(asset_returns.index)
    factor_returns = factor_returns.reindex(common_dates)
    asset_returns  = asset_returns.reindex(common_dates)

    cov_dict = compute_risk_models_over_time_given_factor_returns(
        asset_returns=asset_returns.iloc[:horizon], 
        factor_returns=factor_returns.iloc[:horizon],
        halflife=halflife,
        burnin=2*halflife)
    return asset_returns, cov_dict


@app.cell
def _(asset_returns, cov_dict, extending_covariances_by_KL, halflife, horizon):
    cov_extended_dict = extending_covariances_by_KL(df_returns=asset_returns.iloc[:horizon],
                                                    Sigma_dict=cov_dict,
                                                    burnin=2*halflife,
                                                    H=halflife,
                                                    num_additional_factors=5)
    return (cov_extended_dict,)


@app.cell
def _(asset_returns, cov_dict, cov_extended_dict, horizon, np, pd, start):
    log_likes = pd.DataFrame(np.nan, 
                            index=asset_returns.index[start:horizon-1],
                            columns=["base", "extended"])

    for _date, _next_date in zip(asset_returns.index[start:horizon-1], asset_returns.index[start+1:horizon]):
        print(_date)
        _rets = asset_returns.loc[_next_date]

        Sigmas = {}
        Sigmas["base"] = cov_dict[_date]["Sigma"]
        Sigmas["extended"] = cov_extended_dict[_date]["Sigma"]

        for _type in Sigmas.keys():
            _log_like = - 0.5 * _rets.T @ np.linalg.inv(Sigmas[_type].to_numpy()) @ _rets - 0.5 * np.linalg.slogdet(Sigmas[_type].to_numpy())[1]
            log_likes.loc[_date, _type] = _log_like / Sigmas[_type].to_numpy().shape[0]
    return (log_likes,)


@app.cell
def _(log_likes):
    log_likes.rolling(200).mean().plot()
    return


@app.cell
def _(log_likes):
    log_likes.resample("6ME").mean()
    return


@app.cell
def _(log_likes, np):
    log_likes.resample("6ME").std().div(np.sqrt(log_likes.resample("6ME").size()), axis=0)
    return


if __name__ == "__main__":
    app.run()
