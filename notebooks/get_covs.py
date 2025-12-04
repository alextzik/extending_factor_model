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

    import pickle 
    return (
        compute_risk_models_over_time_given_factor_returns,
        extending_covariances_by_KL,
        load_factor_data,
        np,
        pd,
        pickle,
        plt,
    )


@app.cell
def _():
    horizon = -1
    start = 300
    halflife = 63
    return halflife, horizon, start


@app.cell
def _(factor_returns):
    factor_returns
    return


@app.cell
def _(
    compute_risk_models_over_time_given_factor_returns,
    halflife,
    horizon,
    load_factor_data,
    pd,
):
    factor_returns = load_factor_data()
    factors = ["Mom", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "LT_Rev", "ST_Rev"]
    factor_returns = factor_returns[factors]
    asset_returns  = pd.read_pickle("data/processed/assets/russell_returns.pkl")

    assets = asset_returns.columns

    common_dates = factor_returns.index.intersection(asset_returns.index)
    factor_returns = factor_returns.reindex(common_dates)
    asset_returns  = asset_returns.reindex(common_dates)

    cov_dict = compute_risk_models_over_time_given_factor_returns(
        asset_returns=asset_returns.iloc[:horizon][assets], 
        factor_returns=factor_returns.iloc[:horizon],
        halflife=halflife,
        burnin=2*halflife)

    # with open("basic_risk_model.pkl", "rb") as _f:
    #     cov_dict = pickle.load(_f)
    return asset_returns, assets, cov_dict, factor_returns


@app.cell
def _(
    asset_returns,
    assets,
    cov_dict,
    extending_covariances_by_KL,
    halflife,
    horizon,
):
    cov_extended_dict = extending_covariances_by_KL(df_returns=asset_returns.iloc[:horizon][assets],
                                                    Sigma_dict=cov_dict,
                                                    burnin=2*halflife,
                                                    H=halflife,
                                                    num_additional_factors=5)
    return (cov_extended_dict,)


@app.cell
def _(cov_dict, cov_extended_dict, pickle):
    with open("basic_risk_model_russell.pkl", "wb") as _f:
        pickle.dump(cov_dict, _f)

    with open("extended_risk_model_russell.pkl", "wb") as _f:
        pickle.dump(cov_extended_dict, _f)    
    return


@app.cell
def _(cov_extended_dict):
    cov_extended_dict.keys()
    return


@app.cell
def _(date):
    date
    return


@app.cell
def _(cov_extended_dict, np, pd, plt):
    date = pd.to_datetime("2020-03-20") #list(cov_extended_dict.keys())[1100]
    Sigma = cov_extended_dict[date]

    plt.scatter(np.diag(Sigma["C_rr"]), np.diag(Sigma["Sigma"]))
    plt.plot(np.diag(Sigma["Sigma"]), np.diag(Sigma["Sigma"]), linestyle="--", color="r")
    plt.show()
    return Sigma, date


@app.cell
def _(Sigma, cov_dict, date, np, plt):
    _Sigma = cov_dict[date]
    plt.plot(np.diag(Sigma["Sigma"]), np.diag(Sigma["Sigma"]), linestyle="--", color="r")
    plt.scatter(np.diag(Sigma["Sigma"]), np.diag(_Sigma["Sigma"]))
    return


@app.cell
def _(
    asset_returns,
    assets,
    cov_dict,
    cov_extended_dict,
    horizon,
    np,
    pd,
    start,
):
    log_likes = pd.DataFrame(np.nan, 
                            index=asset_returns.index[start:horizon-1],
                            columns=["base", "extended"])

    for _date, _next_date in zip(asset_returns.index[start:horizon-1], asset_returns.index[start+1:horizon]):
        print(_date)
        _rets = asset_returns.loc[_next_date][assets]

        Sigmas = {}
        Sigmas["base"] = cov_dict[_date]["Sigma"]
        Sigmas["extended"] = cov_extended_dict[_date]["Sigma"]

        for _type in Sigmas.keys():
            _log_like = - 0.5 * _rets.T @ np.linalg.inv(Sigmas[_type].to_numpy()) @ _rets - 0.5 * np.linalg.slogdet(Sigmas[_type].to_numpy())[1]
            log_likes.loc[_date, _type] = _log_like / Sigmas[_type].to_numpy().shape[0]
    return (log_likes,)


@app.cell
def _(log_likes):
    log_likes.mean()
    return


@app.cell
def _(log_likes):
    log_likes.rolling(100).mean().plot()
    return


@app.cell
def _(log_likes):
    log_likes.idxmin(axis=0)
    return


@app.cell
def _(log_likes):
    log_likes.resample("6ME").mean()
    return


@app.cell
def _(log_likes, np):
    log_likes.resample("6ME").std().div(np.sqrt(log_likes.resample("6ME").size()), axis=0)
    return


@app.cell
def _(cov_extended_dict, np, pd):
    _dates = list(cov_extended_dict.keys())
    norms = pd.DataFrame(np.nan, index=_dates, columns=["orig", "added"])
    for _d in _dates:
        F_orig = cov_extended_dict[_d]["F_Omega_sqrt"].iloc[:, :1]
        F_new = cov_extended_dict[_d]["F_Omega_sqrt"].iloc[:, 1:]

        norms.loc[_d, "orig"] = np.linalg.norm(F_orig)
        norms.loc[_d, "added"] = np.linalg.norm(F_new)

    norms.plot()
    return (norms,)


@app.cell
def _(norms):
    norms
    return


@app.cell
def _(asset_returns, assets):
    asset_returns[assets[0:3]].cumsum().plot()
    return


if __name__ == "__main__":
    app.run()
