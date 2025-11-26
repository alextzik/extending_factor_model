import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from extending_factor_model.risk_models.compute_risk_model_from_factor_returns import compute_risk_model_given_factor_returns, compute_risk_models_over_time

    from extending_factor_model.risk_models.compute_risk_model_statistical import extending_covariances_by_KL

    from extending_factor_model.data_loader.load_data import load_factor_data

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    return (
        compute_risk_models_over_time,
        extending_covariances_by_KL,
        load_factor_data,
        pd,
    )


@app.cell
def _(compute_risk_models_over_time, load_factor_data, pd):
    factor_returns = load_factor_data()
    asset_returns  = pd.read_pickle("data/processed/assets/returns_df.pkl")

    common_dates = factor_returns.index.intersection(asset_returns.index)
    factor_returns = factor_returns.reindex(common_dates)
    asset_returns  = asset_returns.reindex(common_dates)

    cov_dict = compute_risk_models_over_time(asset_returns=asset_returns.iloc[:200], factor_returns=factor_returns.iloc[:200])
    return asset_returns, cov_dict


@app.cell
def _(asset_returns, cov_dict, extending_covariances_by_KL):
    cov_extended_dict = extending_covariances_by_KL(df_returns=asset_returns.iloc[:200],
                                                   Sigma_dict=cov_dict,
                                                    burnin=2*63,
                                                   H=63,num_additional_factors=5)
    return


if __name__ == "__main__":
    app.run()
