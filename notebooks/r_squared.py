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
    from joblib import Parallel, delayed
    import pickle
    import statsmodels.api as sm


    from extending_factor_model.data_loader.load_data import load_factor_data
    return load_factor_data, np, pd, pickle, sm


@app.cell
def _(load_factor_data, pickle):
    factor_returns = load_factor_data()
    rf_returns = factor_returns["RF"]
    factor_returns = factor_returns.drop(columns="RF")

    with open("data/processed/assets/russell_returns.pkl", "rb") as _f:
            returns_df = pickle.load(_f)
    return factor_returns, returns_df


@app.cell
def _(factor_returns, np, pd, returns_df, sm):
    r2_scores = pd.Series(np.nan, index=returns_df.columns)
    intersection = returns_df.index.intersection(factor_returns.index)
    for ticker in returns_df.columns:
        y = returns_df[ticker].reindex(intersection)
        X = factor_returns.reindex(intersection)
        X = sm.add_constant(X)

        model = sm.OLS(y, X, missing="drop").fit()
        r2_scores.loc[ticker] = model.rsquared
    return (r2_scores,)


@app.cell
def _(r2_scores):
    r2_scores[r2_scores < 0.3].hist()
    return


if __name__ == "__main__":
    app.run()
