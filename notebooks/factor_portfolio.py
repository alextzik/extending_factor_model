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
    from extending_factor_model.risk_models.compute_risk_model_statistical import covariances_by_EIG
    return covariances_by_EIG, cp, deepcopy, load_factor_data, np, pd, plt


@app.cell
def _(load_factor_data):
    factor_returns = load_factor_data()

    monthly_factor_returns = (1+factor_returns).resample('ME').prod() - 1
    return factor_returns, monthly_factor_returns


@app.cell
def _(covariances_by_EIG, deepcopy, monthly_factor_returns):
    _monthly_returns = deepcopy(monthly_factor_returns)
    _monthly_returns["cash"] = 0.0
    risk_models = covariances_by_EIG(_monthly_returns[["Mkt-RF", "ST_Rev", "cash", "Mom"]], Hvol=10, Hcor=20, rank=None)
    risk_models[list(risk_models.keys())[100]]["Sigma"]
    return (risk_models,)


@app.cell
def _(cp, pd):
    def markowitz(holdings: pd.Series, 
                  risk_model: dict,
                 alpha: pd.Series,
                 vol_ann:float = 0.08) -> tuple[pd.Series, pd.Series]:
        nav = holdings.sum()

        z = cp.Variable(len(holdings))
        w = cp.Variable(len(holdings))

        objective_alpha = alpha.values.T @ w
        objective_aversion = 0. * cp.square(w.T @ risk_model["Sigma"].loc[holdings.index, "Mom"].to_numpy().reshape(-1))
        constraints = [w == holdings.values/nav + z,
                       w >= 0,
                       cp.sum(w) == 1.,
                      cp.quad_form(w, risk_model["Sigma"].loc[holdings.index, holdings.index].to_numpy()) <= vol_ann**2 / 12]

        prob = cp.Problem(cp.Maximize(objective_alpha + objective_aversion), 
                         constraints)
        prob.solve()

        return pd.Series(z.value * nav, index = holdings.index), pd.Series(w.value * nav, index = holdings.index)
    return (markowitz,)


@app.cell
def _(markowitz, monthly_factor_returns, np, pd, risk_models):
    holdings_df = pd.DataFrame(np.nan, index=monthly_factor_returns.index, columns=["Mkt-RF", "ST_Rev", "cash"])
    alpha = pd.DataFrame(np.array([1., 1.5, 0.]), index=holdings_df.columns)

    current_holdings = pd.Series(np.array([0., 0., 1.]), index=holdings_df.columns)

    # Main loop
    for _iter, (_date, _next_date) in enumerate(zip(monthly_factor_returns.index[:-1], monthly_factor_returns.index[1:])):
        holdings_df.loc[_date] = current_holdings

        ##########################################
        # Apply returns
        next_returns = pd.Series(0.0, index = current_holdings.index)
        next_returns.loc[["Mkt-RF", "ST_Rev"]] = monthly_factor_returns.loc[_next_date, ["Mkt-RF", "ST_Rev"]]
        current_holdings = (1 + next_returns) * current_holdings

        if _iter >= 10:
            current_trades, current_holdings = markowitz(current_holdings, risk_models[_next_date], alpha, 0.08)

    return (holdings_df,)


@app.cell
def _(factor_returns, holdings_df, plt):
    (1+factor_returns).cumprod().plot(label=factor_returns.columns)
    holdings_df.dropna().sum(axis=1).plot(label="portfolio")
    plt.yscale("log")
    plt.legend()
    plt.show()
    return


@app.cell
def _(holdings_df, monthly_factor_returns, np, pd):
    def metrics(nav: pd.Series, name: str):
        nav = nav.dropna()
        total_return = (nav[-1] - nav[0]) / nav[0]
        total_ret_ann = 100*((nav[-1] / nav[0]) ** (12 / len(nav)) - 1)
        std_return = 100*nav.pct_change().std() * np.sqrt(12)
        sharpe_ratio = (total_ret_ann / std_return)  if std_return > 0 else 0.0

        cumulative = nav
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = 100*np.abs(drawdown.min())

        return pd.Series(np.array([total_ret_ann, std_return, sharpe_ratio, max_drawdown]), index=["return (%)", "std (%)", "sharpe", "drawdown (%)"])

    monthly_factor_returns_extended = pd.concat([monthly_factor_returns, holdings_df.dropna().sum(axis=1).pct_change().rename("portfolio")], axis=1, join="outer")

    (1+monthly_factor_returns_extended).cumprod().apply(lambda x: metrics(x, x.name), axis=0).T.round(2)
    return (monthly_factor_returns_extended,)


@app.cell
def _(monthly_factor_returns_extended):
    monthly_factor_returns_extended.corr()
    return


if __name__ == "__main__":
    app.run()
