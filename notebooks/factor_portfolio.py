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

    from extending_factor_model.data_loader.load_data import load_factor_data
    from extending_factor_model.risk_models.compute_risk_model_statistical import covariances_by_EIG
    return (
        Parallel,
        covariances_by_EIG,
        cp,
        deepcopy,
        delayed,
        load_factor_data,
        np,
        pd,
        plt,
    )


@app.cell
def _(load_factor_data):
    factor_returns = load_factor_data()
    rf_returns = factor_returns["RF"]
    factor_returns = factor_returns.drop(columns="RF")
    return factor_returns, rf_returns


@app.cell
def _(covariances_by_EIG, deepcopy, factor_returns):
    _returns = deepcopy(factor_returns)
    _returns["cash"] = 0.0
    risk_models = covariances_by_EIG(_returns, Hvol=42, Hcor=126, rank=None)
    print(252*risk_models[list(risk_models.keys())[100]]["Sigma"])
    return (risk_models,)


@app.cell
def _(cp, pd):
    def markowitz(holdings: pd.Series, 
                risk_model: dict,
                alpha: pd.Series,
                vol_ann:float = 0.08,
                 aversion:str="Mom") -> tuple[pd.Series, pd.Series]:
        nav = holdings.sum()

        z = cp.Variable(len(holdings))
        w = cp.Variable(len(holdings))

        objectives = []
        if alpha is not None:
            objectives.append(alpha.values.T @ w)
        else:
            objectives.append( - w[-1])

        if aversion is not None:
            objectives.append(-10 * cp.square(w.T @ risk_model["Sigma"].loc[holdings.index, aversion].to_numpy().reshape(-1) / risk_model["Sigma"].loc[aversion, aversion]))

        constraints = [w == holdings.values/nav + z,
                       w >= 0,
                       cp.sum(w) == 1.,
                      252*cp.quad_form(w, risk_model["Sigma"].loc[holdings.index, holdings.index].to_numpy()) <= vol_ann**2]

        prob = cp.Problem(cp.Maximize(cp.sum(objectives)), 
                         constraints)
        prob.solve()

        return pd.Series(z.value * nav, index = holdings.index), pd.Series(w.value * nav, index = holdings.index)
    return (markowitz,)


@app.cell
def _(factor_returns, np, pd):
    portfolios = {}
    portfolios[("Mkt-RF", "ST_Rev", "Mom", "cash")] = {"alpha": pd.DataFrame(np.array([1., 1.3, 1., 0.]), index=["Mkt-RF", "ST_Rev", "Mom", "cash"]),
                                               "holdings": pd.DataFrame(np.nan, index=factor_returns.index, columns=["Mkt-RF", "ST_Rev", "Mom", "cash"]),
                                               "aversion": None,
                                               "vol_tar": 0.08}

    for _factor in factor_returns.columns:
        portfolios[(_factor, "cash")] = {"alpha": None,
                                               "holdings": pd.DataFrame(np.nan, index=factor_returns.index, columns=[_factor, "cash"]),
                                               "aversion": None,
                                               "vol_tar": 0.08}
    return (portfolios,)


@app.cell
def _(
    Parallel,
    delayed,
    factor_returns,
    markowitz,
    np,
    pd,
    portfolios,
    rf_returns,
    risk_models,
):
    def run_portfolio(_portfolio_name, _portfolio_pars, factor_returns, risk_models):
        current_holdings = pd.Series(np.r_[np.zeros(len(_portfolio_pars["holdings"].columns)-1), 1.],
                                     index=_portfolio_pars["holdings"].columns)

        holdings_out = _portfolio_pars["holdings"].copy()

        for _iter, (_date, _next_date) in enumerate(zip(factor_returns.index[:-1],
                                                        factor_returns.index[1:])):
            print(_date)
            # write holdings
            holdings_out.loc[_date] = current_holdings

            # apply returns
            next_returns = pd.Series(0.0, index=current_holdings.index)
            next_returns.loc[list(_portfolio_name[:-1])] = \
                factor_returns.loc[_next_date, list(_portfolio_name[:-1])]
            next_returns.loc[_portfolio_name[-1]] = rf_returns.loc[_next_date]

            current_holdings = (1 + next_returns) * current_holdings

            # rebalance
            if _iter >= 126:
                current_trades, current_holdings = markowitz(
                    current_holdings,
                    risk_models[_next_date],
                    _portfolio_pars["alpha"],
                    _portfolio_pars["vol_tar"],
                    _portfolio_pars["aversion"]
                )

        # Write last day
        holdings_out.iloc[-1] = current_holdings

        return _portfolio_name, holdings_out

    # Parallel execution
    results = Parallel(n_jobs=9, prefer="threads")(
        delayed(run_portfolio)(_name, _pars, factor_returns, risk_models)
        for _name, _pars in portfolios.items()
    )



    # Reassemble output back into portfolios
    for name, holdings in results:
        portfolios[name]["holdings"] = holdings
    return


@app.cell
def _(np, pd, portfolios):
    def metrics(nav: pd.Series, name: str):
        nav = nav.dropna()
        total_return = (nav[-1] - nav[0]) / nav[0]
        total_ret_ann = 100*((nav[-1] / nav[0]) ** (252 / len(nav)) - 1)
        std_return = 100*nav.pct_change().std() * np.sqrt(252)
        sharpe_ratio = (total_ret_ann / std_return)  if std_return > 0 else 0.0

        cumulative = nav
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = 100*np.abs(drawdown.min())

        return pd.Series(np.array([total_ret_ann, std_return, sharpe_ratio, max_drawdown]), index=["return (%)", "std (%)", "sharpe", "drawdown (%)"])

    factor_returns_extended = pd.concat([_portfolio_pars["holdings"].dropna().sum(axis=1).pct_change().rename(str(_portfolio_name)) for _portfolio_name, _portfolio_pars in portfolios.items()], axis=1, join="outer")

    (1+factor_returns_extended).cumprod().apply(lambda x: metrics(x, x.name), axis=0).T.round(2)
    return (factor_returns_extended,)


@app.cell
def _(factor_returns_extended, plt):
    (1+factor_returns_extended).cumprod().plot()
    plt.yscale("log")
    plt.title("NAV")
    plt.show()
    return


@app.cell
def _(factor_returns_extended):
    factor_returns_extended.corr()
    return


@app.cell
def _(plt, portfolios):
    _holdings = portfolios[('Mkt-RF', 'ST_Rev', "Mom", 'cash')]["holdings"]
    _holdings.div(_holdings.sum(axis=1),axis=0).abs().plot.area(stacked=True)
    plt.title("Weights stackplot")
    return


if __name__ == "__main__":
    app.run()
