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

        if alpha is not None and not alpha.index.equals(holdings.index):
            raise ValueError("Index of holdings and alpha must match.")
        if alpha is not None and not alpha.index[-1] == "cash":
            raise ValueError("Last index entry of alpha must be cash.")
        if not holdings.index[-1] == "cash":
            raise ValueError("Last index entry of holdings must be cash.")

        nav = holdings.sum()

        z = cp.Variable(len(holdings))
        w = cp.Variable(len(holdings))

        objectives = []
        if alpha is not None:
            objectives.append(w.T @ alpha.values)
        else:
            objectives.append( - w[-1])

        if aversion is not None:
            objectives.append(-10 * cp.square(w.T @ risk_model["Sigma"].loc[holdings.index, aversion].to_numpy().reshape(-1) / risk_model["Sigma"].loc[aversion, aversion]))

        constraints = [w == holdings.values/nav + z,
                       w[:-1] >= 0,
                       cp.sum(w) == 1.,
                       cp.sum(cp.abs(w[:-1])) <= 1.3, 
                      252*cp.quad_form(w, risk_model["Sigma"].loc[holdings.index, holdings.index].to_numpy()) <= vol_ann**2]

        prob = cp.Problem(cp.Maximize(cp.sum(objectives)), 
                         constraints)
        prob.solve()

        return pd.Series(z.value * nav, index = holdings.index), pd.Series(w.value * nav, index = holdings.index)
    return (markowitz,)


@app.cell
def _(factor_returns, np, pd):
    alphas = 252*factor_returns.rolling(512).mean().dropna()
    alphas["cash"] = 0.

    alphas_portfolio = alphas[["Mkt-RF", "ST_Rev", "Mom", "RMW", "cash"]]
    alphas_portfolio = alphas_portfolio.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    alphas_portfolio = alphas_portfolio.sub(alphas_portfolio["cash"], axis=0)

    const_portfolio = pd.DataFrame(np.tile(np.array([1.0, 1.0, 1.0, 1.0, 0.0]), (len(factor_returns.index), 1)),
        index=factor_returns.index,
        columns=["Mkt-RF", "ST_Rev", "Mom", "RMW", "cash"]).apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    const_portfolio = const_portfolio.sub(const_portfolio["cash"], axis=0)

    # alphas_combined = 0.0 * alphas_portfolio + 1.0*const_portfolio
    # alphas_combined.loc[:, ["Mkt-RF", "ST_Rev", "Mom"]] = \
    #     alphas_combined.loc[:, ["Mkt-RF", "ST_Rev", "Mom"]].fillna(1.0)

    # alphas_combined.loc[:, ["cash"]] = \
    #     alphas_combined.loc[:, ["cash"]].fillna(0.0)
    alphas_combined = const_portfolio
    return alphas, alphas_combined, alphas_portfolio


@app.cell
def _(alphas_combined):
    alphas_combined.plot()
    return


@app.cell
def _(alphas_portfolio):
    alphas_portfolio.plot()
    return


@app.cell
def _(alphas_combined, factor_returns, np, pd):
    portfolios = {}
    portfolios[("Mkt-RF", "ST_Rev", "Mom", "cash")] = {"alpha": 
                                                       alphas_combined,
                                                                    #alphas[["Mkt-RF", "ST_Rev", "Mom", "cash"]],
                                               "holdings": pd.DataFrame(np.nan, index=factor_returns.index, columns=["Mkt-RF", "ST_Rev", "Mom", "RMW", "cash"]),
                                               "aversion": None,
                                               "vol_tar": 0.05}

    # for _factor in factor_returns.columns:
    #     portfolios[(_factor, "cash")] = {"alpha": None, #alphas[[_factor, "cash"]],
    #                                            "holdings": pd.DataFrame(np.nan, index=factor_returns.index, columns=[_factor, "cash"]),
    #                                            "aversion": None,
    #                                            "vol_tar": 0.05}
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
        if not _portfolio_name[-1] == "cash":
            raise ValueError("Cash must be last.")

        current_holdings = pd.Series(np.r_[np.zeros(len(_portfolio_pars["holdings"].columns)-1), 1.],
                                     index=_portfolio_pars["holdings"].columns)

        holdings_out = _portfolio_pars["holdings"].copy()

        for _iter, (_date, _next_date) in enumerate(zip(_portfolio_pars["holdings"].index[:-1],
                                                        _portfolio_pars["holdings"].index[1:])):
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
                    _portfolio_pars["alpha"].loc[_next_date] if _portfolio_pars["alpha"] is not None else None,
                    _portfolio_pars["vol_tar"],
                    _portfolio_pars["aversion"]
                )

        # Write last day
        holdings_out.iloc[-1] = current_holdings

        return _portfolio_name, holdings_out

    # Parallel execution
    results = Parallel(n_jobs=10, prefer="threads")(
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
        total_return = (nav.iloc[-1] - nav.iloc[0]) / nav.iloc[0]
        total_ret_ann = 100*((nav.iloc[-1] / nav.iloc[0]) ** (252 / len(nav)) - 1)
        std_return = 100*nav.pct_change().std() * np.sqrt(252)
        sharpe_ratio = (total_ret_ann / std_return)  if std_return > 1e-4 else 0.0

        cumulative = nav
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = 100*np.abs(drawdown.min())

        return pd.Series(np.array([total_ret_ann, std_return, sharpe_ratio, max_drawdown]), index=["return (%)", "std (%)", "sharpe", "drawdown (%)"])

    factor_returns_extended = pd.concat([_portfolio_pars["holdings"].dropna().sum(axis=1).pct_change().rename(str(_portfolio_name)) for _portfolio_name, _portfolio_pars in portfolios.items()], axis=1, join="outer")

    navs = (1+factor_returns_extended).cumprod()
    joint_results = navs.apply(lambda x: metrics(x, x.name), axis=0).T.round(2).sort_values("sharpe", ascending=False)
    joint_results
    return factor_returns_extended, joint_results, metrics, navs


@app.cell
def _(navs, plt):
    navs.plot()
    plt.yscale("log")
    plt.title("NAV")
    plt.show()
    return


@app.cell
def _(plt, portfolios):
    _holdings = portfolios[('Mkt-RF', 'ST_Rev', "Mom", 'cash')]["holdings"]
    _holdings.div(_holdings.sum(axis=1),axis=0).abs().plot.area(stacked=True)
    plt.title("Weights stackplot")
    return


@app.cell
def _(factor_returns_extended, metrics, pd):
    out = []

    for period_end, grp in factor_returns_extended.resample("YE"):
        year = period_end.year                      # <-- this is your year label
        # compute metrics per portfolio
        m = (1+grp).cumprod().apply(lambda col: metrics(col, col.name))
        # Build MultiIndex columns = (year, metric)
        m.index = pd.MultiIndex.from_product([[pd.to_datetime(year, format="%Y")], m.index], names=["year", "metric"])
        out.append(m)

    # Concatenate across years
    result = pd.concat(out, axis=0) 
    return (result,)


@app.cell
def _(joint_results, plt, result):
    result[joint_results.index[:4]].xs(key="sharpe", level="metric").plot()
    plt.title("Sharpe per year")
    plt.axhline(0.0, linestyle="--")
    plt.show()
    return


@app.cell
def _(alphas):
    (alphas[["Mom", "Mkt-RF", "ST_Rev"]]).plot()
    return


@app.cell
def _(alphas):
    alphas[["Mom", "Mkt-RF", "ST_Rev"]].mean()
    return


if __name__ == "__main__":
    app.run()
