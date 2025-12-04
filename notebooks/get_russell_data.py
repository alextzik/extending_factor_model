import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import lseg.data as ld
    import pandas as pd
    import copy
    import plotly.express as px


    ld.open_session(app_key="77a15b279b9e475e8033315b8a519c30beb8432f")
    return copy, ld, pd


@app.cell
def _(copy, ld, pd):
    def get_constituents_as_of(ric, date):
        initial_constituents = ld.get_data(universe=[f"0#{ric}({date.replace('-', '')})"], 
                    fields=["TR.PriceClose"],
                    parameters={"SDATE":f"{date}", "EDATE":f"{date}"}
                )
        initial_constituents = initial_constituents['Instrument'].to_list()
        return initial_constituents

    def get_constituent_changes(ric, start, end):
 
        const_changes  = ld.get_data(universe=[ric], 
                    fields = ["TR.IndexJLConstituentChangeDate", "TR.IndexJLConstituentRIC",
                              "TR.IndexJLConstituentName", "TR.IndexJLConstituentituentChange"],
                    parameters={"SDATE":f"{start}","EDATE":f"{end}", 'IC':'B'}
                )
    
        return const_changes

    def add_joiner(init_list, joiner_list):
        for joiner in joiner_list:
            if joiner not in init_list:
                init_list.append(joiner)
            else:
                print(f'{joiner} joiner is already in the list')
        return init_list

    def remove_leaver(init_list, leaver_list):
        for leaver in leaver_list:
            if leaver in init_list:
                init_list.remove(leaver)
            else:
                print(f'{leaver} leaver is not in the list')
        return init_list

    def update_constituents(start, constituents, constitent_changes):
    
        hist_constituents = pd.DataFrame([(start, ric) for ric in constituents], columns=['Date', 'RIC'])
        for date in constitent_changes['Date'].unique():
            const_changes_date = constitent_changes[constitent_changes['Date'] == date]
            joiners = const_changes_date[const_changes_date['Change']=='Joiner']['Constituent RIC'].to_list()
            leavers = const_changes_date[const_changes_date['Change']=='Leaver']['Constituent RIC'].to_list()
            joiners_unique = list(set(joiners) - set(leavers))
            leavers_unique = list(set(leavers) - set(joiners))
            if len(joiners_unique) > 0:
                constituents = add_joiner(constituents, joiners_unique)
            if len(leavers_unique) > 0:
                constituents = remove_leaver(constituents, leavers_unique)
            new_constituents = copy.deepcopy(constituents)
            new_constituents_df =  pd.DataFrame([(str(date)[:10], ric) for ric in new_constituents], columns=['Date', 'RIC'])
            hist_constituents = pd.concat([hist_constituents, new_constituents_df])
        hist_constituents = hist_constituents.reset_index(drop = True)
    
        return hist_constituents

    def get_historical_constituents(index, start, end):
        initial_constituents = get_constituents_as_of(index, start)
        constitent_changes = get_constituent_changes(index, start, end)
        historical_constituents = update_constituents(start, initial_constituents, constitent_changes)
        return historical_constituents
    return (get_historical_constituents,)


@app.cell
def _(get_historical_constituents):
    ric = '.RUA'
    start = '2014-01-01'
    end = '2024-02-15'
    constituents = get_historical_constituents(ric, '2014-01-01', '2024-03-14')
    constituents = constituents.set_index("Date")
    return constituents, end, start


@app.cell
def _(constituents):
    # Group by date and get sets of tickers per date
    sets_per_date = constituents.groupby(constituents.index)["RIC"].apply(set)

    # Take intersection across all sets
    tickers_in_all_dates = list(set.intersection(*sets_per_date))
    len(tickers_in_all_dates)
    return (tickers_in_all_dates,)


@app.cell
def _(end, ld, start, tickers_in_all_dates):
    # Fetch historical data (e.g., daily closing prices)
    df_prices = ld.get_history(
        universe=tickers_in_all_dates[:500],
        fields=["TRDPRC_1"],
        start=start,
        end=end,
        interval="daily"
    )
    return (df_prices,)


@app.cell
def _(df_prices):
    returns = df_prices.dropna(axis=1, how="any").pct_change().dropna()
    returns.to_pickle('data/processed/assets/russell_returns.pkl')
    return


if __name__ == "__main__":
    app.run()
