import os
from dotenv import load_dotenv
from pandas_datareader import data as pdr       # TODO. Use bloomberg terminal data instead
import yfinance as yf
import pandas as pd
import analysis
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pymysql

yf.pdr_override()


def get_adj_close_data(tickers, data_sd, ed=None):  # Calendar 작업과, start ~ end 까지 데이터 무조건 있게 만듦
    extend_sd = str((datetime.strptime(data_sd, '%Y-%m-%d') - relativedelta(days=5)).date())  # 5 days earlier
    if isinstance(tickers, str):
        universe_df = pdr.get_data_yahoo(tickers, extend_sd, ed)['Adj Close']
        while universe_df.empty:
            print(f"Failed to download data for ticker: {tickers}. Retrying...")
            universe_df = pdr.get_data_yahoo(tickers, extend_sd, ed)['Adj Close']

    else:
        data = {}
        for ticker in tickers:
            ticker_data = pdr.get_data_yahoo(ticker, extend_sd, ed)['Adj Close']
            while ticker_data.empty:
                print(f"Failed to download data for ticker: {ticker}. Retrying...")
                ticker_data = pdr.get_data_yahoo(ticker, extend_sd, ed)['Adj Close']
            data[ticker] = ticker_data
        universe_df = pd.concat(data, axis=1, keys=data.keys())

    if ed is None:
        calendar1 = pd.date_range(start=extend_sd, end=universe_df.index[-1])
        calendar2 = pd.date_range(start=data_sd, end=universe_df.index[-1])
    else:
        calendar1 = pd.date_range(start=extend_sd, end=ed)
        calendar2 = pd.date_range(start=data_sd, end=ed)
    universe_df = universe_df.reindex(calendar1)

    universe_df = universe_df.fillna(method='ffill')
    universe_df = universe_df.reindex(calendar2)

    return universe_df


def get_rebalance_date(dataset, trade_sd, lookback=0):
    dataset = dataset.copy()  # deep copy
    diff_month = (dataset.index[-1].year - dataset.index[0].year) * 12 + (dataset.index[-1].month - dataset.index[0].month) - lookback  # TODO. lookback might be None
    if dataset.index[-1].day >= dataset.index[0].day:
        diff_month += 1
    dates = []
    for i in range(-lookback, diff_month):
        dates.append(datetime.strptime(trade_sd, '%Y-%m-%d') + relativedelta(months=i))
    dataset = dataset.reindex(dates)

    return dataset


def get_day_return(adj_close_data):
    day_return_df = (adj_close_data / adj_close_data.shift(1)).fillna(1)

    return day_return_df


def get_final_adj_close_data(universe_old, universe_new, data_sd, data_ed=None):

    universe_new_df = get_adj_close_data(universe_new, data_sd, data_ed)
    reverse_universe_new_df = universe_new_df[::-1]
    last = reverse_universe_new_df.isna().idxmax(axis=0).where(reverse_universe_new_df.isna().any(axis=0))

    # TODO if last date > 2003-01-01, then use ticker in universe_old[index] to extend the time series
    for index, ticker in enumerate(universe_new):

        if last.loc[ticker] > datetime(2003, 1, 1):
            # TODO get data from 2003-01-01 to the next date of the last.loc[ticker]
            universe_old_df = get_adj_close_data(universe_old[index], data_sd, last.loc[ticker] + relativedelta(days=1))
            daily_return_old = get_day_return(universe_old_df)
            reverse_daily_return_old = daily_return_old[::-1]
            reverse_universe_new_df = reverse_universe_new_df.copy()    # TODO copy the dataframe to avoid SettingWithCopyWarning

            # TODO extend time-series by dividing price by return
            for date in reverse_daily_return_old.index:
                reverse_universe_new_df[ticker].loc[date - relativedelta(days=1)] = reverse_universe_new_df[ticker].loc[date] / reverse_daily_return_old.loc[date]
                reverse_universe_new_df = reverse_universe_new_df.copy()    # TODO copy the dataframe to avoid SettingWithCopyWarning

    # TODO reverse the dataframe to get the original order
    universe_new_df = reverse_universe_new_df[::-1]

    return universe_new_df


def get_performance(universe_df, weight_df, universe_name):
    performance_list = [list(weight_df.iloc[0])]
    for i in range(1, len(universe_df)):
        date = str(universe_df.index[i].date())
        if date in weight_df.index:     # rebalancing date
            apply_change = list(universe_df.iloc[i] * performance_list[-1])
            performance_list.append(list(sum(apply_change) * weight_df.loc[date]))
        else:
            performance_list.append(list(universe_df.iloc[i] * performance_list[-1]))

    performance_df = pd.DataFrame(performance_list, index=universe_df.index, columns=universe_name)

    return performance_df


def get_performance_with_tc(universe_df, weight_df, tc, universe_name):
    performance_list = [list(weight_df.iloc[0] * (1 - tc))]
    tc_list = [tc]
    for i in range(1, len(universe_df)):
        date = str(universe_df.index[i].date())
        if date in weight_df.index:
            apply_change = universe_df.iloc[i] * performance_list[-1]
            rebalance = sum(apply_change) * weight_df.loc[date]
            tc_series = abs(apply_change - rebalance) * tc
            total_tc = sum(tc_series) / sum(rebalance)
            performance_list.append(list(rebalance * (1 - total_tc)))
            tc_list.append(total_tc)
        else:
            performance_list.append(list(universe_df.iloc[i] * performance_list[-1]))
            tc_list.append(0)

    performance_df = pd.DataFrame(performance_list, index=universe_df.index, columns=universe_name)
    tc_df = pd.DataFrame(tc_list, index=universe_df.index)
    tc_df = tc_df.reindex(weight_df.index)

    return performance_df, tc_df


def get_total_return(universe_return):      # dataframe을 series로 변경
    return universe_return.sum()


def get_krw(return_df):
    krw_df = get_adj_close_data('KRW=X', str(return_df.index[0].date()), str(return_df.index[-1].date()))
    krw_df = (krw_df / krw_df.iloc[0]).fillna(1)

    return_df = return_df * krw_df

    return return_df


def plot(tr_df, mdd_df, tr_tc_df, mdd_tc_df, tc_df, suptitle, png_name):

    tr_krw_df = get_krw(tr_tc_df)
    mdd_krw_df = (tr_krw_df / tr_krw_df.cummax() - 1) * 100

    analysis_table = analysis.Analyze(tr_df, mdd_df, tr_tc_df, mdd_tc_df, tr_krw_df, mdd_krw_df, tc_df).get_analysis_table()
    annual_ret = analysis.Analyze(tr_df, mdd_df, tr_tc_df, mdd_tc_df, tr_krw_df, mdd_krw_df, tc_df).get_annual_returns()

    fig, axs = plt.subplots(2, 2, figsize=(25, 10), gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle(suptitle)
    axs[0, 0].set_title('Total Return')
    axs[0, 0].plot(tr_df.index, tr_df.values, label='No Transaction Cost')
    axs[0, 0].plot(tr_tc_df.index, tr_tc_df.values, label='USD')
    axs[0, 0].plot(tr_krw_df.index, tr_krw_df.values, label='KRW')
    axs[0, 0].legend()
    axs[1, 0].set_title('Maximum Drawdown')
    axs[1, 0].plot(mdd_df.index, mdd_df.values, label='No Transaction Cost')
    axs[1, 0].plot(mdd_tc_df.index, mdd_tc_df.values, label='USD')
    axs[1, 0].plot(mdd_krw_df.index, mdd_krw_df.values, label='KRW')
    axs[0, 0].legend()
    axs[0, 1].set_title('Performance Table')
    axs[0, 1].axis('off')
    table1 = axs[0, 1].table(cellText=analysis_table.values, rowLabels=analysis_table.index, colLabels=analysis_table.columns, loc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.auto_set_column_width(col=list(range(len(analysis_table.columns))))
    axs[1, 1].set_title('Annual Return (USD)')
    axs[1, 1].axis('off')
    table2 = axs[1, 1].table(cellText=annual_ret.values, rowLabels=annual_ret.index, colLabels=annual_ret.columns, loc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.auto_set_column_width(col=list(range(len(analysis_table.columns))))
    sd = str(tr_df.index[0])[:10]
    ed = str(tr_df.index[-1])[:10]
    try:
        os.makedirs('./Results/{} ~ {}/'.format(sd, ed), exist_ok=True)
        plt.savefig('./Results/{} ~ {}/'.format(sd, ed) + png_name)
    except:
        exit('Specific Folder Name not existing in Results: {} ~ {}'.format(sd, ed))


# TODO function that convert new ticker to old ticker
def get_old_ticker(tickers):
    # TODO new ticker as key and old ticker as value
    ticker = {
        'SPY': 'VFINX',     # 미국 대형주
        'QQQ': 'VIGRX',     # 미국 기술주
        'IEV': 'IEV',       #
        'EWY': 'EWY',
        'VWO': 'VEIEX',     # 신흥국 주식
        'RWX': 'IYR',       # 리츠
        'GLD': 'FKRCX',     # 금
        'DBC': 'PCRAX',     # 원자재
        'AGG': 'VBMFX',     # 미국 종합채권
        'TLT': 'VUSTX',     # 미국 장기국채
        'IEF': 'VFITX',     # 미국 중기국채
        'IWM': 'NAESX',     # 미국 소형주
        'HYG': 'VWEHX',     # 미국 하이일드 채권
        'LQD': 'VWESX',     # 미국 회사채권
        'SHY': 'VFISX',     # 미국 단기국채
        'BIL': 'VFISX',     # 미국 단기국채
        'TIP': 'VIPSX',     # 미국 물가연동채
        'VEA': 'EFA'        # 선진국 주식
    }

    # TODO make old ticker corresponds to new ticker as list
    ticker_old = []
    for new in tickers:
        ticker_old.append(ticker[new])

    return ticker_old


def join_weights_and_get_performances(w_df_list, df_w_list, tc, ed):
    print('Joining strategies')
    weight_df = None
    for i in range(len(w_df_list)):
        temp_df = w_df_list[i] * df_w_list[i]
        if weight_df is None:
            weight_df = temp_df
        else:
            weight_df = pd.concat([weight_df, temp_df], axis=1).groupby(level=0, axis=1).sum()

    if len(weight_df) == 1:
        print('series date: {}'.format(str(weight_df.index[0])[:10]))
        print('Got a single weight series - no backtesting')
        weight_df.to_csv('./Series/{} UAA Weight Series.csv'.format(str(weight_df.index[0])[:10]))

        return weight_df, None, None, None, None, None, None, None

    # TODO weight_df.columns
    old_tickers = get_old_ticker(weight_df.columns)
    join_ret = get_final_adj_close_data(old_tickers, weight_df.columns.tolist(), str(weight_df.index[0])[:10], ed)

    if 'BIL' in join_ret.columns:
        join_ret['BIL'] = join_ret['BIL'].fillna(method='bfill')
    join_ret = (join_ret / join_ret.shift(1)).fillna(1)

    print('start date: {}, end date: {} \n'.format(str(join_ret.index[0])[:10], str(join_ret.index[-1])[:10]))

    weighted_return_df = get_performance(join_ret, weight_df, weight_df.columns)
    weighted_return_tc_df, tc_df = get_performance_with_tc(join_ret, weight_df, tc, weight_df.columns)
    total_return_df = weighted_return_df.apply(get_total_return, axis=1)
    total_return_tc_df = weighted_return_tc_df.apply(get_total_return, axis=1)
    mdd_df = (total_return_df / total_return_df.cummax() - 1) * 100
    mdd_tc_df = (total_return_tc_df / total_return_tc_df.cummax() - 1) * 100

    weight_df.to_csv('./Dataframes/Weights.csv')
    weighted_return_df.to_csv('./Dataframes/Weighted Return.csv')
    weighted_return_tc_df.to_csv('./Dataframes/Weighted Return (TC).csv')
    total_return_df.to_csv('./Dataframes/Total Return.csv')
    total_return_tc_df.to_csv('./Dataframes/Total Return (TC).csv')
    mdd_df.to_csv('./Dataframes/MDD.csv')
    mdd_tc_df.to_csv('./Dataframes/MDD (TC).csv')
    tc_df.to_csv('./Dataframes/Transaction Cost.csv')

    return weight_df, weighted_return_df, weighted_return_tc_df, total_return_df, total_return_tc_df, mdd_df, mdd_tc_df, tc_df


def get_bond(row_data):
    assigned = False
    for i in range(len(row_data)):
        if (not assigned) and (row_data[i] > 0):
            row_data[i] = 1
            assigned = True
        else:
            row_data[i] = 0
        if (not assigned) and (i == len(row_data) - 1):
            row_data[i] = 1

    return row_data


def mix_strategy_bond(w_df, bond_list=None, mix_weight=None, ed=None):
    print('Mixing with bonds')
    if bond_list is None:
        bond_list = ['IEF', 'SHY', 'BIL']
    if mix_weight is None:
        mix_weight = [0.7, 0.3]

    trade_start_d = str(w_df.index[0])[:10]
    data_start_d = str((datetime.strptime(trade_start_d, '%Y-%m-%d') - relativedelta(months=12)).date())

    old_bond = get_old_ticker(bond_list)
    bond_momentum = get_final_adj_close_data(old_bond, bond_list, data_start_d, ed)

    if 'BIL' in bond_list:
        bond_momentum['BIL'] = bond_momentum['BIL'].fillna(method='bfill')
    bond_momentum = get_rebalance_date(bond_momentum, trade_start_d, 12)  # get end of month price
    bond_momentum = 1 * (bond_momentum / bond_momentum.shift(1)) + \
                    1 * (bond_momentum / bond_momentum.shift(3)) + \
                    1 * (bond_momentum / bond_momentum.shift(6)) + \
                    1 * (bond_momentum / bond_momentum.shift(12)) - 4
    bond_momentum.dropna(inplace=True)
    bond_weight = bond_momentum.copy().apply(get_bond, axis=1)

    w_df = w_df * mix_weight[0]
    bond_weight = bond_weight * mix_weight[1]

    weight_df = pd.concat([w_df, bond_weight], axis=1).groupby(level=0, axis=1).sum()

    if len(weight_df) == 1:
        print('series date: {}'.format(str(weight_df.index[0])[:10]))
        print('Got a single weight series - no backtesting')

        if mix_weight == [0.6, 0.4]:
            weight_df.to_csv('./Series/60_40/{} Mixed Weight Series.csv'.format(str(weight_df.index[0])[:10]))
        elif mix_weight == [0.7, 0.3]:
            weight_df.to_csv('./Series/70_30/{} Mixed Weight Series.csv'.format(str(weight_df.index[0])[:10]))
        else:
            weight_df.to_csv('./Series/80_20/{} Mixed Weight Series.csv'.format(str(weight_df.index[0])[:10]))

        return weight_df, None, None, None, None, None, None, None

    old_ticker = get_old_ticker(weight_df.columns)
    universe_df = get_final_adj_close_data(old_ticker, weight_df.columns.tolist(), str(weight_df.index[0])[:10], ed)
    if 'BIL' in universe_df.columns:
        universe_df['BIL'] = universe_df['BIL'].fillna(method='bfill')
    universe_df = (universe_df / universe_df.shift(1)).fillna(1)

    print('start date: {}, end date: {} \n'.format(str(universe_df.index[0])[:10], str(universe_df.index[-1])[:10]))

    weighted_return_df = get_performance(universe_df, weight_df, weight_df.columns)
    weighted_return_tc_df, tc_df = get_performance_with_tc(universe_df, weight_df, 0.0025, weight_df.columns)
    total_return_df = weighted_return_df.apply(get_total_return, axis=1)
    total_return_tc_df = weighted_return_tc_df.apply(get_total_return, axis=1)
    mdd_df = (total_return_df / total_return_df.cummax() - 1) * 100
    mdd_tc_df = (total_return_tc_df / total_return_tc_df.cummax() - 1) * 100
    # TODO depending on the mix_weight, csv file name should be different
    if mix_weight == [0.6, 0.4]:
        weight_df.to_csv('./Dataframes/60_40/60_40_Weights.csv')
        weighted_return_df.to_csv('./Dataframes/60_40/Weighted Return.csv')
        weighted_return_tc_df.to_csv('./Dataframes/60_40/Weighted Return (TC).csv')
        total_return_df.to_csv('./Dataframes/60_40/Total Return.csv')
        total_return_tc_df.to_csv('./Dataframes/60_40/Total Return (TC).csv')
        mdd_df.to_csv('./Dataframes/60_40/MDD.csv')
        mdd_tc_df.to_csv('./Dataframes/60_40/MDD (TC).csv')
        tc_df.to_csv('./Dataframes/60_40/Transaction Cost.csv')
    elif mix_weight == [0.7, 0.3]:
        weight_df.to_csv('./Dataframes/70_30/70_30_Weights.csv')
        weighted_return_df.to_csv('./Dataframes/70_30/Weighted Return.csv')
        weighted_return_tc_df.to_csv('./Dataframes/70_30/Weighted Return (TC).csv')
        total_return_df.to_csv('./Dataframes/70_30/Total Return.csv')
        total_return_tc_df.to_csv('./Dataframes/70_30/Total Return (TC).csv')
        mdd_df.to_csv('./Dataframes/70_30/MDD.csv')
        mdd_tc_df.to_csv('./Dataframes/70_30/MDD (TC).csv')
        tc_df.to_csv('./Dataframes/70_30/Transaction Cost.csv')
    else:
        weight_df.to_csv('./Dataframes/80_20/80_20_Weights.csv')
        weighted_return_df.to_csv('./Dataframes/80_20/Weighted Return.csv')
        weighted_return_tc_df.to_csv('./Dataframes/80_20/Weighted Return (TC).csv')
        total_return_df.to_csv('./Dataframes/80_20/Total Return.csv')
        total_return_tc_df.to_csv('./Dataframes/80_20/Total Return (TC).csv')
        mdd_df.to_csv('./Dataframes/80_20/MDD.csv')
        mdd_tc_df.to_csv('./Dataframes/80_20/MDD (TC).csv')
        tc_df.to_csv('./Dataframes/80_20/Transaction Cost.csv')

    return weight_df, weighted_return_df, weighted_return_tc_df, total_return_df, total_return_tc_df, mdd_df, mdd_tc_df, tc_df


def excel_data(total_return, weights, mdd, test_universe=None):
    if test_universe is None:
        test_universe = ['SPY', 'QQQ', 'AGG']
    total_return = total_return.iloc[1:-1]
    total_return = total_return / total_return.iloc[0]
    total_return.name = 'tr'
    mdd = mdd.iloc[1:-1]
    mdd.name = 'mdd'
    daily_return = (total_return / total_return.shift(1)).fillna(1) - 1
    daily_return.name = 'dr'
    universe_df = get_adj_close_data(test_universe, mdd.index[0].strftime('%Y-%m-%d'), mdd.index[-1].strftime('%Y-%m-%d')) # Timestamp mdd.index[0] => string mdd.index[0].strftime('%Y-%m-%d')
    universe_df = universe_df.reindex(daily_return.index)
    universe_df = universe_df / universe_df.iloc[0]
    uni_daily_ret = (universe_df / universe_df.shift(1)).fillna(1) - 1

    df_final = pd.concat([daily_return, total_return, mdd, uni_daily_ret, universe_df], axis=1)
    drop_index = df_final[df_final['dr'] == 0].index
    df_final.drop(drop_index, inplace=True)

    weights = weights.iloc[1:]

    df_final.to_csv('./Dataframes/수익률.csv')
    weights.to_csv('./Dataframes/비중자료.csv')

    return 0


def insertDB(join_w, product, portfolio):

    load_dotenv()
    conn = pymysql.connect(host=os.environ.get('host'),
                           user=os.environ.get('user'),
                           password=os.environ.get('password'),
                           db=os.environ.get('db'),
                           charset=os.environ.get('charset'))

    sql = "INSERT INTO r_portfolios (product_id, portfolio_id, created_at, symbol, weight) " \
          "VALUES (%s, %s, %s, %s, %s)"

    with conn:
        with conn.cursor() as cur:
            for i in join_w.columns:
                if join_w.iloc[-1][i] == 0:
                    continue
                else:
                    cur.execute(sql, (product, portfolio, join_w.index[-1].strftime('%Y%m%d'), i, join_w.iloc[-1][i]))
        conn.commit()


if __name__ == '__main__':
    join_tr_tc = pd.read_csv('./Dataframes/join_tr_tc.csv', index_col=0)
    join_w = pd.read_csv('./Dataframes/join_w.csv', index_col=0)
    join_mdd_tc = pd.read_csv('./Dataframes/join_mdd_tc.csv', index_col=0)
    a = excel_data(join_tr_tc, join_w, join_mdd_tc)



