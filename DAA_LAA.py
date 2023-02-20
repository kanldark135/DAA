import pandas as pd
import base_function
from fredapi import Fred
import time
from datetime import date
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf

yf.pdr_override()

class LAA:
    def __init__(self, isSeries=False, trade_start_d=None, end_d=None, tc=0.0025, SPY_lookback=10, UE_lookback=12):
        if trade_start_d is None:       # trade_start_d가 지정 되있지 않을떄, 오늘 날짜로 지정
            trade_start_d = str(date.today().strftime('%Y-%m-%d'))
        if isSeries:
            end_d = trade_start_d       # series 찾을 떄는 trade_start_d와 end_d가 같아야 함!

        if time.strptime(trade_start_d, '%Y-%m-%d') < time.strptime('2003-12-01', '%Y-%m-%d'):
            print('Start Date out of bounds: must be after 2003-12-01')
            exit()

        data_start_d = str((datetime.strptime(trade_start_d, '%Y-%m-%d') - relativedelta(months=12)).date())

        self.isSeries = isSeries
        self.data_start_d = data_start_d
        self.trade_start_d = trade_start_d
        self.end_d = end_d
        self.tc = tc  # transaction cost - 25bp
        self.SPY_lookback = SPY_lookback
        self.UE_lookback = UE_lookback

        self.fredapi_key = '715aea79178835c32d989ba4f400146d'       # TODO. make another account and get new fredapi key
        self.spy_ticker = 'SPY'
        self.stock_ticker = 'Stock'
        self.UE_ticker = 'UNRATE'

        self.universe_old = ['VIGRX', 'VFISX', 'VFINX', 'FKRCX', 'VFITX']
        self.universe_new = ['QQQ', 'SHY', 'SPY', 'GLD', 'IEF']

    def execute(self):
        print('LAA running')
        sma_df = pd.DataFrame()
        fred = Fred(api_key=self.fredapi_key)
        ue = pd.concat([pd.Series([None]), fred.get_series(self.UE_ticker, observation_start=self.data_start_d, observation_end=self.end_d)])
        sma_df[self.stock_ticker] = base_function.get_adj_close_data(self.spy_ticker, self.data_start_d, self.end_d)
        sma_df = base_function.get_rebalance_date(sma_df, self.trade_start_d, self.UE_lookback)
        while len(ue) > len(sma_df):
            ue = ue.iloc[:-1]

        # TODO : if len(sma_df.index) > len(ue.index), then add rows and fill with NAN
        while len(sma_df.index) > len(ue.index):
            ue = pd.concat([ue, pd.Series([None])])
        ue = ue.fillna(method='ffill')

        sma_df[self.UE_ticker] = ue.values
        sma_df[self.UE_ticker] = sma_df[self.UE_ticker].fillna(method='bfill')
        sma_df[self.stock_ticker] = (sma_df[self.stock_ticker] / sma_df[self.stock_ticker].rolling(self.SPY_lookback + 1).mean()) - 1
        sma_df[self.UE_ticker] = (sma_df[self.UE_ticker] / sma_df[self.UE_ticker].rolling(self.UE_lookback + 1).mean()) - 1
        sma_df.dropna(inplace=True)

        if self.trade_start_d != str(sma_df.index[0])[:10]:
            exit('SMA data and trade start date unmatched')

        universe_df = base_function.get_final_adj_close_data(self.universe_old, self.universe_new, self.data_start_d)
        weight_df = base_function.get_rebalance_date(universe_df, self.trade_start_d, self.UE_lookback)
        weight_df = pd.concat([weight_df, sma_df], axis=1)
        weight_df.dropna(inplace=True)
        weight_df = weight_df.apply(self.get_weight, axis=1)
        del weight_df[self.stock_ticker]
        del weight_df[self.UE_ticker]

        # TODO percentage 15% on SPY, 10% on VWO
        weight_df_copy = weight_df['SPY'].copy()
        weight_df['SPY'] = weight_df['SPY'] * 0.6
        weight_df['VWO'] = weight_df_copy * 0.4

        # TODO universe_df now includes VWO ticker
        universe_df = base_function.get_final_adj_close_data(self.universe_old + ['VEIEX'], self.universe_new + ['VWO'], self.data_start_d)
        universe_df = universe_df[~(universe_df.index < weight_df.index[0])]
        universe_df = (universe_df / universe_df.shift(1)).fillna(1)

        if self.isSeries:
            print('series date: {}'.format(str(weight_df.index[0])[:10]))
            print('Got a single weight series - no backtesting')
            weight_df.to_csv('./Series/{} LAA Weight Series.csv'.format(str(weight_df.index[0])[:10]))

            return weight_df, None, None, None, None, None, None, None

        print('start date: {}, end date: {} \n'.format(str(universe_df.index[0])[:10], str(universe_df.index[-1])[:10]))

        weighted_return_df = base_function.get_performance(universe_df, weight_df, self.universe_new + ['VWO'])
        weighted_return_tc_df, tc_df = base_function.get_performance_with_tc(universe_df, weight_df, self.tc, self.universe_new + ['VWO'])
        total_return_df = weighted_return_df.apply(base_function.get_total_return, axis=1)
        total_return_tc_df = weighted_return_tc_df.apply(base_function.get_total_return, axis=1)
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

    def get_weight(self, row_data):
        if (row_data[self.stock_ticker] > 0) or (row_data[self.UE_ticker] < 0):
            risky = True
        else:
            risky = False
        for asset in self.universe_new:
            if ((asset == 'QQQ') and (risky is False)) or ((asset == 'SHY') and (risky is True)):
                row_data[asset] = 0
            else:
                row_data[asset] = 0.25

        return row_data


if __name__ == "__main__":
    # Series should be used for recent days (absence of BIL data)
    Series = False
    start_date = '2008-01-01'   # Recommended: '2007-02-25', '2007-12-25', '2008-07-25', '2009-12-25'
    end_date = '2023-02-01'     # Recommended: None, '2020-12-31', '2022-10-31'

    laa = LAA(isSeries=Series, trade_start_d=start_date, end_d=end_date)
    w, wr, wr_tc, tr, tr_tc, mdd, mdd_tc, t = laa.execute()
    if not Series:
        base_function.plot(tr, mdd, tr_tc, mdd_tc, t, 'Lethargic Asset Allocation', 'LAA.png')
