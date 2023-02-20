import pandas as pd
import base_function
import math
import time
from datetime import date
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf

yf.pdr_override()

class FAA:
    """
    Flexible Asset Allocation (Keller, 2012)

    Absolute Momentum, Volatility, Correlation
    """

    def __init__(self, isSeries=False, trade_start_d=None, end_d=None, top=6, tc=0.0025, N=12, wR=1.0, wV=0.5, wC=0.5,
                 R1_lookback=3, R2_lookback=6, R3_lookback=12, VC_lookback=6):
        if trade_start_d is None:       # trade_start_d가 지정 되있지 않을떄, 오늘 날짜로 지정
            trade_start_d = str(date.today().strftime('%Y-%m-%d'))
        if isSeries:
            end_d = trade_start_d       # series 찾을 떄는 trade_start_d와 end_d가 같아야 함!

        if time.strptime(trade_start_d, '%Y-%m-%d') < time.strptime('2003-12-01', '%Y-%m-%d'):
            print('Start Date out of bounds: must be after 2003-12-01')
            exit()

        else:
            self.universe_old = ['VFINX', 'VIGRX', 'IEV', 'EWY', 'VEIEX', 'IYR', 'FKRCX', 'PCRAX', 'VBMFX', 'VUSTX', 'VFITX', 'VFISX']
            self.universe_new = ['SPY', 'QQQ', 'IEV', 'EWY', 'VWO', 'RWX', 'GLD', 'DBC', 'AGG', 'TLT', 'IEF', 'SHY']

        data_start_d = str((datetime.strptime(trade_start_d, '%Y-%m-%d') - relativedelta(months=12)).date())

        self.isSeries = isSeries
        self.data_start_d = data_start_d
        self.trade_start_d = trade_start_d
        self.end_d = end_d
        self.tc = tc                    # transaction cost - 25bp
        self.top = top                    # maximum number of risky assets (fixed)
        self.N = N                      # universe size (fixed)
        self.wR = wR
        self.wV = wV
        self.wC = wC
        self.R1_lookback = R1_lookback
        self.R2_lookback = R2_lookback
        self.R3_lookback = R3_lookback
        self.VC_lookback = VC_lookback

        self.cash_old = ['SHY', 'VFISX']
        self.cash_new = ['SHY', 'BIL']
        self.cash_rename = ['SHY(C)', 'BIL']

    def execute(self):
        print('FAA running')

        universe_df = base_function.get_final_adj_close_data(self.universe_old, self.universe_new, self.data_start_d)
        monthly_df = base_function.get_rebalance_date(universe_df, self.trade_start_d, self.R3_lookback)
        momentum_df = monthly_df / monthly_df.shift(self.R1_lookback) + monthly_df / monthly_df.shift(self.R2_lookback) +\
                      monthly_df / monthly_df.shift(self.R3_lookback) - 3
        momentum_df.dropna(inplace=True)
        universe_df = (universe_df / universe_df.shift(1)).fillna(1)
        universe_df = universe_df - 1   # substitute 1 to get rate of change, for s.d. calculation later

        cash_universe_df = base_function.get_final_adj_close_data(self.cash_old, self.cash_new, self.data_start_d)
        cash_universe_df['BIL'] = cash_universe_df['BIL'].fillna(method='bfill')    # TODO extended BIL length
        cash_df = base_function.get_rebalance_date(cash_universe_df, self.trade_start_d, self.R3_lookback)
        cash_df = cash_df / cash_df.shift(self.R1_lookback) + cash_df / cash_df.shift(self.R2_lookback) +\
                  cash_df / cash_df.shift(self.R3_lookback) - 3
        cash_df.dropna(inplace=True)
        cash_df.columns = self.cash_rename
        cash_universe_df = (cash_universe_df / cash_universe_df.shift(1)).fillna(1)

        absolute_boolean_df = momentum_df > 0       # turn df into True and False
        absolute_boolean_df[absolute_boolean_df == 0] = None        # False into Nan

        volatility_df = []
        correlation_df = []
        for date_ in momentum_df.index:     # rebalancing date마다
            lookback_df = universe_df.loc[date_ - pd.DateOffset(months=self.VC_lookback):date_]     # export 6 months of datas
            volatility_df.append(list(lookback_df.std() * math.sqrt(365)))
            correlation_df.append(list(lookback_df.corr().sum() - 1))
        volatility_df = pd.DataFrame(volatility_df, index=momentum_df.index, columns=self.universe_new)
        correlation_df = pd.DataFrame(correlation_df, index=momentum_df.index, columns=self.universe_new)

        momentum_df = momentum_df * absolute_boolean_df
        volatility_df = volatility_df * absolute_boolean_df
        correlation_df = correlation_df * absolute_boolean_df

        rank_R_df = momentum_df.rank(ascending=False, axis=1)
        rank_V_df = volatility_df.rank(ascending=True, axis=1)
        rank_C_df = correlation_df.rank(ascending=True, axis=1)

        loss_function_df = (self.wR * rank_R_df) + (self.wV * rank_V_df) + (self.wC * rank_C_df)
        loss_function_df = loss_function_df.rank(ascending=True, axis=1, method='min')
        loss_function_df[loss_function_df > self.top] = None
        loss_function_df = pd.concat([loss_function_df, cash_df], axis=1)
        loss_function_df['n'] = loss_function_df[self.universe_new].count(axis=1)
        loss_function_df = loss_function_df.fillna(0)

        weight_df = loss_function_df.apply(self.get_weight, axis=1)
        del weight_df['n']
        del weight_df[self.cash_rename[0]]

        if self.isSeries:
            print('series date: {}'.format(str(weight_df.index[0])[:10]))
            print('Got a single weight series - no backtesting')
            weight_df.to_csv('./Series/{} FAA Weight Series.csv'.format(str(weight_df.index[0])[:10]))

            return weight_df, None, None, None, None, None, None, None

        universe_df = universe_df + 1
        universe_df = pd.concat([universe_df, cash_universe_df[self.cash_new[1]]], axis=1)
        universe_df = universe_df[~(universe_df.index < loss_function_df.index[0])]     # logic?

        print('start date: {}, end date: {} \n'.format(str(universe_df.index[0])[:10], str(universe_df.index[-1])[:10]))

        weighted_return_df = base_function.get_performance(universe_df, weight_df, universe_df.columns)
        weighted_return_tc_df, tc_df = base_function.get_performance_with_tc(universe_df, weight_df, self.tc, universe_df.columns)
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
        if row_data['n'] >= self.top:
            weight = 1 / row_data['n']
            cash_w = 0
        else:
            weight = 1 / self.top
            cash_w = weight * (self.top - row_data['n'])

        for asset in self.universe_new:
            if row_data[asset] != 0:
                row_data[asset] = weight
            else:
                row_data[asset] = 0

        cash_w += row_data[self.cash_new[0]]

        if row_data[self.cash_rename[0]] > row_data[self.cash_rename[1]]:
            row_data[self.cash_new] = [cash_w, 0]
        else:
            row_data[self.cash_new] = [0, cash_w]

        return row_data


if __name__ == "__main__":
    # Series should be used for recent days (absence of BIL data)
    Series = False
    start_date = '2003-12-01'   # Recommended: '2007-02-25', '2007-12-25', '2008-07-25', '2009-12-25'
    end_date = None             # Recommended: None, '2020-12-31', '2022-10-31'

    faa = FAA(isSeries=Series, trade_start_d=start_date, end_d=end_date)
    w, wr, wr_tc, tr, tr_tc, mdd, mdd_tc, t = faa.execute()
    if not Series:
        base_function.plot(tr, mdd, tr_tc, mdd_tc, t, 'Flexible Asset Allocation', 'FAA.png')
