import pandas as pd
import yfinance as yf
import base_function
import time
from datetime import date
from datetime import datetime
from dateutil.relativedelta import relativedelta

yf.pdr_override()


class ADM:
    def __init__(self, isSeries=False, trade_start_d=None, end_d=None, top=6, tc=0.0025, lookback_1=1, lookback_2=3, lookback_3=6):
        if trade_start_d is None:
            trade_start_d = str(date.today().strftime('%Y-%m-%d'))
        if isSeries:
            end_d = trade_start_d       # series 찾을 떄는 trade_start_d와 end_d가 같아야 함!
        if time.strptime(trade_start_d, '%Y-%m-%d') < time.strptime('2003-12-01', '%Y-%m-%d'):
            print('Start Date out of bounds: must be after 2003-12-01')
            exit()

        else:
            self.universe_old = ['VFINX', 'VIGRX', 'IEV', 'EWY', 'VEIEX', 'IYR', 'FKRCX', 'PCRAX', 'VBMFX', 'VUSTX', 'VFITX']
            self.universe_new = ['SPY', 'QQQ', 'IEV', 'EWY', 'VWO', 'RWX', 'GLD', 'DBC', 'AGG', 'TLT', 'IEF']

        data_start_d = str((datetime.strptime(trade_start_d, '%Y-%m-%d') - relativedelta(months=6)).date())

        self.isSeries = isSeries
        self.data_start_d = data_start_d
        self.trade_start_d = trade_start_d
        self.top = top
        self.end_d = end_d
        self.tc = tc
        self.lookback_1 = lookback_1  # momentum lookback period length
        self.lookback_2 = lookback_2
        self.lookback_3 = lookback_3

        self.cash_old = ['VFISX', 'VFISX']
        self.cash_new = ['SHY', 'BIL']

    def execute(self):
        print('ADM running')

        universe_df = base_function.get_final_adj_close_data(self.universe_old, self.universe_new, self.data_start_d, self.end_d)
        cash_df = base_function.get_final_adj_close_data(self.cash_old, self.cash_new, self.data_start_d, self.end_d)
        all_universe_df = pd.concat([universe_df, cash_df], axis=1)

        monthly_df = base_function.get_rebalance_date(all_universe_df, self.trade_start_d, self.lookback_3)     # 월 리벨런싱 일 데이터만 빼오기
        momentum_df = monthly_df / monthly_df.shift(self.lookback_1) + monthly_df / monthly_df.shift(self.lookback_2) +\
                      monthly_df / monthly_df.shift(self.lookback_3) - 3    # momentum 계산식 (1,3,6달 모멘텀)

        momentum_df.dropna(inplace=True)    # 첫 n개 데이터 버리기
        momentum_df = momentum_df.rank(ascending=False, axis=1)     # 모멘텀 랭크 매기기
        weight_df = momentum_df.copy().apply(self.get_weight, axis=1)     # 모멘텀 랭크를 기반으로 weight 지정하기

        if self.isSeries:
            print('series date: {}'.format(str(weight_df.index[0])[:10]))
            print('Got a single weight series - no backtesting')
            weight_df.to_csv('./Series/{} ADM Weight Series.csv'.format(str(weight_df.index[0])[:10]))

            return weight_df, None, None, None, None, None, None, None

        universe_df = all_universe_df.loc[weight_df.index[0]:]
        universe_df = (universe_df / universe_df.shift(1)).fillna(1)

        print('Backtest'
              ''
              ' start date: {}, end date: {} \n'.format(str(universe_df.index[0])[:10], str(universe_df.index[-1])[:10]))

        weighted_return_df = base_function.get_performance(universe_df, weight_df, list(weight_df.columns))
        weighted_return_tc_df, tc_df = base_function.get_performance_with_tc(universe_df, weight_df, self.tc, list(weight_df.columns))
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
        if row_data[self.cash_new[0]] < row_data[self.cash_new[1]]:     # compare SHY and BIL first
            cash = self.cash_new[0]
            cash_rank = row_data[cash]
        else:
            cash = self.cash_new[1]
            cash_rank = row_data[cash]

        for asset in row_data.index:
            if row_data[asset] <= min(cash_rank, self.top):
                row_data[asset] = 1 / self.top
            else:
                row_data[asset] = 0

        if cash_rank < self.top:
            row_data[cash] *= (self.top - cash_rank + 1)

        return row_data


if __name__ == '__main__':
    # Series should be used for recent days (absence of BIL data)
    Series = False
    start_date = '2023-01-20'         # Recommended: '2007-02-25', '2007-12-25', '2008-07-25', '2009-12-25'
    end_date = None            # Recommended: None, '2020-12-31', '2022-10-31'

    adm = ADM(isSeries=Series, trade_start_d=start_date, end_d=end_date)    # class declaration
    w, wr, wr_tc, tr, tr_tc, mdd, mdd_tc, t = adm.execute()
    if not Series:
        base_function.plot(tr, mdd, tr_tc, mdd_tc, t, 'Accelerated Dual Momentum', 'ADM.png')
