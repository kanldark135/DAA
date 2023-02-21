import pandas as pd
import base_function
import time
from datetime import date
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf

yf.pdr_override()


class BAA:
    """
    Bold Asset Allocation (Keller, 2022)
    Protective Universe, 13612W for absolute, SMA(12) for relative
    """
    def __init__(self, isAggressive=True, isSeries=False, trade_start_d=None, end_d=None, tc=0.0025, sma_l=12, B=1, prot_N=4, def_N=7, def_top=3, off_top_12=6, off_top_4=1):
        if trade_start_d is None:
            trade_start_d = str(date.today().strftime('%Y-%m-%d'))
        if isSeries:
            end_d = trade_start_d

        if time.strptime(trade_start_d, '%Y-%m-%d') < time.strptime('2003-12-01', '%Y-%m-%d'):
            print('Start Date out of bounds: must be after 2003-12-01')
            exit()

        else:
            self.protective_universe_old = ['VFINX', 'EFA', 'VEIEX', 'VBMFX']
            self.offensive_universe_4_old = ['VIGRX', 'EFA', 'VEIEX', 'VBMFX']
            self.offensive_universe_12_old = ['VFINX', 'VIGRX', 'NAESX', 'IEV', 'EWY', 'VEIEX', 'IYR', 'PCRAX', 'FKRCX', 'VWEHX', 'VWESX', 'VUSTX']

            self.protective_universe_new = ['SPY', 'VEA', 'VWO', 'AGG']
            self.offensive_universe_4_new = ['QQQ', 'VEA', 'VWO', 'AGG']
            self.offensive_universe_12_new = ['SPY', 'QQQ', 'IWM', 'IEV', 'EWY', 'VWO', 'RWX', 'DBC', 'GLD', 'HYG', 'LQD', 'TLT']

        data_start_d = str((datetime.strptime(trade_start_d, '%Y-%m-%d') - relativedelta(months=12)).date())

        self.isSeries = isSeries
        self.data_start_d = data_start_d
        self.trade_start_d = trade_start_d
        self.end_d = end_d
        self.tc = tc                                            # transaction cost - 25bp
        self.isAggressive = isAggressive                        # Balanced or Aggressive offensive universe - True or False
        self.sma_l = sma_l                                      # lookback period for relative momentum - SMA(L)
        self.B = B                                              # Breadth momentum criteria - cash if B number of ETFs is negative momentum (fixed)
        self.prot_N = prot_N                                    # Number of ETFs in protective universe (fixed)
        self.def_N = def_N                                      # Number of ETFs in defensive universe (fixed)
        self.def_top = def_top                                  # Number of ETFs to allocate weights in defensive universe (fixed)
        self.off_top_12 = off_top_12
        self.off_top_4 = off_top_4

        self.cash = 'BIL'
        self.defensive_universe_old = ['VFISX', 'VFITX', 'VUSTX', 'VBMFX', 'VWESX', 'VIPSX', 'PCRAX']   # Defensive Universe Criterion as Cash
        self.defensive_universe_new = ['BIL', 'IEF', 'TLT', 'AGG', 'LQD', 'TIP', 'DBC']

        self.offdef_universe = None                             # Declared later according to balanced / aggressive
        self.offensive_universe_old = None                          # Declared later according to balanced / aggressive
        self.offensive_universe_new = None                          # Declared later according to balanced / aggressive
        self.off_top = None                                     # Declared later according to balanced / aggressive

    def execute(self):
        """
        Execution of Bold Asset Allocation

        :return: weight_dataframe, investment_result_dataframe
        """
        print('BAA running')

        protective_universe = base_function.get_final_adj_close_data(self.protective_universe_old, self.protective_universe_new, self.data_start_d, self.end_d)
        absolute_momentum = base_function.get_rebalance_date(protective_universe, self.trade_start_d, self.sma_l)
        absolute_momentum = 12 * (absolute_momentum / absolute_momentum.shift(1)) + \
                            4 * (absolute_momentum / absolute_momentum.shift(3)) + \
                            2 * (absolute_momentum / absolute_momentum.shift(6)) + \
                            1 * (absolute_momentum / absolute_momentum.shift(12)) - 19                              # calculate 13612W for absolute momentum
        absolute_momentum.dropna(inplace=True)                                                                      # drop first k data
        absolute_momentum[absolute_momentum < 0] = None                                                             # None if momentum is negative
        absolute_momentum['n'] = absolute_momentum.count(axis=1)                                                    # get 'n' - count number of "good" assets (count if not None)

        if self.isAggressive:
            self.offensive_universe_old = self.offensive_universe_4_old                                               # declare Aggressive Offensive Universe
            self.offensive_universe_new = self.offensive_universe_4_new
            self.off_top = self.off_top_4                                                                           # declare Number of ETFs to allocate weights in Offensive Universe
        else:
            self.offensive_universe_old = self.offensive_universe_12_old                                                # declare Balanced Offensive Univpyerse
            self.offensive_universe_new = self.offensive_universe_12_new
            self.off_top = self.off_top_12                                                                          # declare Number of ETFs to allocate weights in Offensive Universe

        offensive_universe_df = base_function.get_final_adj_close_data(self.offensive_universe_old, self.offensive_universe_new, self.data_start_d, self.end_d)
        defensive_universe_df = base_function.get_final_adj_close_data(self.defensive_universe_old, self.defensive_universe_new, self.data_start_d, self.end_d)
        universe_df = pd.concat([offensive_universe_df, defensive_universe_df], axis=1)
        universe_df = universe_df.loc[:, ~universe_df.columns.duplicated()]                                        # remove duplicated columns

        self.offdef_universe = universe_df.columns.tolist()

        universe_df['BIL'] = universe_df['BIL'].fillna(method='bfill')                                               # TODO extended BIL length like this

        monthly_df = base_function.get_rebalance_date(universe_df, self.trade_start_d, self.sma_l)                           # get end of month price
        relative_momentum = (monthly_df / monthly_df.rolling(self.sma_l + 1).mean()) - 1                        # dataframe: calculate SMA(L) for relative momentum
        relative_momentum.dropna(inplace=True)                                                                      # drop first k data
        universe_df = universe_df[~(universe_df.index < relative_momentum.index[0])]                                # drop first k data
        universe_df = (universe_df / universe_df.shift(1)).fillna(1)                                                    # get daily price change rate
        relative_momentum = pd.concat([relative_momentum, absolute_momentum['n']], axis=1)                          # join absolute momentum measure to dataframe

        weight_df = relative_momentum.copy().apply(self.get_weight, axis=1)                                     # dataframe: get weight according to relative and absolute momentum
        del weight_df['n']                                                                                          # delete 'n' column

        if self.isSeries:
            print('series date: {}'.format(str(weight_df.index[0])[:10]))
            print('Got a single weight series - no backtesting')
            weight_df.to_csv('./Series/{} BAA Weight Series.csv'.format(str(weight_df.index[0])[:10]))

            return weight_df, None, None, None, None, None, None, None

        print('start date: {}, end date: {} \n'.format(str(universe_df.index[0])[:10], str(universe_df.index[-1])[:10]))

        weighted_return_df = base_function.get_performance(universe_df, weight_df, self.offdef_universe)
        weighted_return_tc_df, tc_df = base_function.get_performance_with_tc(universe_df, weight_df, self.tc, self.offdef_universe)
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
        """
        Dataframe Row Operation (apply) - assign weight to Offensive Universe if Absolute Momentum is mostly positive,
        or assign weight to Defensive Universe

        :param row_data: row data with relative momentum values and absolute momentum result for a single date
        :return: update the row data with new weight for offdef_universe
        """
        if row_data['n'] > (4 - self.B):                                                                        # All Absolute Momentum is positive
            asset_rank = row_data[self.offensive_universe_new].rank(ascending=False)                                    # Get Rank for Offensive Universe
            weight = 1 / self.off_top                                                                               # Determine weight according to self.off_top
            for asset in self.offdef_universe:                                                                      # Iteration for offdef_universe
                if asset in self.offensive_universe_new:                                                                    # Check for Offensive Universe
                    if asset_rank[asset] <= self.off_top:                                                                   # If rank within self.off_top, assign weight
                        row_data[asset] = weight
                    else:                                                                                                   # If rank not within self.off_top, assign 0
                        row_data[asset] = 0
                else:                                                                                                   # If not in Offensive Universe, assign 0
                    row_data[asset] = 0
        else:                                                                                                   # Some Absolute Momentum is negative
            asset_rank = row_data[self.defensive_universe_new].rank(ascending=False)                                    # Get Rank for Defensive Universe
            weight = 1 / self.def_top                                                                               # Determine weight according to self.def_top
            cash_rank = asset_rank[self.cash]                                                                       # Get rank for 'BIL'
            if cash_rank <= self.def_top:                                                                           # Assign weight to 'BIL' if within self.def_top
                cash_weight = (self.def_top - cash_rank + 1) * weight
            else:                                                                                                   # Assign 0 to 'BIL' if within self.def_top
                cash_weight = 0
            for asset in self.offdef_universe:                                                                      # Iteration for offdef_universe
                if asset in self.defensive_universe_new:                                                                    # Check for Defensive Universe
                    if asset == self.cash:                                                                                  # If 'BIL', assign cash_weight
                        row_data[asset] = cash_weight
                    elif (asset_rank[asset] <= self.def_top) and (asset_rank[asset] < cash_rank):                           # If rank within self.def_top and better than 'BIL', assign weight
                        row_data[asset] = weight
                    else:                                                                                                   # If rank not within self.def_top or worse than 'BIL', assign 0
                        row_data[asset] = 0
                else:                                                                                                   # If not in Defensive Universe, assign 0
                    row_data[asset] = 0

        return row_data


if __name__ == '__main__':
    # Series should be used for recent days (absence of BIL data)
    Aggressive = False
    Series = False
    start_date = '2003-12-01'   # Recommended: '2007-02-25', '2007-12-25', '2008-07-25', '2009-12-25'
    end_date = None     # Recommended: None, '2020-12-31', '2022-10-31'

    baa = BAA(isAggressive=Aggressive, isSeries=Series, trade_start_d=start_date, end_d=end_date)
    w, wr, wr_tc, tr, tr_tc, mdd, mdd_tc, t = baa.execute()
    if not Series:
        base_function.plot(tr, mdd, tr_tc, mdd_tc, t, 'Bold Asset Allocation (Aggressive)', 'BAA.png')
