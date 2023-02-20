import pandas as pd
import math


def get_cumulative_return(return_df):
    start = return_df.iloc[0]
    end = return_df.iloc[-1]
    cum_ret = ((end - start) / start) * 100

    return cum_ret


def get_cagr(return_df):
    start = return_df.iloc[0]
    end = return_df.iloc[-1]
    yr = len(return_df) / 365
    cagr = ((end / start) ** (1 / yr) - 1) * 100

    return cagr


def get_sd(return_df):
    daily_return = (return_df / return_df.shift(1)).fillna(1)
    daily_return = (daily_return - 1) * 100
    sd = (daily_return.std()) * math.sqrt(365)

    return sd


def get_mdd(mdd_df):
    mdd = mdd_df.min()

    return mdd


class Analyze:
    def __init__(self, r, m, r_tc, m_tc, r_kr, m_kr, tc, col1='X TC', col2='USD', col3='KRW'):
        self.return_df = r
        self.mdd_df = m
        self.return_tc_df = r_tc
        self.mdd_tc_df = m_tc
        self.return_krw_df = r_kr
        self.mdd_krw_df = m_kr
        self.tc_df = tc
        self.row = ['Cumulative Return', 'CAGR', 'Annualized S.D.', 'MDD', 'Sharpe Ratio', 'Calmar Ratio', 'Turnover rate', 'Transaction Cost']
        self.col = [col1, col2, col3]

    def get_analysis_table(self):
        analysis_df = pd.DataFrame(columns=self.col, index=self.row)

        cagr_list = [get_cagr(self.return_df), get_cagr(self.return_tc_df), get_cagr(self.return_krw_df)]
        analysis_df.loc['Cumulative Return'] = [get_cumulative_return(self.return_df), get_cumulative_return(self.return_tc_df), get_cumulative_return(self.return_krw_df)]
        analysis_df.loc['CAGR'] = cagr_list
        analysis_df.loc['Annualized S.D.'] = [get_sd(self.return_df), get_sd(self.return_tc_df), get_sd(self.return_krw_df)]
        analysis_df.loc['MDD'] = [get_mdd(self.mdd_df), get_mdd(self.mdd_tc_df), get_mdd(self.mdd_krw_df)]
        analysis_df.loc['Sharpe Ratio'] = analysis_df.loc['CAGR'] / abs(analysis_df.loc['Annualized S.D.'])
        analysis_df.loc['Calmar Ratio'] = analysis_df.loc['CAGR'] / abs(analysis_df.loc['MDD'])
        analysis_df.loc['Turnover rate'] = self.get_turnover()
        analysis_df.loc['Transaction Cost'] = [cagr_list[0] - cagr_list[1], cagr_list[0] - cagr_list[1], cagr_list[0] - cagr_list[1]]

        analysis_df = analysis_df.astype(float).round(2)
        analysis_df.to_csv('./Dataframes/Analysis.csv')

        return analysis_df

    def get_annual_returns(self):
        rdf = self.return_tc_df.copy()
        krw_df = self.return_krw_df.copy()
        rdf = pd.DataFrame(rdf)
        krw_df = pd.DataFrame(krw_df)
        rdf['year'] = rdf.index.year
        krw_df['year'] = krw_df.index.year
        rdf['month'] = rdf.index.month

        yearly_df = rdf.drop_duplicates(['year'], keep='last')
        del yearly_df['year']
        del yearly_df['month']
        yearly_df = yearly_df / yearly_df.shift(1)
        yearly_df = (yearly_df - 1) * 100
        yearly_df.index = pd.to_datetime(yearly_df.index, format='%Y-%m-%d').year
        yearly_df = yearly_df.round(1)

        krw_df = krw_df.drop_duplicates(['year'], keep='last')
        del krw_df['year']
        krw_df = krw_df / krw_df.shift(1)
        krw_df = (krw_df - 1) * 100
        krw_df.index = pd.to_datetime(krw_df.index, format='%Y-%m-%d').year
        krw_df = krw_df.round(1)

        monthly_df = rdf.drop_duplicates(['year', 'month'], keep='last')
        del monthly_df['year']
        del monthly_df['month']
        monthly_df = monthly_df / monthly_df.shift(1)
        monthly_df.dropna(inplace=True)
        monthly_df = (monthly_df - 1) * 100
        monthly_df['year'] = pd.to_datetime(monthly_df.index, format='%Y-%m-%d').year
        monthly_df['month'] = pd.to_datetime(monthly_df.index, format='%Y-%m-%d').month
        monthly_df = monthly_df.round(1)
        monthly_df.columns = ['rate', 'year', 'month']
        monthly_df = monthly_df.pivot(index='year', columns='month', values='rate')
        monthly_df['USD'] = yearly_df
        monthly_df['KRW'] = krw_df

        monthly_df.to_csv('./Dataframes/Monthly Return.csv')

        return monthly_df

    def get_turnover(self):
        to = (self.tc_df / 0.0025) * 100
        try:
            to_rate = (to.sum()[0] / (len(self.tc_df) / 12)) / 2
        except:
            to_rate = 0

        return [to_rate, to_rate, to_rate]
