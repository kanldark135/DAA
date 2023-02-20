import os

import base_function
import DAA_ADM
import DAA_BAA
import DAA_LAA
import DAA_FAA
import yfinance as yf

yf.pdr_override()

tc = 0.0025

class MAIN:
    def __init__(self, isSeries=False, trade_start_d=None, end_d=None):
        self.Series = isSeries
        self.start_d = trade_start_d
        self.end_d = end_d

    def execute(self):
        print("Main running")
        adm = DAA_ADM.ADM(isSeries=self.Series, trade_start_d=self.start_d, end_d=self.end_d)
        adm_w, adm_wr, adm_wr_tc, adm_tr, adm_tr_tc, adm_mdd, adm_mdd_tc, adm_tc = adm.execute()
        if not self.Series:
            base_function.plot(adm_tr, adm_mdd, adm_tr_tc, adm_mdd_tc, adm_tc, 'Accelerated Dual Momentum', 'ADM.png')

        baa4 = DAA_BAA.BAA(isAggressive=True, isSeries=self.Series, trade_start_d=self.start_d, end_d=self.end_d)
        baa4_w, baa4_wr, baa4_wr_tc, baa4_tr, baa4_tr_tc, baa4_mdd, baa4_mdd_tc, baa4_tc = baa4.execute()
        if not self.Series:
            base_function.plot(baa4_tr, baa4_mdd, baa4_tr_tc, baa4_mdd_tc, baa4_tc, 'Bold Asset Allocation (Aggressive)', 'BAA(A).png')

        baa12 = DAA_BAA.BAA(isAggressive=False, isSeries=self.Series, trade_start_d=self.start_d, end_d=self.end_d)
        baa12_w, baa12_wr, baa12_wr_tc, baa12_tr, baa12_tr_tc, baa12_mdd, baa12_mdd_tc, baa12_tc = baa12.execute()
        if not self.Series:
            base_function.plot(baa12_tr, baa12_mdd, baa12_tr_tc, baa12_mdd_tc, baa12_tc, 'Bold Asset Allocation (Balanced)', 'BAA(B).png')

        faa = DAA_FAA.FAA(isSeries=self.Series, trade_start_d=self.start_d, end_d=self.end_d)
        faa_w, faa_wr, faa_wr_tc, faa_tr, faa_tr_tc, faa_mdd, faa_mdd_tc, faa_tc = faa.execute()
        if not self.Series:
            base_function.plot(faa_tr, faa_mdd, faa_tr_tc, faa_mdd_tc, faa_tc, 'Flexible Asset Allocation', 'FAA.png')

        laa = DAA_LAA.LAA(isSeries=self.Series, trade_start_d=self.start_d, end_d=self.end_d)
        laa_w, laa_wr, laa_wr_tc, laa_tr, laa_tr_tc, laa_mdd, laa_mdd_tc, laa_tc = laa.execute()
        if not self.Series:
            base_function.plot(laa_tr, laa_mdd, laa_tr_tc, laa_mdd_tc, laa_tc, 'Lethargic Asset Allocation', 'LAA.png')

        df_list = [adm_w, baa4_w, baa12_w, faa_w, laa_w]
        w_list = [0.15, 0.35, 0.15, 0.15, 0.2]
        join_w, join_wr, join_wr_tc, join_tr, join_tr_tc, join_mdd, join_mdd_tc, join_tc = base_function.join_weights_and_get_performances(df_list, w_list, tc, self.end_d)
        if not self.Series:
            base_function.plot(join_tr, join_mdd, join_tr_tc, join_mdd_tc, join_tc, 'ADM + BAA(A) + BAA(B) + FAA + LAA', 'UAA (0.15.0.35.0.15.0.15.0.2).png')

        join_w2, _, _, mix_tr, mix_tr_tc, mix_mdd, mix_mdd_tc, mix_tc = base_function.mix_strategy_bond(w_df=join_w, bond_list=['SHY', 'BIL'], mix_weight=[0.6, 0.4], ed=self.end_d)
        if not self.Series:
            base_function.plot(mix_tr, mix_mdd, mix_tr_tc, mix_mdd_tc, mix_tc, 'ADM + BAA(A) + BAA(B) + FAA + LAA (0.6) + Bond(0.4)', 'UAA (0.15.0.35.0.15.0.15.0.2) 0.6.png')

        _, _, _, mix_tr, mix_tr_tc, mix_mdd, mix_mdd_tc, mix_tc = base_function.mix_strategy_bond(w_df=join_w, bond_list=['SHY', 'BIL'], mix_weight=[0.7, 0.3], ed=self.end_d)
        if not self.Series:
            base_function.plot(mix_tr, mix_mdd, mix_tr_tc, mix_mdd_tc, mix_tc, 'ADM + BAA(A) + BAA(B) + FAA + LAA (0.7) + Bond(0.3)', 'UAA (0.15.0.35.0.15.0.15.0.2) 0.7.png')

        join_w3, _, _, mix_tr, mix_tr_tc, mix_mdd, mix_mdd_tc, mix_tc = base_function.mix_strategy_bond(w_df=join_w, bond_list=['SHY', 'BIL'], mix_weight=[0.8, 0.2], ed=self.end_d)
        if not self.Series:
            base_function.plot(mix_tr, mix_mdd, mix_tr_tc, mix_mdd_tc, mix_tc, 'ADM + BAA(A) + BAA(B) + FAA + LAA (0.8) + Bond(0.2)', 'UAA (0.15.0.35.0.15.0.15.0.2) 0.8.png')

        return join_w, join_w3, join_w2     # TODO (적극형, 위험형, 안전형)


if __name__ == "__main__":
    os.makedirs('./Dataframes/', exist_ok=True)
    os.makedirs('./Results/', exist_ok=True)
    os.makedirs('./Dataframes/60_40/', exist_ok=True)
    os.makedirs('./Dataframes/70_30/', exist_ok=True)
    os.makedirs('./Dataframes/80_20/', exist_ok=True)

    # Series should be used for recent days (absence of BIL data)
    Series = False
    start_date = '2022-12-05'   # Recommended: '2007-02-25', '2007-12-25', '2008-07-25', '2009-12-25'
    end_date = None      # Recommended: None, '2020-12-31', '2022-10-31'

    main = MAIN(isSeries=Series, trade_start_d=start_date, end_d=end_date)
    join_w1, join_w2, join_w3 = main.execute()