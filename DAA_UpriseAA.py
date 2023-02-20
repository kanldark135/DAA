import base_function
import DAA_BAA
import DAA_LAA
import DAA_ADM
import DAA_FAA

tc = 0.0025


if __name__ == '__main__':
    # Series should be used for recent days (absence of BIL data)
    Series = False
    start_date = '2003-12-01'   # Recommended: '2007-02-25', '2007-12-25', '2008-07-25', '2009-12-25'
    end_date = None    # Recommended: None, '2020-12-31', '2022-10-31'

    adm = DAA_ADM.ADM(isSeries=Series, trade_start_d=start_date, end_d=end_date)
    adm_w, adm_wr, adm_wr_tc, adm_tr, adm_tr_tc, adm_mdd, adm_mdd_tc, adm_tc = adm.execute

    baa4 = DAA_BAA.BAA(isAggressive=True, isSeries=Series, trade_start_d=start_date, end_d=end_date)
    baa4_w, baa4_wr, baa4_wr_tc, baa4_tr, baa4_tr_tc, baa4_mdd, baa4_mdd_tc, baa4_tc = baa4.execute()

    baa12 = DAA_BAA.BAA(isAggressive=False, isSeries=Series, trade_start_d=start_date, end_d=end_date)
    baa12_w, baa12_wr, baa12_wr_tc, baa12_tr, baa12_tr_tc, baa12_mdd, baa12_mdd_tc, baa12_tc = baa12.execute()

    faa = DAA_FAA.FAA(isSeries=Series, trade_start_d=start_date, end_d=end_date)
    faa_w, faa_wr, faa_wr_tc, faa_tr, faa_tr_tc, faa_mdd, faa_mdd_tc, faa_tc = faa.execute()

    laa = DAA_LAA.LAA(isSeries=Series, trade_start_d=start_date, end_d=end_date)
    laa_w, laa_wr, laa_wr_tc, laa_tr, laa_tr_tc, laa_mdd, laa_mdd_tc, laa_tc = laa.execute()

    df_list = [adm_w, baa4_w, baa12_w, faa_w, laa_w]
    w_list = [0.15, 0.35, 0.15, 0.15, 0.2]

    join_w, join_wr, join_wr_tc, join_tr, join_tr_tc, join_mdd, join_mdd_tc, join_tc = base_function.join_weights_and_get_performances(df_list, w_list, tc, end_date)

    if not Series:
        base_function.excel_data(join_tr_tc, join_w, join_mdd_tc)
        base_function.plot(join_tr, join_mdd, join_tr_tc, join_mdd_tc, join_tc, 'ADM + BAA(A) + BAA(B) + FAA + LAA', 'UAA.png')

