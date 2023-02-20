import main
import base_function

if __name__ == '__main__':

    Series = False
    start_date = '2023-01-09'
    end_date = None

    main = main.MAIN(isSeries=Series, trade_start_d=start_date, end_d=end_date)
    join_w1, join_w2, join_w3 = main.execute()  # TODO join_w1(100_0) join_w2(80_20), join_w3(60_40)

    # TODO insert into mySQL DB
    base_function.insertDB(join_w1, 7, 1)
    base_function.insertDB(join_w2, 7, 2)
    base_function.insertDB(join_w3, 7, 3)