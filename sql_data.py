# -*-coding:utf-8-*-

import pymysql as mdb
# import re
from jqdatasdk import *
import datetime
from scipy import stats
import numpy as np


def sql_news(type_name, start_date, end_date):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    cur = con.cursor(cursor=mdb.cursors.DictCursor)

    sql = "SELECT a.uuid, a.date, a.stock_code as code FROM work.news as a " \
          "left join work.news_type as b on a.uuid = b.id_news where b.type = '{}' and a.date > '{}' " \
          "and a.date < '{}';" .format(type_name, start_date, end_date)

    # print(sql)

    try:
        cur.execute(sql)
        con.commit()
        return cur.fetchall()

    except Exception as e:
        print(e)
        return None

    finally:
        con.close()


def sql_news2(type_name, start_date, end_date):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    cur = con.cursor(cursor=mdb.cursors.DictCursor)

    sql = "SELECT a.uuid, a.date, a.stock_code as code, c.stock_pool_name FROM work.news as a " \
          "left join work.news_type as b on a.uuid = b.id_news left join work.stock_pool as c " \
          "on a.stock_code = c.stock_code where b.type = '{}' and c.stock_pool_name = '沪深300' " \
          "and a.date > '{}' and a.date < '{}';".format(type_name, start_date, end_date)

    # print(sql)

    try:
        cur.execute(sql)
        con.commit()
        return cur.fetchall()

    except Exception as e:
        print(e)
        return None

    finally:
        con.close()


def stock_size_pattern(text):

    percent = re.findall(r"\d+\.?\d*%", text)
    if len(percent) > 0:
        return "|".join(percent), "percent"

    stock = re.findall(r"[1-9]+[0-9,]*\.?[0-9,]*[亿|万]?股", text)
    if len(stock) > 0:
        return "|".join(stock), "stock"

    cash = re.findall(r"[1-9]+[0-9,]*\.?[0-9,]*[亿|万]?元|[1-9]+[0-9,]*\.?[0-9,]*[亿|万]", text)
    if len(cash) > 0:
        return "|".join(cash), "cash"

    return "", "other"


def position_pattern(text):

    person = re.findall(r"(董监高|董事|监事|高管|总裁|实控人|管理人员|副总|控制人|管理层|总经理|员工|总监|董秘)", text)
    if len(person) > 0:
        return "person"

    company = re.findall(r"(股东|证金|基金|证券)", text)
    if len(company) > 0:
        return "company"

    return "other"


def write_data(path, news_type):

    auth("18612754762", "xyz117")

    with open(path, "w") as f3:

        news_data = sql_news(news_type, "2014-1-1", "2015-1-1")

        drop_duplicate = set()

        for num, item in enumerate(news_data):

            if num % 10 == 0:
                print(num)

            uuid = item["uuid"]
            date = datetime.datetime.strftime(item["date"], "%Y-%m-%d")
            stock = item["code"]

            if date + stock not in drop_duplicate:

                drop_duplicate.add(date + stock)

                if stock[0] == "6":
                    stock += ".XSHG"

                else:
                    stock += ".XSHE"

                try:
                    data = jq_bars_data(stock, date)
                    f3.write(uuid + " " + " ".join(data) + "\n")

                except Exception as e:
                    # print(uuid, stock, e)
                    pass

            else:
                continue


def jq_val_data(date):

    return get_fundamentals(query(valuation.code, valuation.circulating_cap.label("vol"),
                                  valuation.turnover_ratio.label("turnover"),
                                  valuation.circulating_market_cap.label("cap"), valuation.pe_ratio.label("pe"),
                                  valuation.pb_ratio.label("pb"), valuation.ps_ratio.label("ps"),
                                  valuation.pcf_ratio.label("pcf")), date).fillna(0)


def jq_indicator_data(stat_date):

    return get_fundamentals(query(valuation.code, indicator.roe, indicator.roa,
                                  indicator.gross_profit_margin.label("gross"),
                                  indicator.inc_revenue_year_on_year.label("income"),
                                  indicator.inc_net_profit_year_on_year.label("return")), stat_date).fillna(0)


def jq_bars_data(stock_in, date_in):

    price = get_bars(stock_in, 22, unit='1d',
                     fields=['date', 'close', 'volume'],
                     include_now=True, end_dt=date_in, fq_ref_date=None)

    price = price.fillna(method="ffill")

    pct_1d = "%.2f" % float(price["close"][21] / price["close"][20] - 1.)
    pct_5d = "%.2f" % float(price["close"][21] / price["close"][16] - 1.)
    pct_22d = "%.2f" % float(price["close"][21] / price["close"][0] - 1.)

    pct_1d_v = "%.2f" % float(price["volume"][21] / price["volume"][20] - 1.)
    pct_5d_v = "%.2f" % float(price["volume"][21] / price["volume"][16] - 1.)
    pct_22d_v = "%.2f" % float(price["volume"][21] / price["volume"][0] - 1.)

    return [stock_in, pct_1d, pct_5d, pct_22d, pct_1d_v, pct_5d_v, pct_22d_v]


def jq_bars_data2(stock_in, date_e, bars):

    price = get_bars(stock_in, bars[0] + 1, unit='1d',
                     fields=['date', 'close'],
                     include_now=False, end_dt=date_e, fq_ref_date=None)

    price = price.fillna(method="ffill")

    p1 = price["close"][bars[2]] / price["close"][bars[1]] - 1.
    p2 = price["close"][bars[3]] / price["close"][bars[1]] - 1.
    p3 = price["close"][bars[4]] / price["close"][bars[1]] - 1.

    return p1, p2, p3


def stat_price(news_data, trade_days, mark, bars, with_mark=True):

    drop_duplicate = set()

    signs = [0]*3

    return1 = []
    return2 = []
    return3 = []

    count = 0

    for num, item in enumerate(news_data):

        if num % 100 == 0:
            # print(num)
            pass

        date_ori = item["date"].date()

        if item["date"].hour >= 15:
            dates_after = [item for item in trade_days if item > date_ori]

        else:
            dates_after = [item for item in trade_days if item >= date_ori]

        date_p = [dates_after[ind].strftime('%Y-%m-%d') for ind in bars[1:]]
        date_end = dates_after[bars[0] + 1].strftime('%Y-%m-%d')

        if not with_mark:
            mr1, mr2, mr3 = 0, 0, 0

        else:
            # print(date1, date2, date3, date0)
            mr1 = mark[date_p[1]] / mark[date_p[0]] - 1
            mr2 = mark[date_p[2]] / mark[date_p[0]] - 1
            mr3 = mark[date_p[3]] / mark[date_p[0]] - 1

        stock = item["code"]

        if datetime.datetime.strftime(date_ori, "%Y-%m-%d") + stock not in drop_duplicate:

            drop_duplicate.add(datetime.datetime.strftime(date_ori, "%Y-%m-%d") + stock)

            if stock[0] == "6":
                stock += ".XSHG"

            else:
                stock += ".XSHE"

            try:
                r1, r2, r3 = jq_bars_data2(stock, date_end, bars)

                return1.append(r1 - mr1)
                return2.append(r2 - mr2)
                return3.append(r3 - mr3)

                if r1 - mr1 > 0.:
                    signs[0] += 1

                if r2 - mr2 > 0.:
                    signs[1] += 1

                if r3 - mr3 > 0.:
                    signs[2] += 1

                count += 1

            except Exception as e:
                # print(item["uuid"], stock, e)
                pass

        else:
            continue

    t_val1, p_val1 = stats.ttest_1samp(return1, 0)
    t_val2, p_val2 = stats.ttest_1samp(return2, 0)
    t_val3, p_val3 = stats.ttest_1samp(return3, 0)

    # print(return1)
    # print(np.mean(return1), np.mean(return2), np.mean(return3))
    # print(np.std(return1), np.std(return2), np.std(return3))
    # print(Counter(dates))

    if count == 0:
        print(type_name, year + start_dates[md], year + end_dates[md],
              np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0)

        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0

    rate1 = signs[0] / count
    rate2 = signs[1] / count
    rate3 = signs[2] / count

    print(type_name, year + start_dates[md], year + end_dates[md],
          rate1, rate2, rate3, np.mean(return1), np.mean(return2), np.mean(return3), p_val1, p_val2, p_val3, count)

    return rate1, rate2, rate3, np.mean(return1), np.mean(return2), np.mean(return3), p_val1, p_val2, p_val3, count


def get_bench(start_date, end_date, index_name):

    out = get_price(index_name, start_date=start_date, end_date=end_date, frequency='daily', fields=None,
                    skip_paused=False, fq='pre')["close"]

    # print(out)
    date_index = []
    for d in out.index:
        date_index.append(d.to_pydatetime().strftime("%Y-%m-%d"))

    out.index = date_index

    return out


if __name__ == "__main__":

    # auth("15811211802", "211802")
    auth("18612754762", "xyz117")
    print(get_query_count())
    
    tradeDays = get_trade_days(start_date="2011-1-1", end_date="2020-10-1")
    bench = get_bench("2011-1-1", "2020-9-20", "000842.XSHG")

    # start_dates = ["2012-1-1", "2013-1-1", "2014-1-1", "2015-1-1", "2016-1-1", "2017-1-1", "2018-1-1"]
    # end_dates = ["2013-1-1", "2014-1-1", "2015-1-1", "2016-1-1", "2017-1-1", " 2018-1-1", "2019-9-1"]

    # start_dates = ["2019-1-1", "2019-2-1", "2019-3-1", "2019-4-1", "2019-5-1", "2019-6-1", "2019-7-1", "2019-8-1"]
    # end_dates = ["2019-2-1", "2019-3-1", "2019-4-1", "2019-5-1", "2019-6-1", "2019-7-1", "2019-8-1", "2019-9-1"]

    start_dates = ["-1-1", "-2-1", "-3-1", "-4-1", "-5-1", "-6-1", "-7-1", "-8-1", "-9-1", "-10-1", "-11-1", "-12-1"]
    end_dates = ["-2-1", "-3-1", "-4-1", "-5-1", "-6-1", "-7-1", "-8-1", "-9-1", "-10-1", "-11-1", "-12-1", "-12-31"]

    # start_dates = ["-1-1"]
    # end_dates = ["-12-31"]

    # start_dates = ["2018-9-1", "2018-10-1", "2018-11-1", "2018-12-1"]
    # end_dates = ["2018-10-1", "2018-11-1", "2018-12-1", "2019-1-1"]

    # years = ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019"]
    # years = ["2017"]
    years = ["2018"]
    type_name = "龙虎榜"

    for year in years:
        for md in range(len(start_dates)):

            newsData = sql_news(type_name, year + start_dates[md], year + end_dates[md])

            stat_price(newsData, tradeDays, bench, [5, 0, 1, 3, 5], True)
