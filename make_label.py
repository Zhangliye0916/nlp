# -*-coding:utf-8-*-

import pandas as pd
import re
import datetime
import pymysql as mdb
from collections import defaultdict
import codecs
import gc


def sql_news(stock_code):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    cur = con.cursor(cursor=mdb.cursors.DictCursor)

    sql = "select uuid as id, date, stock_code as code from work.news where date > '2018-3-28' " \
          "and stock_code = '%s' order by date;" % stock_code

    print(sql)
    try:
        cur.execute(sql)
        con.commit()
        return cur.fetchall()

    except Exception as e:
        print(e)
        con.rollback()

    finally:
        con.close()


def sql_price(stock_code):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    cur = con.cursor(cursor=mdb.cursors.DictCursor)

    sql = "select date_time as date, stock_code as code, price_close as price from work.stock_price " \
          "where stock_code = '%s';" % stock_code

    try:
        cur.execute(sql)
        con.commit()
        return cur.fetchall()

    except Exception as e:
        print(e)
        con.rollback()

    finally:
        con.close()


def sql_date():

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    cur = con.cursor(cursor=mdb.cursors.DictCursor)

    sql = "select distinct date_time from work.stock_price order by date_time;"

    try:
        cur.execute(sql)
        con.commit()
        return cur.fetchall()

    except Exception as e:
        print(e)
        con.rollback()

    finally:
        con.close()


def insert_data(id_news, label):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    # 所有的查询，都在连接con的一个模块cursor上面运行的
    cur = con.cursor(cursor=mdb.cursors.DictCursor)
    # 执行一个查询

    sql = "update work.news_label set 240_min_direction = '%s' where id_news in %s" % (label, id_news)
    try:
        print(sql)
        cur.execute(sql)
        con.commit()

    except Exception as e:
        print(e)
        con.rollback()

    finally:
        con.close()


def sql_stock():

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    # 所有的查询，都在连接con的一个模块cursor上面运行的
    cur = con.cursor(cursor=mdb.cursors.DictCursor)
    # 执行一个查询

    sql = "SELECT distinct stock_code from work.news order by stock_code;"
    try:

        cur.execute(sql)
        con.commit()

        return cur.fetchall()

    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":

    time_predict = 240

    stock_codes = sql_stock()

    codes = []
    for item in stock_codes:
        codes.append(item["stock_code"])

    print(codes)
    for code in codes:

        print(code, datetime.datetime.now())
        news_data = sql_news(code)

        if len(news_data) == 0:
            print(code, "no news data")
            continue

        price_data = sql_price(code)

        if len(price_data) == 0:
            print(code, "no price data")
            continue

        all_price = {}

        for line in price_data:
            date = line["date"]
            price = line["price"]
            all_price[date] = price

        del price_data
        gc.collect()

        label_posi = []
        label_neg = []

        for line in news_data:

            Id = line["id"]
            date = line["date"]

            temp = sorted([t for t in all_price.keys() if t > date])
            if len(temp) < int(time_predict/5):
                continue

            date_start = temp[0]
            date_end = temp[int(time_predict/5) - 1]

            if date_start - date > datetime.timedelta(hours=66.5):
                continue

            elif date_end - date_start > datetime.timedelta(hours=72):
                continue

            price_start = all_price[date_start]
            price_end = all_price[date_end]

            if price_end - price_start >= 0:
                label_posi.append(Id)

            else:
                label_neg.append(Id)

        insert_data(tuple(label_posi), 1)
        insert_data(tuple(label_neg), 0)
