# -*-coding:utf-8-*-

import os
import pandas as pd
import re
import datetime
import pymysql as mdb
import codecs


def insert_data(file_data, code):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    cur = con.cursor()

    for ind in file_data.index[:-1]:

        line = file_data.loc[ind]
        date = line[0].split("/")
        hours = int(line[1]/100)
        minites = int(line[1]) % 100
        date_time = datetime.datetime(int(date[0]), int(date[1]), int(date[2]), hours, minites)
        close = line[5]
        now = datetime.datetime.now()
        # data.append((600000, date_time, 100, now))
        sql = "INSERT INTO work.stock_price (idstock_price, stock_code, date_time, price_close, ktype, updated_time) \
                 VALUES (replace(uuid(),'-',''), '%s', '%s', '%s', '5min', '%s');" % (code, date_time, close, now)

        try:
            cur.execute(sql)
            con.commit()

        except Exception as e:
            print(e)
            con.rollback()

    con.close()


def insert_data_batch(file_data, code):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    cur = con.cursor()

    sql = "INSERT INTO work.stock_price (idstock_price, stock_code, date_time, price_close, ktype, updated_time) \
             VALUES (replace(uuid(),'-',''), %s, %s, %s, '5min', %s);"

    data = []
    for ind in file_data.index[:-1]:

        line = file_data.loc[ind]
        date = line[0].split("/")
        hour = int(line[1]/100)
        min = int(line[1]) % 100
        date_time = datetime.datetime(int(date[0]), int(date[1]), int(date[2]), hour, min)
        close = line[5]
        now = datetime.datetime.now()
        data.append((code, date_time, str(close), now))

    try:
        cur.executemany(sql, data)
        con.commit()

    except Exception as e:
        print(e)
        con.rollback()

    con.close()


if __name__ == "__main__":

    Path1 = 'D:/数据/通达信5minA股价格数据（截至20190807）'
    Path2 = 'C:/Users/text/Desktop/text_classifier/'
    f = codecs.open(Path2 + "codes.txt", encoding="utf-8-sig")
    codes = f.read().split("\n")

    file_names = []
    for Code in codes:
        file_names.append("SH#" + Code.replace("\r", "").zfill(6))
        file_names.append("SZ#" + Code.replace("\r", "").zfill(6))

    print(file_names)
    for num, file_name in enumerate(file_names):

        stock_code = "".join(re.findall(r"\d", file_name))
        print(num, stock_code, datetime.datetime.now())

        try:
            FileData = pd.read_csv(Path1 + "/" + file_name + ".csv", encoding="gbk", header=None)[[0, 1, 5]]

        except IOError:
            print(file_name)
            continue

        try:
            insert_data_batch(FileData, str(stock_code))
            # insert_data(FileData, str(stock_code))
        except Exception as e:
            print(stock_code, e)


