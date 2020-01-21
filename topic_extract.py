# -*-coding:utf-8-*-

import pymysql as mdb
import datetime
import jieba
import numpy as np
import re
from jieba import posseg
from collections import defaultdict


def sql_news(start, end):

    con = None
    try:
        con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
        cur = con.cursor()

        sql = "SELECT a.title FROM work.news as a where a.date > '%s' and a.date < '%s';" % (start, end)

        cur.execute(sql)
        result = cur.fetchall()

        return result

    finally:
        if con:
            con.close()


if __name__ == "__main__":

    test = sql_news("2019-7-27", "2019-8-27")

    data = []
    for num, item in enumerate(test):
        if "融资融券" in item[0]:
            continue

        text = posseg.cut(item[0])
        for x in text:
            if x.flag[0] in ("n", "v"):
                data.append(x.word)

    data_2gram = []

    for i in range(len(data) - 1):
        data_2gram.append(data[i] + data[i + 1])

    output = defaultdict(int)

    for word in data_2gram:
        output[word] += 1

    output = sorted(output.items(), key=lambda y: y[1], reverse=True)

    for item in output[: 1000]:
        print(item[0], item[1])
