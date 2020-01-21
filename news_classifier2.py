# -*-coding:utf-8-*-

import pymysql as mdb
import datetime
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import jieba
import warnings
import gc
from sklearn.utils import shuffle
import codecs
from collections import defaultdict
import re

# 连接mysql的方法：connect('ip','user','password','db_name')
con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
# 所有的查询，都在连接con的一个模块cursor上面运行的
cur = con.cursor(cursor=mdb.cursors.DictCursor)


def sql_data(stock_code):
    con = None

    try:
        # 连接mysql的方法：connect('ip','user','password','db_name')
        con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
        # 所有的查询，都在连接con的一个模块cursor上面运行的
        cur = con.cursor(cursor=mdb.cursors.DictCursor)
        # 执行一个查询

        sql = "SELECT news.uuid as id, news.title as title \
               from work.news where stock_code = '%s';" % stock_code

        cur.execute(sql)
        con.commit()
        result = cur.fetchall()
        return result

    finally:
        con.close()


def sql_stock():

    sql = "SELECT distinct stock_code from work.news order by stock_code;"

    try:

        cur.execute(sql)
        con.commit()

        return cur.fetchall()

    except Exception as e:
        print(e)
        return None


def insert_data(ids, type, title, stock_code):

    sql = "INSERT INTO work.news_type (idnews_type, id_news, title, type, " \
          "stock_code) VALUES (replace(uuid(),'-',''), '%s', '%s', " \
          "'%s', '%s');" % (ids, title, type, stock_code)

    try:
        cur.execute(sql)
        con.commit()

    except Exception as e:
        print(e)
        con.rollback()


def rules(title):

    if "大宗交易" in title:
        return "大宗交易", True

    elif "月" in title and "日" in title and \
            ("快速反弹" in title or "开盘涨幅达" in title or "快速上涨" in title or "开盘涨停" in title or "盘中涨幅达"
             in title or "火箭发射" in title or "打开跌停" in title or "盘中涨停" in title):
        return "盘口大涨", True

    elif "月" in title and "日" in title and \
            ("打开涨停" in title or "加速下跌" in title or "开盘跌幅达" in title or "快速回调" in title or "盘中跌幅达"
             in title or "盘中跌停" in title or "开盘跌停" in title or "高台跳水" in title):
        return "盘口大跌", True

    elif "融资融券" in title:
        return "融资融券", True

    elif "龙虎榜" in title:
        return "龙虎榜", True

    elif "超大单流入排名" in title:
        return "主力流入", True

    elif "超大单流出排名" in title:
        return "主力流出", True

    elif "问询" in title and ("上交所" in title or "深交所" in title):
        return "交易所问询", True

    elif "增持" in title and "评级" not in title and ("股" in title or "%" in title):
        return "股东增持", True

    elif ("减持" in title or "抛" in title) and "评级" not in title and ("股" in title or "%" in title):
        return "股东减持", True

    elif ("证券：" in title or "中金" in title or "建投" in title) and "评级" in title:
        return "券商研报", True

    elif "实控人" in title and "变更" in title:
        return "股权变更", True

    elif "辞职" in title or "接任" in title or "人事" in title or "选举" in title:
        return "高管变更", True

    elif "停牌" in title:
        return "交易停牌", True

    elif "解禁" in title:
        return "限售股解禁", True

    elif "股价" in title and "跌" in title and "不跌" not in title:
        return "股价下跌", True

    elif "股价" in title and "涨" in title and "跌" not in title:
        return "股价上涨", True

    elif "拟" in title and "派" in title:
        return "分红派息", True

    elif "定增" in title or "增发" in title:
        return "定向增发", True

    elif "并购" in title or "重组" in title or "收购" in title:
        return "并购重组", True

    elif ("中报" in title or "年" in title or "季" in title or "净利" in title) and \
            ("降" in title or "下滑" in title or "亏" in title or "减" in title):
        return "业绩预降", True

    elif ("年" in title or "季" in title) and ("净利" in title or "盈利" in title or "赚" in title):
        return "业绩预增", True

    elif "会议" in title or "公告" in title or "报告" in title:
        return "会议公告", True

    elif "增资" in title or "入股" in title or "投资" in title or "举牌" in title:
        return "增资入股", True

    elif "质押" in title and "解除" not in title:
        return "股权质押", True

    elif "回购" in title or ("解除" in title and "质押" in title):
        return "股权解质押", True

    elif "股权" in title and "转让" in title:
        return "股权转让", True

    elif "涉嫌" in title or "违法" in title or "违规" in title or "造假" in title or "投诉" in title:
        return "违规造假", True

    else:
        return "公司新闻", False


if __name__ == "__main__":

    Path1 = 'C:/Users/text/Desktop/text_classifier/'
    Path2 = 'C:/Users/text/Desktop/data_news/'

    stock_codes = sql_stock()

    codes = []
    for item in stock_codes:
        codes.append(item["stock_code"])

    for Code in codes:

        code = Code.replace("\r", "").zfill(6)

        test_data = sql_data(code)

        if len(test_data) == 0:
            print(code, "lost news data!")
            continue

        test_text_list = []
        test_title = []

        all_label = defaultdict(list)

        for item in test_data:
            Title = item["title"]
            id = item["id"]
            if Title is np.nan:
                continue

            label, flag = rules(Title)
            all_label[label].append(Title)

            insert_data(id, label, Title, Code)

        count = [(item[0], len(item[1])) for item in all_label.items()]
        print(Code, sum([x[1] for x in count if x[0] != "公司新闻"]), len(all_label["公司新闻"]), datetime.datetime.now())




