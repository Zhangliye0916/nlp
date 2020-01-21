# -*-coding:utf-8-*-

import pymysql as mdb
import pandas as pd
import datetime
import jieba
import numpy as np
import tushare as ts


def single_company_news(company_name):

    con = None

    try:
        # 连接mysql的方法：connect('ip','user','password','db_name')
        con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
        # 所有的查询，都在连接con的一个模块cursor上面运行的
        cur = con.cursor()
        # 执行一个查询
        cur.execute("SELECT content, title, date from work.news where company = '" + company_name + "'")
        # 取得上个查询的结果，是单个结果
        data = cur.rowcount
        print(data)

    finally:
        if con:
            # 无论如何，连接记得关闭
            con.close()


def create_table():

    con = None

    try:
        # 连接mysql的方法：connect('ip','user','password','db_name')
        con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
        # 所有的查询，都在连接con的一个模块cursor上面运行的
        cur = con.cursor()
        # 执行一个查询
        cur.execute("SELECT content, title, date from work.news where company = '" + company_name + "'")
        # 取得上个查询的结果，是单个结果
        data = cur.rowcount
        sql = """CREATE TABLE news_label (
          uuid int(11) NOT NULL AUTO_INCREMENT,
          `datetime` varchar(20) DEFAULT NULL,
          `ironincome` decimal(20,2) DEFAULT NULL,
          `generalincome` decimal(20,2) DEFAULT NULL,
          `baiincome` decimal(20,2) DEFAULT NULL,
          PRIMARY KEY (`id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8;
        """

    finally:
        if con:
            # 无论如何，连接记得关闭
            con.close()


def sql_stock_list(stock_pool):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    # 所有的查询，都在连接con的一个模块cursor上面运行的
    cur = con.cursor()
    # 执行一个查询
    sql = "select stock_code from work.stock_pool where stock_pool_name = '%s'" % stock_pool

    try:
        cur.execute(sql)
        con.commit()
        result = cur.fetchall()

        return [item[0] for item in result]

    except:
        con.rollback()

    finally:
        con.close()


def sql_insert(stock_pool_name):
    con = None

    try:
        # 连接mysql的方法：connect('ip','user','password','db_name')
        con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
        # 所有的查询，都在连接con的一个模块cursor上面运行的
        cur = con.cursor()
        # 执行一个查询
        sql = "SELECT content, title, date from work.news where company = '" + company_name + "'"
        cur.execute(sql)
        con.commit()
    except:
        con.rollback()

    finally:
        con.close()


def insert_data_stock_pool(path):

    data = pd.read_csv(path, encoding="utf-8-sig")
    print(data.head())

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    # 所有的查询，都在连接con的一个模块cursor上面运行的
    cur = con.cursor()
    # 执行一个查询

    for ind in data.index:
        line = data.loc[ind]
        stock_pool_name = line["指数名称"]
        stock_code = line["股票代码"]
        stock_name = line["股票名称"]
        updated_time = datetime.datetime.now()

        sql = "INSERT INTO work.stock_pool(idstock_pool,\
              stock_pool_name, stock_code, stock_name, updated_time)\
              VALUES (replace(uuid(), '-',''), '%s', '%s', '%s', '%s' )" % \
                (stock_pool_name, stock_code, stock_name, updated_time)

        try:
            cur.execute(sql)
            con.commit()
            pass

        except:
            con.rollback()

    con.close()


def sql_news():

    con = None

    try:
        # 连接mysql的方法：connect('ip','user','password','db_name')
        con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
        # 所有的查询，都在连接con的一个模块cursor上面运行的
        cur = con.cursor(cursor=mdb.cursors.DictCursor)
        # 执行一个查询
        # sql = "SELECT uuid, content, title, reader, remark from work.news;"
        sql = "select uuid, content, title, reader, remark from work.news \
                where work.news.uuid not in (select id_news from work.news_cut)"
        cur.execute(sql)
        con.commit()
        result = cur.fetchall()
        return result

    except:
        con.rollback()

    finally:
        con.close()


def insert_data_topic(path):

    data = pd.read_csv(path, encoding="utf-8-sig")
    print(data.head())

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    # 所有的查询，都在连接con的一个模块cursor上面运行的
    cur = con.cursor()
    # 执行一个查询

    for ind in data.index:
        line = data.loc[ind]
        topic_name = line["topic_name"]
        topic_index = line["index"]
        rate = line["rate"]
        period = line["period"]
        updated_time = datetime.datetime.now()

        sql = "INSERT INTO work.news_topic(idnews_topic,\
              topic_name, topic_index, rate, period, updated_time)\
              VALUES (replace(uuid(), '-',''), '%s', '%s', '%s', '%s', '%s' )" % \
                (topic_name, topic_index, rate, period, updated_time)

        print (sql)
        try:
            cur.execute(sql)
            con.commit()

        except:
            con.rollback()

    con.close()



def insert_news_cut():

    jieba.load_userdict(Path1 + "userdict.txt")

    all_data = sql_news()

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    # 所有的查询，都在连接con的一个模块cursor上面运行的
    cur = con.cursor()
    # 执行一个查询

    count = 0
    for line in all_data:

        count += 1
        if count % 10 == 0:
            print(count)

        if line["content"] is np.nan or line["title"] is np.nan:
            continue

        id_news = line["uuid"]
        DateTime = datetime.datetime.now()

        title = line["title"]
        content = line["content"]

        title_cut = " ".join(jieba.lcut(title.replace("\n", "")))
        content_cut = " ".join(jieba.lcut(content.replace("\n", "")))

        sql = "INSERT INTO work.news_cut(idnews_cut,\
                  id_news, news_title_cut, news_content_cut, updated_time)\
                  VALUES (replace(uuid(), '-',''), '%s', '%s', '%s', '%s' )" % \
              (id_news, title_cut, content_cut, DateTime)

        try:
            cur.execute(sql)
            con.commit()

        except:
            print(id_news)
            con.rollback()

    con.close()


def insert_stock_price():
    pass


if __name__ == "__main__":

    Path1 = "C:/Users/text/Desktop/data_news/"
    insert_data_topic(Path1 + "topic.csv")




