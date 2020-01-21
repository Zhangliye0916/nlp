# -*-coding:utf-8-*-

import pandas as pd
import re
import datetime
import pymysql as mdb
from collections import defaultdict
import codecs


def update_pool(stock_code):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    # 所有的查询，都在连接con的一个模块cursor上面运行的
    cur = con.cursor(cursor=mdb.cursors.DictCursor)
    # 执行一个查询

    sql = "update work.news_prediction_online set hs300 = 1 where stock_code = '%s'" % stock_code
    try:
        print(sql)
        cur.execute(sql)
        con.commit()

    except Exception as e:
        print(e)
        con.rollback()

    finally:
        con.close()


def update_type(pool_name, stock_code):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    # 所有的查询，都在连接con的一个模块cursor上面运行的
    cur = con.cursor(cursor=mdb.cursors.DictCursor)
    # 执行一个查询

    sql = "update work.news_type set stock_industry = '%s' where stock_code = '%s'" % (pool_name, stock_code)
    try:
        print(sql)
        cur.execute(sql)
        con.commit()

    except Exception as e:
        print(e)
        con.rollback()

    finally:
        con.close()
        

def sql_stock_pool(pool_name):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    # 所有的查询，都在连接con的一个模块cursor上面运行的
    cur = con.cursor(cursor=mdb.cursors.DictCursor)
    # 执行一个查询

    sql = "SELECT stock_code as code from work.stock_pool where stock_pool_name = '%s';" % pool_name
    try:

        cur.execute(sql)
        con.commit()

        return cur.fetchall()

    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":

    """
    all_pool = ["SW.采掘", 
                "SW.传媒", 
                "SW.电气设备", 
                "SW.电子", 
                "SW.房地产", 
                "SW.纺织服装", 
                "SW.非银金融", 
                "SW.钢铁", 
                "SW.公用事业", 
                "SW.国防军工", 
                "SW.化工", 
                "SW.机械设备", 
                "SW.计算机", 
                "SW.家用电器", 
                "SW.建筑材料", 
                "SW.建筑装饰", 
                "SW.交通运输", 
                "SW.农林牧渔", 
                "SW.汽车", 
                "SW.轻工制造", 
                "SW.商业贸易", 
                "SW.食品饮料", 
                "SW.通信", 
                "SW.休闲服务", 
                "SW.医药生物", 
                "SW.银行", 
                "SW.有色金属", 
                "SW.综合", 
                ]
    
    for pool in all_pool:

        for item in sql_stock_pool(pool):
    
            update_type(pool, item["code"].zfill(6))
            print(item["code"].zfill(6), datetime.datetime.now())
            
    """

    

