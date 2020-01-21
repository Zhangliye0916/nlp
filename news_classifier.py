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
import lightgbm as lgb
import warnings
import gc
from sklearn.utils import shuffle
import codecs
from collections import defaultdict
import re


# 朴素贝叶斯算法
def nb_model(train, train_label, test, test_label):
    clf_model = MultinomialNB(alpha=0.01)
    clf_model.fit(train, train_label)
    predict_results = clf_model.predict(test)

    print("nb_model_precision_score.................... ")
    print(classification_report(test_label, predict_results))

    return predict_results


# K近邻算法
def knn_model(train, train_label, test, test_label):
    k_n_n_model = KNeighborsClassifier()
    k_n_n_model.fit(train, train_label)
    predict_results = k_n_n_model.predict(test)

    print("knn_model_precision_score.................... ")
    print(classification_report(test_label, predict_results))

    return predict_results


# 支持向量机算法
def svm_model(train, train_label, test, test_label):
    svm_clf = SVC(kernel="linear", verbose=False)
    svm_clf.fit(train, train_label)
    predict_results = svm_clf.predict(test)

    print("svm_model_precision_score.................... ")
    print(classification_report(test_label, predict_results))

    return predict_results


def lgb_model(train, train_label, test):
    data_train = train.toarray()
    data_test = test.toarray()

    gbm = lgb.sklearn.LGBMClassifier()
    gbm.fit(data_train, train_label)
    predict_results = gbm.predict(data_test)

    return predict_results


def sql_data(stock_code):

    con = None

    try:
        # 连接mysql的方法：connect('ip','user','password','db_name')
        con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
        # 所有的查询，都在连接con的一个模块cursor上面运行的
        cur = con.cursor(cursor=mdb.cursors.DictCursor)
        # 执行一个查询

        sql = "SELECT news.uuid as id, news.title as title, news.content as content \
               from work.news where stock_code = '%s';" % stock_code

        cur.execute(sql)
        con.commit()
        result = cur.fetchall()
        return result

    finally:
        con.close()


def sql_stock():
    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    # 所有的查询，都在连接con的一个模块cursor上面运行的
    cur = con.cursor(cursor=mdb.cursors.DictCursor)
    # 执行一个查询
    sql = "SELECT count(1) as count, news.stock_code from work.news group by stock_code;"

    try:

        cur.execute(sql)
        con.commit()
        sql_result = cur.fetchall()
        stock_count = {}
        for item in sql_result:
            stock_count[item["stock_code"]] = item["count"]

        return stock_count

    except Exception as e:
        print(e)
        return None

    finally:
        if con:
            con.close()


def insert_data(ids, results, model):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    # 所有的查询，都在连接con的一个模块cursor上面运行的
    cur = con.cursor(cursor=mdb.cursors.DictCursor)
    # 执行一个查询

    for i in range(len(ids)):

        id = ids[i]
        result = results[i]
        date_time = datetime.datetime.now()

        sql = "INSERT INTO work.news_prediction (idnews_prediction, id_news, model, method, object, " \
              "result, updated_time) VALUES (replace(uuid(),'-',''), '%s', '%s', 'relative', 'direction'" \
              ", '%s', '%s');" % (id, model, result, date_time)

        # print (sql)
        try:
            cur.execute(sql)
            con.commit()

        except Exception as e:
            print(e)
            con.rollback()

    con.close()


def insert_label(id_news, label):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    # 所有的查询，都在连接con的一个模块cursor上面运行的
    cur = con.cursor(cursor=mdb.cursors.DictCursor)
    # 执行一个查询

    sql = "update work.news_label set direction_label_240min = '%s' where id_news in %s" % (label, id_news)
    # print (sql)
    try:
        cur.execute(sql)
        con.commit()

    except Exception as e:
        print(e)
        con.rollback()

    finally:
        con.close()


def rules(title):

    if "大宗交易" in title:
        return "大宗交易", True

    elif "月" in title and "日" in title and \
            ("盘" in title or "打开" in title in title or "速" in title or "火箭发射" in title):
        return "盘口异动", True

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

    elif "证券：" in title and "评级" in title:
        return "券商研报", True

    elif "实控人" in title and "变更" in title:
        return "股权变更", True

    elif "辞职" in title or "接任" in title or "人事" in title or "离职" in title or "离任" in title:
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

    elif ("年" in title or "季" in title or "净利" in title) and ("增" in title or "涨" in title or "扭亏" in title):
        return "业绩预增", True

    elif ("年" in title or "季" in title or "净利" in title) and \
            ("降" in title or "下滑" in title or "亏损" in title or "减少" in title):
        return "业绩预降", True

    elif "会议" in title or "公告" in title or "报告" in title:
        return "会议公告", True

    elif "增资" in title or "入股" in title or "投资" in title or "举牌" in title:
        return "增资入股", True

    elif "质押" in title and "解除" not in title:
        return "股权质押", True

    elif "回购" in title or ("解除" in title and "质押" in title):
        return "股权解质押", True

    else:
        return "", False


if __name__ == "__main__":

    Path1 = 'C:/Users/text/Desktop/text_classifier/'
    Path2 = 'C:/Users/text/Desktop/data_news/'

    warnings.filterwarnings(module="sklearn*", action="ignore", category=DeprecationWarning)

    train_data = pd.read_csv(Path2 + "新闻分类3.csv", encoding="utf-8")

    train_text_list = []
    train_label_list = []

    for i in range(len(train_data)):
        line = train_data.iloc[i]
        label = line["type"]
        title = line["title"]
        content = line["content"]
        if content is np.nan or title is np.nan or label is None:
            continue

        text_cut = " ".join(jieba.lcut((title + content).replace("\n", "")))
        # text_cut = " ".join(jieba.lcut(title.replace("\n", "")))

        train_text_list.append(text_cut)
        train_label_list.append(label)

    # 基于构建的词典分别统计训练集/测试集词频, 即每个词出现1次、2次、3次等
    count_vec = CountVectorizer(min_df=3)
    tf_train = count_vec.fit_transform(train_text_list)
    print(tf_train.shape)

    # 进一步计算词频-逆文档频率
    tfidf_transformer = TfidfTransformer().fit(tf_train)
    tfidf_train = tfidf_transformer.transform(tf_train)

    f = codecs.open(Path1 + "codes.txt", encoding="utf-8-sig")
    codes = f.read().split("\n")

    for Code in codes[100:]:

        code = Code.replace("\r", "").zfill(6)

        test_data = sql_data(code)

        if len(test_data) == 0:
            print(code, "lost news data!")
            continue

        test_text_list = []
        test_title = []

        ids = []
        all_label = defaultdict(list)

        for item in test_data:
            Title = item["title"]
            Content = item["content"]
            id = item["id"]
            if Content is np.nan or Title is np.nan:
                continue

            label, flag = rules(Title)

            all_label[label].append(Title)
            continue

        print(Code, all_label["盘口异动"])
        count = [(item[0], len(item[1])) for item in all_label.items()]
        print(sum([x[1] for x in count if x[0] != ""]), len(all_label[""]))

        """
            if flag:
                all_label[label].append(Title)
                continue

            text_cut = " ".join(jieba.lcut((Title + Content).replace("\n", "")))
            # text_cut = " ".join(jieba.lcut(Title.replace("\n", "")))

            test_text_list.append(text_cut)
            test_title.append(Title)
            ids.append(id)

        tf_test = count_vec.transform(test_text_list)
        tfidf_test = tfidf_transformer.transform(tf_test)

        # lightgbm算法
        lgb_result = lgb_model(tfidf_train, train_label_list, tfidf_test)

        for i in range(len(lgb_result)):
            all_label[lgb_result[i]].append(test_title[i])

        print(all_label)
        print(code, datetime.datetime.now())

        del test_data
        del test_text_list
        del tf_test
        del tfidf_test
        del all_label
        gc.collect()
        
        """


