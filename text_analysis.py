# -*-coding:utf-8-*-

import pymysql as mdb
# import re
from jqdatasdk import *
import time
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import warnings
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np


def sql_news(type_name):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    cur = con.cursor(cursor=mdb.cursors.DictCursor)

    sql = "SELECT a.uuid, a.title, a.content, a.date, a.stock_code as code, a.company  FROM work.news as a " \
          "left join work.news_type as b on a.uuid = b.id_news where b.type = '{}' " \
          "and a.date > '2018-03-01';" .format(type_name)

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


def write_data():

    auth("18612754762", "xyz117")

    with open(Path1 + "text.txt", "w") as f3:

        news_titles = sql_news("股东增持")

        for num, item in enumerate(news_titles):

            if num % 10 == 0:
                print(num)

            id = item["uuid"]
            Date = item["date"]
            Stock = item["code"]

            if Stock[0] == "6":
                Stock += ".XSHG"

            else:
                Stock += ".XSHE"

            try:
                data = jq_bars_data(Stock, Date)
                f3.write(id + " " + " ".join(data) + "\n")

            except:
                print(id, Stock)


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


def jq_bars_data(stock, date):

    price = get_bars(stock, 22, unit='1d',
                     fields=['date', 'close', 'volume'],
                     include_now=False, end_dt=date, fq_ref_date=None)

    price = price.fillna(method="ffill")

    pct_1d = "%.2f" % float(price["close"][21] / price["close"][20] - 1.)
    pct_5d = "%.2f" % float(price["close"][21] / price["close"][16] - 1.)
    pct_22d = "%.2f" % float(price["close"][21] / price["close"][0] - 1.)

    pct_1d_v = "%.2f" % float(price["volume"][21] / price["volume"][20] - 1.)
    pct_5d_v = "%.2f" % float(price["volume"][21] / price["volume"][16] - 1.)
    pct_22d_v = "%.2f" % float(price["volume"][21] / price["volume"][0] - 1.)

    return [stock, pct_1d, pct_5d, pct_22d, pct_1d_v, pct_5d_v, pct_22d_v]


def sql_label(ids):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    cur = con.cursor(cursor=mdb.cursors.DictCursor)

    sql = "SELECT a.id_news as id, a.240_min_direction as label, a.stock_industry as industry " \
          "FROM work.news_label as a where a.id_news in {}" .format(ids)

    try:
        cur.execute(sql)
        con.commit()
        return cur.fetchall()

    except Exception as e:
        print(e)
        return None

    finally:
        con.close()


def sql_content(ids):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    cur = con.cursor(cursor=mdb.cursors.DictCursor)

    sql = "SELECT a.uuid as id, a.title, a.content FROM work.news as a where a.uuid in {}" .format(ids)

    try:
        cur.execute(sql)
        con.commit()
        return cur.fetchall()

    except Exception as e:
        print(e)
        return None

    finally:
        con.close()


# 梯度提升树算法
def lgb_model(train, train_label, test, test_label, show_result=True):

    gbm = lgb.sklearn.LGBMClassifier()
    gbm.fit(train, train_label)
    predict_results = gbm.predict(test)

    if show_result:
        print("lgb_model_precision_score.................... ")
        print(classification_report(test_label, predict_results))

    return predict_results


# 朴素贝叶斯算法
def nb_model(train, train_label, test, test_label, show_result=True):
    clf_model = MultinomialNB(alpha=0.01)
    clf_model.fit(train, train_label)
    predict_results = clf_model.predict(test)

    if show_result:
        print("nb_model_precision_score.................... ")
        print(classification_report(test_label, predict_results))

    return predict_results


# K近邻算法
def knn_model(train, train_label, test, test_label, show_result=True):
    k_n_n_model = KNeighborsClassifier()
    k_n_n_model.fit(train, train_label)
    predict_results = k_n_n_model.predict(test)

    if show_result:
        print("knn_model_precision_score.................... ")
        print(classification_report(test_label, predict_results))

    return predict_results


# 支持向量机算法
def svm_model(train, train_label, test, test_label, show_result=True):
    svm_clf = SVC(kernel="linear", verbose=False)
    svm_clf.fit(train, train_label)
    predict_results = svm_clf.predict(test)

    if show_result:
        print("svm_model_precision_score.................... ")
        print(classification_report(test_label, predict_results))

    return predict_results


# 决策树算法
def dt_model(train, train_label, test, test_label, show_result=True):
    dt_clf = DTC(criterion="entropy")
    dt_clf.fit(train, train_label)
    predict_results = dt_clf.predict(test)

    if show_result:
        print("dt_model_precision_score.................... ")
        print(classification_report(test_label, predict_results))

    return predict_results


if __name__ == "__main__":

    warnings.filterwarnings(module="sklearn*", action="ignore", category=DeprecationWarning)

    t1 = time.time()

    Path1 = "C:/Users/text/Desktop/news_analysis/"
    Path2 = 'C:/Users/text/Desktop/data_news/'

    jieba.load_userdict(Path2 + "userdict.txt")

    with open(Path1 + "盘口大跌.txt", "r") as f4:
        ids = []
        for item in f4.readlines():
            line = item.replace("\n", "").split(" ")
            ids.append(line[0])

    # print(market_data)
    Ids = tuple(ids)
    label_data = sql_label(Ids)

    """............................文本分析方法........................"""

    AllData = sql_content(Ids)

    all_data = {}

    for line in AllData:

        title = line["title"]
        content = line["content"]
        if content is np.nan or title is np.nan:
            continue

        text_cut = " ".join(jieba.lcut((title + content).replace("\n", "")))
        # text_cut = 10*title + content

        all_data[line["id"]] = text_cut

    x_data = []
    y_data = []
    for item in label_data:
        if item["label"] is not None:
            x_data.append(all_data[item["id"]])
            y_data.append(item["label"])

        else:
            del all_data[item["id"]]

    length = len(y_data)

    for term in range(10):
        x_train = [x_data[i] for i in range(length) if i % 10 != term]
        x_test = [x_data[i] for i in range(length) if i % 10 == term]
        y_train = [y_data[i] for i in range(length) if i % 10 != term]
        y_test = [y_data[i] for i in range(length) if i % 10 == term]

        count_vec = CountVectorizer(min_df=1)
        tf_train = count_vec.fit_transform(x_train)

        tfidf_transformer = TfidfTransformer().fit(tf_train)
        tfidf_train = tfidf_transformer.transform(tf_train)
        tf_test = count_vec.transform(x_test)
        tfidf_test = tfidf_transformer.transform(tf_test)

        lgb_model(tfidf_train, y_train, tfidf_test, y_test)
        # nb_model(x_train, y_train, x_test, y_test)
        knn_model(tfidf_train, y_train, tfidf_test, y_test)
        # svm_model(x_train, y_train, x_test, y_test)
        dt_model(tfidf_train, y_train, tfidf_test, y_test)

