# -*-coding:utf-8-*-

import pymysql as mdb
import pandas as pd
import datetime
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


def lgb_model(train, train_label, test, test_label, show_result=True):
    data_train = train.toarray()
    data_test = test.toarray()

    gbm = lgb.sklearn.LGBMClassifier()
    gbm.fit(data_train, train_label)
    predict_results = gbm.predict(data_test)

    if show_result:
        print("lgb_model_precision_score.................... ")
        print(classification_report(test_label, predict_results))

    return predict_results


def sql_data(code, label_type):

    con = None

    if label_type == "heat":
        t_label = "relative_heat_label"

        sql = "SELECT news.uuid as id, news.title as title, news.content as content, news.date as date, \
                news_label.%s as label from work.news join work.news_label on \
                news.uuid = news_label.id_news where stock_code = '%s';" % (t_label, code)

    elif label_type == "direction":
        t_label = "240min_direction"
        sql = "SELECT news.uuid as id, news.title as title, news.content as content, news.date as date, \
                news_label.%s as label from work.news join work.news_label on \
                news.uuid = news_label.id_news where stock_code = '%s';" % (t_label, code)

    else:
        print("input error label type!")
        return None

    try:
        # 连接mysql的方法：connect('ip','user','password','db_name')
        con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
        # 所有的查询，都在连接con的一个模块cursor上面运行的
        cur = con.cursor(cursor=mdb.cursors.DictCursor)
        # 执行一个查询
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
    sql = "SELECT distinct stock_code from work.news order by stock_code;"

    try:

        cur.execute(sql)
        con.commit()
        sql_result = cur.fetchall()
        stock_codes = []
        for item in sql_result:
            stock_codes.append(item["stock_code"])

        return stock_codes

    except Exception as e:
        print(e)
        return None

    finally:
        if con:
            con.close()


def insert_data(ids, code, title_list, date_list, results, model):

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
    # 所有的查询，都在连接con的一个模块cursor上面运行的
    cur = con.cursor(cursor=mdb.cursors.DictCursor)
    # 执行一个查询

    for i in range(len(ids)):

        date_time = datetime.datetime.now()

        sql = "INSERT INTO work.news_prediction_online (id_news, model, " \
              "result, stock_code, title, date, updated_time) VALUES ('%s', '%s'," \
              "'%d', '%s', '%s', '%s', '%s');" % (ids[i], model, int(results[i]), code, title_list[i], date_list[i], date_time)

        # print(sql)
        try:
            cur.execute(sql)
            con.commit()

        except Exception as e:
            print(e)
            con.rollback()

    con.close()


if __name__ == "__main__":

    warnings.filterwarnings(module="sklearn*", action="ignore", category=DeprecationWarning)

    StockCodes = sql_stock()
    print(StockCodes)

    insert = True
    ShowResult = True

    start_ind = StockCodes.index("601668")

    for num, StockCode in enumerate(StockCodes[start_ind:]):

        AllData = sql_data(StockCode, "heat")
        random.shuffle(AllData)

        total_text_list = []
        total_label_list = []

        total_id_list = []
        total_title_list = []
        total_date_list = []

        for line in AllData:

            label = line["label"]
            title = line["title"]
            content = line["content"]
            news_id = line["id"]
            date = line["date"]
            if content is np.nan or title is np.nan or label is None:
                continue

            text_cut = " ".join(jieba.lcut((title + content).replace("\n", "")))
            # text_cut = 10*title + content

            total_text_list.append(text_cut)
            total_label_list.append(label)
            total_id_list.append(news_id)
            total_title_list.append(title)
            total_date_list.append(date)

        print(num, StockCode, len(total_label_list), datetime.datetime.now())

        if len(total_label_list) < 200:
            del total_text_list
            del total_label_list
            del AllData
            del total_id_list
            gc.collect()

            continue

        length = len(total_label_list)

        train_text_list = total_text_list[:int(length / 2)]
        test_text_list = total_text_list[int(length / 2):]

        train_label_list = total_label_list[:int(length / 2)]
        test_label_list = total_label_list[int(length / 2):]

        train_id_list = total_id_list[:int(length / 2)]
        test_id_list = total_id_list[int(length / 2):]

        train_title_list = total_title_list[:int(length / 2)]
        test_title_list = total_title_list[int(length / 2):]

        train_date_list = total_date_list[:int(length / 2)]
        test_date_list = total_date_list[int(length / 2):]

        """
        # 构建词典
        vec_total = CountVectorizer()
        vec_total.fit_transform(total_text_list)

        # 基于构建的词典分别统计训练集/测试集词频, 即每个词出现1次、2次、3次等
        vec_train = CountVectorizer(vocabulary=vec_total.vocabulary_)
        tf_train = vec_train.fit_transform(train_text_list)
        vec_test = CountVectorizer(vocabulary=vec_total.vocabulary_)
        tf_test = vec_test.fit_transform(test_text_list)

        # 进一步计算词频-逆文档频率
        tfidf_transformer = TfidfTransformer()
        tfidf_train = tfidf_transformer.fit(tf_train).transform(tf_train)
        tfidf_test = tfidf_transformer.fit(tf_test).transform(tf_test)
        """

        count_vec = CountVectorizer(min_df=1)
        tf_train = count_vec.fit_transform(train_text_list)

        tfidf_transformer = TfidfTransformer().fit(tf_train)
        tfidf_train = tfidf_transformer.transform(tf_train)
        tf_test = count_vec.transform(test_text_list)
        tfidf_test = tfidf_transformer.transform(tf_test)

        # 朴素贝叶斯算法
        nb_result = nb_model(tfidf_train, train_label_list, tfidf_test, test_label_list, ShowResult)
        if insert:
            insert_data(test_id_list, StockCode, test_title_list, test_date_list, nb_result, "nb")
        # K近邻算法
        knn_result = knn_model(tfidf_train, train_label_list, tfidf_test, test_label_list, ShowResult)
        if insert:
            insert_data(test_id_list, StockCode, test_title_list, test_date_list, knn_result, "knn")
        # 支持向量机算法
        svm_result = svm_model(tfidf_train, train_label_list, tfidf_test, test_label_list, ShowResult)
        if insert:
            insert_data(test_id_list, StockCode, test_title_list, test_date_list, svm_result, "svm")
        # lightgbm算法
        lgb_result = lgb_model(tfidf_train, train_label_list, tfidf_test, test_label_list, ShowResult)
        if insert:
            insert_data(test_id_list, StockCode, test_title_list, test_date_list, lgb_result, "lgb")
        """............................................................................................."""

        # 朴素贝叶斯算法
        nb_result = nb_model(tfidf_test, test_label_list, tfidf_train, train_label_list, ShowResult)
        if insert:
            insert_data(train_id_list, StockCode, train_title_list, train_date_list, nb_result, "nb")
        # K近邻算法
        knn_result = knn_model(tfidf_test, test_label_list, tfidf_train, train_label_list, ShowResult)
        if insert:
            insert_data(train_id_list, StockCode, train_title_list, train_date_list, knn_result, "knn")
        # 支持向量机算法
        svm_result = svm_model(tfidf_test, test_label_list, tfidf_train, train_label_list, ShowResult)
        if insert:
            insert_data(train_id_list, StockCode, train_title_list, train_date_list, svm_result, "svm")
        # lightgbm算法
        lgb_result = lgb_model(tfidf_test, test_label_list, tfidf_train, train_label_list, ShowResult)
        if insert:
            insert_data(train_id_list, StockCode, train_title_list, train_date_list, lgb_result, "lgb")

        del AllData

        del total_text_list
        del total_label_list
        del total_id_list

        del train_text_list
        del train_label_list
        del train_id_list
        del tf_train
        del tfidf_train

        del test_text_list
        del test_label_list
        del test_id_list
        del tf_test
        del tfidf_test

        del tfidf_transformer
        del count_vec
        gc.collect()
