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


def sql_data(label_type, start, size):

    con = None

    if label_type == "heat":

        sql = "SELECT news.uuid as id, news.title as title, news.content as content, " \
              "news.date as date, news.stock_code as code, \
                news_label.relative_heat_label as label from work.news join work.news_label on \
                news.uuid = news_label.id_news where news.date > '2018-03-28' order by news.date limit %d, %d" \
              % (start, size)

    elif label_type == "direction":
        sql = "SELECT news.uuid as id, news.title as title, news.content as content, " \
              "news.date as date, news.stock_code as code, \
                news_label.240_min_direction as label from work.news join work.news_label on \
                news.uuid = news_label.id_news where news_label.240_min_direction" \
              " is not null order by news.date limit %d, %d" % (start, size)

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

        sql = "INSERT INTO work.news_prediction2_online (id_news, model, " \
              "result, stock_code, title, date, updated_time) VALUES ('%s', '%s'," \
              "'%d', '%s', '%s', '%s', '%s');" \
              % (ids[i], model, int(results[i]), code[i], title_list[i], date_list[i], date_time)

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

    train_size = 5000
    test_size = 5000
    max_num = 886367

    insert = True
    ShowResult = True

    for start_num in range(60000, max_num, train_size + test_size):

        AllData = sql_data("direction", start_num, min(train_size + test_size, max_num - start_num))
        random.shuffle(AllData)

        total_text_list = []
        total_label_list = []

        total_id_list = []
        total_title_list = []
        total_date_list = []
        total_code_list = []

        for line in AllData:

            if line["content"] is np.nan or line["title"] is np.nan or line["label"] is None:
                continue

            text_cut = " ".join(jieba.lcut((line["title"] + line["content"]).replace("\n", "")))
            # text_cut = 10*title + content

            total_text_list.append(text_cut)
            total_label_list.append(line["label"])
            total_id_list.append(line["id"])
            total_title_list.append(line["title"])
            total_date_list.append(line["date"])
            total_code_list.append(line["code"])

        print(start_num, datetime.datetime.now())

        train_text_list = total_text_list[:train_size]
        test_text_list = total_text_list[train_size:]

        train_label_list = total_label_list[:train_size]
        test_label_list = total_label_list[train_size:]

        train_id_list = total_id_list[:train_size]
        test_id_list = total_id_list[train_size:]

        train_title_list = total_title_list[:train_size]
        test_title_list = total_title_list[train_size:]

        train_date_list = total_date_list[:train_size]
        test_date_list = total_date_list[train_size:]

        train_code_list = total_code_list[:train_size]
        test_code_list = total_code_list[train_size:]

        count_vec = CountVectorizer(min_df=1)
        tf_train = count_vec.fit_transform(train_text_list)

        tfidf_transformer = TfidfTransformer().fit(tf_train)
        tfidf_train = tfidf_transformer.transform(tf_train)
        tf_test = count_vec.transform(test_text_list)
        tfidf_test = tfidf_transformer.transform(tf_test)

        del AllData
        del total_text_list
        del total_label_list
        del total_id_list
        del total_title_list
        del total_date_list
        del total_code_list

        del train_text_list
        del test_text_list
        del tf_train
        del tf_test
        gc.collect()

        # 朴素贝叶斯算法
        nb_result = nb_model(tfidf_train, train_label_list, tfidf_test, test_label_list, ShowResult)
        if insert:
            insert_data(test_id_list, test_code_list, test_title_list, test_date_list, nb_result, "nb")
        # K近邻算法
        knn_result = knn_model(tfidf_train, train_label_list, tfidf_test, test_label_list, ShowResult)
        if insert:
            insert_data(test_id_list, test_code_list, test_title_list, test_date_list, knn_result, "knn")
        # 支持向量机算法
        svm_result = svm_model(tfidf_train, train_label_list, tfidf_test, test_label_list, ShowResult)
        if insert:
            insert_data(test_id_list, test_code_list, test_title_list, test_date_list, svm_result, "svm")
        # lightgbm算法
        lgb_result = lgb_model(tfidf_train, train_label_list, tfidf_test, test_label_list, ShowResult)
        if insert:
            insert_data(test_id_list, test_code_list, test_title_list, test_date_list, lgb_result, "lgb")
        """............................................................................................."""

        # 朴素贝叶斯算法
        nb_result = nb_model(tfidf_test, test_label_list, tfidf_train, train_label_list, ShowResult)
        if insert:
            insert_data(train_id_list, train_code_list, train_title_list, train_date_list, nb_result, "nb")
        # K近邻算法
        knn_result = knn_model(tfidf_test, test_label_list, tfidf_train, train_label_list, ShowResult)
        if insert:
            insert_data(train_id_list, train_code_list, train_title_list, train_date_list, knn_result, "knn")
        # 支持向量机算法
        svm_result = svm_model(tfidf_test, test_label_list, tfidf_train, train_label_list, ShowResult)
        if insert:
            insert_data(train_id_list, train_code_list, train_title_list, train_date_list, svm_result, "svm")
        # lightgbm算法
        lgb_result = lgb_model(tfidf_test, test_label_list, tfidf_train, train_label_list, ShowResult)
        if insert:
            insert_data(train_id_list, train_code_list, train_title_list, train_date_list, lgb_result, "lgb")

        del train_label_list
        del train_id_list
        del tfidf_train

        del test_label_list
        del test_id_list
        del tfidf_test

        del tfidf_transformer
        del count_vec
        gc.collect()
