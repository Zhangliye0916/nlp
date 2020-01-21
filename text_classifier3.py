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
def nb_model(train, train_label, test, test_label):
    clf_model = MultinomialNB(alpha=0.01)
    clf_model.fit(train, train_label)
    predict_results = clf_model.predict(test)

    print("nb_model_precision_score.................... ")
    print(classification_report(test_label, predict_results))

    del train
    del train_label
    del test
    del test_label
    gc.collect()


# K近邻算法
def knn_model(train, train_label, test, test_label):
    k_n_n_model = KNeighborsClassifier()
    k_n_n_model.fit(train, train_label)
    predict_results = k_n_n_model.predict(test)

    print("knn_model_precision_score.................... ")
    print(classification_report(test_label, predict_results))

    del train
    del train_label
    del test
    del test_label
    gc.collect()


# 支持向量机算法
def svm_model(train, train_label, test, test_label):
    svm_clf = SVC(kernel="linear", verbose=False)
    svm_clf.fit(train, train_label)
    predict_results = svm_clf.predict(test)

    print("svm_model_precision_score.................... ")
    print(classification_report(test_label, predict_results))

    del train
    del train_label
    del test
    del test_label
    gc.collect()


def lgb_model(train, train_label, test, test_label):
    data_train = train.toarray()
    data_test = test.toarray()

    gbm = lgb.sklearn.LGBMClassifier()
    gbm.fit(data_train, train_label)
    gbm_predictions = gbm.predict(data_test)

    print("lgb_model_precision_score.................... ")
    print(classification_report(test_label, gbm_predictions))

    del train
    del train_label
    del test
    del test_label
    del data_train
    del data_test
    gc.collect()


def text_classification(total_label_list, total_text_list,
                        train_label_list, train_text_list,
                        test_label_list, test_text_list):

    print("start building vector model...")
    # 构建词典
    vec_total = CountVectorizer()
    vec_total.fit_transform(total_text_list)
    print (len(vec_total.get_feature_names()))
    # 基于构建的词典分别统计训练集/测试集词频, 即每个词出现1次、2次、3次等
    vec_train = CountVectorizer(vocabulary=vec_total.vocabulary_)
    tf_train = vec_train.fit_transform(train_text_list)
    vec_test = CountVectorizer(vocabulary=vec_total.vocabulary_)
    tf_test = vec_test.fit_transform(test_text_list)

    del total_label_list
    del total_text_list
    del train_text_list
    del test_text_list
    gc.collect()

    # 进一步计算词频-逆文档频率
    tfidf_transformer = TfidfTransformer()
    tfidf_train = tfidf_transformer.fit(tf_train).transform(tf_train)
    tfidf_test = tfidf_transformer.fit(tf_test).transform(tf_test)
    print("building vector model is finished...")

    # 朴素贝叶斯算法
    nb_model(tfidf_train, train_label_list, tfidf_test, test_label_list)
    # K近邻算法
    knn_model(tfidf_train, train_label_list, tfidf_test, test_label_list)
    # 支持向量机算法
    svm_model(tfidf_train, train_label_list, tfidf_test, test_label_list)
    # lightgbm算法
    lgb_model(tfidf_train, train_label_list, tfidf_test, test_label_list)

    print("building predict model is finished...")


def sql_data(code):

    con = None

    try:
        # 连接mysql的方法：connect('ip','user','password','db_name')
        con = mdb.connect('192.168.10.85', 'zly', 'zly@isi2019', 'work')
        # 所有的查询，都在连接con的一个模块cursor上面运行的
        cur = con.cursor(cursor=mdb.cursors.DictCursor)
        # 执行一个查询
        """
        sql = "SELECT news_cut.news_title_cut as title, news_cut.news_content_cut as content, \
                news_label.relative_heat_label as label from work.news_cut join work.news_label on \
                news_cut.id_news = news_label.id_news where stock_code = '000568';"
        """
        sql = "SELECT news.uuid as id, news.title as title, news.content as content, \
                        news_label.relative_heat_label as label from work.news join work.news_label on \
                        news.uuid = news_label.id_news where stock_code = '%s'" % code

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

    except:
        return None

    finally:
        if con:
            con.close()


if __name__ == "__main__":

    Path1 = 'C:/Users/text/Desktop/text_classifier/'
    Path2 = 'C:/Users/text/Desktop/data_news/'

    warnings.filterwarnings(module="sklearn*", action="ignore", category=DeprecationWarning)

    AllData = sql_data("000799")
    random.shuffle(AllData)

    TotalTextList = []
    TotalLabelList = []

    for line in AllData:

        label = line["label"]
        title = line["title"]
        content = line["content"]

        if content is np.nan or title is np.nan:
            continue

        if "融资融券" in title or "大宗交易" in title or "盘口异动" in content:
            # continue
            pass

        text_cut = " ".join(jieba.lcut((title + content).replace("\n", "")))
        # text_cut = 10*title + content

        TotalTextList.append(text_cut)
        TotalLabelList.append(label)

    print ("load data is finished...")

    length = len(TotalLabelList)

    TrainTextList = TotalTextList[:int(length/2)]
    TestTextList = TotalTextList[int(length/2):]

    TrainLabelList = TotalLabelList[:int(length/2)]
    TestLabelList = TotalLabelList[int(length/2):]

    text_classification(TotalLabelList, TestTextList,
                        TrainLabelList, TrainTextList,
                        TestLabelList, TrainTextList)

