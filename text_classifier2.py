# -*-coding:utf-8-*-
import csv
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import jieba
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score


def text_classification():

    print("start building vector model...")
    # 构建词典
    vectorizer = CountVectorizer()
    tf_idf_transformer = TfidfTransformer()
    tfidf_train = tf_idf_transformer.fit_transform(vectorizer.fit_transform(train_text_list))
    tfidf_test = tf_idf_transformer.transform(vectorizer.transform(test_text_list))

    print("building vector model is finished...")

    data_train = tfidf_train.toarray()
    data_test = tfidf_test.toarray()

    # print (vectorizer.get_feature_names())# 特征
    """
    # 将数据转化为DMatrix类型
    d_train = xgb.DMatrix(data_train, label=train_label_list)
    d_test = xgb.DMatrix(data_test, label=test_label_list)

    param = {'max_depth': 10, 'eta': 0.2, 'eval_metric': 'mlogloss', 'silent': 1,
             'objective': "multi:softmax", "num_class": 2}  # 参数
    evallist = [(d_train, 'train'), (d_test, 'test')]  # 这步可以不要，用于测试效果

    num_round = 100  # 循环次数
    bst = xgb.train(param, d_train, num_round, evallist)

    prediction = bst.predict(d_test)

    print (classification_report(test_label_list, prediction))
    """

    gbm = lgb.sklearn.LGBMClassifier()
    gbm.fit(data_train, train_label_list)
    gbm_predictions = gbm.predict(data_test)
    print (classification_report(test_label_list, gbm_predictions))


if __name__ == "__main__":

    Path1 = 'C:/Users/text/Desktop/text_classifier/'
    Path2 = 'C:/Users/text/Desktop/data_news/'

    jieba.load_userdict(Path2 + "userdict.txt")
    # StopWords = [line.strip() for line in open(Path2 + 'stopwords.txt', encoding="utf-8-sig").readlines()]

    print("start loading data...")

    file = pd.read_csv(Path1 + "sentiment.csv", encoding="utf-8-sig")

    total_text_list = []
    total_label_list = []

    order = list(file.index)
    random.shuffle(order)

    for ind in order:
        line = file.loc[ind]
        label = line["label"]
        # content = "".join(re.findall(r"\D", line["text"]))
        content = line["text"]
        if content is np.nan:
            continue

        # 不去除停用词
        text_cut = " ".join(jieba.lcut(content.replace("\n", "")))
        # 去除停用词
        # text_cut = " ".join(list(filter
        #                          (lambda x: x not in StopWords, jieba.lcut((title*10 + content).replace("\n", "")))))

        total_text_list.append(text_cut)
        # total_label_list.append(int((label*label + label)/2))
        total_label_list.append(label)
    """
        probability = random.random()
        if probability > 0.1:
            train_text_list.append(text_cut)
            train_label_list.append(label)

        else:
            test_text_list.append(text_cut)
            test_label_list.append(label)

    text_classification()
    """

    print("load data is finished...")

    # 划分训练集和测试集
    length = len(total_label_list)

    for term in range(10):

        test_text_list = [total_text_list[i] for i in range(length) if i % 10 == term]
        train_text_list = [total_text_list[i] for i in range(length) if i % 10 != term]
        test_label_list = np.array([total_label_list[i] for i in range(length) if i % 10 == term])
        train_label_list = np.array([total_label_list[i] for i in range(length) if i % 10 != term])

        text_classification()
