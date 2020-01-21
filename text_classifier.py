# 使用fastText的文本分类
import csv
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import jieba
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import fasttext
import pandas as pd
import re
import numpy as np
import lightgbm as lgb
import warnings


# 朴素贝叶斯算法
def nb_model(train, train_label, test, test_label):
    clf_model = MultinomialNB(alpha=0.01)
    clf_model.fit(train, train_label)
    predict_results = clf_model.predict(test)

    print("nb_model_precision_score.................... ")
    print(classification_report(test_label, predict_results))


# K近邻算法
def knn_model(train, train_label, test, test_label):
    k_n_n_model = KNeighborsClassifier(n_neighbors=8)
    k_n_n_model.fit(train, train_label)
    predict_results = k_n_n_model.predict(test)

    print("knn_model_precision_score.................... ")
    print(classification_report(test_label, predict_results))


# 支持向量机算法
def svm_model(train, train_label, test, test_label):
    svm_clf = SVC(kernel="linear", verbose=False)
    svm_clf.fit(train, train_label)
    predict_results = svm_clf.predict(test)

    print("svm_model_precision_score.................... ")
    print(classification_report(test_label, predict_results))


def lgb_model(train, train_label, test, test_label):
    data_train = train.toarray()
    data_test = test.toarray()

    gbm = lgb.sklearn.LGBMClassifier()
    gbm.fit(data_train, train_label)
    prediction_result = gbm.predict(data_test)

    print("lgb_model_precision_score.................... ")
    print(classification_report(test_label, prediction_result))


def text_classification():

    print("start building vector model...")
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


def fasttext_model():

    classifier = fasttext.train_supervised(Path1 + "data_train.txt", label_prefix="__label__", dim=50, epoch=100,
                                           word_ngrams=1, loss="hs", bucket=2000000)

    result = []
    for sample in classifier.predict(test_text_list)[0]:
        result.append(sample[0].replace("__label__", ""))

    print(classification_report(test_label_list, result))


if __name__ == "__main__":

    Path1 = 'C:/Users/text/Desktop/text_classifier/'
    Path2 = 'C:/Users/text/Desktop/data_news/'

    warnings.filterwarnings(module="sklearn*", action="ignore", category=DeprecationWarning)

    # jieba.load_userdict(Path2 + "userdict.txt")
    # StopWords = [line.strip() for line in open(Path2 + 'stopwords.txt', encoding="utf-8-sig").readlines()]

    print("start loading data...")

    file = pd.read_csv(Path1 + "company13.csv", encoding="utf-8-sig")

    total_text_list = []
    total_label_list = []

    order = list(file.index)
    random.shuffle(order)

    for ind in order:
        line = file.loc[ind]
        label = line["label"]
        title = line["title"]

        if line["content"] is np.nan or line["title"] is np.nan:
            continue

        content = line["content"]
        # content = "".join(re.findall(r"\D", line["content"]))

        if "融资融券" in title or "大宗交易" in title or "盘口异动" in content:
            # continue
            pass

        # 不去除停用词
        text_cut = " ".join(jieba.lcut((title + content).replace("\n", "")))
        # 去除停用词
        # text_cut = " ".join(list(filter(lambda x:
        # x not in StopWords, jieba.lcut((title*10 + content).replace("\n", "")))))

        total_text_list.append(text_cut)
        total_label_list.append(str(label))

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
        test_label_list = [total_label_list[i] for i in range(length) if i % 10 == term]
        train_label_list = [total_label_list[i] for i in range(length) if i % 10 != term]

        output_test = open(Path1 + "data_test.txt", 'w', encoding='utf-8-sig')
        output_train = open(Path1 + "data_train.txt", 'w', encoding='utf-8-sig')

        for num in range(len(train_label_list)):
            output_train.write("__label__" + str(train_label_list[num]) + "   " + train_text_list[num] + "\n")

        for num in range(len(test_label_list)):
            output_test.write("__label__" + str(test_label_list[num]) + "   " + test_text_list[num] + "\n")

        text_classification()
        print("fasttext model result:")
        # fasttext_model()

        output_train.flush()
        output_train.close()
        output_test.flush()
        output_test.close()
