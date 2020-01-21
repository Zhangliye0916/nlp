# -*-coding:utf-8-*-

import os
import uniout
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
from jieba import analyse

'''
def file_name2(file_dir):

    return [filename for filename in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, filename))]

path = unicode('C:/Users/text/Desktop/数据挖掘/能力', "utf-8")
print file_name2(path)

with open(unicode("C:/Users/text/Desktop/name.txt", "utf-8"), 'w') as f:
    for name in file_name2(path):
        f.write(name.encode('gbk') + '\n')


with open(unicode("C:/Users/text/PycharmProjects/fin_network/data/name_vector.txt", "utf-8")) as f:

    text = f.readlines()
    data = []
    for line in text:
        vector = []
        for num in line.split(','):
            vector.append(float(num))

        data.append(vector)


estimator = KMeans(n_clusters=3)#构造聚类器

estimator.fit(data)#聚类

label_pred = estimator.labels_ #获取聚类标签

for j in label_pred:
    print j

print len(label_pred)

'''


# 引入TF-IDF关键词抽取接口
tfidf = analyse.extract_tags

with open(unicode("C:/Users/text/PycharmProjects/fin_network/data/names.txt", "utf-8")) as f:

    text = f.read()
    # 基于TF-IDF算法进行关键词抽取
    keywords = tfidf(text.decode('gbk'), topK=100)
    # print "keywords by tfidf:"
    # 输出抽取出的关键词
    for keyword in keywords:
        print keyword
        

'''
with open(unicode("C:/Users/text/PycharmProjects/fin_network/data/name_vector2.txt", "utf-8")) as f:

    text = f.readlines()
    data = []
    for line in text:
        vector = []

        for num in line.split(line[1]):
            vector.append(float(num))
        # print vector
        data.append(vector)


estimator = KMeans(n_clusters=7)#构造聚类器

estimator.fit(data)#聚类

label_pred = estimator.labels_ #获取聚类标签

for j in label_pred:
    print j

print len(label_pred)

'''


