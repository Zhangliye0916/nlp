# -*- coding: utf-8 -*-

import jieba
import codecs
from jieba.analyse import extract_tags
import re
import gensim
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import uniout
from collections import defaultdict


def data_clear():

    with codecs.open((Path1 + "pinglun.txt").encode('utf-8'), encoding='utf-8') as f:
        with codecs.open((Path1 + "pinglun_clear.txt").encode('utf-8'), 'a', encoding='utf-8') as f2:

            text = f.readlines()
            for line in text:
                sentence = line.replace("【#水滴筹回应吴鹤臣众筹#】", "").replace("【#水滴筹回应吴鹤臣众筹#】", "") \
                    .replace("【德云社吴鹤臣众筹百万引质疑 水滴筹：没资格审核发起人车产房产】", "")
                f2.write(sentence)


if __name__ == "__main__":

    key_words = []
    Path1 = "C:/Users/text/Desktop/"
    sentences = []
    ClusterNum = 10


    with codecs.open((Path1 + "pinglun_clear.txt").encode('utf-8'), encoding='utf-8') as f:

        text = f.readlines()
        counter = 0
        for line in text:
            key_word = extract_tags(line, topK=10)
            # print(key_word)
            key_words += key_word

        # print(len(set(key_words)))
        # print(sentences)

    data = pd.read_csv(Path1 + "sentence_vector.csv").set_index("Unnamed: 0")
    data.index.name = 'num'
    MyVector = data.values

    ClusterModel = KMeans(n_clusters=ClusterNum)
    ClusterResult = ClusterModel.fit_predict(MyVector)

    WordDict = defaultdict(int)
    doc2ind = {}
    with codecs.open(Path1 + "doc2ind.txt", encoding="utf-8") as f:

        for line in f.readlines():
            # print line
            num, doc = line.split("|")[: 2]
            doc2ind[int(num)] = doc
            for word in jieba.lcut(doc):
                WordDict[word] += 1
                # print WordDict[word]
            # print num, doc

    f.close()

    for key, value in WordDict.items():

        if value > 300:
            print key, value


    '''
    with codecs.open(Path1 + "cluster.txt",  "w", encoding="utf-8") as f2:
        cluster = {}
        for i in range(ClusterNum):
            cluster[i] = []
            for j in range(len(ClusterResult)):
                if ClusterResult[j] == i:
                    cluster[i].append(doc2ind[j])
                    f2.write(str(i) + "|" + doc2ind[j])
            # print('label_' + str(i) + ':' + str(cluster[i]))
            print extract_tags("".join(cluster[i]), topK=30)
            
    '''
