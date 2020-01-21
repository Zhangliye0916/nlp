# encoding=utf-8

import jieba
import uniout
import jieba.analyse


f = open('C:/Users/text/PycharmProjects/fin_network/data/result_sent.txt')

for sentence in f.readlines():

    print sentence.decode('gbk')

    keywords = jieba.analyse.extract_tags(sentence, topK=5, withWeight=True, allowPOS=('n', 'nr', 'ns'))

    # print(type(keywords))

    # <class 'list'>

    for item in keywords:
        print(item[0], item[1])

