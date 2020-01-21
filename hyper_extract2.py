# -*- coding: utf-8 -*-

import uniout
import sys


f1 = open(u'C:/Users/text/PycharmProjects/fin_network/data/经济高频词.txt')
f2 = open(u'C:/Users/text/PycharmProjects/fin_network/data/HIT-IRLab-同义词词林（扩展版）_full_2005.3.3.txt')
f3 = open(u'C:/Users/text/PycharmProjects/fin_network/data/经济高频词vs同义词.txt', 'w')

text = f2.readlines()
count = 0
for word in f1.readlines():
    # print word.decode('gbk')
    word = word.replace('\n', '')
    loc = ' '
    for word2 in text:
        word2 = word2.replace('\n', '').split(' ')
        # print word2[1].decode('gbk')
        if word in word2:
            # print word.decode('gbk'), word2[0]
            loc = loc + "/" + word2[0]

    count += 1
    print count
    # print word.decode('gbk'), word2.decode('gbk'), type(word), type(word2)
    f3.write(word + loc + '\n')

f3.close()



