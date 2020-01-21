# -*- coding: utf-8 -*-

import uniout
import re
import pandas as pd
import codecs

def find_id(findword, text):

    return [i.start() for i in re.compile(findword).finditer(text)]

'''
with open(u'C:/Users/text/PycharmProjects/fin_network/data/HIT-IRLab-同义词词林（扩展版）_full_2005.3.3.txt') as f:

    lines = f.readlines()
    dict = {}
    all_word = []

    for line in lines:
        line = line.decode('gbk')
        item = line.replace('\n', '').split(' ')
        dict[item[0]] = item[1:]

        for word in line.split(' ')[1:]:
            if word not in all_word:

                # f2 = open(u'C:/Users/text/PycharmProjects/fin_network/data/同义词词林重排.txt', 'a')

                for id in find_id(word, text):
                    # print word
                    # print text[text[:id].rfind("#"): id], 'aaa'
                    pass
                    
'''

'''
with open(u'C:/Users/text/PycharmProjects/fin_network/data/HIT-IRLab-同义词词林（扩展版）_full_2005.3.3.txt') as f:

    lines = f.readlines()
    all_word = ''

    for line in lines:
        line = line.decode('gbk')
        item = line.replace('\n', '').split(' ')
        for word in item[1:]:
            all_word += word + ' '+ item[0] + '\n'

    f2 = codecs.open(u'C:/Users/text/PycharmProjects/fin_network/data/同义词词林重排.txt', 'w',"UTF-8")
    print all_word
    print type(all_word)
    f2.write(all_word)
    
'''

# data = pd.read_csv(u'C:/Users/text/PycharmProjects/fin_network/data/同义词词林重排.txt')


with open(u'C:/Users/text/PycharmProjects/fin_network/data/百科爬取疑似上位词.txt') as f:

    lines = f.readlines()
    all_word = ''

    for line in lines:
        line = line
        item = line.replace('\n', '').split(':')
        for word in item[1].split(' '):
            all_word += item[0] + ' '+ word + '\n'

    f2 = codecs.open(u'C:/Users/text/PycharmProjects/fin_network/data/同义词词林重排2.txt', 'w')
    print all_word
    print type(all_word)
    f2.write(all_word)
