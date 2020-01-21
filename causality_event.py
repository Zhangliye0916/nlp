# -*-coding:utf-8-*-

import jieba
import pandas as pd
import time
import uniout
import jieba.posseg as psg
import numpy as np


def verbs():
    return [u'v', u'vd', u'vn', u'vshi', u'vyou', u'vf', u'vx', u'vi', u'vl', u'vg']


def nouns():
    return [u'n', u'nr', u'nr1', u'nr2', u'nrj', u'nrf', u'ns', u'nsf', u'nt', u'nz', u'nl', u'ng']


def event_extract(sentence):

    if not sentence:
        return u''

    if sentence != sentence:
        return u''

    sentence_cut = psg.lcut(sentence)
    sentence_cut.reverse()

    out = []
    flag = 0
    for loop in sentence_cut:
        if loop.flag in nouns() and flag == 0:
            out.append(loop.word)
            flag = 1

        elif loop.flag in verbs() + [u'l', u'm']:
            out.append(loop.word)
            flag = 0

    out.reverse()
    return ''.join(out)


'''
data = pd.read_csv('C:/Users/text/Desktop/data/zhi_right.csv')[['key', 'cause', 'result', 'title']]

causality = pd.DataFrame([], columns=['cause', 'effect'])
i = 0

for loop1 in data.values:

    cause = event_extract(loop1[1])
    effect = event_extract(loop1[2])

    causality.loc[i] = [cause, effect]
    i += 1

# print causality
causality.to_csv('C:/Users/text/Desktop/data/causality.csv', encoding='utf-8')


'''

sentence = u'美国发生私人飞机坠机事件'
word = []
flag = []
for loop in psg.cut(sentence):
    word.append(loop.word)
    flag.append(loop.flag)

print word
print flag


