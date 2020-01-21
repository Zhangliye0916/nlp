# -*-coding:utf-8-*-

import jieba
import pandas as pd
import time
import uniout
import re


t1 = time.time()


def key_words():

    return {u'因', u'致', u'致使'}


def punct_c():

    return [u'，', u';', u'。', u'!']


def stop_yins():

    return [u'被', u'致', u'受', u'原因']


def sen_pseg(sentences):

    out = []
    for sentence in sentences:
        out.append(pseg.lcut(sentence))

    return pd.Series(out)


def cause2result_cr(sentence_cut, index_c, index_r, index_s=None, Type=0):

    if Type == 0:
        return ''.join(sentence_cut[index_c + 1: index_r]), ''.join(sentence_cut[index_r: index_s])

    else:
        return ''.join(sentence_cut[index_c + 1: index_s]), ''.join(sentence_cut[index_r: index_c])


def cause2result(sentence_cut):

    print sentence_cut
    stop_yin = [x for i, x in enumerate(stop_yins()) if x in sentence_cut]

    if u'因' in sentence_cut and stop_yin:

        index1 = sentence_cut.index(u'因')
        index2 = sentence_cut.index(stop_yin[0])

        if index2 > index1:
            type = 0
            stop_words = set(sentence_cut[index2:]) & set(punct_c())

        else:
            type = 1
            stop_words = set(sentence_cut[index1:]) & set(punct_c())

        if len(stop_words) == 0:
            index3 = None

        else:
            index3 = []

            if type == 0:
                for stop_word in stop_words:
                    index3.append(sentence_cut[index2:].index(stop_word))

                index3 = min(index3) + index2

            else:
                for stop_word in stop_words:
                    index3.append(sentence_cut[index1:].index(stop_word))

                index3 = min(index3) + index1

        if stop_yin[0] in [u'原因']:
            return u'因', cause2result_cr(sentence_cut, index1, index2 + 1, index3, type)

        return u'因', cause2result_cr(sentence_cut, index1, index2, index3, type)

    return


# print cause2result(jieba.lcut(u'新华保险董事长万峰因个人年龄原因辞去董事长兼首席执行官等职务'))

sentences = pd.read_csv('C:/Users/text/Desktop/data/news2.csv')['title']
print sen_pseg(sentences).values[0]