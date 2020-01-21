# -*-coding:utf-8-*-

import jieba
import pandas as pd
import time
import uniout
import jieba.posseg as psg
import numpy as np
from word_similarity import WordSimilarity2010, SimilarBase
from itertools import permutations as pt
import word_extract


def my_path():
    path1 = {'cilin': 'C:/Users/text/PycharmProjects/fin_network/data/cilin_ex.txt',
             'stopwords': 'C:/Users/text/PycharmProjects/fin_network/data/stopwords.txt',
             'title': 'C:/Users/text/PycharmProjects/fin_network/data/title.txt'
             }
    return path1


ws_tool = WordSimilarity2010()

s1 = u'官方称哈尔滨社保局工作失误'
s2 = u'加拿大多伦多街头突发枪击案 '


def EMD(x):
    x_n = len(x[0])
    out = []

    for loop1 in pt(range(x_n), x_n):
        sign = list(loop1)
        dis = 0

        for loop2 in loop1:
            max1 = 0
            index1 = sign[0]

            for loop3 in sign:
                if x[loop2][loop3] > max1:
                    max1 = x[loop2][loop3]
                    index1 = loop3

            dis += max1
            sign.remove(index1)

        out.append(dis)

    return max(out)


def load_cilin(path=my_path()['cilin']):
    f = open(path, 'r')
    f.readline()
    cilin = {}

    for loop in f:
        temp = loop.replace('\n', '').split(' ')
        cilin[temp[0][:-1]] = temp[1:]

    f.close()

    return cilin


cilin = load_cilin(my_path()['cilin'])


def stopwordslist(filepath):

    return [line.strip().decode('gb2312') for line in open(filepath, 'r').readlines()]


def drop_stopwords(sentence):

    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist(my_path()['stopwords'])  # 这里加载停用词的路径
    outstr = []

    for word in sentence_seged:
        if word not in stopwords + ['\t']:
            outstr.append(word)

    return outstr


def generalize(word, layer, cilin):

    Including = filter(lambda x: word in x[1], cilin.items())
    out = []

    for loop in Including:
        if layer == 1:
            key = ''.join(parse_code(loop[0])[:1]) + 'a01A01'

        elif layer == 2:
            key = ''.join(parse_code(loop[0])[:2]) + '01A01'

        elif layer == 3:
            key = ''.join(parse_code(loop[0])[:3]) + 'A01'

        else:
            print 'Please input the right layer!'
            return

        try:
            out.append(cilin[key][0])

        except:
            pass

    return out


def parse_code(c):

    return [c[0], c[1], c[2:4], c[4], c[5:7]]


def drop_stop_word(s):

    pair = psg.lcut(s)
    out = []
    for loop in pair:

        if loop.flag not in ['y', 'uj', 'm', 'x']:
            out.append(loop.word)

    return out


def sentence_similarity(s1, s2, cilin):

    long1 = drop_stop_word(s1)
    long2 = drop_stop_word(s2)

    # print s1, s2

    l1 = []
    for loop in long1:
        temp = generalize(loop.encode('utf-8'), 3, cilin)
        if temp:
            l1.append(temp[0])

        else:
            l1.append(loop)

    l2 = []
    for loop in long2:
        temp = generalize(loop.encode('utf-8'), 3, cilin)
        if temp:
            l2.append(temp[0])

        else:
            l2.append(loop.encode('utf-8'))

    if len(l1) < len(l2):
        l1 = l1 + ['']*(len(l2) - len(l1))

    elif len(l2) < len(l1):
        l2 = l2 + ['']*(len(l1) - len(l2))

    # print s2, l2

    sim_mat = []
    for loop1 in l1:
        sim_arr = []

        for loop2 in l2:
            sim_arr.append(ws_tool.similarity(loop1, loop2))

        sim_mat.append(sim_arr)

    return EMD(sim_mat)/(1.0*len(sim_mat))


def hyper_weight(word_hyper, word, w1=0.8, w2=0.4):

    out = {}
    for ind in word_hyper.index:
        if word_hyper['fst_hypernym'][ind] == word:
            out[word_hyper['sec_hypernym'][ind]] = w1

        else:
            if word_hyper['sec_hypernym'][ind] not in out.keys():
                out[word_hyper['sec_hypernym'][ind]] = w2

    return out


def sen_sim(s1, s2):

    word1 = drop_stopwords(s1)
    word2 = drop_stopwords(s2)

    word_sim = SimilarBase()
    out_cilin = {}
    baike = word_extract.SemanticBaike()
    for word in word1 + word2:
        if word.encode('utf-8') not in word_sim._data and word not in out_cilin.keys():
            print word
            hyper = baike.extract_main(word.encode('utf-8'))
            out_cilin[word] = hyper_weight(hyper, word)

    for loop1 in out_cilin.items():
        word_in = loop1[1].keys()

        if not word_in:
            continue

        if loop1[0] in word1:
            word_out = word1

        else:
            word_out = word2

        sim_sums = {}
        for loop2 in loop1[1].keys():
            if loop2 not in word_sim._data:
                continue

            sim_sum = 0
            for loop3 in word_in + word_out:
                sim_sum += pow(ws_tool.similarity(loop2, loop3), 2)

            sim_sums[loop2] = sim_sum

        if not sim_sums:
            continue

        hyper = sorted(sim_sums.items(), key=lambda item:item[1])[-1][0]
        word1 = ' '.join(word1).replace(loop1[0], hyper).split(' ')
        word2 = ' '.join(word2).replace(loop1[0], hyper).split(' ')

    # print word1, word2

    if len(word1) < len(word2):
        word1 = word1 + ['']*(len(word2) - len(word1))

    elif len(word2) < len(word1):
        word2 = word2 + ['']*(len(word1) - len(word2))

    sim_mat = []
    for loop1 in word1:
        sim_arr = []

        for loop2 in word2:
            if loop1 == loop2:
                sim_arr.append(1.0)

            else:
                sim_arr.append(ws_tool.similarity(loop1.encode('utf-8'), loop2.encode('utf-8')))

        sim_mat.append(sim_arr)

    return EMD(sim_mat)/(1.0*len(sim_mat)), out_cilin, sim_mat


'''
data = pd.read_csv('C:/Users/text/Desktop/data/df1.csv')['sent']

ss = pd.DataFrame(index=range(len(data)))
for s1 in data:
    temp = []

    for s2 in data:
        temp.append(sentence_similarity(s1.decode("gb2312"), s2.decode("gb2312"), cilin))

    ss[s1.decode("gb2312")] = temp
    print s1.decode("gb2312")
    print temp

ss.to_csv('C:/Users/text/Desktop/data/df2.csv', encoding='utf-8')

'''


if __name__ == '__main__':

    # print sentence_similarity(s1, s2, cilin)
    # print sen_sim(s1, s2)
    # print ws_tool.similarity('哈尔滨', '加拿大')
    # word_sim = SimilarBase()
    # print '枪' in word_sim._data
    # baike = word_extract.SemanticBaike()
    # print baike.extract_main('枪击案')
    path = my_path()['title']
    f = open(path, 'r')
    f.readline()

    sentences = []
    for loop in f:
        sentences.append(unicode(loop, 'gbk'))

    # print sentences
    out_cilins = []
    for sent1 in sentences:
        for sent2 in sentences:
            [sims, out_cilin, mat] = sen_sim(sent1, sent2)
            out_cilins.append(out_cilin)
            print sent1, sent2, sims


