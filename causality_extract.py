# -*-coding:utf-8-*-

import jieba
import pandas as pd
import time
import uniout


t1 = time.time()


def key_words():

    return {u'因', u'致', u'致使'}


def punct_c():

    return [u'，', u'；', u'。', u'!', u' ', u'：', u' ', u'？', u':', u',', u'！', u'?', u'　']


def stop_yins():

    return [u'被', u'致', u'受', u'原因']


def c_e_yin2(sentence_cut, index_c, index_r, index_s=None, Type=0):

    if Type == 0:
        return ''.join(sentence_cut[index_c + 1: index_r]), ''.join(sentence_cut[index_r: index_s])

    else:
        return ''.join(sentence_cut[index_c + 1: index_s]), ''.join(sentence_cut[index_r: index_c])


def c_e_yin(sentence_cut):

    # print sentence_cut
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
            return u'因', c_e_yin2(sentence_cut, index1, index2 + 1, index3, type)

        return u'因', c_e_yin2(sentence_cut, index1, index2, index3, type)

    return


def c_e_zhi(sentence_cut):
    # print sentence_cut

    index1 = sentence_cut.index(u'致')
    if index1 == 0:
        return []

    if sentence_cut[index1 - 1] in [u'―', u'—']:
        return []

    stop1 = [sentence_cut[index1 - 2:: -1].index(x)
             for i, x in enumerate(punct_c()) if x in sentence_cut[: index1 - 1]]
    stop2 = [sentence_cut[index1 + 1:].index(x)
             for i, x in enumerate(punct_c()) if x in sentence_cut[index1 + 1: ]]

    stop1.append(index1 - 1)
    stop2.append(len(sentence_cut) - index1)
    stop1 = min(stop1)
    stop2 = min(stop2)

    if stop2 + index1 + 1 <= len(sentence_cut):
        if sentence_cut[stop2 + index1 + 1] in [u':', u'：']:
            return []

    if [x for i, x in enumerate([u'信', u'词', u'电'])
        if x in ''.join(sentence_cut[index1 + 1: stop2 + index1 + 1])]:
        return []

    if u'“' in sentence_cut[index1 - stop1 - 1: index1] \
            and u'”' in sentence_cut[index1 + 1: stop2 + index1 + 1]:
        return []

    if sentence_cut[index1 -1] in punct_c():
        return [u'致', ''.join(sentence_cut[index1 - stop1 - 1: index1]),
               ''.join(sentence_cut[index1 + 1: stop2 + index1 + 1])]

    return [u'致', ''.join(sentence_cut[index1 - stop1 - 1: index1]),
           ''.join(sentence_cut[index1 + 1: stop2 + index1 + 1])]


# print c_e_zhi(jieba.lcut(u'经济观察报改版致读者：这仍是花开的季节'))

data = pd.read_csv('C:/Users/text/Desktop/data/news39_cut.csv').values

right = pd.DataFrame(columns=['key', 'cause', 'result', 'title'])
wrong = pd.DataFrame(columns=['title'])

i = 0
j = 0
for loop in data:
    key = loop[0]

    if key.decode('utf-8') == u'致':
        sentence_cut = jieba.lcut(loop[1].decode('utf-8'))

        result = c_e_zhi(sentence_cut)

        if result:
            right.loc[i] = result + [loop[1]]
            i += 1

        else:
            wrong.loc[j] = [loop[1]]
            j += 1


# print right
# print wrong
right.to_csv('C:/Users/text/Desktop/data/zhi_right.csv', encoding='utf-8')
wrong.to_csv('C:/Users/text/Desktop/data/zhi_wrong.csv', encoding='utf-8')


