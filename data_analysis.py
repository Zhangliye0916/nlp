# -*-coding:utf-8-*-

import jieba
import pandas as pd
import time
import uniout


t1 = time.time()


def load_data(path):

    return pd.read_csv(path)


def origin_words():

    origin_word = {u'所以',
                   u'以致',
                   u'因此',
                   u'因而',
                   u'因为',
                   u'由于',
                   u'从而',
                   u'故',
                   u'故而',
                   u'结果',
                   u'为是',
                   u'惟其',
                   u'为此',
                   u'以至',
                   u'以至于',
                   u'因',
                   u'因之',
                   u'于是',
                   u'之所以',
                   u'致',
                   u'致使'}

    return origin_word


sentences = load_data('C:/Users/text/Desktop/data/news.csv')['title']
sentences = sentences.drop_duplicates()

sentences_cut = []

for sentence in sentences:
    sentences_cut.append(jieba.lcut(sentence))

data = pd.DataFrame(columns=['key', 'title'])
i = 0
for word in list(origin_words()):
    for sentence in sentences_cut:
        if word in sentence:
            data.loc[i] = [word, ''.join(sentence)]
            i += 1

# new_data.to_csv('C:/Users/text/Desktop/data/news_out.csv', encoding='utf-8')
data.to_csv('C:/Users/text/Desktop/data/news39_cut.csv', encoding='utf-8', index=False)
# print data

t2 = time.time()
print t2 - t1




