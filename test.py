# -*- coding: utf-8 -*-

# import gensim
import jieba
# import uniout
from gensim.models import word2vec
from collections import Counter
import pandas as pd
from smart_open import open as op


def seg_sentence(text):

    return jieba.lcut(text)


def load_data(path):

    with op(path) as f:
        text = f.readlines()

        return text


def save_data(model, path1, path2):

    with open(path1, "r") as f:
        AllWord = []
        for sentence in f.readlines():
            AllWord += sentence.split(" ")

    WordVector = pd.DataFrame()
    counter = Counter(AllWord)
    for word in sorted(counter, key=counter.get, reverse=True):
        try:
            WordVector[word] = model[word.decode('utf-8')]

        except KeyError:
            print(word)

    Data = WordVector.T
    Data.to_csv(path2, encoding='utf-8_sig')


if __name__ == "__main__":

    Path1 = 'C:/Users/text/Desktop/data_news/'

    with open(Path1 + "sentences4_out.txt", "a") as f:

        for Ind, Text in enumerate(load_data(Path1 + 'sentences4.txt')):
            try:
                f.write(' '.join(seg_sentence(Text)).encode('utf-8'))

            except UnicodeEncodeError:
                print (Ind)

    sentences = word2vec.LineSentence(Path1 + 'sentences4_out.txt')
    model = word2vec.Word2Vec(sentences, hs=1, min_count=1,window=3,size=100)
    model.save(Path1 + 'sentences4.model')

    model = word2vec.Word2Vec.load(Path1 + 'sentences4.model')
    print(model.similar_by_word(u'中签率'))
    print(model.similar_by_word(u'下跌'))
    print(model.similar_by_word(u'上涨'))
    print(model.similar_by_word(u'暴跌'))
    print(model.similar_by_word(u'网上'))
    print(model.similar_by_word(u'投资界'))
    # print(model[u'投资界'])


