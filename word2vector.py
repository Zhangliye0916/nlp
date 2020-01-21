# -*- coding: utf-8 -*-

# import gensim
import jieba
# import uniout
from gensim.models import word2vec
from collections import Counter
import pandas as pd
import codecs


def seg_sentence(text):

    return jieba.lcut(text)


def load_data(path):

    with open(path) as f:
        text = f.readlines()

        return text


def save_data(model, path1, path2):

    with open(path1, "r") as f:
        AllWord = []
        for sentence in f.readlines():
            AllWord += sentence.split(" ")

    print (len(AllWord))
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

    sentences = word2vec.LineSentence(Path1 + 'allsentence_clear.txt')
    print (sentences)
    model = word2vec.Word2Vec(sentences, hs=1, min_count=1,window=3,size=100)
    print ('model')
    model.save(Path1 + 'allsentences.model')

    save_data(model, Path1 + 'allsentence_clear.txt', Path1 + 'wordvector2.csv')

    model = word2vec.Word2Vec.load(Path1 + 'sentences4.model')
    print(model.similar_by_word(u'中签率'))
    print(model.similar_by_word(u'下跌'))
    print(model.similar_by_word(u'上涨'))
    print(model.similar_by_word(u'暴跌'))
    print(model.similar_by_word(u'网上'))
    print(model.similar_by_word(u'投资界'))
    # print(model[u'投资界'])
