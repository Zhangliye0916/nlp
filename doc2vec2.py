# coding:utf-8
import os
import sys
import gensim
import jieba
import numpy as np
from jieba import analyse
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import pandas as pd
import codecs
from string import punctuation
import re
import uniout

TaggededDocument = gensim.models.doc2vec.TaggedDocument

Path1 = "C:/Users/text/Desktop/"
Path2 = "C:/Users/text/PycharmProjects/fin_network/data/"


def chinese_punctuation():

    return "！“”‘’#￥&（）*+，。/：；《=》？@【、】……——、{|}~"


def get_datasest():
    fin = codecs.open(Path1 + "pinglun_clear.txt", "r", encoding="utf-8").readlines()

    # 添加自定义的词库用于分割或重组模块不能处理的词组。
    # jieba.load_userdict("userdict.txt")
    # 添加自定义的停用词库，去除句子中的停用词。
    stopwords = set([line.strip() for line in
                     codecs.open(Path2 + "stopwords.txt", "r", encoding='utf-8').readlines()])   #读入停用词

    # 去掉停用词中的词和标点
    pattern = []
    for s in chinese_punctuation() + punctuation:
        pattern.append(s)

    pattern = "[^" + "|".join(pattern) + "]"

    x_train = []
    for i,sub_list in enumerate(fin):
        # sub_list = "".join(re.findall(pattern, sub_list))
        # sentence_cut = [x for x in jieba.lcut(sub_list) if x not in stopwords]
        sentence_cut = jieba.lcut(sub_list)
        # print (sentence_cut)
        document = TaggededDocument(sentence_cut, tags=[i])
        # document是一个Tupple,形式为：TaggedDocument( 杨千嬅 现在 教育 变成 一种 生意 , [42732])
        # print(document)
        x_train.append(document)

    return x_train


def getVecs(model, corpus, size):

    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, size)) for z in corpus]

    return np.concatenate(vecs)


def train(x_train, size=50):
    # D2V参数解释：
    # min_count：忽略所有单词中单词频率小于这个值的单词。
    # window：窗口的尺寸。（句子中当前和预测单词之间的最大距离）
    # vector_size:特征向量的维度
    # sample：高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)。
    # negative: 如果>0,则会采用negativesampling，用于设置多少个noise words（一般是5-20）。默认值是5。
    # workers：用于控制训练的并行数。
    model_dm = Doc2Vec(x_train, min_count=1, window=5, vector_size=size, sample=1e-3, negative=5, workers=4, hs=1,
                       epochs=6)
    # total_examples：统计句子数
    # epochs：在语料库上的迭代次数(epochs)。

    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save(Path1 + "model_test.model")

    return model_dm


def test():

    model_dm = Doc2Vec.load(Path1 + "model_test.model")
    test_ = '伤亡'
    # 读入停用词
    stopwords = set([line.strip() for line in
                     codecs.open(Path2 + "stopwords.txt", "r", encoding='utf-8').readlines()])
    # 去掉停用词中的词
    test_text = [x for x in jieba.lcut(test_) if x not in stopwords]
    # print text
    # 获得对应的输入句子的向量
    inferred_vector_dm = model_dm.infer_vector(doc_words=test_text)
    # print(inferred_vector_dm)
    # 返回相似的句子
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)

    return sims


def similar_sentence(text):

    model_dm = Doc2Vec.load(Path1 + "model_test.model")
    f2 = open(Path1 + "doc2vec.txt", 'w')
    out1 = pd.DataFrame(index=range(50))
    # out2 = pd.DataFrame(index=range(11))
    tag = 0
    f3 = codecs.open(Path1 + "doc2ind.txt", "w", encoding="utf-8")
    for loop in text:
        sentence_cut = loop.words
        # 获得对应的输入句子的向量
        inferred_vector_dm = model_dm.infer_vector(doc_words=sentence_cut)
        # print(inferred_vector_dm)
        out1[tag] = list(inferred_vector_dm)
        f2.write(str(inferred_vector_dm))
        f3.write(str(tag) + "|" + ''.join(loop.words))
        tag += 1

        # 返回相似的句子
        sim_sentence = [''.join(sentence_cut)]
        sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
        # print sentence_cut
        for tags, sim in sims:
            # print (loop.words, ''.join(text[tags].words))
            sim_sentence.append(''.join(text[tags].words).replace('/n', ''))

    out1 = out1.T
    out1.to_csv(Path1 + "sentence_vector.csv", encoding='utf-8_sig')

    f2.close()


if __name__ == '__main__':
    x_train = get_datasest()
    # print x_train
    # print type(x_train[0])
    model_dm = train(x_train)
    # out =  getVecs(model_dm, x_train, 200)
    # print len(out[0]), len(out)

    # sims = test(x_train)
    similar_sentence(x_train)

    # print('sims:'+str(sims))
