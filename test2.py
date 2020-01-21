# -*- coding: utf-8 -*-


'''
import os
import jieba
import uniout
import jieba.posseg as psg
from jieba import analyse
# 引入TF-IDF关键词抽取接口
tfidf = analyse.extract_tags

# 原始文本
f = open('C:/Users/text/PycharmProjects/fin_network/data/result.txt')

text = f.read()

# 基于TF-IDF算法进行关键词抽取
jieba.load_userdict('C:/Users/text/PycharmProjects/fin_network/data/newdict.txt')
keywords = tfidf(text, topK=50)
print "keywords by tfidf:"
# 输出抽取出的关键词
for keyword in keywords:
    print keyword + "/",

'''
# from nltk.parse import stanford

from stanfordcorenlp import StanfordCoreNLP
import uniout
import jieba.posseg as pseg

nlp = StanfordCoreNLP(r'C:\ProgramData\Anaconda2\stanfordNLP\stanford-corenlp-full-2018-10-05', lang='zh')# 这里改成你stanford-corenlp所在的目录
sentence = '对公司未来业绩造成不利影响'
print [(word, flag) for (word, flag) in pseg.cut(sentence)]
# print 'Tokenize:', nlp.word_tokenize(sentence)
# print 'Part of Speech:', nlp.pos_tag(sentence)
# print 'Named Entities:', nlp.ner(sentence)
print 'Constituency Parsing:', nlp.parse(sentence)
# print 'Dependency Parsing:', nlp.dependency_parse(sentence)
nlp.close() # Do not forget to close! The backend server w
