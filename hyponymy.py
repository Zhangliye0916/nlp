# -*-coding:utf-8-*-

import jieba
import pandas as pd
import time
import uniout
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sinica_treebank


print(sinica_treebank.words()[100:120])
# sinica_treebank.parsed_sents()[33].draw()

# nltk.download()
# wn.syssets('love')

# print wn.synsets(u'摩托车')
# print wn.synset('car.n.01').lemma_names