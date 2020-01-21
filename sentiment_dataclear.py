# -*-coding:utf-8-*-

import re
from string import punctuation
import pandas as pd
import jieba
import codecs
import numpy as np


def chinese_punctuation():

    return u"！“”‘’#￥&（）*+，。/：；《=》？@【、】……——、{|}~"


def drop_punctuation(text):

    pattern = []
    for s in chinese_punctuation() + punctuation:
        pattern.append(s)

    pattern = "[^" + "|".join(pattern) + "]"

    text_drop = []

    for sentence in text:
        text_drop.append("".join(re.findall(pattern, sentence)))

    return text_drop


def load_sentences(path):

    with codecs.open(path, encoding='utf-8-sig') as f:
        text = f.read()
    # print text
    lines = text.split('\n')

    return lines


def word_cut(text, path):

    with codecs.open(path, encoding="utf-8-sig", mode='w') as f:
        for line in text:
            if len(line.split("|")) == 2:
                ind, comment = line.split("|")
                f.write(ind + "|" + " ".join(jieba.lcut(comment)) + '\n')


if __name__ == "__main__":

    Path1 = 'C:/Users/text/Desktop/data_news/'
    Path2 = 'C:/Users/text/Desktop/text_classifier/'
    jieba.load_userdict(Path1 + "userdict.txt")
    Text = load_sentences(Path2 + "300003.txt")
    word_cut(Text, Path2 + "300003_cut.txt")
    # word_cut(drop_punctuation(Text), Path1 + "allsentence_clear.txt")
