# -*-coding:utf-8-*-

import re
from string import punctuation
import pandas as pd
import jieba
import codecs
import numpy as np
import codecs
from collections import defaultdict


def load_sentences(path):

    with codecs.open(path, encoding='utf-8-sig') as f1:
        text = f1.read()

    return text.replace("\r", "").split("\n")


def load_words(path):

    with codecs.open(path, encoding="utf-8-sig") as f:
        text = f.readlines()
        my_dict = {}

        for line in text:
            word, weight = line.replace('\n', '').split('\t')
            try:
                my_dict[word] = float(weight)

            except ValueError:
                continue

        return my_dict


def split_punctuation():

    return [u'.', u'。', u',', u'，', u'!', u'！', u'?', u'？', u';', u'；']


def seg_sentences(sentence, punctuations):

    punctuation_in = re.findall('[' + '|'.join(punctuations) + ']', sentence)
    sent_slit = re.split('[' + '|'.join(punctuations) + ']', sentence)
    seg_sentence = []

    for i in range(len(sent_slit) - 1):
        seg_sentence.append(sent_slit[i] + punctuation_in[i])

    seg_sentence.append(sent_slit[-1])

    return seg_sentence


def write_sentence():

    with codecs.open(Path1 + "all_sentence_cut.txt", encoding="utf-8-sig") as f1:
        text = f1.readlines()
    f1.close()

    f2 = codecs.open(Path1 + "positive_sentences.txt", mode='w', encoding="utf-8-sig")
    f3 = codecs.open(Path1 + "negative_sentences.txt", mode='w', encoding="utf-8-sig")
    f4 = codecs.open(Path1 + "neutral_sentences.txt", mode='w', encoding="utf-8-sig")
    f5 = codecs.open(Path1 + "all_sentences.txt", mode='w', encoding="utf-8-sig")

    for comment in text:

        final_score = 0
        # 如果评论长度大于100个词语，则剔除该评论
        if len(comment) > 100:
            continue

        for sentence in (seg_sentences(comment, split_punctuation())):
            sentence_cut = sentence.split(" ")

            inner_score = len([word for word in sentence_cut if word in OriginalPositiveWord
                               ]) - len([word for word in sentence_cut if word in OriginalNegativeWord])

            if inner_score != 0:
                for word in sentence_cut:
                    if word in DenyWords:
                        inner_score *= -1

                    if word in DegreeWords.keys():
                        inner_score *= DegreeWords[word]

            final_score += inner_score

        if final_score > 0:
            f2.write(comment + '\n')

        if final_score < 0:
            f3.write(comment + '\n')

        if final_score == 0:
            f4.write(comment + '\n')

        f5.write(str(final_score) + "|" + comment + '\n')

        """
        if key_word_positive and len(deny_word) % 2 == 0:
            # 上涨后面不能有数字
            try:
                location_sz = sentence_cut.index(u"上涨")
                if len(re.findall(r"\d", sentence_cut[location_sz + 1])) >= 1:
                    break

            except IndexError:
                pass

            except ValueError:
                pass
        """

    f2.close()
    f3.close()


def find_new_word(cut_num1=10, c1=0.2, cut_num2=10, c2=0.2):

    positive_sentences = []
    positive_words = defaultdict(int)

    with codecs.open(Path1 + "positive_sentences.txt", encoding="utf-8-sig") as f1:
        for line in f1.readlines():
            line = line.replace(" \r\n", "")
            positive_sentences.append(line)
            for word in line.split(" "):
                positive_words[word] += 1

    negative_sentences = []
    negative_words = defaultdict(int)

    with codecs.open(Path1 + "negative_sentences.txt", encoding="utf-8-sig") as f1:
        for line in f1.readlines():
            line = line.replace(" \r\n", "")
            negative_sentences.append(line)
            for word in line.split(" "):
                negative_words[word] += 1

    f2 = codecs.open(Path1 + "new_word.txt", mode="w", encoding="utf-8-sig")
    for word, count in positive_words.items():
        if count > cut_num1:
            counter = 0
            for sentence in negative_sentences:
                if word in sentence:
                    counter += 1

            if float(counter) / float(count) < c1 and word not in OriginalPositiveWord:
                print(word, count, counter)
                f2.write("positive" + " " + word + " " + str(count) + " " + str(counter) + "\r\n")

    print ("_____________________________")
    for word, count in negative_words.items():
        if count > cut_num2:
            counter = 0
            for sentence in positive_sentences:
                if word in sentence:
                    counter += 1

            if float(counter) / float(count) < c2 and word not in OriginalNegativeWord:
                print(word, count, counter)
                f2.write("negative" + " " + word + " " + str(count) + " " + str(counter) + "\r\n")

    f2.close()


def test_new_word():

    new_word = {"positive_word": [], "negative_word": []}
    with codecs.open(Path1 + "new_word.txt", encoding="utf-8-sig") as f1:
        for line in f1.readlines():
            direction, word = line.replace("\r\n", "").split(" ")[: 2]

            if direction == u"positive":
                new_word["positive_word"].append(word)

            if direction == u"negative":
                new_word["negative_word"].append(word)

    f3 = codecs.open(Path1 + "test_sentences.txt", mode="w", encoding="utf-8-sig")

    with codecs.open(Path1 + "neutral_sentences.txt", encoding="utf-8-sig") as f2:
        for line in f2.readlines():
            line = line.replace(" \r\n", "")

            for word in line.split(" "):
                if word in new_word["positive_word"]:
                    f3.write("positive" + " " + word + "|" + line + "\r\n")
                if word in new_word["negative_word"]:
                    f3.write("negative" + " " + word + "|" + line + "\r\n")


if __name__ == "__main__":

    Path1 = 'C:/Users/text/Desktop/data_news/'

    OriginalPositiveWord = load_sentences(Path1 + "positive_word.txt")
    OriginalNegativeWord = load_sentences(Path1 + "negative_word.txt")
    DenyWords = load_words(Path1 + 'deny_word.txt').keys()
    DegreeWords = load_words(Path1 + 'degree_word.txt')

    # write_sentence()
    # find_new_word(cut_num1=5, c1=0.1, cut_num2=5, c2=0.1)
    # test_new_word()

    TestSentence = u"跌倒 5 元 以下 ， 只是 时间 问题 。 "
    SentenceCut = TestSentence.split(" ")
    KeyWordPositive = [word for word in SentenceCut if word in OriginalPositiveWord]
    KeyWordNegative = [word for word in SentenceCut if word in OriginalNegativeWord]
    DenyWord = [word for word in SentenceCut if word in DenyWords]
    DegreeWord = [word for word in SentenceCut if word in DegreeWords]

    print (KeyWordPositive)
    print (KeyWordNegative)
    print (DenyWord)
    print (DegreeWord)

    FinalScore = 0

    for Sentence in (seg_sentences(TestSentence, split_punctuation())):
        SentenceCut = Sentence.split(" ")

        InnerScore = len([word for word in SentenceCut if word in OriginalPositiveWord
                          ]) - len([word for word in SentenceCut if word in OriginalNegativeWord])

        if InnerScore != 0:
            for word in SentenceCut:
                if word in DenyWords:
                    InnerScore *= -1

                if word in DegreeWords.keys():
                    InnerScore *= DegreeWords[word]

        FinalScore += InnerScore

    print(FinalScore)

    NewWord = {"positive_word": [u"问题"],
               "negative_word": []}
    with codecs.open(Path1 + "neutral_sentences.txt", encoding="utf-8-sig") as F:
        for Line in F.readlines():
            Line = Line.replace(" \r\n", "")

            for word in Line.split(" "):
                if word in NewWord["positive_word"]:
                    print ("positive word: %s" % word, Line)
                if word in NewWord["negative_word"]:
                    print ("negative word: %s" % word, Line)
