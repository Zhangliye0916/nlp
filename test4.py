# -*- coding: utf-8 -*-

from snownlp import SnowNLP
import codecs


def load_sentences(path):

    with codecs.open(path, encoding='utf-8-sig') as f1:
        text = f1.read()

    return text.replace("\r", "").split("\n")


for sent in s.sentences:
    print (sent, SnowNLP(sent).sentiments)


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
    Text = codecs.open(Path1 + "all_sentence_cut.txt", encoding="utf-8=sig")
