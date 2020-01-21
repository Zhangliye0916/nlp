# -*-coding:utf-8-*-

import os
import uniout
import re
import pandas as pd


def file_name(file_dir):

    return [filename for filename in os.listdir(file_dir)
            if os.path.isfile(os.path.join(file_dir, filename))]


def follow_zhuyao():

    return [u'主要系', u'主要是', u'主要因', u'主要原因', u'主要由于', u'主要源于']


def punct_c():

    return [u'。', u'！', u'？', u' ?']


def end_word():

    return [u'所致']


def start_word():

    return [u'不适用', u'适用', u'说明']


def sentence_cut(text, punct, forward=True):

    if forward:
        keys_id = [text.find(key) for key in punct if text.find(key) != -1]
        if keys_id:
            return text[: min(keys_id)]

    else:
        keys_id = [text.rfind(key) + len(punct) for key in punct if text.find(key) != -1]
        # keys_id = [i.end() for i in re.compile(r"[" + "|".join(punct) + "]").finditer(text)]
        if keys_id:
            return text[max(keys_id) - 1:]


def find_id(findword, text):

    return [i.start() for i in re.compile(findword).finditer(text)]


if __name__ == "__main__":

    path1 = unicode(r'D:/上市公司定期报告/test', "utf-8")

    text_name = []
    key_word = []
    cause = []
    result = []

    for file in file_name(path1):
        # print file_name(path1)
        # print file

        f = open(os.path.join(path1, file))

        source = f.read()
        text = source.decode('utf-8')

        for findword in follow_zhuyao():

            for ind1 in find_id(findword, text):
                # print findword, sentence_cut(text[:ind1], punct_c(), False), sentence_cut(text[ind1: ], punct_c())
                # print types, elements
                # print file_name(path1)[7], findword, sentence_cut(text[ind1:], punct_c()).replace('\n', ''),
                # sentence_cut(text[:ind1], punct_c() + start_word(), False).replace('\n', '')
                # print file
                single_cause = sentence_cut(text[ind1:], punct_c() + end_word())
                single_result = sentence_cut(text[:ind1], punct_c() + start_word(), False)

                if len(single_cause) > 800 or len(single_result) > 800:
                    continue

                if len(re.findall(r'\s-?(\d+,?)*.?(\d+,?)*%?', single_cause)) > 3:
                    continue

                if len(re.findall(r'\s-?(\d+,?)*.?(\d+,?)*%?', single_result)) > 3:
                    continue

                if len(single_result) > 0 and single_result[0]:
                    pass

                single_cause = re.compile(r'\s').sub('', single_cause)
                single_result = re.compile(r'\s').sub('', single_result)

                if len(single_cause) >= 1 and len(single_result) >= 1:
                    # print file
                    text_name.append(file)
                    key_word.append(findword)
                    cause.append(single_cause)
                    result.append(single_result)

    causality = pd.DataFrame({'text': text_name, 'key_word':key_word, 'cause': cause, 'result': result})
    causality.to_csv('C:/Users/text/PycharmProjects/fin_network/data/causality_20190524.csv', encoding='utf_8_sig')
