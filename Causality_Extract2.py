# -*-coding:GBK-*-

import os
import uniout
import re
import pandas as pd


def file_name(file_dir):

    return [filename for filename in os.listdir(file_dir)
            if os.path.isfile(os.path.join(file_dir, filename))]


def follow_zhuyao():

    return [u'主要系', u'主要是', u'主要因', u'主要原因', u'主要为', u'主要由于', u'主要源于']


def punct_c():

    return [u'。', u'！', u'？', u'；']


def end_word():

    return [u'所致']


def start_word():

    return [u'不适用', u'说明', u'：', u':', u'\d+']


def sentence_cut(text, punct, forward=True):

    if forward:
        keys_id = [text.find(key) for key in punct if text.find(key) != -1]
        if keys_id:
            return text[: min(keys_id)]

    else:
        keys_id = [text.rfind(key) for key in punct if text.find(key) != -1]
        if keys_id:
            return text[max(keys_id) + 1:]


def find_id(findword, text):

    return [i.start() for i in re.compile(findword).finditer(text)]


def judge_number(text):

    if not text:
        return [0], ['']

    text_list = text.replace('\n', ' ').split(' ')
    types = []
    elements = []

    for element in text_list:
        if element:
            elements.append(element)
            try:
                float(element.replace('%', '').replace(',', '').replace(u'，', ''))

            except:
                types.append(0)

            else:
                types.append(1)

    return types, elements


def select_table(types, elements):

    try:
        id1 = types.index(1)
        types.reverse()
        id2 = len(types) - types.index(1) - 1

        return elements[id1 - 1: id2 + 1]

    except:
        pass


def drop_table(types, elements):

    try:
        id1 = types.index(1)

        return elements[id1]

    except:
        pass


if __name__ == "__main__":

    path1 = unicode('D:\上市公司定期报告\年报文本2', "gbk")

    with open('C:/Users/text/PycharmProjects/fin_network/data/jiayongdianqi.txt') as f:
        company_name = f.read().decode('gbk')

    text_name = []
    key_word = []
    cause = []
    result = []

    for file in file_name(path1):
        if file in company_name:
            # print file

            f = open(os.path.join(path1, file))

            source = f.read()
            text = source.decode('utf-8')

            for findword in follow_zhuyao():

                for ind1 in find_id(findword, text):
                    # print findword, sentence_cut(text[:ind1], punct_c(), False), sentence_cut(text[ind1: ], punct_c())

                    types, elements = judge_number(sentence_cut(text[:ind1], punct_c(), False))
                    # print types, elements
                    # print file_name(path1)[7], findword, sentence_cut(text[ind1:], punct_c()).replace('\n', ''), sentence_cut(text[:ind1], punct_c() + start_word(), False).replace('\n', '')

                    # print file

                    if sentence_cut(text[ind1:], punct_c()) != None and sentence_cut(text[:ind1], punct_c() + start_word(), False) != None:
                        print file
                        text_name.append(file)
                        key_word.append(findword)
                        cause.append(sentence_cut(text[ind1:], punct_c()).replace('\n', ''))
                        result.append(sentence_cut(text[:ind1], punct_c() + start_word(), False).replace('\n', ''))
                        break

    causality = pd.DataFrame({'text': text_name, 'key_word':key_word, 'cause': cause, 'result': result})
    causality.to_csv('C:/Users/text/PycharmProjects/fin_network/data/causality_20190403.csv', encoding='utf_8_sig')
