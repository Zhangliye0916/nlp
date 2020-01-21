# -*-coding:utf-8-*-

import os


def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):

        return files


def punct_c1():

    return ['，', '；', '。', '：', '？', '！', '　',  ' ',
            ',', ';',  '.', ':', '?', '!', ' ',  '\n']


def punct_c2():

    return ['。', '？', '！',
            '?', '!', '\n']


def sentence_cut(punct, text, method=0):

    if method == 0:
        id1 = [text.find(p) for p in punct]
        id2 = [id for id in id1 if id != -1]
        return min(id2)

    else:
        id1 = [find_all(p, text) for p in punct]
        id2 = [max(l) for l in id1 if len(l) > 0]
        return max(id2)


def find_all(str, text):

    inds = []
    ind = 0
    ind_abs = 0

    while ind != -1:
        ind = text[ind_abs:].find(str)
        ind_abs += ind + 1
        inds.append(ind_abs)

    return inds[:-1]


if __name__ == '__main__':

    path_in = unicode('C:/Users/text/Desktop/上市公司定期报告/年报文本节选', "utf-8")
    files = file_name(path_in)

    for single_file in files[12:13]:
        try:
            f = open(path_in + "/" + single_file)

        except IOError:
            print path_in + "/" + single_file
            continue

        text = f.read()

        # print text.count('主要是'), text.count('主要原因'), text.count('主要系'), text.count('所致')
        ids = find_all('主要原因为', text)
        print text
        text1 = text[ids[0] - 100: ids[0]]
        id = sentence_cut(punct_c2(), text1, 1)
        print id, len(text1)
        print text1
        print text1[id -2: -1]


