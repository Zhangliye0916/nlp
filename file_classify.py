# -*-coding:utf-8-*-

import os
import uniout
import pandas as pd

path1 = unicode('C:/Users/text/Desktop/上市公司定期报告/年报文本节选4', "utf-8")
path2 = unicode('C:/Users/text/Desktop/上市公司定期报告/年报文本节选——张紫乾', "utf-8")


def file_name(rootDir):

    return [filename for filename in os.listdir(rootDir)]


if __name__ == "__main__":

    data = file_name(path1)
    # print data

    f = open('C:/Users/text/PycharmProjects/fin_network/data/file_name6.txt', 'w')
    counter = 0
    for single_file in data:
        print type(single_file), single_file
        single_file.split("_")
        counter += 1
        print counter
        f.write(single_file.encode("utf-8") + '\n' )
        # f.write(sentence_new + '     ' + unicode(sentence, 'gbk') + '\n')
    f.close()





