# -*-coding:utf-8-*-

import uniout
import time
import word_extract


if __name__ == '__main__':

    with open(u'C:/Users/text/PycharmProjects/fin_network/data/待处理词汇2.txt') as f:

        handler = word_extract.SemanticBaike()
        for word in f.readlines():

            f2 = open(u'C:/Users/text/PycharmProjects/fin_network/data/百科爬取疑似上位词2.txt', 'a')
            word = word.replace('\n', '')
            hyper_words = handler.extract_main(word)
            print word + ':' + ' '.join(hyper_words)

            for loop in hyper_words:
                f2.write(word + ':' + loop + '\n')

            f2.close()




