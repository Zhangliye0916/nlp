# -*- coding: utf-8 -*-
import jieba.posseg as pseg
import uniout
import jieba
import sys

reload(sys)
sys.setdefaultencoding('utf8')

if __name__ == '__main__':

    to_drop = ['本报告期内', '报告期内', '报告期末', '报告期', '本期']
    to_add = [u'增加', u'减少', u'下降', u'上升', u'大', u'小',
              u'冷淡', u'火爆', u'短期', u'长期', u'及', u'和', u'与',
              u'负面', u'不利', u'影响', u'长期', u'及', u'和', u'与']
    jieba.load_userdict('C:/Users/text/PycharmProjects/fin_network/data/newdict.txt')
    f = open('C:/Users/text/PycharmProjects/fin_network/data/cause.txt')

    # stopwords = set([line.strip().decode('gb2312') for line
                     # in open('C:/Users/text/PycharmProjects/fin_network/data/stopwords.txt', 'r').readlines()])  # 读入停用词

    f2 = open('C:/Users/text/PycharmProjects/fin_network/data/text_new2.txt', 'w')
    for sentence in f.readlines():
        sentence.strip()

        for word in to_drop:
            sentence.replace(word, '')

        # 去掉停用词中的词
        # sentence_cut = [x for x in jieba.lcut(sentence) if x not in stopwords]
        sentence_new = ''.join([word for (word, flag) in pseg.cut(sentence) if flag[0] in [u'n', u'v'] or word in to_add])
        # print sentence, type(sentence)
        f2.write(sentence_new + '     ' + unicode(sentence, 'gbk') + '\n')

    f2.close()
