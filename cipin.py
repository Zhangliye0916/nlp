# -*-coding:utf-8-*-
from collections import Counter
import jieba
import uniout
import codecs

jieba.load_userdict(unicode('C:/Users/text/PycharmProjects/fin_network/data/经济自定义词典.txt', 'utf-8'))  # 加载自己的词典
# 创建停用词list


def stopwordslist(filepath):
    stopwords = [line.strip().decode('gb2312') for line in open(filepath, 'r').readlines()]
    return stopwords


# 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('C:/Users/text/PycharmProjects/fin_network/data/stopwords.txt')# 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


outputs = codecs.open('C:/Users/text/PycharmProjects/fin_network/data/out.txt', 'a', encoding='utf-8')  # 加载处理后的文件路径

import os
file_dir = unicode('C:/Users/text/Desktop/上市公司定期报告/年报文本', 'utf-8')


def file_name2(file_dir):  # file_dir文件的路径
   return [os.path.join(file_dir, filename) for filename in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, filename))]


lll = file_name2(file_dir)
print(lll)
# 打开中文文件
count = 0

for i in lll[:100]:
    input = codecs.open(i, encoding='utf-8')  # 打开文件
    print count
    print i
    count += 1
    outputs.write(input.read().replace('\n', ''))
    input.close()

outputs.close()

# WordCount
with codecs.open('C:/Users/text/PycharmProjects/fin_network/data/out.txt', 'r') as fr:  # 读入已经去除停用词的文件
    data = jieba.cut(fr.read())
    print '结巴分词完成'
data = dict(Counter(data))

print '词频统计完成'

with codecs.open('C:/Users/text/PycharmProjects/fin_network/data/cipin.txt', 'w', encoding='utf-8') as fw:  # 读入存储wordcount的文件路径
    for k, v in data.items():
        fw.write('%s,  %d\n' % (k, v) + '\n')


