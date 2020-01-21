# 利用搜狐新闻语料库计算每个词语的idf值，
# -*-coding:utf-8-*-
import numpy as np
import math
from collections import defaultdict

doc_num = 0
doc_frequency = defaultdict(int)
with open('sohu_train.txt', encoding='utf-8') as trainText:
    for line in trainText:
        id, catgre, body = line.split('^_^')
        # if doc_num>100000:break
        doc_num += 1
        for word in set(body.split('    ')):
            word = word.replace('\n', '').strip()
            # if word in stopword :continue
            if word == '' or word == '': continue
            doc_frequency[word] += 1

fw = open('idf-1.txt', 'w', encoding='utf-8')
for word in doc_frequency:
    idf = math.log(doc_num / (doc_frequency[word] + 1))
    fw.write(word + ' ' + str(idf) + '\n')
    print(word, doc_frequency[word])
fw.close()
print('procesing completed')

# 加载已经训练好的idf值，计算部分文章的tfidf，返回相应关键词
idf_dict = defaultdict(int)
with open('idf-1.txt', encoding='utf-8') as idf_dict_text:
    for line in idf_dict_text:
        word, value = line.split(' ')
        idf_dict[word] = float(value)

doc_num = 0
with open('sohu_train.txt', encoding='utf-8') as trainText:
    for line in trainText:
        id, catgre, body = line.split('^_^')
        # 仅抽取前5篇文档的关键词
        if doc_num > 5:
            break
        else:
            doc_num += 1
        word_num = 0
        word_frequency = defaultdict(int)
        for word in body.split('    '):  # 每篇文档中词频统计
            word = word.replace('\n', '').strip()
            if word == '' or word == '': continue
            word_frequency[word] += 1
            word_num += 1

        for word in word_frequency:  # 计算当前文章中每个词的tfidf值
            # print(idf_dict[word],type(idf_dict[word]))
            tfidf = idf_dict[word] * word_frequency[word] / word_num
            word_frequency[word] = tfidf
        word_sorted = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
        print('document:', body.strip().replace('    ', ''))
        print('keywords:', word_sorted[:5])