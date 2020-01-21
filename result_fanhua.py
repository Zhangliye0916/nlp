# encoding=utf-8

import jieba
import uniout
import jieba.analyse
import pandas as pd


f = open('C:/Users/text/PycharmProjects/fin_network/data/result_sent.txt')

classes = {}

for loop in f.readlines():

    sentence = loop.decode('gbk')
    if sentence not in classes.keys():

        classes[sentence] = []

        if u"投资" in sentence and u"现金流" in sentence:
            classes[sentence].append(u"投资活动现金流量变动" + u"&")

        if u"筹资" in sentence and u"现金流" in sentence:
            classes[sentence].append(u"筹资活动现金流量变动" + u"&")

        if u"经营" in sentence and u"现金流" in sentence:
            classes[sentence].append(u"经营活动现金流量变动" + u"&")

        if u"营业" in sentence and u"收入" in sentence:
            classes[sentence].append(u"营业收入变动" + u"&")

        if u"现金及现金等价物" in sentence:
            classes[sentence].append(u"现金及现金等价物变动" + u"&")

        if u"产量" in sentence or u"产销量" in sentence:
            classes[sentence].append(u"生产量变动" + u"&")

        if u"销售量" in sentence or u"产销量" in sentence:
            classes[sentence].append(u"销售量变动" + u"&")

        if u"库存量" in sentence:
            classes[sentence].append(u"库存量变动" + u"&")

        if u"研发" in sentence:
            classes[sentence].append(u"研发投入变动" + u"&")

        if u"销售费用" in sentence:
            classes[sentence].append(u"销售费用变动" + u"&")

        if u"财务费用" in sentence:
            classes[sentence].append(u"财务费用变动" + u"&")

        if u"管理费用" in sentence:
            classes[sentence].append(u"管理费用变动" + u"&")

        if u"借款" in sentence:
            classes[sentence].append(u"借款变动" + u"&")

        if u"营业外收入" in sentence:
            classes[sentence].append(u"营业外收入变动" + u"&")

        if u"营业外支出" in sentence:
            classes[sentence].append(u"营业外支出变动" + u"&")

        if u"营业成本" in sentence:
            classes[sentence].append(u"营业成本" + u"&")

        if u"资产减值" in sentence:
            classes[sentence].append(u"资产减值损失变动" + u"&")


print classes

out = pd.Series(classes)
out.to_csv('C:/Users/text/PycharmProjects/fin_network/data/classes_20190408.csv', encoding='utf_8_sig')



