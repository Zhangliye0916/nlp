# -*-coding:utf-8-*-
import pandas as pd
import uniout
import jieba


path0 = u'C:/Users/text/PycharmProjects/fin_network/data/公司简称-公司编码.txt'
path1 = u'C:/Users/text/PycharmProjects/fin_network/data/家电产业链图谱.txt'
path2 = u'C:/Users/text/PycharmProjects/fin_network/data/实体类型.txt'
path3 = u'C:/Users/text/PycharmProjects/fin_network/data/申万三级行业-产品.txt'
path4 = u'C:/Users/text/PycharmProjects/fin_network/data/公司编码-申万三级行业.txt'
path5 = u'C:/Users/text/PycharmProjects/fin_network/data/经济自定义词典.txt'
path6 = u'C:/Users/text/PycharmProjects/fin_network/data/同义词词表.txt'


jieba.load_userdict(path5)


def load_data(path, cols=[0, 2, 1]):

    lines = open(path).readlines()[1:]
    dict = {}

    for (i, text) in enumerate(lines):
        temp = text.replace('\n', '').split('	')
        dict[i] = [temp[col] for col in cols]

    return pd.DataFrame(dict, index=cols).T


info = '华帝股份格力电器长虹美菱电视玻璃超市小家电产量上升'


# 分词and同义词归并
synonyms = load_data(path6, cols=[0, 1])
sent_cut = [word.encode('utf-8') for word in jieba.lcut(info)]
info_new = [synonyms[1][list(synonyms[0].values).index(word)]
            if word in synonyms[0].values else word for word in sent_cut]

# 实体识别
obj = load_data(path2, cols=[0, 1])


def obj_reconize(info_new, obj):
    out = []
    for word in info_new:

        temp = [(word, obj[1][i]) for i in obj.index if obj[0][i] == word]
        if temp:
            out.append(temp)

    return out


obj_in = obj_reconize(info_new, obj)
# 识别上市公司
company_in = [x[0][0] for x in obj_in if x[0][1] == "上市公司简称"]

# 识别产品
product_in = [x[0][0] for x in obj_in if x[0][1] == "主营产品"]

# 识别行业
industry_in = [x[0][0] for x in obj_in if x[0][1] == "申万三级行业"]

#
if company_in:
    companycode2industry = load_data(path4, cols=[0, 1, 3])
    companycode2industry.columns = ['公司编码', '申万三级行业', '所占比例']
    company2companycode = load_data(path0, cols=[0, 1])
    company2companycode.columns = ['上市公司简称', '公司编码']

    company2industry_in = pd.merge(pd.DataFrame(company_in, columns=['上市公司简称']), company2companycode, how='inner')
    company2industry_in = pd.merge(company2industry_in, companycode2industry, how='inner')
    company2industry_in['所占比例'] = company2industry_in['所占比例'].astype('float')
    company2industry_in = company2industry_in[company2industry_in['所占比例'] >= 0.7] # 行业阈值

    industry_chain = load_data(path1, cols=[0, 1, 2])
    industry_chain.columns = ['申万三级行业', '相关行业', '关系属性']
    all_data = pd.merge(company2industry_in, industry_chain, how='inner')

    print all_data




