# -*- coding: utf-8 -*-
import jieba
import pandas as pd
import uniout
import jieba.posseg as pseg
import re


if __name__ == "__main__":

    jieba.load_userdict(unicode(r"C:\Users\text\PycharmProjects\fin_network\data\经济自定义词典.txt", "utf-8"))
    with open(r"C:\Users\text\PycharmProjects\fin_network\data\direct_word.txt", 'r') as f1:
        word_dict = {}
        for i, word in enumerate(f1.readlines()):
            word_dict[i] = word.replace('\n', '').split('\t')

        direct_word = pd.DataFrame(word_dict, index=['word', 'sign']).T
        up_word = list(direct_word[direct_word['sign'] == '1']['word'].values)
        down_word = list(direct_word[direct_word['sign'] == '-1']['word'].values)

    # print jieba.lcut('经营活动现金流')

    jieba.suggest_freq(('成本', '上升'), True)
    jieba.suggest_freq(('价格', '上涨'), True)
    jieba.suggest_freq(('价格', '调整'), True)

    with open(r"C:\Users\text\PycharmProjects\fin_network\data\test_sent5.txt", 'r') as f3:
        text = f3.readlines()
        f4 = open(r"C:\Users\text\PycharmProjects\fin_network\data\test_sent7.txt", 'a')

        for single_line in text:
            # print single_line
            number = single_line.split('	')[0]
            sent = single_line.split('	')[1]
            # 剔除原句中可能存在的数字
            OriginalSent = sent.replace('\n', '').replace('%', '')
            # sent = ''.join([word for word, prop in pseg.lcut(sent) if prop[0] is not 'm'])
            for num_in in list(set(re.findall('\d*,?\d*\.?\d*,?\d*', OriginalSent))):
                sent = OriginalSent.replace(num_in, '')

            print sent

            old_ind = 0
            for item in re.compile(r'|'.join(up_word + down_word)).finditer(sent):
                ind_end = item.start()
                ind_start = max([i.end() for i in re.compile(r',|，|此外|使得|导致|从而|引起|致使|原因|影响|公司').finditer(sent[:ind_end])
                                 ] + [old_ind])
                old_ind = item.end()
                ind_split = [i.end() + ind_start for i in re.compile(r'与|和|以及|及|且|、').finditer(sent[ind_start: ind_end])]
                inds = [ind_start] + ind_split + [ind_end]
                inds.sort()

                for myid in range(len(inds) - 1):
                    inner_sent = sent[inds[myid]: inds[myid + 1]]
                    word_prop = pseg.lcut(inner_sent)
                    event = [word for word, prop in word_prop if prop[0] in ['v', 'n']]
                    content = single_line.replace('\n', '') + '@' + ''.join(event).encode('utf-8') + sent[item.start(): item.end()] + '\n'
                    f4.write(content)

