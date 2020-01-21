#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: Baike_extract.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-8-11

import urllib2
import urlparse
from lxml import etree
from urllib import quote
import jieba.posseg as pseg
import os
import uniout
import pandas as pd


def drop_words():
    return ['歌', '电影', '书', '人', '动漫', '音乐', '角色', '曲', '作品']


class BaiduBaike:

    def get_html(self, url):
        return urllib2.urlopen(url).read().decode('utf-8').replace('&nbsp;', '')

    def info_extract_baidu(self, word):  # 百度百科
        url = "http://baike.baidu.com/item/%s" % quote(word)
        selector = etree.HTML(self.get_html(url))
        info_list = list()
        info_list.append(self.extract_baidu(selector, word))
        polysemantics = self.checkbaidu_polysemantic(selector, word)
        if polysemantics:
            info_list += polysemantics
        infos = [info for info in info_list if len(info) > 2]

        return infos

    def extract_baidu(self, selector, word):
        info_data = {}
        if selector.xpath('//h2/text()'):
            info_data['current_semantic'] = selector.xpath('//h2/text()')[0].encode("utf-8").replace('    ', '').replace('（','').replace('）','')
        else:
            info_data['current_semantic'] = unicode(word, 'utf-8')

        if info_data['current_semantic'] in [unicode('目录', 'utf-8'), '目录']:
            info_data['current_semantic'] = unicode(word, 'utf-8')

        info_data['tags'] = [item.replace('\n', '') for item in selector.xpath('//span[@class="taglist"]/text()')]
        info_data['tags'] += [item.replace('\n', '') for item in selector.xpath('//span[@class="taglist"]/a[@target="_blank"]/text()')]

        if selector.xpath("//div[starts-with(@class,'basic-info')]"):
            for li_result in selector.xpath("//div[starts-with(@class,'basic-info')]")[0].xpath('./dl'):
                attributes = [attribute.xpath('string(.)').replace('\n', '') for attribute in li_result.xpath('./dt')]
                values = [value.xpath('string(.)').replace('\n', '') for value in li_result.xpath('./dd')]
                for item in zip(attributes, values):
                    info_data[item[0].replace(' ', '')] = item[1].replace(' ', '')

        return info_data

    def checkbaidu_polysemantic(self, selector, word):
        semantics = ['https://baike.baidu.com' + sem for sem in
                     selector.xpath("//ul[starts-with(@class,'polysemantList-wrapper')]/li/a/@href")]
        names = [name for name in selector.xpath("//ul[starts-with(@class,'polysemantList-wrapper')]/li/a/text()")]
        info_list = names
        if semantics:
            for item in zip(names, semantics):
                selector = etree.HTML(self.get_html(item[1]))
                info_data = self.extract_baidu(selector, word)
                info_data['current_semantic'] = item[0].replace(u' ', '').replace(u'（', '').replace(u'）', '')
                if info_data:
                    info_list.append(info_data)

        return info_list


class HudongBaike:
    def get_html(self, url):
        return urllib2.urlopen(url).read().decode('utf-8').replace('&nbsp;', '')

    def info_extract_hudong(self, word):  # 互动百科
        url = "http://www.baike.com/wiki/%s" % quote(word)
        selector = etree.HTML(self.get_html(url))
        info_list = list()
        info_data = self.extract_hudong(selector)
        if selector.xpath('//li[@class="current"]/strong/text()'):
            info_data['current_semantic'] = selector.xpath('//li[@class="current"]/strong/text()')[0].replace(u'    ', '').replace(u'（','').replace(u'）','')
        else:
            info_data['current_semantic'] = unicode(word, 'utf-8')
        info_list.append(info_data)
        polysemantics = self.checkhudong_polysemantic(selector)
        if polysemantics:
            info_list += polysemantics
        infos = [info for info in info_list if len(info) > 2]

        return infos

    def extract_hudong(self, selector):
        info_data = {}
        info_data['desc'] = selector.xpath('//div[@id="content"]')[0].xpath('string(.)')
        info_data['intro'] = selector.xpath('//div[@class="summary"]')[0].xpath('string(.)').replace(u'编辑摘要', '')
        info_data['tags'] = [item.replace('\n', '') for item in selector.xpath('//p[@id="openCatp"]/a/text()')]
        for info in selector.xpath('//td'):
            attribute = info.xpath('./strong/text()')
            val = info.xpath('./span')
            if attribute and val:
                value = val[0].xpath('string(.)')
                info_data[attribute[0].replace(u'：', '')] = value.replace(u'\n', '').replace(u'  ', '').replace(u'    ', '')
        return info_data

    def checkhudong_polysemantic(self, selector):
        semantics = [sem for sem in selector.xpath("//ul[@id='polysemyAll']/li/a/@href") if 'doc_title' not in sem]
        names = [name for name in selector.xpath("//ul[@id='polysemyAll']/li/a/text()")]
        info_list = list()
        if semantics:
            for item in zip(names, semantics):
                if "http:" not in item[1]:
                    str_in = "http:" + item[1].encode('utf-8')
                else:
                    str_in = item[1]
                selector = etree.HTML(self.get_html(str_in))
                info_data = self.extract_hudong(selector)
                info_data['current_semantic'] = item[0].replace(u'（','').replace(u'）','')
                if info_data:
                    info_list.append(info_data)
        return info_list


class SemanticBaike:
    def __init__(self):
        cur = '/'.join(os.path.realpath(__file__).split('/')[:-1])
        self.tmp_file = os.path.join(cur, 'word_concept.txt')

    '''根据instance本身抽取其概念'''
    def extract_concept(self, word):
        wds = [w.word for w in pseg.cut(word) if w.flag[0] in ['n']]
        if not wds:
            return ''
        else:
            return wds[-1]

    '''对三大百科得到的semantic概念进行对齐'''
    def extract_main(self, word):
        try:
            baidu = BaiduBaike()
            info = baidu.info_extract_baidu(word)
            baidu_info = []
            for loop in info:
                if isinstance(loop, dict):
                    baidu_info.append([loop['current_semantic'], loop['tags']])

        except:
            baidu_info = []

        try:
            hudong = HudongBaike()
            info = hudong.info_extract_hudong(word)
            hudong_info = []
            for loop in info:
                if isinstance(loop, dict):
                    hudong_info.append([loop['current_semantic'], loop['tags']])

        except:
            hudong_info = []

        semantic_dict = {}
        semantics = []
        tuples = []
        concepts_all = []

        semantics += baidu_info
        semantics += hudong_info

        for i in semantics:
            instance = i[0]
            concept = i[1]

            if not instance:
                # print concept
                continue

            if instance not in semantic_dict:
                semantic_dict[instance] = concept

            else:
                semantic_dict[instance] += concept

        # 对从百科知识库中抽取得到的上下位关系进行抽取
        for instance, concepts in semantic_dict.items():
            concepts = set([i for i in concepts if i not in ['', ' ']])
            concept_pre = self.extract_concept(instance)
            concepts_all += concepts
            concepts_all += [concept_pre]
            tuples.append(instance)
            tuples.append(concept_pre)
            tuples += concepts

        # 对词汇本身上下位义进行上下位抽取
        tmps = [j for i in concepts_all for j in concepts_all if j in i and i and j]
        tuples += tmps
        out = []

        for tuple in tuples:
            if isinstance(tuple, unicode):
                tuple = tuple.encode('utf-8')

            if word == tuple:
                continue

            if not any([drop_word in tuple for drop_word in drop_words()]):
                if '的' in tuple:
                    out.append(tuple.split('的')[-1])

                else:
                    out.append(tuple)

        return list(set(out))


if __name__ == '__main__':

    handler = SemanticBaike()
    print handler.extract_main('苹果')



