# -*-coding:utf-8-*-

import tushare as ts

ts.set_token('6f24f949abfe007a10ce411138abe846f334ea882f5f96c8aacfe4c1')

pro = ts.pro_api()

df = pro.cctv_news(date='20190115')

print df

