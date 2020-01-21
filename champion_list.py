# -*-coding:utf-8-*-

from jqdatasdk import *
# import pandas as pd
import numpy as np
from collections import Counter


# auth("15811211802", "211802")
auth("18612754762", "xyz117")
print(get_query_count())

df = get_billboard_list(None, "2017-2-1", "2017-3-1")
print(np.sum(df["buy_value"]), np.sum(df["sell_value"]), np.sum(df["amount"]))
print(Counter(df["abnormal_name"]))


df = get_billboard_list(None, "2017-3-1", "2017-4-1")
print(np.sum(df["buy_value"]), np.sum(df["sell_value"]), np.sum(df["amount"]))
print(Counter(df["abnormal_name"]))


df = get_billboard_list(None, "2018-7-1", "2018-8-1")
print(np.sum(df["buy_value"]), np.sum(df["sell_value"]), np.sum(df["amount"]))
print(Counter(df["abnormal_name"]))

df = get_billboard_list(None, "2018-11-1", "2018-12-1")
print(np.sum(df["buy_value"]), np.sum(df["sell_value"]), np.sum(df["amount"]))
print(Counter(df["abnormal_name"]))