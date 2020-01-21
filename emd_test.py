# -*-coding:utf-8-*-
import numpy as np
from itertools import permutations as pt


# p、q是两个矩阵，第一列表示权值，后面三列表示直方图或数量
def EMD(x):

    x_n = len(x[0])
    out = []

    for loop1 in pt(range(x_n), 3):
        sign = list(loop1)
        dis = 0

        for loop2 in loop1:
            max1 = 0
            index1 = sign[0]

            for loop3 in sign:
                if x[loop2][loop3] > max1:
                    max1 = x[loop2][loop3]
                    index1 = loop3

            dis += max1
            sign.remove(index1)

        out.append(dis)

    return max(out)







