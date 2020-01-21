# -*- coding: utf-8 -*-
import numpy as np


def energy_f(c, e, a, b):

    return 0.5 * np.square(np.linalg.norm(np.dot(a, c) - e, ord=2)) \
           + 0.5 * np.square(np.linalg.norm(np.dot(b, e) - c, ord=2))


if __name__ == "__main__":

    A = np.loadtxt(r"D:/工作/金融事理图谱/event/data/result_A.txt", dtype=np.float, delimiter=',')
    B = np.loadtxt(r"D:/工作/金融事理图谱/event/data/result_B.txt", dtype=np.float, delimiter=',')

    with open(r"D:/工作/金融事理图谱/event/data/测试集向量.txt") as f2:

        cause = {}
        effect = {}
        for Text in f2.readlines():

            item = Text.replace('\n', '').split('	')
            my_key = item[0]
            my_value = item[1].split(',')
            col = []
            for Num in my_value:
                col.append(float(Num))

            if my_key[0] == 'c':
                cause[my_key] = col

            elif my_key[0] == 'e':
                effect[my_key] = col

    acc1 = 0
    topK = 1  # 可选参数topK
    for key_c in cause.keys():

        energy = {}
        for key_e in effect.keys():

            energy[key_e] = energy_f(cause[key_c], effect[key_e], A, B)

        sorted_f = sorted(energy.values())
        # print (sorted_f)
        if sorted_f.index(energy['e' + key_c[1:]]) <= topK - 1:
            acc1 += 1

    print("由因到果的topK命中率为：", acc1 * 1.0/len(cause))

    acc2 = 0
    for key_e in effect.keys():

        energy = {}
        for key_c in cause.keys():

            energy[key_c] = energy_f(cause[key_c], effect[key_e], A, B)

        sorted_f = sorted(energy.values())
        if sorted_f.index(energy['c' + key_e[1:]]) <= topK - 1:
            acc2 += 1
        # print sorted_f

    print("由果到因的topK命中率为：", acc2 * 1.0 / len(cause))
