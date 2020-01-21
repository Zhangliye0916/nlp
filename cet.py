# -*- coding: utf-8 -*-


def energy_f(c ,e , t, tao):

    out = 0
    for i in range(len(tao)):
        out += abs(c[i] + t[i] - e[i]) + abs(e[i] + tao[i] - c[i])

    return out


if __name__ == "__main__":

    with open(r"C:\Users\text\Desktop/event/data/result.txt") as f:

        text = f.read().split('\n')

        t1 = text[0].split(',')
        tao1 = text[1].split(',')

        t= []
        for num in t1:
            t.append(float(num))

        tao = []
        for num in tao1:
            tao.append(float(num))

    with open(r"C:\Users\text\PycharmProjects\fin_network\data\cet3.txt") as f2:

        cause = []
        effect = []
        counter = 0
        for text in f2.readlines():

            l = text.replace('\n', '').split(',')
            col = []
            for num in l:
                col.append(float(num))

            if counter%2 == 0:
                cause.append(col)

            else:
                effect.append(col)

            counter += 1
    acc = 0
    topK = 1
    for i in range(len(cause)):

        energy = []
        for j in range(len(cause)):

            energy.append(energy_f(cause[i], effect[j], t, tao))

        temp = energy[i]

        energy.sort()
        # print energy.index(temp)
        if energy.index(temp) <= topK - 1:
            acc += 1

    print acc * 1.0/len(cause)

    acc2 = 0
    for i in range(len(cause)):

        energy = []
        for j in range(len(cause)):

            energy.append(energy_f(cause[j], effect[i], t, tao))

        temp = energy[i]

        energy.sort()
        # print energy
        # print energy.index(temp)
        if energy.index(temp) <= topK - 1:
            acc2 += 1

    print acc2 * 1.0 / len(cause)








