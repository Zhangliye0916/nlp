# -*- coding: utf-8 -*-


def energy_f(c, e, t, tao):

    out = 0
    for i in range(len(tao)):
        out += (abs(c[i] + t[i] - e[i]) + abs(e[i] + tao[i] - c[i]))

    return out


def load_data(path):

    with open(path) as f:
        t = []
        tao = []
        text = f.readlines()
        for i in range(len(text)):
            temp = []
            for num in text[i].split(','):
                temp.append(float(num))

            if i % 2 == 0:
                t.append(temp)

            else:
                tao.append(temp)

    return t, tao


if __name__ == "__main__":

    all_t, all_tao = load_data(r"C:\Users\text\Desktop/event/data/result1.txt")

    with open(r"C:\Users\text\PycharmProjects\fin_network\data\cet2.txt") as f2:

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

    for loop in range(len(all_t)):

        T = all_t[loop]
        Tao = all_tao[loop]
        acc1 = 0
        topK = 10
        for key_c in cause.keys():

            energy = {}
            for key_e in effect.keys():

                energy[key_e] = energy_f(cause[key_c], effect[key_e], T, Tao)

            sorted_f = sorted(energy.values())
            # print sorted_f
            if sorted_f.index(energy['e' + key_c[1:]]) <= topK - 1:
                acc1 += 1

            # print energy.values().count(min(energy.values()))
        print (loop, acc1 * 1.0/len(cause))

        acc2 = 0
        for key_e in effect.keys():

            energy = {}
            for key_c in cause.keys():

                energy[key_c] = energy_f(cause[key_c], effect[key_e], T, Tao)

            sorted_f = sorted(energy.values())
            if sorted_f.index(energy['c' + key_e[1:]]) <= topK - 1:
                acc2 += 1
            # print sorted_f
        print (loop, acc2 * 1.0 / len(cause))
