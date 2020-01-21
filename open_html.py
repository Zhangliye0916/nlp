# -*-coding:utf-8-*-

import os
import uniout


def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):

        return files


def dir_name(rootDir):

    return [os.path.join(rootDir, filename) for filename in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, filename))]


def file_name2(file_dir):

    return [filename for filename in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, filename))]


'''
if __name__ == '__main__':

    path_in = unicode('C:/Users/text/Desktop/上市公司定期报告/20160331（600236）', "utf-8")
    files = file_name(path_in)

    for single_file in files:

        try:
            f = open(path_in + "/" + single_file)

        except IOError:
            print path_in + "/" + single_file
            continue

        text = f.read()

        ind1 = text.find('<h3 >字段1_文本</h3><p>')
        ind2 = text.find('</p><h3 >标题</h3><p>')
        ind3 = text.find('</p><h3 >时间</h3><p>')
        ind4 = text.find('</p><h3 >字段2</h3><p>')

        if any(ind == -1 for ind in [ind1, ind2, ind3, ind4]):
            continue

        stock = text[ind1 + 27: ind2]
        title = text[ind2 + 23: ind3].replace("\n", "").replace(" ", "").replace("查看PDF原文", "")
        date = text[ind3 + 23: ind4]

        str_remove = ["摘要", "更正", "英文版", "补充", "半年度"]
        if "年度报告" in title and not any(name in title for name in str_remove):
            # print title

            sign_start1 = ' 经营情况讨论与分析\n'
            sign_end1 = '第五节重要事项\n'
            sign_start2 = '第四节经营情况讨论与分析\n'
            sign_end2 = ' 重要事项\n'

            not_fund = False

            if text.count(sign_start1) == 1:
                ind5 = text.find(sign_start1)

            elif text.count(sign_start2) == 1:
                ind5 = text.find(sign_start2)

            else:
                not_fund = True
                # print single_file, title
                continue

            if text.count(sign_end1) == 1:
                ind6 = text.find(sign_end1)

            elif text.count(sign_end2) == 1:
                ind6 = text.find(sign_end2)

            else:
                not_fund = True
                # print single_file, title
                continue

            if not not_fund:
                content = text[ind5: ind6]

                title = title.replace("*", "_").replace(":", "_").replace("/", "_")
                print title
                with open(unicode("C:/Users/text/Desktop/上市公司定期报告/年报文本节选3/"
                                  + str(date) + "_" + str(title) + ".txt", "utf-8"), "w") as f2:
                    f2.write(title + '\n')
                    f2.write(stock + '\n')
                    f2.write(date + '\n')
                    f2.write(content + '\n')

'''


if __name__ == '__main__':

    path_in1 = unicode('D:\数据\创业板定期公告\创业板', "utf-8")

    # print files1
    count = 0
    num = 0
    for single_dir in dir_name(path_in1):

        print single_dir

        for single_file in file_name(single_dir):
            try:
                f = open(single_dir + '/' + single_file)

            except IOError:
                print single_dir + '/' + single_file
                continue

            text = f.read()

            ind1 = text.find('<h3 >字段1</h3><p>')
            ind2 = text.find('</p><h3 >标题</h3><p>')
            ind3 = text.find('</p><h3 >时间</h3><p>')
            ind4 = text.find('</p><h3 >字段2</h3><p>')

            if any(ind == -1 for ind in [ind1, ind2, ind3, ind4]):
                # print count
                count += 1
                continue

            stock = text[ind1 + 23: ind2]
            title = text[ind2 + 23: ind3].replace("\n", "").replace(" ", "").replace("查看PDF原文", "")
            date = text[ind3 + 23: ind4]
            str_remove = ["摘要", "更正", "英文版", "补充", "半年度", "说明", "半年报"]
            # print stock
            if ("年度报告" in title or "年报" in title) and not any(name in title for name in str_remove):
                content = text[ind4: ]

                title = title.replace("*", "_").replace(":", "_").replace("/", "_")
                stock = stock.replace("*", "+").replace(":", "+").replace("/", "+")

                with open(unicode("D:\上市公司定期报告\年报文本节选4/"
                                  + str(date) + "_" + str(stock) + "_" + str(title) + ".txt", "utf-8"), "w") as f2:
                    f2.write(title + '\n')
                    f2.write(stock + '\n')
                    f2.write(date + '\n')
                    f2.write(content + '\n')
                    num += 1
                    
    print num
                    


'''


path_in1 = unicode('C:/Users/text/Desktop/上市公司定期报告', "utf-8")

files1 = file_name(path_in1)
# print files1

for single_dir in dir_name(path_in1):
    print single_dir

    for single_file in file_name(single_dir):
        try:
            f = open(single_dir + "/" + single_file)

        except IOError:
            print single_dir + "/" + single_file
            continue

        text = f.read()

        ind1 = text.find('<h3 >字段1_文本</h3><p>')
        ind2 = text.find('</p><h3 >标题</h3><p>')
        ind3 = text.find('</p><h3 >时间</h3><p>')
        ind4 = text.find('</p><h3 >字段2</h3><p>')

        if any(ind == -1 for ind in [ind1, ind2, ind3, ind4]):
            continue

        stock = text[ind1 + 27: ind2]
        title = text[ind2 + 23: ind3].replace("\n", "").replace(" ", "").replace("查看PDF原文", "")
        date = text[ind3 + 23: ind4]

        str_remove = ["摘要", "更正", "英文版", "补充", "半年度"]
        if ("年度报告" in title or "年年报" in title) and not any(name in title for name in str_remove):
            # print title

            sign_start1 = ' 经营情况讨论与分析\n'
            sign_end1 = '第五节重要事项\n'
            sign_start2 = '第四节经营情况讨论与分析\n'
            sign_end2 = ' 重要事项\n'

            not_fund = False

            if text.count(sign_start1) == 1:
                ind5 = text.find(sign_start1)

            elif text.count(sign_start2) == 1:
                ind5 = text.find(sign_start2)

            else:
                not_fund = True
                # print single_file, title
                continue

            if text.count(sign_end1) == 1:
                ind6 = text.find(sign_end1)

            elif text.count(sign_end2) == 1:
                ind6 = text.find(sign_end2)

            else:
                not_fund = True
                # print single_file, title
                continue

            if not not_fund:
                content = text[ind5: ind6]

                title = title.replace("*", "_").replace(":", "_").replace("/", "_")
                print title
                with open(unicode("C:/Users/text/Desktop/上市公司定期报告/年报文本节选2/"
                                  + str(date) + "_" + str(title) + ".txt", "utf-8"), "w") as f2:
                    f2.write(title + '\n')
                    f2.write(stock + '\n')
                    f2.write(date + '\n')
                    f2.write(content + '\n')

'''

'''
if __name__ == '__main__':

    path_in1 = unicode('D:\数据\深证A股定期公告\张紫乾', "utf-8")

    files1 = file_name(path_in1)
    # print files1

    count = 0

    for single_dir in dir_name(path_in1):
        print single_dir

        for single_file in file_name(single_dir):
            try:
                f = open(single_dir + '/' + single_file)

            except IOError:
                print single_dir + '/' + single_file
                continue

            text = f.read()

            ind1 = text.find('</p><h3 >字段1</h3><p>')
            ind2 = text.find('<h3 >标题')
            ind3 = text.find('</p><h3>时间</h3><p>')
            ind4 = text.find('</p><h3 >字段2</h3><p>')
            ind5 = text[ind3 + 16: ind3 + 50].find('日')

            print text[ind1 + 24: ind4].replace("\n", "").replace(" ", "")
            print text[ind2 + 19: ind1].replace("\n", "").replace(" ", "").replace("查看PDF原文", "")
            print text[ind3: ind5]

            if any(ind == -1 for ind in [ind1, ind2, ind3, ind4]):
                print count
                count += 1
                continue

            stock = text[ind1 + 27: ind2]
            title = text[ind2 + 19: ind1].replace("\n", "").replace(" ", "").replace("查看PDF原文", "")
            date = text[ind3 + 9: ind3 + 13 + 9]
            str_remove = ["摘要", "更正", "英文版", "补充", "半年度", "说明", "半年报"]
            # print stock
            if ("年度报告" in title or "年报" in title) and not any(name in title for name in str_remove):
                # print title
                content = text[ind4: ]

                title = title.replace("*", "_").replace(":", "_").replace("/", "_")
                stock = stock.replace("*", "+").replace(":", "+").replace("/", "+")

                with open(unicode("C:/Users/text/Desktop/上市公司定期报告/年报文本节选4/"
                                  + str(date) + "_" + str(title) + ".txt", "utf-8"), "w") as f2:
                    f2.write(title + '\n')
                    f2.write(stock + '\n')
                    f2.write(date + '\n')
                    f2.write(content + '\n')

'''

