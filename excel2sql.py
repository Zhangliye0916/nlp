# coding:utf-8
import pymysql as mdb


def insert_price(data, id, code):

    for line in data:

        sql = "INSERT INTO finance_knowledge_graph.stock_price (entity_id, stock_code, date, open, high, " \
              "low, close, volume, amount) VALUES ('%d', '%s', '%s', '%f', '%f', '%f', '%f', '%f', '%f');" \
              % (id, code, line[0], float(line[1]), float(line[2]), float(line[3]), float(line[4]),
                 float(line[5]), float(line[6]))

        print(sql)
        try:
            cur.execute(sql)
            con.commit()

        except Exception as e:
            print(e)
            con.rollback()


def open_file(path):

    with open(path) as f:
        content = f.readlines()

        output = []
        for row in content:
            cells = row.replace("\n", "").split(",")
            output.append(cells)

        return output


if __name__ == "__main__":

    # 连接mysql的方法：connect('ip','user','password','db_name')
    con = mdb.connect('192.168.10.85', 'caiji', 'cj%2018', 'finance_knowledge_graph')
    # 所有的查询，都在连接con的一个模块cursor上面运行的
    cur = con.cursor(cursor=mdb.cursors.DictCursor)

    Content = open_file("D:/temp_data/SH#600519.csv")

    insert_price(Content, 4, "600519")

    cur.close()
