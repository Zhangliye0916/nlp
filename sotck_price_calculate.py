# -*-coding:utf-8-*-

import tushare as ts
from sklearn import linear_model
import pandas as pd
import datetime
import time
import re
import numpy as np


def load_daily_data(stock_code, date_start, date_end, period="D", benchmark="sh"):

    date_range = pd.date_range(date_start, date_end).to_pydatetime()
    stock_price = ts.get_hist_data(stock_code, date_start, date_end)["close"]

    dates = []
    for i in range(len(stock_price)):
        dates.append(datetime.datetime.strptime(stock_price.index[i], "%Y-%m-%d"))

    stock_price.index = dates
    benchmark_price = ts.get_hist_data(benchmark, date_start, date_end)["close"]

    dates = []
    for i in range(len(benchmark_price)):
        dates.append(datetime.datetime.strptime(benchmark_price.index[i], "%Y-%m-%d"))

    benchmark_price.index = dates

    data = pd.DataFrame(index=date_range)
    data['stock_price'] = stock_price
    data['benchmark_price'] = benchmark_price

    data = data.fillna(method="ffill")
    date_range2 = pd.date_range(date_start, date_end, freq=period).to_pydatetime()
    data = data.loc[date_range2]

    data["stock_return"] = data['stock_price'].pct_change(1)
    data["benchmark_return"] = data['benchmark_price'].pct_change(1)
    data = data.fillna(0)

    return data


def linear_model_main(x_series, y_series):

    x_parameters = x_series.values.reshape(len(x_series), 1)
    y_parameters = y_series.values.tolist()

    model = linear_model.LinearRegression()
    model.fit(x_parameters, y_parameters, sample_weight=None)

    residuals = y_parameters - model.predict(x_parameters)

    # print (model.intercept_, model.coef_)

    return pd.Series(residuals, index=x_series.index)


def download_all_stock(end=datetime.datetime.now(), period="5", benchmark="sh"):

    end_time = end.strftime("%Y-%m-%d %H:%M:%S")
    data = ts.get_k_data(benchmark, end=end_time, ktype=period).set_index("date")[["close"]]
    data.columns = ["benchmark"]
    error_code = []

    for num, code in enumerate(All_Stock_Code):

        if num % 10 == 0:
            print ("%.2f percent data is download." % (num / len(All_Stock_Code)*100), time.strftime("%H:%M:%S"))

        try:
            stock_price = ts.get_k_data(code, end=end_time, ktype=period, retry_count=1, pause=0.0001).set_index("date")
            data[code] = stock_price["close"]

        except IOError:
            error_code.append(code)
            print("Stock price download error %s" % code)

    data.to_csv(Path1 + "stock_price_" + period + "min_" +
                end_time.replace(" ", "_").replace(":", "_").replace("-", "_") + ".csv")

    with open(Path1 + "error_code_" + period + "min_" + end_time.replace(" ", "_").replace(":", "_").replace("-", "_")
              + ".csv", "w", encoding="utf-8-sig") as f:
        f.write("\t\n".join(error_code))

    print ("These stock fail to download: %s" % " ".join(error_code))

    return data


def load_data(path):

    data = pd.read_csv(path, encoding="utf-8-sig")
    data.drop_duplicates()

    return data


def data_clear(data):

    data.columns = ["code", "title", "date", "hot", "text"]
    codes = {}
    dates = {}
    hots = {}

    for ind in data.index:
        row = data.loc[ind]

        try:
            codes[ind] = re.findall(r"\d{6}.s[h|z]", row["code"])[0]

        except IndexError:
            codes[ind] = ""
        except TypeError:
            codes[ind] = ""

        try:
            dates[ind] = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", row["date"])[0]
            hots[ind] = "|".join(re.findall(r"\d+", row["hot"]))

        except IndexError:
            dates[ind] = ""
        except TypeError:
            dates[ind] = ""

        try:
            hots[ind] = "|".join(re.findall(r"\d+", row["hot"]))

        except IndexError:
            hots[ind] = ""
        except TypeError:
            hots[ind] = ""

    data["code"] = pd.Series(codes)
    data["date"] = pd.Series(dates)
    data["hot"] = pd.Series(hots)

    data[data["code"] != ""].to_csv(Path2 + "news_clear_20190722_v1.csv", encoding="utf-8-sig")


def calculate_return(path_in, path_out=None, save=False):

    data = pd.read_csv(path_in, encoding="utf=8-sig").fillna(method="ffill").set_index("date")
    stock_return = data.pct_change(1).fillna(0)
    if save and path_out is None:
        print ("Please input the path to save.")

    elif save:
        stock_return.to_csv(path_out, encoding="utf=8-sig")

    return stock_return


def calculate_residuals(benchmark, stock_return, path_out=None, save=False):

    stock_residuals = pd.DataFrame([], index=benchmark.index)
    for stock_code in stock_return.columns:
        stock_residuals[stock_code] = linear_model_main(benchmark, stock_return[stock_code])

    if save and path_out is None:
        print ("Please input the path to save.")

    elif save:
        stock_residuals.to_csv(path_out, encoding="utf-8-sig")

    return stock_residuals


def make_label(text, digital_data, duration=240):

    times = []
    for ind in digital_data.index:
        times.append(datetime.datetime.strptime(ind, "%Y-%m-%d %H:%M"))

    digital_data.index = times
    sorted(times)
    labels = [2]*len(text.index)
    digital = [0] * len(text.index)
    for num, ind in enumerate(text.index):
        line = text.loc[ind]
        code = re.findall(r"\d{6}", line["code"])[0]

        try:
            report_time = datetime.datetime.strptime(line["date"], "%Y/%m/%d %H:%M")

        except ValueError:
            report_time = datetime.datetime.strptime(line["date"], "%Y-%m-%d %H:%M:%S")

        if report_time < datetime.datetime(2019, 7, 2, 0, 0):
            continue

        if code not in digital_data.columns:
            continue

        if line["text"] is np.nan:
            continue

        if u"异动" in line["text"]:
            # continue
            pass

        duration_time = [t for t in times if t > report_time + datetime.timedelta(minutes=5)]

        if len(duration_time) >= duration/5 + 1:

            total_residual = digital_data[code][duration_time[: int(duration/5 + 1)]].sum()
            digital[num] = total_residual

            if total_residual > 0.00:
                labels[num] = 1

            elif total_residual < -0.00:
                labels[num] = -1

            else:
                labels[num] = 1

    text["label"] = labels
    text["digital"] = digital
    text[text["label"] != 2].drop_duplicates().to_csv(Path2 + "news_label_20190722_v1.csv", encoding="utf-8-sig")


if __name__ == "__main__":

    Path1 = "C:/Users/text/Desktop/data_news/"
    Path2 = "C:/Users/text/Desktop/text_classifier/"
    # All_Stock_Code = open(Path2 + "all_stock_code.txt", encoding="utf-8-sig").read().split("\n")

    """
    
    Stock_code = "600519"
    Start_date = "2017-01-01"
    End_date = "2019-07-15"

    Data = load_daily_data(Stock_code, Start_date, End_date, period="D")

    x = Data["benchmark_return"]
    y = Data["stock_return"]

    residuals = linear_model_main(x, y)

    Data["residual"] = residuals

    Data.to_csv(Path1 + Stock_code + "_return_residuals.csv", encoding="utf-8-sig")
    """

    # download_all_stock()

    Data = load_data(Path2 + "news_20190722_v1.csv")
    data_clear(Data)

    # StockReturn = calculate_return(Path1 + "stock_price_5min_2019_07_17_15_04_24.csv",
    #                                Path1 + "stock_return_5min_2019_07_17_15_04_24.csv", save=False)
    # Benchmark = StockReturn["benchmark"]

    # calculate_residuals(Benchmark, StockReturn, Path1 + "stock_residuals_5min_2019_07_17_15_04_24.csv", save==False)

    DigitalData = pd.read_csv(Path1 + "stock_return_5min_2019_07_17_15_04_24.csv",
                              encoding="utf-8-sig").set_index("date")
    news = pd.read_csv(Path2 + "news_clear_20190722_v1.csv", encoding="utf-8-sig").set_index("Unnamed: 0")

    make_label(news, DigitalData)
