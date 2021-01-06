import datetime

import numpy as np
import pandas as pd

dataset = "train"

df = pd.read_csv("data/{0}.csv".format(dataset))

SEARCHES_NAMES = ["wo", "tnt", "1st", "r1", "sts", "news", "rentv"]

result = pd.DataFrame(columns = [])

for name in SEARCHES_NAMES:
    df_searches_init = pd.read_csv("data/searches/searches_{0}.csv".format(name))

    df_searches = pd.DataFrame(np.array([list(df_searches_init["Категория: Все категории"].index)[1 :], list(df_searches_init["Категория: Все категории"])[1 :]]).T, columns = ["week", "searches"])

    weeks_time = df_searches["week"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

    weeks_year = pd.Series(weeks_time.apply(lambda x: x.year), name = "year")
    weeks_month = pd.Series(weeks_time.apply(lambda x: x.month), name = "month")
    weeks_day = pd.Series(weeks_time.apply(lambda x: x.day), name = "day")
    weeks_ymd = pd.merge(pd.merge(weeks_year, weeks_month, right_index = True, left_index = True), weeks_day, right_index = True, left_index = True)

    df_searches = pd.merge(df_searches, weeks_ymd, right_index = True, left_index = True)

    all_datas_searches = pd.DataFrame(pd.date_range(start = "2018-01-01", end = "2021-01-01"), columns = ["date"])

    all_datas_searches["searches"] = np.empty(len(all_datas_searches["date"]))
    all_datas_searches["searches"][:] = np.nan


    current_searches = df_searches["searches"][0]

    for dataIndex in range(len(list(all_datas_searches["date"]))):
        date_str = datetime.datetime.strftime(all_datas_searches["date"].iloc[dataIndex], "%Y-%m-%d")

        if (len(df_searches[df_searches["week"] == date_str]) > 0):
            current_searches = df_searches[df_searches["week"] == date_str]["searches"]

        all_datas_searches["searches"][dataIndex] = current_searches


    days_time = df["Дата"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    days_year = pd.Series(days_time.apply(lambda x: x.year), name = "year")
    days_month = pd.Series(days_time.apply(lambda x: x.month), name = "month")
    days_day = pd.Series(days_time.apply(lambda x: x.day), name = "day")

    days_fullymd = pd.merge(pd.merge(days_year, days_month, right_index = True, left_index = True), days_day, right_index = True, left_index = True)

    days_fullymd["searches"] = np.empty(len(days_fullymd["year"]))
    days_fullymd["searches"][:] = np.nan

    all_datas_searches["year"] = pd.Series(all_datas_searches["date"].apply(lambda x: x.year), name = "year")
    all_datas_searches["month"] = pd.Series(all_datas_searches["date"].apply(lambda x: x.month), name = "month")
    all_datas_searches["day"] = pd.Series(all_datas_searches["date"].apply(lambda x: x.day), name = "day")


    for year in [2018, 2019, 2020]:
        for monthIndex in range(12):
            this_searches = all_datas_searches[(all_datas_searches["year"] == year) & (all_datas_searches["month"] == monthIndex + 1)]
            days_this = days_fullymd[(days_fullymd["year"] == year) & (days_fullymd["month"] == monthIndex + 1)]
            if (len(days_this) != 0):
                for day in set(days_this["day"]):
                    searches_ = this_searches.loc[this_searches["day"] == day, "searches"]
                    if (len(searches_) == 0): continue
                    days_this.loc[days_this["day"] == day, "searches"] = list(searches_)[0]
            days_fullymd[(days_fullymd["year"] == year) & (days_fullymd["month"] == monthIndex + 1)] = days_this

    result["searches_{0}".format(name)] = days_fullymd["searches"]

    # print(all_datas_searches[(all_datas_searches["year"] == 2019) & (all_datas_searches["month"] == 4) & (all_datas_searches["day"] == 4)])
    # print(result[result["searches_wo"] == 65.0])
    # input()


result.to_csv("data/searches_full_{0}.csv".format(dataset))
