import json, requests, datetime

from bs4 import BeautifulSoup

import pandas as pd

def int_check(x):
    try:
        int(str(x))
        return True
    except ValueError:
        return False

dataset = "train"

df = pd.read_csv("../data/{0}.csv".format(dataset))
days = df["Дата_День"].apply(lambda x: datetime.datetime.strftime(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), "%d.%m.%Y")).unique()

print("Unique days loaded.")

all_views = {}

for dayIndex in range(len(days)):
    url_ = "https://neftegaz.ru/archive/news/?date={0}".format(str(days[dayIndex]))
    request_ = requests.get(url_)
    q_ = BeautifulSoup(request_.text)
    views_html_ = q_.findAll("div", {"class": "views2"})
    views_sum_ = sum(list(map(lambda x: list(map(lambda x: int(x), filter(lambda x_: int_check(x_), str(x).split("\t"))))[0], views_html_)))

    all_views[days[dayIndex]] = views_sum_

    print("Progress: {0}".format(dayIndex / len(days) * 100))

with open("../data/news_data_{0}.txt".format(dataset), "w+") as file:
    file.write(json.dumps(all_views))
