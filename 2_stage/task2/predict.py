from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np

import datetime, json

tvr_18 = "Все 18+_TVR"
tvr_55 = "Все 55+_TVR"

share_18 = "Все 18+_Share"
share_55 = "Все 55+_Share"

targets = [tvr_18, tvr_55, share_18, share_55]

df_test = pd.read_csv("data/test_preresult.csv")

df_train = pd.read_csv("data/train_result.csv")

channels = df_test["Канал"].unique()

# for target in targets:
#     for channel in channels:
#         mean = df_test.groupby(["Канал", "Месяц"])[target].mean()[channel]
#         for month in df_train["Месяц"].unique():
#             df_test.loc[(df_test["Канал"] == channel) & (df_test["Месяц"] == month), "rolling_{0}".format(target)] = mean[month]


monthes = ["month_{0}".format(i) for i in range(12)]
df_test['month'] = df_test["Дата"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").month)

onehotObject_month = OneHotEncoder(handle_unknown = 'ignore')
onehotObject_month.fit(np.array(df_test['month']).reshape(-1, 1))
df_test = pd.merge(df_test, pd.DataFrame(onehotObject_month.transform(np.array(df_test['month']).reshape(-1, 1)).toarray().astype(int), columns = monthes), left_index = True, right_index = True)

monthes = ["month_{0}".format(i) for i in range(12)]

df_test["date"] = df_test["Дата"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

df_weeks_mean = pd.read_csv("data/weeks_mean.csv")
df_weeks_mean["date"] = df_weeks_mean["date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

df_test = pd.merge(df_test, df_weeks_mean, on = 'date', how = 'left')

weeks_mean = ["mean_index_target_{0}".format(target) for target in targets]

for target in targets:
    for channelIndex in range(len(channels)):
        model = CatBoostRegressor()
        model.load_model("data/models/model_{0}_{1}".format(target, channelIndex))

        with open("data/models/backup/model_features_{0}_{1}.txt".format(target, channelIndex), "r") as file:
            features = json.loads(file.read()) + monthes + weeks_mean

        X = df_test[(df_test["Канал"] == channels[channelIndex])][features]

        Y = np.absolute(model.predict(X))

        df_test.loc[df_test["Канал"] == channels[channelIndex], target] = Y

df_test.to_csv("data/result/test.csv")
