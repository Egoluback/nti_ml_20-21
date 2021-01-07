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

df_train = pd.read_csv("data/train_result.csv")


weekdays = ["weekday_{0}".format(i) for i in range(1, 8)]

onehotObject_weekday = OneHotEncoder(handle_unknown = 'ignore')
onehotObject_weekday.fit(np.array(df_train['weekday']).reshape(-1, 1))
df_train = pd.merge(df_train, pd.DataFrame(onehotObject_weekday.transform(np.array(df_train['weekday']).reshape(-1, 1)).toarray().astype(int), columns = weekdays), left_index = True, right_index = True)
df_train.drop("weekday", inplace = True, axis = 1)


hours_onehot = ["hour_{0}".format(i) for i in range(24)]

df_train["hour"] = df_train["Дата"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)


onehotObject_time = OneHotEncoder(handle_unknown = 'ignore')
onehotObject_time.fit(np.array(df_train["hour"]).reshape(-1, 1))
df_train = pd.merge(df_train, pd.DataFrame(onehotObject_time.transform(np.array(df_train["hour"]).reshape(-1, 1)).toarray().astype(int), columns = hours_onehot), left_index = True, right_index = True)


hour_mean_tvr18 = df_train.groupby("hour")[tvr_18].mean()
hour_mean_tvr55 = df_train.groupby("hour")[tvr_55].mean()

hour_mean_share18 = df_train.groupby("hour")[share_18].mean()
hour_mean_share55 = df_train.groupby("hour")[share_55].mean()

df_train["hour_mean_tvr18"] = df_train["hour"].apply(lambda x: hour_mean_tvr18[x])
df_train["hour_mean_tvr55"] = df_train["hour"].apply(lambda x: hour_mean_tvr55[x])

df_train["hour_mean_share18"] = df_train["hour"].apply(lambda x: hour_mean_share18[x])
df_train["hour_mean_share55"] = df_train["hour"].apply(lambda x: hour_mean_share55[x])


print(df_train)
print("PREPROCESSING COMPLETED")
print("TRAINING")

# best 500
random_state_ = 500

channels = df_train["Канал"].unique()

targets = [tvr_18, tvr_55, share_18, share_55]

for target in targets:
    print("------{0}------".format(target))
    for channelIndex in range(len(channels)):
        print("---{0}---".format(channels[channelIndex]))
        # best 800
        model = CatBoostRegressor(random_state = random_state_, silent = True, n_estimators = 800, max_depth = 8)

        with open("data/models/model_features_{0}_{1}.txt".format(target, channelIndex), "r") as file:
            features = json.loads(file.read())

        print("TRAINING")
        XTrain, XTest, YTrain, YTest = train_test_split(df_train[df_train["Канал"] == channels[channelIndex]][features].fillna(0), df_train[df_train["Канал"] == channels[channelIndex]][target], test_size = 0.33, random_state = random_state_)
        model.fit(XTrain, YTrain)

        print("SAVING")
        model.save_model("data/models/model_{0}_{1}".format(target, channelIndex))

        print("{0} saved".format(channelIndex))
