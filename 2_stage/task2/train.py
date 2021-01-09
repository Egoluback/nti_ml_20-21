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

from model import Model

tvr_18 = "Все 18+_TVR"
tvr_55 = "Все 55+_TVR"

share_18 = "Все 18+_Share"
share_55 = "Все 55+_Share"

df_train = pd.read_csv("data/train_result.csv")
df_weeks_mean = pd.read_csv("data/weeks_mean.csv")
# df_weeks_oneday = pd.read_csv("data/oneday_mean.csv")

df_weeks_mean["date"] = df_weeks_mean["date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
# df_weeks_oneday["date"] = df_weeks_oneday["date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

weekdays = ["weekday_{0}".format(i) for i in range(1, 8)]

onehotObject_weekday = OneHotEncoder(handle_unknown = 'ignore')
onehotObject_weekday.fit(np.array(df_train['weekday']).reshape(-1, 1))
df_train = pd.merge(df_train, pd.DataFrame(onehotObject_weekday.transform(np.array(df_train['weekday']).reshape(-1, 1)).toarray().astype(int), columns = weekdays), left_index = True, right_index = True)

hours_onehot = ["hour_{0}".format(i) for i in range(24)]

df_train["date"] = df_train["Дата"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
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

monthes = ["month_{0}".format(i) for i in range(12)]
df_train['month'] = df_train["Дата"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").month)

onehotObject_month = OneHotEncoder(handle_unknown = 'ignore')
onehotObject_month.fit(np.array(df_train['month']).reshape(-1, 1))
df_train = pd.merge(df_train, pd.DataFrame(onehotObject_month.transform(np.array(df_train['month']).reshape(-1, 1)).toarray().astype(int), columns = monthes), left_index = True, right_index = True)

df_train.drop("weekday", inplace = True, axis = 1)

channels = df_train["Канал"].unique()
targets = [tvr_18, tvr_55, share_18, share_55]


# df_train.index = df_train["date"]
# df_weeks_mean.index = df_weeks_mean["date"]
df_train = pd.merge(df_train, df_weeks_mean, on = 'date', how = 'left')

# df_train = pd.merge(df_train, df_weeks_oneday, left_on = 'date', right_on = 'date', how = 'left', suffixes = ('', '_oneday'))

# print(pd.merge(df_train, df_weeks_mean, left_on = 'date', right_on = 'date'))

print(df_train)
input()
print("PREPROCESSING COMPLETED")
print("TRAINING")

# best 500
random_state_ = 500

weeks_mean = ["mean_index_target_{0}".format(target) for target in targets]
# oneday_mean = ["mean_index_target_{0}_oneday".format(target) for target in targets]

for target in targets:
    print("------{0}------".format(target))
    for channelIndex in range(len(channels)):
        print("---{0}---".format(channels[channelIndex]))

        with open("data/models/backup/model_features_{0}_{1}.txt".format(target, channelIndex), "r") as file:
            features = json.loads(file.read())

        print("TRAINING")
        XTrain, XTest, YTrain, YTest = train_test_split(df_train[df_train["Канал"] == channels[channelIndex]][features].fillna(0), df_train[df_train["Канал"] == channels[channelIndex]][target], test_size = 0.33, random_state = random_state_)

        # best 835 8 0.045
        # model_cbr = CatBoostRegressor(random_state = random_state_, silent = True, n_estimators = 800, max_depth = 8)
        model_cbr = CatBoostRegressor(random_state = random_state_, silent = True, n_estimators = 835, max_depth = 8, learning_rate = 0.045)
        model_cbr.fit(XTrain, YTrain)

        print("Test MSLE: {0}".format(mean_squared_log_error(YTest, np.absolute(model_cbr.predict(XTest)))))

        features_ = features + monthes + weeks_mean

        XTrain, XTest, YTrain, YTest = train_test_split(df_train[df_train["Канал"] == channels[channelIndex]][features_].fillna(0), df_train[df_train["Канал"] == channels[channelIndex]][target], test_size = 0.33, random_state = random_state_)
        # best 835 8 0.045
        # model_cbr = CatBoostRegressor(random_state = random_state_, silent = True, n_estimators = 800, max_depth = 8)
        model_cbr = CatBoostRegressor(random_state = random_state_, silent = True, n_estimators = 1100, max_depth = 8, learning_rate = 0.045)
        model_cbr.fit(XTrain, YTrain)

        # print("Train MSLE: {0}".format(mean_squared_log_error(YTrain, np.absolute(model_cbr.predict(XTrain)))))
        print("Test MSLE: {0}".format(mean_squared_log_error(YTest, np.absolute(model_cbr.predict(XTest)))))

        print("SAVING")
        model_cbr.save_model("data/models/model_{0}_{1}".format(target, channelIndex))

        print("{0} saved".format(channelIndex))
