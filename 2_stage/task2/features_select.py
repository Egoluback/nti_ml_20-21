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

channels = df_train["Канал"].unique()
targets = [tvr_18, tvr_55, share_18, share_55]

for target in targets:
    for channel in channels:
        mean = df_train.groupby(["Канал", "Месяц"])[target].mean()[channel]
        for month in df_train["Месяц"].unique():
            df_train.loc[(df_train["Канал"] == channel) & (df_train["Месяц"] == month), "rolling_{0}".format(target)] = mean[month]

# print(df_train["mean_{0}".format(tvr_18)])
# print(df_train.groupby("Месяц")["rolling_{0}".format(tvr_18)].value_counts())
# input()
print("PREPROCESSING COMPLETED")
print("TRAINING")

rolling_names = ["rolling_{0}".format(target) for target in targets]
mean_names = ["mean_{0}".format(target) for target in targets]

columns_search = np.array([weekdays, hours_onehot, rolling_names,
        'hour_mean_tvr18', 'hour_mean_tvr55', 'hour_mean_share18', 'hour_mean_share55',
        'news_views', 'news_views_shift', 'temp_Moscow',
        'temp_Peter', 'temp_Novosib', 'temp_Ekb', 'searches_wo',
        'searches_tnt', 'searches_1st', 'searches_r1', 'searches_sts',
        'searches_news', 'searches_rentv', 'lockdown_index', 'working_day',
        'oil_price', 'dollar_rate'])

cat_features_ = ['working_day'] + weekdays + hours_onehot

random_state_ = 500

for target in targets:
    print("-------{0}-------".format(target))

    models_target = []

    for channel in channels:
        errors = []

        print("---{0}---".format(channel))

        for column in columns_search:

            print("---")
            if (type(column) == str): column = [column]

            XTrain, XTest, YTrain, YTest = train_test_split(df_train[df_train["Канал"] == channel][column].fillna(0), df_train[df_train["Канал"] == channel][target], test_size = 0.33, random_state = 42)

            # if (len(list(set(column) & set(cat_features_))) > 0):
            #     print("cat feature")
            #     model = CatBoostRegressor(silent = True, cat_features = column, random_state = random_state_)
            # else:
            #     model = CatBoostRegressor(silent = True, random_state = random_state_)
            model = CatBoostRegressor(silent = True)

            model.fit(XTrain, YTrain)

            error = mean_squared_log_error(YTest, np.absolute(model.predict(XTest)))
            print("Train MSLE: {0}; Column: {1}".format(mean_squared_log_error(YTrain, np.absolute(model.predict(XTrain))), column))
            print("Test MSLE: {0}; Column: {1}".format(error, column))

            errors.append(error)

        print("Columns brut force completed")

        indexes_sorted = list(map(lambda x: errors.index(x), list(sorted(set(errors)))))
        print("Sorted indexes: {0}".format(indexes_sorted))

        out_indexes = []
        current_features = []
        all_features = columns_search.copy()

        errors = []
        models = []

        models_errors = []

        print("Features search started")

        for i in range(len(indexes_sorted)):
            print("---")
            if (type(columns_search[indexes_sorted[i]]) == str):
                current_features.append(columns_search[indexes_sorted[i]])
            else:
                current_features += columns_search[indexes_sorted[i]]

            # print("Current features: {0}".format(current_features))

            XTrain, XTest, YTrain, YTest = train_test_split(df_train[df_train["Канал"] == channel][current_features].fillna(0), df_train[df_train["Канал"] == channel][target], test_size = 0.33, random_state = 42)

            # if (len(list(set(current_features) & set(cat_features_))) > 0):
            #     model = CatBoostRegressor(silent = True, cat_features = list(set(current_features) & set(cat_features_)), random_state = random_state_)
            # else:
            #     model = CatBoostRegressor(silent = True, random_state = random_state_)
            model = CatBoostRegressor(silent = True)

            model.fit(XTrain, YTrain)

            error = mean_squared_log_error(YTest, np.absolute(model.predict(XTest)))

            print("Train MSLE: {0}; Column: {1}".format(mean_squared_log_error(YTrain, np.absolute(model.predict(XTrain))), current_features[-1]))
            print("Test MSLE: {0}; Column: {1}".format(error, current_features[-1]))

            errors.append(error)

            # if (i >= 1 and errors[i] > errors[i - 1]):
            if (i >= 1 and errors[i] > models_errors[-1]):
                # print("---making someone out---")
                print("{0} was ejected.".format(current_features[-1]))
                current_features.pop(len(current_features) - 1)
            else:
                models.append(model)
                models_errors.append(error)

        print("Last model MSLE error: {0}".format(mean_squared_log_error(YTest, np.absolute(models[-1].predict(XTest)))))
        models_target.append([models[-1], current_features])

    print("SAVING")
    for modelIndex in range(len(models_target)):
        models_target[modelIndex][0].save_model("data/models/model_{0}_{1}".format(target, modelIndex))
        with open("data/models/model_features_{0}_{1}.txt".format(target, modelIndex), "w+") as file:
            file.write(json.dumps(models_target[modelIndex][1]))
        print("{0} saved".format(modelIndex))
