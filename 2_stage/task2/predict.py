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

channels = df_test["Канал"].unique()

for target in targets:
    for channelIndex in range(len(channels)):
        model = CatBoostRegressor()
        model.load_model("data/models/model_{0}_{1}".format(target, channelIndex))

        with open("data/models/model_features_{0}_{1}.txt".format(target, channelIndex), "r") as file:
            features = json.loads(file.read())

        X = df_test[(df_test["Канал"] == channels[channelIndex])][features]

        Y = np.absolute(model.predict(X))

        df_test.loc[df_test["Канал"] == channels[channelIndex], target] = Y

df_test.to_csv("data/result/test.csv")
