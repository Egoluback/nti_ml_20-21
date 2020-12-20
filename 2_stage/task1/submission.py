import numpy as np

import json, pickle

# from joblib import load

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def eval(data):
    # x_test = np.array(json.loads(''.join(list(map(lambda x: x.strip(), data)))))
    x_test = np.array(data)

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    # model = load("model.joblib")

    result = model.predict_proba(x_test).T[1]

    return np.array2string(np.array(result),separator=",",precision=20 ).replace("\\n", "\n").replace("\n", " ")


# x_test = ""
#
# with open("data/x_train.txt") as file:
#     x_test = file.read()
#
# print(eval(x_test))
