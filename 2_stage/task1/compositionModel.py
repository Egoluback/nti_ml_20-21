import numpy as np

import json, pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, random_state):
        self.random_state = random_state
        self.hyperparams = {
                              "RFC": (100, 9),
#                             "GBC": (1000, 3),
#                             "XGB": (80, 5),
                            "CBC": (60, 5)}
#                             "LGBM": (70, 10)}
        self.models_features = {
            "RFC": [2,  5, 15, 19, 21, 22],
#             "GBC": [2,  5,  7, 15, 19, 21, 26],
#             "XGB": [0,  2,  5,  7, 11, 15, 16, 19, 21, 22, 25, 26],
            "CBC": [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]}
#             "LGBM": [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]}

    def load(self, path_arr):
        x_train = []
        y_train = []

        with open(path_arr[0], "r") as file:
            x_train = np.array(json.loads(''.join(list(map(lambda x: x.strip(), file.readlines())))))

        with open(path_arr[1], "r") as file:
            y_train = np.array(json.loads(''.join(list(map(lambda x: x.strip(), file.readlines())))))

        self.XTrain, self.XComposition, self.YTrain, self.YComposition = train_test_split(x_train, y_train, test_size = 0.5, random_state = self.random_state)

    def save(self):
        return {"models_vector": self.models_vector, "composition_model": self.composition_model, "models_features": self.models_features}

    def train_models(self):
        self.model_RFC = RandomForestClassifier(n_estimators = self.hyperparams["RFC"][0], max_depth = self.hyperparams["RFC"][1], random_state = self.random_state)
        self.model_RFC.fit(self.XTrain[:, self.models_features["RFC"]], self.YTrain)

        print("/* RFR trained */")
        print("Score: {0} ".format((roc_auc_score(self.YTrain, self.model_RFC.predict_proba(self.XTrain[:, self.models_features["RFC"]]).T[1]), roc_auc_score(self.YComposition, self.model_RFC.predict_proba(self.XComposition[:, self.models_features["RFC"]]).T[1]))))

#         self.model_GBC = GradientBoostingClassifier(n_estimators = self.hyperparams["GBC"][0], max_depth = self.hyperparams["GBC"][1], random_state = self.random_state)
#         self.model_GBC.fit(self.XTrain[:, self.models_features["GBC"]], self.YTrain)

#         print("/* GBC trained */")
#         print("Score: {0} ".format((roc_auc_score(self.YTrain, self.model_GBC.predict_proba(self.XTrain[:, self.models_features["GBC"]]).T[1]), roc_auc_score(self.YComposition, self.model_GBC.predict_proba(self.XComposition[:, self.models_features["GBC"]]).T[1]))))

#         self.model_XGB = XGBClassifier(n_estimators = self.hyperparams["XGB"][0], max_depth = self.hyperparams["XGB"][1], random_state = self.random_state)
#         self.model_XGB.fit(self.XTrain[:, self.models_features["XGB"]], self.YTrain)

#         print("/* XGB trained */")
#         print("Score: {0} ".format((roc_auc_score(self.YTrain, self.model_XGB.predict_proba(self.XTrain[:, self.models_features["XGB"]]).T[1]), roc_auc_score(self.YComposition, self.model_XGB.predict_proba(self.XComposition[:, self.models_features["XGB"]]).T[1]))))

        self.model_CBC = CatBoostClassifier(n_estimators = self.hyperparams["CBC"][0], max_depth = self.hyperparams["CBC"][1], random_state = self.random_state, silent = True)
        self.model_CBC.fit(self.XTrain[:, self.models_features["CBC"]], self.YTrain)

        print("/* CBC trained */")
        print("Score: {0} ".format((roc_auc_score(self.YTrain, self.model_CBC.predict_proba(self.XTrain[:, self.models_features["CBC"]]).T[1]), roc_auc_score(self.YComposition, self.model_CBC.predict_proba(self.XComposition[:, self.models_features["CBC"]]).T[1]))))

#         self.model_LGBM = LGBMClassifier(n_estimators = self.hyperparams["LGBM"][0], max_depth = self.hyperparams["LGBM"][1], random_state = self.random_state)
#         self.model_LGBM.fit(self.XTrain[:, self.models_features["LGBM"]], self.YTrain)

#         print("/* LGBM trained */")
#         print("Score: {0} ".format((roc_auc_score(self.YTrain, self.model_LGBM.predict_proba(self.XTrain[:, self.models_features["LGBM"]]).T[1]), roc_auc_score(self.YComposition, self.model_LGBM.predict_proba(self.XComposition[:, self.models_features["LGBM"]]).T[1]))))

#         self.models_vector = np.array([self.model_RFC, self.model_GBC, self.model_XGB, self.model_CBC, self.model_LGBM], dtype = object)
        self.models_vector = np.array([self.model_RFC, self.model_CBC], dtype = object)

    def composition_predict(self, X):
        return np.array([self.models_vector[modelIndex].predict(X[:, self.models_features[list(self.models_features.keys())[modelIndex]]]) for modelIndex in range(len(self.models_features.keys()))])

    def composition_predict_proba(self, X):
        return np.array([self.models_vector[modelIndex].predict_proba(X[:, self.models_features[list(self.models_features.keys())[modelIndex]]]).T[1] for modelIndex in range(len(self.models_features.keys()))])

    def composition_models(self):
        self.composition_arr = self.composition_predict_proba(self.XComposition).T

        self.composition_model = LogisticRegression(random_state = self.random_state).fit(self.composition_arr, self.YComposition)
        print("Coef: {0}".format(self.composition_model.coef_))
#         self.composition_model = RandomForestClassifier().fit(self.composition_arr, self.YComposition)
#         print("Feature importances: {0}".format(self.composition_model.feature_importances_))

    def fit(self):
        self.train_models()
        print("---training completed---")
        self.composition_models()

        print("Score: {0} ".format((roc_auc_score(self.YTrain, self.predict_proba(self.XTrain)), roc_auc_score(self.YComposition, self.predict_proba(self.XComposition)))))
        return (roc_auc_score(self.YTrain, self.predict_proba(self.XTrain)), roc_auc_score(self.YComposition, self.predict_proba(self.XComposition)))

    def predict(self, X):
        enter_composition = self.composition_predict(X).T

        return self.composition_model.predict(enter_composition)

    def predict_proba(self, X):
        enter_composition = self.composition_predict_proba(X).T

        return self.composition_model.predict_proba(enter_composition).T[1]

if (__name__ == "__main__"):
    model_comp = Model(random_state = 300)

    model_comp.load(["data/x_train.txt", "data/y_train.txt"])
    print("---datasets loaded---")

    model_comp.fit()
    print("---models fitted---")

    data = model_comp.save()

    with open("model.pkl", "wb") as file:
        pickle.dump(data, file)
