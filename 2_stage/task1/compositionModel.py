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

    def load(self, path_arr):
        x_train = []
        y_train = []

        with open(path_arr[0], "r") as file:
            x_train = np.array(json.loads(''.join(list(map(lambda x: x.strip(), file.readlines())))))

        with open(path_arr[1], "r") as file:
            y_train = np.array(json.loads(''.join(list(map(lambda x: x.strip(), file.readlines())))))

        self.XTrain, self.XComposition, self.YTrain, self.YComposition = train_test_split(x_train, y_train, test_size = 0.5, random_state = 42)

    def save(self):
        return {"models_vector": self.models_vector, "composition_model": self.composition_model}

    def train_models(self):
        # TODO выделить фичи: лучшие колонки датасета для каждой модели

        self.model_RFC = RandomForestClassifier(n_estimators = 100, max_depth = 9, random_state = 42)
        self.model_RFC.fit(self.XTrain, self.YTrain)

        print("RFR trained")

        self.model_GBC = GradientBoostingClassifier(n_estimators = 200, max_depth = 10, random_state = 42)
        self.model_GBC.fit(self.XTrain, self.YTrain)

        print("GBC trained")

        self.model_XGB = XGBClassifier(n_estimators = 100, max_depth = 10, random_state = 42)
        self.model_XGB.fit(self.XTrain, self.YTrain)

        print("XGB trained")

        self.model_CBC = CatBoostClassifier(silent = True)
        self.model_CBC.fit(self.XTrain, self.YTrain)

        print("CBC trained")

        self.model_LGBM = LGBMClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 100)
        self.model_LGBM.fit(self.XTrain, self.YTrain)

        print("LGBM trained")

        self.models_vector = np.array([self.model_RFC, self.model_GBC, self.model_XGB, self.model_CBC, self.model_LGBM], dtype = object)

    def composition_predict(self, X):
        return np.array([model.predict(X) for model in self.models_vector])

    def composition_predict_proba(self, X):
        return np.array([model.predict_proba(X).T[1] for model in self.models_vector])

    def composition_models(self):
        self.composition_arr = self.composition_predict_proba(self.XComposition).T

        self.composition_model = LogisticRegression().fit(self.composition_arr, self.YComposition)

    def fit(self):
        self.train_models()
        print("training completed")
        self.composition_models()

    def predict(self, X):
        enter_composition = self.composition_predict(X).T

        return self.composition_model.predict(enter_composition)

    def predict_proba(self, X):
        enter_composition = self.composition_predict_proba(X)

        return self.composition_model.predict_proba(enter_composition).T[1]

if (__name__ == "__main__"):
    model_comp = Model(random_state = 42)

    model_comp.load(["data/x_train.txt", "data/y_train.txt"])
    print("---datasets loaded---")

    model_comp.fit()
    print("---models fitted---")

    data = model_comp.save()

    with open("model.pkl", "wb") as file:
        pickle.dump(data, file)
