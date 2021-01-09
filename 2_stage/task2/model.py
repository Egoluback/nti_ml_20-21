from catboost import CatBoostRegressor
# from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import train_test_split

import numpy as np

class Model:
    def __init__(self, n_estimators, max_depth, random_state_):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state_ = random_state_

        self.models_vector = [CatBoostRegressor(silent = True, n_estimators = self.n_estimators, max_depth = self.max_depth, random_state = self.random_state_)]
        self.composition_model = LinearRegression()

    def train_models(self, X, y):
        for model in self.models_vector:
            model.fit(X, y)
            print("-one model trained-")

    def composition_predict(self, X):
        return np.array([model.predict(X) for model in self.models_vector]).T

    def composition_models(self, X, y):
        self.composition_arr = self.composition_predict(X)

        self.composition_model.fit(self.composition_arr, y)

    def fit(self, X, y):
        XTrain, XComposition, YTrain, YComposition = train_test_split(X, y, test_size = 0.11, random_state = self.random_state_)

        self.train_models(XTrain, YTrain)
        self.composition_models(XComposition, YComposition)

    def predict(self, X):
        composition = self.composition_predict(X)

        return self.composition_model.predict(composition)
