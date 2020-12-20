from xgboost import XGBClassifier

import pickle, json
import numpy as np

x_train = []
y_train = []

with open("data/x_train.txt", "r") as file:
    x_train = np.array(json.loads(''.join(list(map(lambda x: x.strip(), file.readlines())))))

with open("data/y_train.txt", "r") as file:
    y_train = np.array(json.loads(''.join(list(map(lambda x: x.strip(), file.readlines())))))


model = XGBClassifier(n_estimators = 500, max_depth = 11, random_state = 42)
model.fit(x_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))
