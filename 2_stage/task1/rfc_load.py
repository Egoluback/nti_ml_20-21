from sklearn.ensemble import RandomForestClassifier

import pickle, json
import numpy as np

x_train = []
y_train = []

with open("data/x_train.txt", "r") as file:
    x_train = np.array(json.loads(''.join(list(map(lambda x: x.strip(), file.readlines())))))

with open("data/y_train.txt", "r") as file:
    y_train = np.array(json.loads(''.join(list(map(lambda x: x.strip(), file.readlines())))))


model = RandomForestClassifier(n_estimators = 100, max_depth = 9, random_state = 42)
model.fit(x_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))
