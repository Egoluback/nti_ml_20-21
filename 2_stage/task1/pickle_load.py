from model import Model

import pickle
# from joblib import dump

model = Model(random_state = 42)

model.load(["data/x_train.txt", "data/y_train.txt"])
model.train_models()
model.fit()

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# dump(model, "model.joblib")
