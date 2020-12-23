import numpy as np

import json, pickle

def predict(X, models_vector, composition_model):
    enter_composition = np.array([model.predict(X) for model in models_vector]).T

    return composition_model.predict(enter_composition)

def eval(data):
    # x_test = np.array(json.loads(''.join(list(map(lambda x: x.strip(), data)))))
    x_test = np.array(data)

    with open("model.pkl", "rb") as file:
        object = pickle.load(file)

    result = predict(x_test, object['models_vector'], object['composition_model'])
    # result = object.predict(x_test)
    # result = object.predict_proba(x_test).T[1]

    return np.array2string(np.array(result), separator = ",", precision = 20).replace("\\n", "\n").replace("\n", " ")

# x_test = ""
#
# with open("data/x_train.txt") as file:
#     x_test = file.read()
#
# print(eval(x_test))
