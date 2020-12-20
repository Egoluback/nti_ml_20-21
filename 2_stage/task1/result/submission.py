import pickle, json

import numpy as np

def eval(data):
    x_test = np.array(json.loads(''.join(list(map(lambda x: x.strip(), data)))))

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    result = model.predict(x_test)

    return np.array2string(np.array(result),separator=",",precision=20 ).replace("\\n", "\n").replace("\n", " ")

# x_test = ""
#
# with open("data/x_train.txt") as file:
#     x_test = file.readlines()
#
# print(eval(x_test))
