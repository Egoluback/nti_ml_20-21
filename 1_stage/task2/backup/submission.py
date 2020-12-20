import numpy as np

def eval(data):
    data_np = np.array(data)

    k2 = data_np[0][0]
    k1 = data_np[1][1] / np.sin(k2)

    data_2000 = np.arange(1000, 2000)

    pr_cos = (k2 * np.cos(k1 * data_2000))
    pr_sin = (k1 * np.sin(k2 * data_2000))

    np_prediction = np.array([pr_cos, pr_sin]).T

    return list(map(lambda x: tuple(x), list(np_prediction)))
