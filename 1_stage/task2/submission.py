import numpy as np

def eval(data):
    data_np = np.array(data)

    k2 = data_np[0][0]
    k1 = data_np[1][1] / np.sin(k2)

    step_k1 = 2 * k1 * 0.01 / 100
    step_k2 = 2 * k2 * 0.01 / 100

    border_k1 = abs(2 * k1 * 20 / 100)
    border_k2 = abs(2 * k2 * 20 / 100)

    k_b = (k1, k2)
    k2s = []
    loss = []
    min_loss = -1

    iter_count = 0

    k2 -= border_k2
    for j in range(5000):
        loss.append(np.array([(data_np[i][1] - k1 * np.sin(k2 * i)) ** 2 for i in range(1000)]).mean())
        if min_loss < 0 or loss[-1] < min_loss:
            min_loss = loss[-1]
            best_k2 = k2
        k2s.append(k2)

        k2 += step_k2
        iter_count += 1

    k1s = []
    loss = []
    min_loss = -1
    iter_count = 0

    k2 = best_k2

    k1 -= border_k1
    for j in range(5000):
        loss.append(np.array([(data_np[i][1] - k1 * np.sin(k2 * i)) ** 2 for i in range(1000)]).mean())
        if min_loss < 0 or loss[-1] < min_loss:
            min_loss = loss[-1]
            best_k1 = k1
        k1s.append(k1)

        k1 += step_k1
        iter_count += 1

    k1 = best_k1

    data_2000 = np.arange(1000, 2000)

    pr_cos = (k2 * np.cos(k1 * data_2000))
    pr_sin = (k1 * np.sin(k2 * data_2000))

    np_prediction = np.array([pr_cos, pr_sin]).T

    return list(map(lambda x: tuple(x), list(np_prediction)))
