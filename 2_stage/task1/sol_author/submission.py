import random
import numpy as np
import sklearn
import math
import pickle
from sklearn import linear_model
import sys
np.set_printoptions(threshold=sys.maxsize)

def eval(data):

    with open("model.pkl","rb") as f:
        model = pickle.load(f)



    res = model.predict_proba(np.array(data))[:,1]
    return np.array2string(np.array(res),separator=",",precision=20 ).replace("\\n", "\n").replace("\n", " ")
