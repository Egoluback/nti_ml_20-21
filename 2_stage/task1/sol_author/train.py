import math
import random
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model
import pickle
import lightgbm
import ast


with open('data/x_train.txt',"r") as f:
    x_train_1  =  np.array(ast.literal_eval(f.read()))
with open('data/y_train.txt',"r") as f:
    y_train_1  =  np.array(ast.literal_eval(f.read()))

lr = linear_model.LogisticRegression()

lr.fit(x_train_1,y_train_1)

with open("model.pkl","wb") as f:
    pickle.dump(lr,f)
