import numpy as np
import sys
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)

def eval(data):

    X_train = pd.read_csv("train.csv")
    X_train["Дата"] = pd.to_datetime(X_train["Дата"])
    X_test = pd.read_csv("test.csv")
    X_test["Дата"] = pd.to_datetime(X_test["Дата"])
    X_all = X_train.append(X_test,sort=True)
    X_all.sort_values(by=["Дата","Канал"],inplace=True)

    X_all.set_index("Дата", inplace=True)
    value_cols = ['Все 18+_TVR', 'Все 55+_TVR','Все 18+_Share', 'Все 55+_Share']
    chanels = X_test["Канал"].unique().tolist()
    for n in value_cols:
        #print(n,X_all[n].isna().sum())
        for c in chanels:
            X_all.loc[X_all["Канал"]==c,n] = X_all.loc[X_all["Канал"]==c,n].rolling(
                                                    73,min_periods=1).mean()
        #print("rolls",X_all[n].isna().sum())

    X_res_2 = X_all.loc[
        X_all["Дата_День"].isin(X_test["Дата_День"])].reset_index().sort_values(by=["Дата","Канал"]
        )
    str_x_res = X_res_2.to_csv()
    
    return str_x_res