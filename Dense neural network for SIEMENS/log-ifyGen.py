import pandas as pd
import numpy as np

df_train = pd.read_csv("first_round_testing_data.csv")
df_train_B = df_train.drop(df_train.columns[[0,1,2,3,4,5,6,7,8,9,10,20]], axis=1, )

df_train_B["Attribute4"] = np.log10(df_train_B["Attribute4"])
df_train_B["Attribute5"] = np.log10(df_train_B["Attribute5"])
df_train_B["Attribute6"] = np.log10(df_train_B["Attribute6"])
df_train_B["Attribute7"] = np.log10(df_train_B["Attribute7"])
df_train_B["Attribute8"] = np.log10(df_train_B["Attribute8"])
df_train_B["Attribute9"] = np.log10(df_train_B["Attribute9"])
df_train_B["Attribute10"] = np.log10(df_train_B["Attribute10"])

def find(row):
    if row < 0.47:
        return 1
    elif row < 1.75:
        return 2
    else:
        return 3
df_train_B["Attribute4"] = df_train_B["Attribute4"].apply(find)

def find(row):
    if row < -0.784:
        return 1
    elif row < 2.01:
        return 2
    else:
        return 3
df_train_B["Attribute5"] = df_train_B["Attribute5"].apply(find)

def find(row):
    if row <-2.195:
        return 1
    elif row < 2:
        return 2
    else:
        return 3
df_train_B["Attribute6"] = df_train_B["Attribute6"].apply(find)

def find(row):
    if row < 1.2:
        return 1
    elif row < 2.15:
        return 2
    else:
        return 3
df_train_B["Attribute7"] = df_train_B["Attribute7"].apply(find)

def find(row):
    if row < -1.43:
        return 1
    elif row < 0.2:
        return 2
    else:
        return 3
df_train_B["Attribute8"] = df_train_B["Attribute8"].apply(find)


def find(row):
    if row < 1.31:
        return 1
    elif row < 2.39:
        return 2
    elif row < 3.45:
        return 3
    else:
        return 4
df_train_B["Attribute9"] = df_train_B["Attribute9"].apply(find)

def find(row):
    if row < -1.40:
        return 1
    elif row < 2.28:
        return 2
    elif row < 4.3:
        return 3
    else:
        return 4
df_train_B["Attribute10"] = df_train_B["Attribute10"].apply(find)

print(df_train_B.shape)
df_train_B.to_csv("test_B_transfered.csv",index = False)
