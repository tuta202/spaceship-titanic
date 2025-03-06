import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data/train.csv")
x_test = pd.read_csv("data/test.csv", index_col="PassengerId")

target = "Transported"

data.set_index("PassengerId", inplace=True)

# categorical feature
cols = ["HomePlanet","CryoSleep","Cabin","Destination","VIP"]
n_rows = 2
n_cols = 3
fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.5, n_rows*3.5))
for r in range(0, n_rows):
  for c in range(0, n_cols):
    i = r*n_rows + c
    if i < len(cols):
      ax_i = ax[r, c]
      sns.countplot(data=data, x=cols[i], hue=target, ax=ax_i)
      ax_i.set_title(cols[i])
plt.show()



