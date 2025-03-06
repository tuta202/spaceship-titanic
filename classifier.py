import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data/train.csv")
x_test = pd.read_csv("data/test.csv", index_col="PassengerId")

target = "Transported"

data.set_index("PassengerId", inplace=True)
