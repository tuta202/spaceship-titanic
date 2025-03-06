import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("data/train.csv")
x_test = pd.read_csv("data/test.csv", index_col="PassengerId")

target = "Transported"

# categorical feature
# cols = ["HomePlanet","CryoSleep","Cabin","Destination","VIP"]
# n_rows = 2
# n_cols = 3
# fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.5, n_rows*3.5))
# for r in range(0, n_rows):
#   for c in range(0, n_cols):
#     i = r*n_cols + c
#     if i < len(cols):
#       ax_i = ax[r, c]
#       sns.countplot(data=data, x=cols[i], hue=target, ax=ax_i)
# plt.show()

data.set_index("PassengerId", inplace=True)
data.drop("Name", axis=1, inplace=True)
x_test.drop("Name", axis=1, inplace=True)

spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
data["TotalSpending"] = data[spending_cols].sum(axis=1)
x_test["TotalSpending"] = x_test[spending_cols].sum(axis=1)

data[['Deck', 'CabinNum', 'Side']] = data['Cabin'].str.split('/', expand=True)
x_test[['Deck', 'CabinNum', 'Side']] = x_test['Cabin'].str.split('/', expand=True)

x = data.drop(target, axis=1)
y = data[target]
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

num_features = ["Age","TotalSpending", "CabinNum"]
nom_features = ["HomePlanet","CryoSleep","Cabin","Destination","VIP", "Deck", "Side"]

num_transformer = Pipeline(steps=[
  ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
  ("scaler", StandardScaler()),
])

nom_transformer = Pipeline(steps=[
  ("imputer", SimpleImputer(strategy="most_frequent")),
  ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
  ("num_feature", num_transformer, num_features),
  ("nom_feature", nom_transformer, nom_features),
])

cls = Pipeline(steps=[
  ("preprocessor", preprocessor),
  ("model", RandomForestClassifier()),
])
params = {
  "model__n_estimators": [100, 200, 300, 500],
  "model__criterion": ["gini", "entropy", "log_loss"],
  "model__max_depth": [None, 2],
}

grid_search = GridSearchCV(estimator=cls, param_grid=params, cv=4, scoring="accuracy", verbose=2, n_jobs=-1)
grid_search.fit(x_train, y_train)
y_valid_predicted = grid_search.predict(x_valid)

print(classification_report(y_valid, y_valid_predicted))
print(grid_search.best_score_)
print(grid_search.best_params_)

y_test_predicted = grid_search.predict(x_test)

submission = pd.DataFrame({
  "PassengerId": x_test.index,
  "Transported": y_test_predicted
})
submission.to_csv("data/sample_submission.csv", index=False)