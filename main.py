import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = pd.read_csv("data/student_combined.csv")

print("Rows and columns:", data.shape)

y = data["G3"]
X = data.drop(columns=["G3"])
X = pd.get_dummies(X, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train:", x_train.shape, "Test:", x_test.shape)

lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)

models = [("Linear Regression", lr), ("Decision Tree", dt), ("Random Forest", rf)]

for name, model in models:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print(name)
    print("  R2  :", round(r2_score(y_test, pred), 3))
    print("  MAE :", round(mean_absolute_error(y_test, pred), 3))
    print("  RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 3))

print()
print("Overfitting check (Random Forest)")
train_pred = rf.predict(x_train)
test_pred = rf.predict(x_test)
print("  Train R2:", round(r2_score(y_train, train_pred), 3))
print("  Test R2 :", round(r2_score(y_test, test_pred), 3))

joblib.dump(rf, "model.pkl")
joblib.dump(list(X.columns), "columns.pkl")

columns = joblib.load("columns.pkl")
model = joblib.load("model.pkl")


def predict_grade(student):
    row = pd.DataFrame([student])
    row = pd.get_dummies(row, drop_first=True)
    row = row.reindex(columns=columns, fill_value=0)

    marks = model.predict(row)[0]
    marks = round(float(marks), 1)

    if marks < 10:
        label = "At risk"
    else:
        label = "Pass"

    return marks, label


print()
print("Sample predictions")

good = data[data["G3"] >= 15].iloc[0].drop("G3").to_dict()
weak = data[data["G3"] <= 6].iloc[0].drop("G3").to_dict()

weak["studytime"] = 1
weak["absences"] = 15

for name, s in [("Good student", good), ("Weak student", weak)]:
    marks, label = predict_grade(s)
    print(f"  {name}: {marks} / 20  ->  {label}")
