import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

current_dir = Path(__file__).parent

data = pd.read_csv(
    current_dir /
    "../03_Dataset/gravity_dataset_ml.csv"
)

X = data[["x"]]

y = data["Phi_GR"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = make_pipeline(
    PolynomialFeatures(degree=5),
    LinearRegression()
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

print("\n===== POLYNOMIAL REGRESSION =====\n")

print(
    "MAE =",
    mean_absolute_error(
        y_test,
        predictions
    )
)

print(
    "RMSE =",
    mean_squared_error(
        y_test,
        predictions
    ) ** 0.5
)

print(
    "R² =",
    r2_score(
        y_test,
        predictions
    )
)