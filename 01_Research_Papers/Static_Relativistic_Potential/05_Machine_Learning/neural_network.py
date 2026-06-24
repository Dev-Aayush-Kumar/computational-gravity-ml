import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

current_dir = Path(__file__).parent
df = pd.read_csv(
    current_dir / "../03_Dataset/gravity_dataset.csv"
)

X = df[["x"]]
y = df["Phi_GR"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = MLPRegressor(
    hidden_layer_sizes=(50,50),
    max_iter=5000,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\n===== NEURAL NETWORK =====\n")

print(
    "MAE =",
    mean_absolute_error(y_test, pred)
)

print(
    "RMSE =",
    np.sqrt(
        mean_squared_error(y_test, pred)
    )
)

print(
    "R² =",
    r2_score(y_test, pred)
)