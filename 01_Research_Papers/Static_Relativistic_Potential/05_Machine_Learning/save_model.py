import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

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

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

joblib.dump(
    model,
    current_dir /
    "saved_models/random_forest.pkl"
)

print("Model saved.")