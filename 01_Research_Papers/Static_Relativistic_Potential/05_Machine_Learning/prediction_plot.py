import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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

pred = model.predict(X_test)

plt.figure(figsize=(7,7))

plt.scatter(
    y_test,
    pred,
    s=5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel(
    "Analytical Phi_GR"
)

plt.ylabel(
    "Predicted Phi_GR"
)

plt.title(
    "Random Forest Predictions vs Analytical Solution"
)

plt.tight_layout()

plt.savefig(
    current_dir /
    "../04_Graphs_and_Figures/random_forest_prediction.png",
    dpi=300
)

plt.show()

print(
    "Prediction plot generated."
)