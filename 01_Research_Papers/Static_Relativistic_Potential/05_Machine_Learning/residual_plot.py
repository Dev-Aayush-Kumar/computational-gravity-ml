import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

current_dir = Path(__file__).parent
df = pd.read_csv(
    current_dir / "../03_Dataset/gravity_dataset_ml.csv"
)

X = df[["x"]]
y = df["Phi_GR"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

residuals = (
    y_test -
    predictions
)

plt.figure(figsize=(8,6))

plt.scatter(
    predictions,
    residuals,
    s=10
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Phi_GR"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Residual Plot for Random Forest"
)

plt.tight_layout()

plt.savefig(
    current_dir /
    "../04_Graphs_and_Figures/residual_plot.png",
    dpi=300,
    bbox_inches="tight"
)
