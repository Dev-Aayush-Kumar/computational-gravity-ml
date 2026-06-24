import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor


current_dir = Path(__file__).parent

df = pd.read_csv(
    current_dir / "../03_Dataset/gravity_dataset_ml.csv"
)


X = df[["x"]]
y = df["Phi_GR"]


model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


model.fit(
    X,
    y
)


importance = model.feature_importances_


plt.figure(figsize=(6,4))


plt.bar(
    ["x"],
    importance
)


plt.ylabel(
    "Importance"
)


plt.xlabel(
    "Feature"
)


plt.title(
    "Random Forest Feature Importance"
)


plt.tight_layout()


plt.savefig(
    current_dir /
    "../04_Graphs_and_Figures/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nFeature Importance:",
    importance
)
print("Saved Figure")