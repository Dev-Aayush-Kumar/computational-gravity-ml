import pandas as pd
from pathlib import Path

current_dir = Path(__file__).parent

data = pd.read_csv(
    current_dir / "../03_Dataset/gravity_dataset.csv"
)

max_diff_row = data.loc[
    data["Difference"].abs().idxmax()
]

print("\n===== MAXIMUM DIFFERENCE =====")

print(
    f"x = {max_diff_row['x']:.6f}"
)

print(
    f"Phi_Newton = {max_diff_row['Phi_Newton']:.6f}"
)

print(
    f"Phi_GR = {max_diff_row['Phi_GR']:.6f}"
)

print(
    f"Difference = {max_diff_row['Difference']:.6f}"
)

print(
    f"Relative Error (%) = "
    f"{max_diff_row['Relative_Error_Percent']:.6f}"
)