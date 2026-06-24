import pandas as pd
from pathlib import Path

current_dir = Path(__file__).parent

data = pd.read_csv(
    current_dir / "../03_Dataset/gravity_dataset.csv"
)

thresholds = [10, 5, 1]

print("\n===== ERROR ACCURACY THRESHOLDS =====\n")

for threshold in thresholds:

    valid_rows = data[
        data["Relative_Error_Percent"] <= threshold
    ]

    if valid_rows.empty:
        print(
            f"Error never falls below {threshold}% "
            f"in the sampled range."
        )
        print()
        continue

    first_row = valid_rows.iloc[0]

    print(
        f"Error falls below {threshold}% at:"
    )

    print(
        f"x = {first_row['x']:.6f}"
    )

    print(
        f"Relative Error = "
        f"{first_row['Relative_Error_Percent']:.6f}%"
    )

    print()