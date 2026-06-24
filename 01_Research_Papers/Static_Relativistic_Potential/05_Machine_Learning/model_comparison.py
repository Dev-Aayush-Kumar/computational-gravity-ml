import pandas as pd
from pathlib import Path

results = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Polynomial Regression",
            "Random Forest",
            "Neural Network"
        ],

        "MAE": [
            0.055793482392297596,
            0.056237566913354985,
            2.8172743048775894e-05,
            0.007449017485819647
        ],

        "RMSE": [
            0.07708868785145973,
            0.07907947841256932,
            8.418063687689875e-05,
            0.010378527608299803
        ],

        "R2": [
            0.26914395493361154,
            0.23090829236617672,
            0.9999991284848306,
            0.9834737194083518
        ]
    }
)
current_dir = Path(__file__).parent

output_file = (
    current_dir /
    "model_comparison.csv"
)

results.to_csv(
    output_file,
    index=False
)

print("\n===== MODEL COMPARISON =====\n")
print(results)

print(
    "\nSaved to:",
    output_file
)