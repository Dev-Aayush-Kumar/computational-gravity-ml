import numpy as np
import pandas as pd
from pathlib import Path

x = np.logspace(
    np.log10(1.01),
    np.log10(100),
    10000
)

phi_newton = -1 / (2 * x)

phi_gr = 0.5 * (np.sqrt(1 - 1/x) - 1)

difference = phi_gr - phi_newton

relative_error = np.abs(difference / phi_gr) * 100

data = pd.DataFrame({
    "x": x,
    "Phi_Newton": phi_newton,
    "Phi_GR": phi_gr,
    "Difference": difference,
    "Relative_Error_Percent": relative_error
})

current_dir = Path(__file__).parent

data.to_csv(
    current_dir / "../03_Dataset/gravity_dataset_ml.csv",
    index=False
)

print("Dataset generated successfully.")
print(data.head())