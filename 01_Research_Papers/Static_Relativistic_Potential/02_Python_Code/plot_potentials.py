import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

current_dir = Path(__file__).parent

data = pd.read_csv(
    current_dir / "../03_Dataset/gravity_dataset.csv"
)

plt.figure(figsize=(8,6))

plt.plot(
    data["x"],
    data["Phi_Newton"],
    label="Newtonian Potential"
)

plt.plot(
    data["x"],
    data["Phi_GR"],
    label="Relativistic Potential"
)

plt.xscale("log")

plt.xlabel(r"$x=r/r_s$")
plt.ylabel("Dimensionless Potential")

plt.title(
    "Comparison of Newtonian and Relativistic Potentials"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    current_dir /
    "../04_Graphs_and_Figures/potential_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure generated successfully.")