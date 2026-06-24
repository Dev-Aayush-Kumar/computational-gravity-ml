import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Current folder
current_dir = Path(__file__).parent


# Load dataset
data = pd.read_csv(
    current_dir / "../03_Dataset/gravity_dataset.csv"
)


# Extract values

x = data["x"]

difference = data["Difference"]


# Analytical maximum

x_max = 4/3

delta_max = 1/8


# Plot

plt.figure(figsize=(8,5))


plt.plot(
    x,
    difference,
    label="ΔΦ = Φ_GR - Φ_N"
)


# Mark analytical point

plt.scatter(
    [x_max],
    [delta_max]
)


plt.annotate(
    "Maximum deviation\nx = 4/3\nΔΦ = 1/8",
    xy=(x_max, delta_max),
    xytext=(10,0.11),
    arrowprops=dict(
        arrowstyle="->"
    )
)


plt.xlabel(
    "Dimensionless distance x = r/r_s"
)

plt.ylabel(
    "Potential Difference ΔΦ"
)

plt.title(
    "Maximum Deviation Between Newtonian and Relativistic Potentials"
)

plt.grid(True)

plt.legend()

plt.savefig(
    current_dir /
    "../04_Graphs_and_Figures/annotated_difference_plot.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()