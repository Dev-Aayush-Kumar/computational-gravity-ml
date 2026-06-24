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
    data["Difference"]
)

plt.xscale("log")

plt.xlabel(r"$x=r/r_s$")
plt.ylabel(r"$\Phi_{GR}-\Phi_N$")

plt.title(
    "Difference Between Relativistic and Newtonian Potentials"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    current_dir /
    "../04_Graphs_and_Figures/difference_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Difference plot generated.")