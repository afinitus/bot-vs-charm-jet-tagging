import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ftag.hdf5 import H5Reader
from puma import Roc, RocPlot
from puma.metrics import calc_rej

# File path
fname = "/pscratch/sd/n/nishank/shapiro_pi2/salt/logs/GN2_20240630-T040612/ckpts/epoch=028-val_loss=0.60921__test_large.h5"

# Load the data
reader = H5Reader(fname, batch_size=1_000)

# Use the correct variable names
variable_names = ["pt", "eta", "flavour_label", "GN2_pu", "GN2_pc", "GN2_pb"]
df = pd.DataFrame(reader.load({"jets": variable_names}, num_jets=10_000)['jets'])

# Print unique values in the flavour_label column
print("Unique flavour labels in the dataset:", df["flavour_label"].unique())

def disc_fct(arr: np.ndarray, f_c: float = 0.2) -> np.ndarray:
    return np.log(arr[2] / (f_c * arr[1] + (1 - f_c) * arr[0]))

discs_gn2 = np.apply_along_axis(
    disc_fct, 1, df[["GN2_pu", "GN2_pc", "GN2_pb"]].values
)

# Plot the discriminants
plt.figure(figsize=(10, 6))
plt.hist(discs_gn2[df["flavour_label"] == 2], bins=50, alpha=0.5, label='Light jets')
plt.hist(discs_gn2[df["flavour_label"] == 1], bins=50, alpha=0.5, label='c jets')
plt.hist(discs_gn2[df["flavour_label"] == 0], bins=50, alpha=0.5, label='b jets')
plt.xlabel('Discriminant')
plt.ylabel('Frequency')
plt.legend()
plt.title('Discriminant Distributions')
plt.savefig("plots/discriminant_distributions.png")
plt.show()

# Plot the pt distributions
plt.figure(figsize=(10, 6))
plt.hist(df[df["flavour_label"] == 2]["pt"], bins=50, alpha=0.5, label='Light jets')
plt.hist(df[df["flavour_label"] == 1]["pt"], bins=50, alpha=0.5, label='c jets')
plt.hist(df[df["flavour_label"] == 0]["pt"], bins=50, alpha=0.5, label='b jets')
plt.xlabel('pt')
plt.ylabel('Frequency')
plt.legend()
plt.title('pt Distributions')
plt.savefig("plots/pt_distributions.png")
plt.show()

sig_eff = np.linspace(0.49, 1, 20)
is_light = df["flavour_label"] == 2
is_charm = df["flavour_label"] == 1
is_bottom = df["flavour_label"] == 0

# Debugging prints
print(f"Number of light jets: {is_light.sum()}")
print(f"Number of c jets: {is_charm.sum()}")
print(f"Number of b jets: {is_bottom.sum()}")

n_jets_light = sum(is_light)
n_jets_charm = sum(is_charm)
if n_jets_light == 0:
    raise ValueError("No light jets found in the dataset.")
if n_jets_charm == 0:
    raise ValueError("No charm jets found in the dataset.")
if sum(is_bottom) == 0:
    raise ValueError("No bottom jets found in the dataset.")

gn2_light_rej = calc_rej(discs_gn2[is_bottom], discs_gn2[is_light], sig_eff)
gn2_charm_rej = calc_rej(discs_gn2[is_bottom], discs_gn2[is_charm], sig_eff)

plot_roc = RocPlot(
    n_ratio_panels=2,
    ylabel="Background rejection",
    xlabel="$b$-jet efficiency",
    atlas_second_tag="$\\sqrt{s}=13$ TeV \nLarge Sample",
    figsize=(6.5, 6),
    y_scale=1.4,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        gn2_light_rej,
        #n_test=n_jets_light,
        rej_class="ujets",
        signal_class="bjets",
        label="GN2",
    ),
    reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        gn2_charm_rej,
        #n_test=n_jets_charm,
        rej_class="cjets",
        signal_class="bjets",
        label="GN2",
    ),
    reference=True,
)

plot_roc.set_ratio_class(1, "ujets")
plot_roc.set_ratio_class(2, "cjets")
plot_roc.draw()
plot_roc.savefig("plots/bottom_roc.png", transparent=False)
