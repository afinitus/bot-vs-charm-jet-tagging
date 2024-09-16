import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ftag.hdf5 import H5Reader
from puma import Roc, RocPlot
from puma.metrics import calc_rej

fname = "/pscratch/sd/n/nishank/shapiro_pi2/GN2_fold0_20240226-T063759/ckpts/epoch=034-val_loss=0.57278__test_large.h5"
reader = H5Reader(fname, batch_size=1_000)
df = pd.DataFrame(reader.load({"jets": ["pt", "eta", "flavour_label", "GN2_fold0_pu", "GN2_fold0_pc", "GN2_fold0_pb"]}, num_jets=10_000)['jets'])

print("Unique flavour labels in the dataset:", df["flavour_label"].unique())

def disc_fct_c(arr: np.ndarray, f_b: float = 0.2) -> np.ndarray:
    return np.log(arr[1] / (f_b * arr[2] + (1 - f_b) * arr[0]))

discs_gn2_c = np.apply_along_axis(
    disc_fct_c, 1, df[["GN2_fold0_pu", "GN2_fold0_pc", "GN2_fold0_pb"]].values
)

# Plot the c-tagging discriminants
plt.figure(figsize=(10, 6))
plt.hist(discs_gn2_c[df["flavour_label"] == 2], bins=50, alpha=0.5, label='Light jets')
plt.hist(discs_gn2_c[df["flavour_label"] == 1], bins=50, alpha=0.5, label='c jets')
plt.hist(discs_gn2_c[df["flavour_label"] == 0], bins=50, alpha=0.5, label='b jets')
plt.xlabel('c-Tagging Discriminant')
plt.ylabel('Frequency')
plt.legend()
plt.title('c-Tagging Discriminant Distributions')
plt.savefig("plots/c_tagging_discriminant_distributions.png")
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
n_jets_bottom = sum(is_bottom)
if n_jets_light == 0:
    raise ValueError("No light jets found in the dataset.")
if n_jets_charm == 0:
    raise ValueError("No charm jets found in the dataset.")
if n_jets_bottom == 0:
    raise ValueError("No bottom jets found in the dataset.")

gn2_light_rej = calc_rej(discs_gn2_c[is_charm], discs_gn2_c[is_light], sig_eff)
gn2_bottom_rej = calc_rej(discs_gn2_c[is_charm], discs_gn2_c[is_bottom], sig_eff)


print("gn2_light_rej:", gn2_light_rej)
print("gn2_bottom_rej:", gn2_bottom_rej)

plot_roc = RocPlot(
    n_ratio_panels=2,
    ylabel="Background rejection",
    xlabel="$c$-jet efficiency",
    atlas_second_tag="$\\sqrt{s}=13$ TeV \nLarge Sample",
    figsize=(6.5, 6),
    y_scale=1.4,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        gn2_light_rej,
        n_test=n_jets_light,
        rej_class="ujets",
        signal_class="cjets",
        label="Light jets",
    ),
    reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        gn2_bottom_rej,
        n_test=n_jets_bottom,
        rej_class="bjets",
        signal_class="cjets",
        label="Bottom jets",
    ),
    reference=True,
)

plot_roc.set_ratio_class(1, "ujets")
plot_roc.set_ratio_class(2, "bjets")
plot_roc.draw()
plot_roc.savefig("plots/roc_charm_jet_efficiency.png", transparent=False)
