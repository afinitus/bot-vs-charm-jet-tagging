import numpy as np
import pandas as pd
import h5py
from ftag.hdf5 import H5Reader

# Load data
fname = "/pscratch/sd/n/nishank/shapiro_pi2/GN2_fold0_20240226-T063759/ckpts/epoch=034-val_loss=0.57278__test_large.h5"
reader = H5Reader(fname, batch_size=1_000)
data = reader.load({"jets": ["pt", "eta", "flavour_label", "GN2_fold0_pu", "GN2_fold0_pc", "GN2_fold0_pb"]}, num_jets=10_000)
df = pd.DataFrame(data['jets'])

# Define the b-tagging discriminant function
def disc_fct(arr: np.ndarray, f_c: float = 0.018) -> np.ndarray:
    return np.log(arr[2] / (f_c * arr[1] + (1 - f_c) * arr[0]))

df["disc_gn2"] = np.apply_along_axis(
    disc_fct, 1, df[["GN2_fold0_pu", "GN2_fold0_pc", "GN2_fold0_pb"]].values
)

# Save the data in a HDF5 file with correct structure
with h5py.File("tagger_data.h5", "w") as f:
    jets_group = f.create_group("jets")
    for column in df.columns:
        jets_group.create_dataset(column, data=df[column].values)
