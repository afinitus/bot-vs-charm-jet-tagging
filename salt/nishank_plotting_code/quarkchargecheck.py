import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to iterate through directories and process files
def process_files(directory, max_files=15):
    quark_charges = []
    pdg_ids = []
    file_count = 0
    
    # Iterate through all files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                try:
                    with h5py.File(file_path, 'r') as f:
                        if 'jets' in f:
                            if 'quarkCharge' in f['jets'].dtype.names:
                                quark_charges.extend(f['jets']['quarkCharge'][:])
                            if 'HadronConeExclTruthLabelPdgId' in f['jets'].dtype.names:
                                pdg_ids.extend(f['jets']['HadronConeExclTruthLabelPdgId'][:])
                    file_count += 1
                    print(f"Processed file {file_count}: {file_path}")
                    if file_count >= max_files:
                        return quark_charges, pdg_ids
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    
    return quark_charges, pdg_ids

# Define the directory containing the files
directory = "/global/cfs/cdirs/atlas/jmw464/nishank_pi2/tdd_samples_mod/user.jmwagner.508979.e8382_s3681_r13144_p5981.tdd.EMPFlow.25_2_17.24-07-16-T205230_output.h5"

# Process files and get quark charges and PDG IDs (limit to 15 files)
quark_charges, pdg_ids = process_files(directory, max_files=15)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot quark charges
ax1.hist(quark_charges, bins=np.arange(-2, 3, 1), edgecolor='black')
ax1.set_xlabel('Quark Charge')
ax1.set_ylabel('Frequency')
ax1.set_title('Histogram of Quark Charges')
ax1.grid(True)

# Plot PDG IDs
ax2.hist(pdg_ids, bins=np.arange(-20000, 20001, 1000), edgecolor='black')
ax2.set_xlabel('PDG ID')
ax2.set_ylabel('Frequency')
ax2.set_title('Histogram of PDG IDs')
ax2.set_xlim(-20000, 20000)
ax2.grid(True)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

# Save the figure
plt.savefig('quark_charge_and_pdg_id_histograms_15_files.png')

# Print some statistics
print(f"Total quark charges processed: {len(quark_charges)}")
print(f"Total PDG IDs processed: {len(pdg_ids)}")