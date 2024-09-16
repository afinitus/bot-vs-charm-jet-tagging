import h5py
import numpy as np
from puma import Roc
from puma.utils import logger
import matplotlib.pyplot as plt

def get_discriminant(probs, signal="cjets"):
    """Calculate the discriminant score."""
    if signal == "cjets":
        return probs[:, 1] / (1 - probs[:, 2])
    elif signal == "bjets":
        return probs[:, 2] / (1 - probs[:, 1])
    else:
        raise ValueError("Signal must be either 'cjets' or 'bjets'")

def find_nearest_index(array, value):
    """Find the index of the nearest value in an array."""
    return (np.abs(array - value)).argmin()

# File path
file_path = "/pscratch/sd/n/nishank/shapiro_pi2/GN2_fold0_20240226-T063759/ckpts/epoch=034-val_loss=0.57278__test_ttbar.h5"

# Load data
with h5py.File(file_path, 'r') as f:
    jets = f['jets'][:]
    
    # Extract probabilities
    probs = np.column_stack([
        jets['GN2_fold0_pb'],
        jets['GN2_fold0_pc'],
        jets['GN2_fold0_pu']
    ])
    
    # Extract PDG IDs
    hadron_pdg = jets['HadronConeExclTruthLabelPdgId']
    child_pdg = jets['HadronConeExclTruthLabelChildPdgId']

# Calculate discriminant scores for charm jets
discs = get_discriminant(probs, signal="cjets")

# Create truth labels (1 for charm jets, 0 for others)
truth = (jets['flavour_label'] == 4).astype(int)

# Create a Roc object
roc = Roc(discs, truth)

# Calculate signal efficiency and background rejection
sig_eff = roc.sig_eff
bkg_rej = roc.bkg_rej

# Find the index closest to 50% signal efficiency
wp_50_index = find_nearest_index(sig_eff, 0.5)

# Get the discriminant value at this working point
wp_50 = discs[wp_50_index]

# Filter jets at 50% working point
mask_50wp = discs >= wp_50

# Filter PDG IDs in the range 400-500
mask_pdg_range = (np.abs(hadron_pdg) >= 400) & (np.abs(hadron_pdg) <= 500)

# Combine masks
final_mask = mask_50wp & mask_pdg_range

# Get filtered PDG IDs
filtered_hadron_pdg = hadron_pdg[final_mask]
filtered_child_pdg = child_pdg[final_mask]

# Plot histograms
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.hist(filtered_hadron_pdg, bins=50, range=(400, 550))
plt.title('HadronConeExclTruthLabelPdgId Distribution')
plt.xlabel('PDG ID')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
plt.hist(filtered_child_pdg, bins=50, range=(400, 550))
plt.title('HadronConeExclTruthLabelChildPdgId Distribution')
plt.xlabel('PDG ID')
plt.ylabel('Count')

plt.subplot(1, 3, 3)
plt.hist(np.abs(filtered_hadron_pdg), bins=50, range=(400, 550))
plt.title('Absolute HadronConeExclTruthLabelPdgId Distribution')
plt.xlabel('|PDG ID|')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('pdg_id_distributions.png')
plt.close()

logger.info(f"Total jets at 50% charm efficiency: {np.sum(mask_50wp)}")
logger.info(f"Jets with |PDG ID| in range 400-500 at 50% charm efficiency: {np.sum(final_mask)}")

# Additional plot for absolute values of PDG IDs with focus on 440
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(np.abs(filtered_hadron_pdg), bins=51, range=(400, 550))
plt.title('Absolute HadronConeExclTruthLabelPdgId Distribution')
plt.xlabel('|PDG ID|')
plt.ylabel('Count')

# Highlight the 440 bin
bin_440 = np.searchsorted(bins, 440) - 1
plt.bar(bins[bin_440], counts[bin_440], width=bins[1]-bins[0], color='red', alpha=0.7)

plt.text(440, counts[bin_440], f'{int(counts[bin_440])}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('pdg_id_abs_distribution_focus_440.png')
plt.close()

# Log the count for |PDG ID| = 440
count_440 = counts[bin_440]
logger.info(f"Count for |PDG ID| = 440: {int(count_440)}")

# Define the PDG IDs we're interested in
charmed_pdg_ids = [411, 421, 413, 423, 431, 433]
bottom_pdg_ids = [511, 521, 513, 523, 531]
all_pdg_ids = charmed_pdg_ids + bottom_pdg_ids

# Filter PDG IDs for our particles of interest
mask_pdg_interest = np.isin(np.abs(hadron_pdg), all_pdg_ids)

# Combine masks
final_mask = mask_50wp & mask_pdg_interest

# Get filtered PDG IDs
filtered_hadron_pdg = hadron_pdg[final_mask]

# Plot histogram
plt.figure(figsize=(12, 6))
counts, bins, patches = plt.hist(np.abs(filtered_hadron_pdg), bins=len(all_pdg_ids), 
                                 range=(min(all_pdg_ids)-5, max(all_pdg_ids)+5), 
                                 align='mid', rwidth=0.8)
plt.title('Distribution of Charmed and Bottom Hadron PDG IDs')
plt.xlabel('|PDG ID|')
plt.ylabel('Count')

# Highlight charmed particles
for pdg in charmed_pdg_ids:
    bin_index = np.searchsorted(bins, pdg) - 1
    patches[bin_index].set_facecolor('red')

# Highlight bottom particles
for pdg in bottom_pdg_ids:
    bin_index = np.searchsorted(bins, pdg) - 1
    patches[bin_index].set_facecolor('blue')

# Add labels for each bar
for i, count in enumerate(counts):
    plt.text(bins[i], count, f'{int(count)}', ha='center', va='bottom')

plt.xticks(all_pdg_ids, rotation=45)
plt.tight_layout()
plt.savefig('charmed_bottom_pdg_distribution.png')
plt.close()

# Log the counts for each PDG ID
for pdg in all_pdg_ids:
    count = np.sum(np.abs(filtered_hadron_pdg) == pdg)
    logger.info(f"Count for |PDG ID| = {pdg}: {count}")

logger.info(f"Total jets at 50% charm efficiency: {np.sum(mask_50wp)}")
logger.info(f"Jets with charmed/bottom PDG IDs at 50% charm efficiency: {np.sum(final_mask)}")