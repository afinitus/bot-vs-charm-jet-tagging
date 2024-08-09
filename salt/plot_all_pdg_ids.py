import h5py
import numpy as np
from puma import Roc
from puma.utils import logger
import matplotlib.pyplot as plt

# File path
file_path = "/pscratch/sd/n/nishank/shapiro_pi2/GN2_fold0_20240226-T063759/ckpts/epoch=034-val_loss=0.57278__test_ttbar.h5"

# Load data
with h5py.File(file_path, 'r') as f:
    jets = f['jets'][:]
    
    # Extract PDG IDs
    hadron_pdg = jets['HadronConeExclTruthLabelPdgId']
    child_pdg = jets['HadronConeExclTruthLabelChildPdgId']

# Plot histograms
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.hist(hadron_pdg, bins=50, range=(400, 550))
plt.title('HadronConeExclTruthLabelPdgId Distribution')
plt.xlabel('PDG ID')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
plt.hist(child_pdg, bins=50, range=(400, 550))
plt.title('HadronConeExclTruthLabelChildPdgId Distribution')
plt.xlabel('PDG ID')
plt.ylabel('Count')

plt.subplot(1, 3, 3)
plt.hist(np.abs(hadron_pdg), bins=50, range=(400, 550))
plt.title('Absolute HadronConeExclTruthLabelPdgId Distribution')
plt.xlabel('|PDG ID|')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('pdg_id_distributions_all.png')
plt.close()

logger.info(f"Total jets: {len(hadron_pdg)}")
logger.info(f"Jets with |PDG ID| in range 400-500: {np.sum((np.abs(hadron_pdg) >= 400) & (np.abs(hadron_pdg) <= 500))}")

# Additional plot for absolute values of PDG IDs with focus on 440
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(np.abs(hadron_pdg), bins=51, range=(400, 550))
plt.title('Absolute HadronConeExclTruthLabelPdgId Distribution')
plt.xlabel('|PDG ID|')
plt.ylabel('Count')

# Highlight the 440 bin
bin_440 = np.searchsorted(bins, 440) - 1
plt.bar(bins[bin_440], counts[bin_440], width=bins[1]-bins[0], color='red', alpha=0.7)

plt.text(440, counts[bin_440], f'{int(counts[bin_440])}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('pdg_id_abs_distribution_focus_440_all.png')
plt.close()

# Log the count for |PDG ID| = 440
count_440 = counts[bin_440]
logger.info(f"Count for |PDG ID| = 440: {int(count_440)}")

# Define the PDG IDs we're interested in
charmed_pdg_ids = [411, 421, 413, 423, 431, 433]
bottom_pdg_ids = [511, 521, 513, 523, 531]
all_pdg_ids = charmed_pdg_ids + bottom_pdg_ids

# Plot histogram
plt.figure(figsize=(12, 6))
counts, bins, patches = plt.hist(np.abs(hadron_pdg), bins=len(all_pdg_ids), 
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
plt.savefig('charmed_bottom_pdg_distribution_all.png')
plt.close()

# Log the counts for each PDG ID
for pdg in all_pdg_ids:
    count = np.sum(np.abs(hadron_pdg) == pdg)
    logger.info(f"Count for |PDG ID| = {pdg}: {count}")

logger.info(f"Total jets: {len(hadron_pdg)}")
logger.info(f"Jets with charmed/bottom PDG IDs: {np.sum(np.isin(np.abs(hadron_pdg), all_pdg_ids))}")