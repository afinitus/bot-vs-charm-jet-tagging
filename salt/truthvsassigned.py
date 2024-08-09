import h5py
import numpy as np
import matplotlib.pyplot as plt
from puma import Roc
from puma.utils import logger

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

def process_file(file_path):
    with h5py.File(file_path, 'r') as f:
        jets = f['jets'][:]
        
        probs = np.column_stack([
            jets['4_pb'],
            jets['4_pc'],
            jets['4_pu']
        ])
        
        pdg_ids = jets['HadronConeExclTruthLabelPdgId']
        flavor_labels = jets['flavour_label']
        
    return probs, pdg_ids, flavor_labels

def create_plot(probs, pdg_ids, flavor_labels, file_name):
    # Calculate discriminant scores for charm jets
    discs = get_discriminant(probs, signal="cjets")

    # Create truth labels (1 for charm jets, 0 for others)
    truth = (flavor_labels == 4).astype(int)

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

    # Apply the mask to pdg_ids and flavor_labels
    pdg_ids_filtered = pdg_ids[mask_50wp]
    flavor_labels_filtered = flavor_labels[mask_50wp]

    # Define flavor categories
    flavor_categories = {5: 'bottom', 4: 'charm', 0: 'light'}

    # Create separate PDG ID arrays for each flavor
    bottom_pdg_ids = pdg_ids_filtered[flavor_labels_filtered == 0]
    charm_pdg_ids = pdg_ids_filtered[flavor_labels_filtered == 1]
    light_pdg_ids = pdg_ids_filtered[flavor_labels_filtered == 2]

    # Create histogram bins
    bins = np.linspace(-1000, 1000, 201)  # Adjust range as needed

    # Create the plot
    plt.figure(figsize=(12, 8))

    plt.hist(bottom_pdg_ids, bins=bins, alpha=0.5, label='Bottom', density=True)
    plt.hist(charm_pdg_ids, bins=bins, alpha=0.5, label='Charm', density=True)
    plt.hist(light_pdg_ids, bins=bins, alpha=0.5, label='Light', density=True)

    plt.xlabel('PDG ID')
    plt.ylabel('Density')
    plt.title(f'PDG ID Distribution by Assigned Flavor Label at 50% Charm Efficiency\n{file_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Use log scale for y-axis as distributions might vary greatly
    plt.yscale('log')

    # Zoom in on the central region
    plt.xlim(-600, 600)

    plt.tight_layout()
    plt.savefig(f'pdg_id_distribution_by_flavor_{file_name}.png')
    plt.close()

    # Print some statistics
    logger.info(f"Statistics for {file_name}:")
    for flavor, name in flavor_categories.items():
        count = np.sum(flavor_labels_filtered == flavor)
        logger.info(f"Total {name} jets: {count}")
        if count > 0:
            pdgs = pdg_ids_filtered[flavor_labels_filtered == flavor]
            logger.info(f"  PDG ID range: {np.min(pdgs)} to {np.max(pdgs)}")
            logger.info(f"  Most common PDG IDs: {np.bincount(np.abs(pdgs)).argsort()[-5:][::-1]}")
        logger.info("")

# File paths
fnames_wjets = ["/pscratch/sd/n/nishank/shapiro_pi2/GN2_fold0_20240226-T063759/ckpts/epoch=034-val_loss=0.57278__test_combined_output.h5", "/pscratch/sd/n/nishank/shapiro_pi2/salt/logs/GN2_20240701-T041601/ckpts/epoch=037-val_loss=0.60887__test_combined_output.h5"]
fnames_ttbar = [
    "/pscratch/sd/n/nishank/shapiro_pi2/GN2_fold0_20240226-T063759/ckpts/epoch=034-val_loss=0.57278__test_ttbar.h5",
    "/pscratch/sd/n/nishank/shapiro_pi2/salt/logs/4_20240724-T022822/ckpts/epoch=036-val_loss=0.56414__test_ttbar.h5"
]
fnames_wjets_resamp = ["/pscratch/sd/n/nishank/shapiro_pi2/GN2_fold0_20240226-T063759/ckpts/epoch=034-val_loss=0.57278__test_test.h5", "/pscratch/sd/n/nishank/shapiro_pi2/salt/logs/4_20240722-T172922/ckpts/epoch=037-val_loss=0.60936__test_test.h5"]

# Process each file and create plots
for file_path in [fnames_ttbar[1], fnames_wjets_resamp[1]]:
    file_name = file_path.split('/')[-1].replace('.h5', '')
    probs, pdg_ids, flavor_labels = process_file(file_path)
    create_plot(probs, pdg_ids, flavor_labels, file_name)