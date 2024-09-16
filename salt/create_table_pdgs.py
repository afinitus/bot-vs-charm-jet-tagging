import h5py
import numpy as np
from puma import Roc
import pandas as pd

def get_discriminant(probs, signal="cjets"):
    """calculate the discriminant score"""
    if signal == "cjets":
        return probs[:, 1] / (0.3 * probs[:, 0] + 0.7 * probs[:, 2])
    elif signal == "bjets":
        return probs[:, 0] / (0.2 * probs[:, 1] + 0.8 * probs[:, 2])
    else:
        raise ValueError("no sig")

def find_nearest_index(array, value):
    """find the index of the nearest value in an array"""
    return (np.abs(array - value)).argmin()

def create_particle_table(jets, probs, particle_type):
    """create a table for charmed or bottom particles"""
    if particle_type == "charm":
        weakly_decaying_ids = [411, 421, 431, 4122, 4132, 4232, 4212, 4332]
        baryon_ids = [4122, 4132, 4232, 4212, 4332]
    elif particle_type == "bottom":
        weakly_decaying_ids = [511, 521, 531, 541, 5122, 5132, 5232, 5112, 5212, 5222, 5332]
        baryon_ids = [5122, 5132, 5232, 5112, 5212, 5222, 5332]
    else:
        raise ValueError("light")

    c_discs = get_discriminant(probs, signal="cjets")
    b_discs = get_discriminant(probs, signal="bjets")
    
    # mask so ur only selecting on b or c
    charm_mask = jets['flavour_label'] == 1
    bottom_mask = jets['flavour_label'] == 0

    # 50% c eff
    c_disc_sorted = np.sort(c_discs[charm_mask])
    c_cut = c_disc_sorted[len(c_disc_sorted)//2]
    
    # 70% b eff
    b_disc_sorted = np.sort(b_discs[bottom_mask])
    b_cut = b_disc_sorted[int(len(b_disc_sorted)*0.1)]
    
    # Masking
    mask_c_efficiency = c_discs > c_cut
    mask_b_efficiency = b_discs < b_cut

    table = pd.DataFrame(index=[
        'All Events',
        'Fraction of All Events',
        f'Events (c-eff > 50%)',
        f'Fraction (c-eff > 50%)',
        f'Events (c-eff > 50%, b-eff < 70%)',
        f'Fraction (c-eff > 50%, b-eff < 70%)'
    ])

    all_events = {}
    events_c_eff = {}
    events_c_eff_b_eff = {}

    for pdg_id in weakly_decaying_ids + ['Baryons']:
        if pdg_id == 'Baryons':
            mask_pdg = np.isin(np.abs(jets['HadronConeExclTruthLabelPdgId']), baryon_ids)
        else:
            mask_pdg = np.abs(jets['HadronConeExclTruthLabelPdgId']) == pdg_id
        
        # all events, so no cut
        all_events[pdg_id] = np.sum(mask_pdg)
        # c-eff > 50%
        events_c_eff[pdg_id] = np.sum(mask_pdg & mask_c_efficiency)
        # c-eff > 50% and b-eff < 70%
        events_c_eff_b_eff[pdg_id] = np.sum(mask_pdg & mask_c_efficiency & mask_b_efficiency)

    total_all_events = sum(all_events.values())
    total_events_c_eff = sum(events_c_eff.values())
    total_events_c_eff_b_eff = sum(events_c_eff_b_eff.values())

    for pdg_id in weakly_decaying_ids + ['Baryons']:
        column_name = pdg_id if pdg_id != 'Baryons' else f'{particle_type.capitalize()} Baryons'
        table[column_name] = [
            all_events[pdg_id],
            all_events[pdg_id] / total_all_events if total_all_events > 0 else 0,
            events_c_eff[pdg_id],
            events_c_eff[pdg_id] / total_events_c_eff if total_events_c_eff > 0 else 0,
            events_c_eff_b_eff[pdg_id],
            events_c_eff_b_eff[pdg_id] / total_events_c_eff_b_eff if total_events_c_eff_b_eff > 0 else 0
        ]

    return table

file_path = "/pscratch/sd/n/nishank/shapiro_pi2/salt/logs/4_20240724-T022822/ckpts/epoch=036-val_loss=0.56414__test_ttbar.h5"

with h5py.File(file_path, 'r') as f:
    jets = f['jets'][:]
    
    probs = np.column_stack([
        jets['4_pb'],
        jets['4_pc'],
        jets['4_pu']
    ])

charm_table = create_particle_table(jets, probs, "charm")
bottom_table = create_particle_table(jets, probs, "bottom")

print("Charmed Particles Table:")
print(charm_table.to_string())
print("\nBottom Particles Table:")
print(bottom_table.to_string())

charm_table.to_csv('charm_particles_table.csv')
bottom_table.to_csv('bottom_particles_table.csv')