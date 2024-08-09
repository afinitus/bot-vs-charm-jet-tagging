from __future__ import annotations

from puma.hlplots import Results, Tagger
from puma.utils import logger

# Define file paths
fnames_wjets = ["/pscratch/sd/n/nishank/shapiro_pi2/GN2_fold0_20240226-T063759/ckpts/epoch=034-val_loss=0.57278__test_combined_output.h5", "/pscratch/sd/n/nishank/shapiro_pi2/salt/logs/GN2_20240701-T041601/ckpts/epoch=037-val_loss=0.60887__test_combined_output.h5"]


fnames_ttbar = [
    "/pscratch/sd/n/nishank/shapiro_pi2/GN2_fold0_20240226-T063759/ckpts/epoch=034-val_loss=0.57278__test_ttbar.h5",
    "/pscratch/sd/n/nishank/shapiro_pi2/salt/logs/4_20240724-T022822/ckpts/epoch=036-val_loss=0.56414__test_ttbar.h5"
]

fnames_wjets_resamp = ["/pscratch/sd/n/nishank/shapiro_pi2/GN2_fold0_20240226-T063759/ckpts/epoch=034-val_loss=0.57278__test_test.h5", "/pscratch/sd/n/nishank/shapiro_pi2/salt/logs/4_20240722-T172922/ckpts/epoch=037-val_loss=0.60936__test_test.h5"]

# Define the taggers
gn2_fold0 = Tagger(
    name="GN2",
    label="Pre-Trained W+jet GN2",
    fxs={"fc": 0.2, "fb": 0.2},
    colour="#AA3377",
    reference=True
)

gn2_fold1 = Tagger(
    name="4",
    label="Pre-Trained W+jet Resamp GN2",
    fxs={"fc": 0.2, "fb": 0.2},
    colour="#44AA99",
    reference=False
)

# Create the Results object
results = Results(signal="cjets", sample="anti-kt")

# Load taggers from the files with the pt >= 50 GeV cut
logger.info("Loading taggers.")
results.load_taggers_from_file(
    [gn2_fold0],
    fnames_wjets[1]
    #cuts=[("pt", "<=", 75000)]
)
results.load_taggers_from_file(
    [gn2_fold1],
    fnames_wjets_resamp[1]
    #cuts=[("pt", "<=", 75000)]
)

results.atlas_second_tag = (
    "$\\sqrt{s}=13$ TeV, anti-kt jets" #\n W + jets resampled"#\n W + jets resampled"#,  $p_{T} \leq 75$ GeV"
)

# ROC curves
logger.info("Plotting ROC curves.")
results.plot_rocs()

# Tagger probability distributions
results.plot_probs(logy=True, bins=40)

# Tagger discriminant distributions
logger.info("Plotting tagger discriminant plots.")
results.plot_discs(logy=False, wp_vlines=[60, 85])
results.plot_discs(logy=True, wp_vlines=[60, 85], suffix="log")

# Efficiency/rejection vs. variable plots
logger.info("Plotting efficiency/rejection vs pT curves.")
results.atlas_second_tag = "$\\sqrt{s}=13$ TeV, anti-kt jets" #\n W + jets resampled"#\n W + jets"#, $p_{T} \leq 75$ GeV"
results.plot_var_perf(
    working_point=0.7,
    bins=[10, 20, 30, 40, 60, 85, 110, 140, 175, 250],
    flat_per_bin=False,
)
results.atlas_second_tag = "$\\sqrt{s}=13$ TeV, anti-kt jets" #\nW + jets"
results.plot_var_perf(
    bins=[10, 20, 30, 40, 60, 85, 110, 140, 175, 250],
    flat_per_bin=True,
    working_point=0.7,
    h_line=0.7,
    disc_cut=None,
)

# Flat rejection vs. variable plots
results.atlas_second_tag = "$\\sqrt{s}=13$ TeV, anti-kt jets"  #\n W + jets resampled"#\n W + jets"
results.plot_flat_rej_var_perf(
    fixed_rejections={"cjets": 2.2, "ujets": 1.2},
    bins=[10, 20, 30, 40, 60, 85, 110, 140, 175, 250],
)

# Fraction scan plots
logger.info("Plotting fraction scans.")
results.atlas_second_tag = "$\\sqrt{s}=13$ TeV, anti-kt jets"  #\n W + jets resampled\n70% WP"#\n W + jets"\nW + jets\n70% WP"
results.plot_fraction_scans(efficiency=0.7, rej=False)



