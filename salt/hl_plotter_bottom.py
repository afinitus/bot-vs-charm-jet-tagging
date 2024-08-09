"""Produce roc curves from tagger output and labels."""

from __future__ import annotations

from puma.hlplots import Results, Tagger
from puma.utils import get_dummy_2_taggers, logger
import numpy as np
import pandas as pd
import h5py

# The line below generates dummy data which is similar to a NN output
#file = get_dummy_2_taggers(add_pt=True, return_file=True)

# define jet selections
#cuts = [("pt", ">=", 50000)]

# define the taggers

fname = "/pscratch/sd/n/nishank/shapiro_pi2/salt/logs/GN2_20240701-T041601/ckpts/epoch=037-val_loss=0.60887__test_combined_output.h5"

gn2 = Tagger(
    name="GN2",
    label="W decay data ($f_{c}=0.2$)",
    fxs={"fc": 0.2, "fb": 0.2},
    colour="#AA3377",
    reference=True
)

# create the Results object
# for c-tagging use signal="cjets"
# for Xbb/cc-tagging use signal="hbb"/"hcc"
results = Results(signal="bjets", sample="anti-kt")

# load taggers from the file object
logger.info("Loading taggers.")
results.load_taggers_from_file(
    [gn2],
    fname
    #cuts=cuts
    #num_jets=9000000
)

results.atlas_second_tag = (
    "$\\sqrt{s}=13$ TeV, anti-kt jets \n W + jets, $10$ GeV $< p_{T} <250$ GeV"
    #"$\\sqrt{s}=13$ TeV, anti-kt jets \n W + jets, $ p_{T} \geq 50$ GeV"
)

# tagger probability distributions
results.plot_probs(logy=True, bins=40)

# tagger discriminant distributions
logger.info("Plotting tagger discriminant plots.")
results.plot_discs(logy=False, wp_vlines=[60, 85])
results.plot_discs(logy=True, wp_vlines=[60, 85], suffix="log")

# ROC curves
logger.info("Plotting ROC curves.")
results.plot_rocs()

# eff/rej vs. variable plots
logger.info("Plotting efficiency/rejection vs pT curves.")
results.atlas_second_tag = "$\\sqrt{s}=13$ TeV, anti-kt jets \n W + jets, $10$ GeV $< p_{T} <250$ GeV"
#"$\\sqrt{s}=13$ TeV, anti-kt jets \nW + jets\n $p_T \geq 50 \, GeV$"

# or alternatively also pass the argument `working_point` to the plot_var_perf function.
# specifying the `disc_cut` per tagger is also possible.
results.plot_var_perf(
    working_point=0.7,
    bins=[10, 20, 30, 40, 60, 85, 110, 140, 175, 250],
    flat_per_bin=False,
)

results.atlas_second_tag = "$\\sqrt{s}=13$ TeV, anti-kt jets \nW + jets"
results.plot_var_perf(
    bins=[10, 20, 30, 40, 60, 85, 110, 140, 175, 250],
    flat_per_bin=True,
    working_point=0.7,
    h_line=0.7,
    disc_cut=None,
)
# flat rej vs. variable plots, a third tag is added relating to the fixed
#  rejection per bin
results.atlas_second_tag = "$\\sqrt{s}=13$ TeV, anti-kt jets \nW + jets"
results.plot_flat_rej_var_perf(
    fixed_rejections={"cjets": 2.2, "ujets": 1.2},
    bins=[10, 20, 30, 40, 60, 85, 110, 140, 175, 250],
)

# fraction scan plots
logger.info("Plotting fraction scans.")
results.atlas_second_tag = "$\\sqrt{s}=13$ TeV, anti-kt jets \nW + jets\n70% WP"
results.plot_fraction_scans(efficiency=0.7, rej=False)