#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 14:29:51 2025

@author: zachary
"""
"""
Minimal example script for the fetal OPM preprocessing pipeline.

This script shows the recommended usage of the functions in
`fetal_opm_pipeline2.py`:

1. Build a Config object with paths and block names.
2. Run `run_pipeline` to get a cleaned Raw object
3. Detect events and create epochs.
4. Clean epochs using a simple peak-to-peak percentile threshold.
5. Compute per-condition evoked responses and top-5 sensors + PC1.
6. Plot evokeds using the sensor layout.

Edit only the USER PARAMETERS section below for your own data.
"""

from pathlib import Path

from fetal_opm_pipeline2 import (
    Config,
    run_pipeline,
    detect_event_id,
    make_epochs,
    clean_epochs_ptp,
    evokeds_from_epochs,
    drop_noisy_channels_evoked,
    compute_top5_and_pc1_matstyle,
    plot_evokeds_with_layout,
)


# ---------------------------------------------------------------------
# USER PARAMETERS – ADAPT THIS SECTION
# ---------------------------------------------------------------------
ROOT = Path("/path/to/fif_root")      # e.g. Path("/data/fopm")
SUBJECT = "sub-01"

# Block labels used in your FIF filenames:
# expects files like <ROOT>/<SUBJECT>/<SUBJECT>_<BLOCK>.fif
BLOCKS = [
    "ses-01_task-visual_run-01",
    "ses-01_task-visual_run-02",
]

STIM_CH = "misc16"                   # name of the trigger channel
TMIN, TMAX = -0.2, 1.0               # epoch window (s)
BASELINE = (None, 0.0)               # baseline interval for evokeds
FIT_WINDOW = (0.05, 0.6)             # window for selecting top-5 sensors
MSC_MODE = "fwer"                    # 'fwer' or 'fdr' for motion ICA
PTP_PERCENTILE = 80.0                # peak-to-peak percentile for epoch rejection
# ---------------------------------------------------------------------


def main() -> None:
    # 1) Build configuration
    cfg = Config(
        root=ROOT,
        sub=SUBJECT,
        blocks=BLOCKS,
    )

    # 2) Run the simplified precision-only pipeline
    print(">> Running fetal OPM preprocessing...")
    raw_clean = run_pipeline(cfg, msc_mode=MSC_MODE)
    print(raw_clean)

    # 3) Detect events and epoch the cleaned data
    print(f">> Detecting events on stim channel: {STIM_CH}")
    event_id = detect_event_id(raw_clean, stim_ch=STIM_CH)
    print("Event codes:", event_id)

    print(">> Epoching...")
    epochs = make_epochs(raw_clean, event_id=event_id, stim_ch=STIM_CH, tmin=TMIN, tmax=TMAX)
    print(epochs)

    # 4) Simple peak-to-peak epoch rejection
    print(f">> Cleaning epochs with peak-to-peak percentile {PTP_PERCENTILE}...")
    epochs_clean = clean_epochs_ptp(epochs, percentile=PTP_PERCENTILE)
    print(epochs_clean)

    # 5) Compute evokeds and drop noisy channels
    print(">> Computing evokeds and dropping noisy channels...")
    evokeds = evokeds_from_epochs(epochs_clean)
    evokeds, bads = drop_noisy_channels_evoked(
        evokeds,
        pre_stim_max=0.0,
        mad_sigma=2.5,
        min_keep=5,
        verbose=True,
    )
    print("Marked bad channels:", bads)

    # 6) Compute top-5 sensors and PC1 per condition (MATLAB-style)
    print(">> Computing top-5 sensors and PC1 per condition...")
    top5_pc1 = {}
    for cond in evokeds.keys():
        print(f"   - condition: {cond}")
        idx5, pc1, evk = compute_top5_and_pc1_matstyle(
            epochs_clean,
            condition=cond,
            baseline=BASELINE,
            fit_window=FIT_WINDOW,
            detrend_baseline=False,   # set True to remove baseline slope
        )
        top5_pc1[cond] = {"top5_idx": idx5, "pc1": pc1}
        # use the baseline-corrected evoked for plotting
        evokeds[cond] = evk

    # 7) Plot evokeds with the sensor layout
    print(">> Plotting evokeds with layout...")
    plot_evokeds_with_layout(evokeds, baseline=BASELINE, fit_window=FIT_WINDOW)

    print("Done.")


if __name__ == "__main__":
    main()
