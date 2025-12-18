#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 14:19:43 2025

@author: zachary
"""
"""Fetal OPM preprocessing pipeline.

This module implements a minimal preprocessing chain for
fetal optically-pumped magnetometer (OPM) recordings using MNE-Python.

Main public entry points
------------------------
- Config : dataclass with paths and parameters.
- run_pipeline(cfg, msc_mode='fwer') -> Raw
    Load blocks, precision-clean them, run motion ICA, and return a
    cleaned Raw object.

Helper functions are provided for:
- detecting events
- epoching
- simple peak-to-peak epoch rejection
- computing evoked responses
- dropping noisy channels
- computing top-5 sensors + PC1 (MATLAB-style)
- plotting evokeds with a simple topomap

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import warnings
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import psd_array_welch
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import robust_scale
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from scipy.signal import coherence


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------


@dataclass
class Config:
    """Configuration for the fetal OPM preprocessing pipeline."""

    root: pathlib.Path
    sub: str
    blocks: List[str]

    # layout / channels
    layout_name: str = "layout.lout"
    accel_chs: Sequence[str] = ("misc2", "misc3", "misc4")
    stim_ch: str = "misc16"

    # filtering / resampling
    l_freq: float = 1.0
    h_freq: float = 40.0
    lowpass_for_resampling: float = 100.0
    resample_sfreq: float = 600.0

    # LOF bad-channel detection
    lof_neighbors: int = 20
    lof_contamination: float = 0.1

    # heartbeat ICA
    ica_n_components: int = 20
    ica_random_state: int = 97

    # motion-ICA MSC options
    msc_fmin: float = 0.1
    msc_fmax: float = 5.0
    msc_nperseg_s: float = 2.0
    msc_n_permutations: int = 1000
    msc_alpha: float = 0.05

    # plotting
    show_plots: bool = True


# -------------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------------


def rename_channels_from_layout(raw: mne.io.BaseRaw, layout_file: pathlib.Path) -> mne.io.BaseRaw:
    """Rename channels based on a FieldTrip-style ``.lout`` layout.

    The layout is expected to be readable by :func:`mne.channels.read_layout`.
    Only channels present in both the layout and the Raw will be renamed.
    """
    try:
        layout = mne.channels.read_layout(str(layout_file))
    except Exception as exc:  # pragma: no cover - layout is optional
        warnings.warn(f"Could not read layout file {layout_file}: {exc}")
        return raw

    mapping = {}
    for old, new in zip(raw.ch_names, layout.names):
        mapping[old] = new
    return raw.rename_channels(mapping)


def lof_bad_channels(
    raw: mne.io.BaseRaw,
    n_neighbors: int = 20,
    contamination: float = 0.1,
    fmin: float = 1.0,
    fmax: float = 40.0,
) -> List[str]:
    """Detect bad channels with Local Outlier Factor on log-PSD."""
    picks = mne.pick_types(raw.info, meg=True, stim=False)
    if len(picks) == 0:
        return []

    data = raw.get_data(picks=picks)
    sfreq = raw.info["sfreq"]
    _, psd = psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_fft=int(2.0 * sfreq),
        average="median",
    )
    X = robust_scale(np.log(psd + np.finfo(float).eps), axis=1)

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = lof.fit_predict(X)
    bad_idx = np.where(labels == -1)[0]
    return [raw.ch_names[picks[i]] for i in bad_idx]


def drop_channels_with_lof(raw: mne.io.BaseRaw, cfg: Config) -> mne.io.BaseRaw:
    """Apply LOF-based bad channel detection and drop flagged channels."""
    bads = lof_bad_channels(
        raw,
        n_neighbors=cfg.lof_neighbors,
        contamination=cfg.lof_contamination,
        fmin=cfg.l_freq,
        fmax=cfg.h_freq,
    )
    if bads:
        print(f"LOF: dropping channels {bads}")
        raw = raw.copy().drop_channels(bads)
    return raw


def heartbeat_ica(raw: mne.io.BaseRaw, cfg: Config) -> mne.io.BaseRaw:
    """Run a simple heartbeat ICA and remove heartbeat-like components."""
    picks_meg = mne.pick_types(raw.info, meg=True, stim=False)
    if len(picks_meg) == 0:
        return raw

    ica = mne.preprocessing.ICA(
        n_components=cfg.ica_n_components,
        random_state=cfg.ica_random_state,
        max_iter="auto",
        method="fastica",
    )
    ica.fit(raw, picks=picks_meg)

    sfreq = raw.info["sfreq"]
    src = ica.get_sources(raw).get_data()
    _, psd = psd_array_welch(
        src,
        sfreq=sfreq,
        fmin=1.0,
        fmax=2.5,
        n_fft=int(2.0 * sfreq),
        average="median",
    )
    power = psd.sum(axis=-1)
    thresh = np.percentile(power, 90)
    hb_idx = np.where(power > thresh)[0]

    if len(hb_idx) == 0:
        return raw

    print(f"Heartbeat ICA: removing components {hb_idx.tolist()}")
    ica.exclude = list(hb_idx)
    return ica.apply(raw.copy())


def auto_notch_raw(raw: mne.io.BaseRaw, line_freq: float = 50.0) -> mne.io.BaseRaw:
    """Apply notch filters at line frequency and its harmonics."""
    sfreq = raw.info["sfreq"]
    freqs = np.arange(line_freq, sfreq / 2.0, line_freq)
    return raw.copy().notch_filter(freqs=freqs, picks="meg")


def precision_filter_raw(
    raw: mne.io.BaseRaw,
    radius_mm: float = 35.0,
    alpha: float = 0.0,
) -> mne.io.BaseRaw:
    """Apply a precision-based spatial filter to MEG channels."""
    picks_meg = mne.pick_types(raw.info, meg=True, stim=False)
    if len(picks_meg) == 0:
        return raw

    pos = np.array([raw.info["chs"][p]["loc"][:3] for p in picks_meg])
    radius = radius_mm / 1000.0
    tree = cKDTree(pos)

    data = raw.get_data(picks=picks_meg)
    new = np.zeros_like(data)

    for i in range(len(picks_meg)):
        dists, idx = tree.query(pos[i], k=len(picks_meg), distance_upper_bound=radius)
        mask = np.isfinite(dists)
        idx = idx[mask]
        if len(idx) < 3:
            new[i] = data[i]
            continue
        neigh = data[idx]
        lw = LedoitWolf().fit(neigh.T)
        C = lw.covariance_ + alpha * np.eye(neigh.shape[0])
        prec = np.linalg.pinv(C)
        w = prec[0] / (prec[0].sum() + np.finfo(float).eps)
        new[i] = neigh.T @ w

    raw_filt = raw.copy()
    raw_filt._data[picks_meg] = new
    return raw_filt


# -------------------------------------------------------------------------
# Block-level preprocessing
# -------------------------------------------------------------------------


def preprocess_block(fif_path: pathlib.Path, cfg: Config, tag: str = "") -> mne.io.BaseRaw:
    """Preprocess a single block and return a precision-filtered Raw."""
    print(f"Preprocessing block {tag} from {fif_path}")
    raw = mne.io.read_raw_fif(fif_path, preload=True)

    # Layout-based renaming (optional)
    layout_file = cfg.root / cfg.sub / cfg.layout_name
    if layout_file.exists():
        raw = rename_channels_from_layout(raw, layout_file)

    # LOF bad channels on MEG
    raw = drop_channels_with_lof(raw, cfg)

    # Heartbeat ICA
    raw = heartbeat_ica(raw, cfg)

    # Notch at line noise
    raw = auto_notch_raw(raw, line_freq=50.0)

    # Low-pass before resampling (MEG only)
    raw = raw.filter(l_freq=None, h_freq=cfg.lowpass_for_resampling, picks="meg", fir_design="firwin")

    # Resample all channels
    raw = raw.resample(cfg.resample_sfreq)

    # Final band-pass (MEG only)
    raw = raw.filter(
        l_freq=cfg.l_freq,
        h_freq=cfg.h_freq,
        picks="meg",
        fir_design="firwin",
    )

    # Precision spatial filter on MEG
    raw = precision_filter_raw(raw, radius_mm=35.0, alpha=0.0)

    if cfg.show_plots:
        raw.plot_psd(fmax=60.0)
        plt.show()

    return raw


# -------------------------------------------------------------------------
# Motion ICA via MSC with accelerometers
# -------------------------------------------------------------------------


def _peak_msc(
    x: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    nperseg: int,
) -> float:
    """Peak magnitude-squared coherence between 1 component and 1 accel."""
    f, cxy = coherence(x, y, fs=sfreq, nperseg=nperseg)
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return 0.0
    return float(np.max(cxy[mask]))


def _perm_null_msc(
    sources: np.ndarray,
    accels: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    nperseg: int,
    n_perm: int,
) -> np.ndarray:
    """Permutation null distribution of peak MSC over components."""
    n_comp = sources.shape[0]
    null = np.zeros((n_perm, n_comp))
    rng = np.random.default_rng(0)

    for p in range(n_perm):
        shift = rng.integers(low=int(0.1 * sfreq), high=int(0.9 * sfreq))
        acc_shift = np.roll(accels, shift=shift, axis=1)
        for k in range(n_comp):
            null[p, k] = _peak_msc(sources[k], acc_shift.mean(axis=0), sfreq, fmin, fmax, nperseg)
    return null


def _fdr_bh(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini–Hochberg FDR control."""
    pvals = np.asarray(pvals)
    n = pvals.size
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    thresh = alpha * np.arange(1, n + 1) / n
    below = sorted_p <= thresh
    keep = np.zeros_like(pvals, dtype=bool)
    if below.any():
        k = np.max(np.where(below)[0])
        cutoff = sorted_p[k]
        keep = pvals <= cutoff
    return keep


def ica_motion_reject_with_perm(
    raw: mne.io.BaseRaw,
    cfg: Config,
    mode: str = "fwer",
    make_plots: bool = True,
) -> mne.io.BaseRaw:
    """Remove motion-related ICA components using MSC with accelerometers."""
    picks_meg = mne.pick_types(raw.info, meg=True, stim=False)
    picks_acc = mne.pick_channels(raw.info["ch_names"], list(cfg.accel_chs))

    if len(picks_meg) == 0 or len(picks_acc) == 0:
        warnings.warn("No MEG or accelerometer channels found; skipping motion ICA.")
        return raw

    raw_meg = raw.copy().pick(picks_meg)
    raw_acc = raw.copy().pick(picks_acc)

    ica = mne.preprocessing.ICA(
        n_components=None,
        random_state=cfg.ica_random_state,
        max_iter="auto",
        method="fastica",
    )
    ica.fit(raw_meg)

    sources = ica.get_sources(raw_meg).get_data()
    accels = raw_acc.get_data()
    sfreq = raw.info["sfreq"]
    nperseg = int(cfg.msc_nperseg_s * sfreq)

    peak_emp = np.array(
        [_peak_msc(sources[k], accels.mean(axis=0), sfreq, cfg.msc_fmin, cfg.msc_fmax, nperseg)
         for k in range(sources.shape[0])]
    )

    null = _perm_null_msc(
        sources,
        accels,
        sfreq=sfreq,
        fmin=cfg.msc_fmin,
        fmax=cfg.msc_fmax,
        nperseg=nperseg,
        n_perm=cfg.msc_n_permutations,
    )
    pvals = np.mean(null >= peak_emp[None, :], axis=0)

    if mode == "fwer":
        alpha = cfg.msc_alpha / len(pvals)
        bad_comp = np.where(pvals < alpha)[0]
    elif mode == "fdr":
        keep = _fdr_bh(pvals, alpha=cfg.msc_alpha)
        bad_comp = np.where(keep)[0]
    else:
        raise ValueError(f"Unknown msc mode: {mode}")

    print(f"Motion ICA: rejecting components {bad_comp.tolist()}")
    if make_plots and cfg.show_plots and len(bad_comp) > 0:
        fig, ax = plt.subplots()
        ax.stem(peak_emp)
        ax.set_xlabel("Component")
        ax.set_ylabel("Peak MSC")
        ax.set_title("Motion-related ICA components (MSC)")
        plt.show()

    ica.exclude = list(bad_comp)
    return ica.apply(raw.copy())


# -------------------------------------------------------------------------
# Epoching and evoked helpers
# -------------------------------------------------------------------------


def detect_event_id(raw: mne.io.BaseRaw, stim_ch: str) -> Dict[str, int]:
    """Detect event codes present in the stimulus channel."""
    events = mne.find_events(raw, stim_channel=stim_ch, min_duration=0.0)
    codes = sorted(int(c) for c in np.unique(events[:, 2]) if c != 0)
    if not codes:
        raise RuntimeError(f"No non-zero events found on stim channel {stim_ch}.")
    return {str(c): c for c in codes}


def make_epochs(
    raw: mne.io.BaseRaw,
    event_id: Mapping[str, int],
    stim_ch: str,
    tmin: float = -0.2,
    tmax: float = 1.0,
) -> mne.Epochs:
    """Create epochs without baseline correction (MATLAB-style)."""
    events = mne.find_events(raw, stim_channel=stim_ch, min_duration=0.0)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        reject_by_annotation=False,
        picks=mne.pick_types(raw.info, meg=True, stim=False),
    )
    return epochs


def pct_ptp_reject(
    epochs: mne.Epochs,
    percentile: float = 80.0,
    picks: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Boolean mask of epochs to keep based on peak-to-peak percentile."""
    if picks is None:
        picks = mne.pick_types(epochs.info, meg=True, stim=False)

    keep = np.zeros(len(epochs), dtype=bool)
    for cond, code in epochs.event_id.items():
        sub = epochs[cond]
        data = sub.get_data(picks=picks)  # (n_ep, n_ch, n_t)
        ptp = np.ptp(data, axis=-1).max(axis=-1)  # max over channels
        thr = np.percentile(ptp, percentile)
        cond_mask = ptp < thr
        keep[sub.selection[cond_mask]] = True
    return keep


def clean_epochs_ptp(epochs: mne.Epochs, percentile: float = 80.0) -> mne.Epochs:
    """Return a copy of epochs with peak-to-peak outliers removed."""
    mask = pct_ptp_reject(epochs, percentile=percentile)
    return epochs[mask]


def evokeds_from_epochs(epochs: mne.Epochs) -> Dict[str, mne.Evoked]:
    """Compute evoked responses per condition."""
    return {cond: epochs[cond].average() for cond in epochs.event_id}


def drop_noisy_channels_evoked(
    evokeds: Dict[str, mne.Evoked],
    pre_stim_max: float = 0.0,
    mad_sigma: float = 2.5,
    min_keep: int = 5,
    verbose: bool = True,
) -> Tuple[Dict[str, mne.Evoked], List[str]]:
    """Drop channels that are consistently noisy across conditions."""
    if not evokeds:
        return evokeds, []

    evk0 = next(iter(evokeds.values()))
    ch_names = evk0.ch_names
    all_pre = []

    for evk in evokeds.values():
        mask = evk.times <= pre_stim_max
        all_pre.append(evk.data[:, mask])
    all_pre = np.concatenate(all_pre, axis=1)

    rms = np.sqrt(np.mean(all_pre**2, axis=1))
    med = np.median(rms)
    mad = np.median(np.abs(rms - med))
    z = (rms - med) / (1.4826 * mad + np.finfo(float).eps)
    bad_idx = np.where(np.abs(z) > mad_sigma)[0]

    bads: List[str] = []
    if len(bad_idx) > 0 and (len(ch_names) - len(bad_idx)) >= min_keep:
        bads = [ch_names[i] for i in bad_idx]
        if verbose:
            print(f"Dropping noisy channels: {bads}")
        for key in evokeds:
            evokeds[key].info["bads"].extend(bads)
            evokeds[key].pick_types(meg=True, exclude="bads")

    return evokeds, bads


def compute_top5_and_pc1_matstyle(
    epochs: mne.Epochs,
    condition: str | int,
    baseline: Tuple[Optional[float], Optional[float]] = (-0.2, 0.0),
    fit_window: Tuple[float, float] = (0.05, 0.6),
    detrend_baseline: bool = False,
) -> Tuple[np.ndarray, np.ndarray, mne.Evoked]:
    """MATLAB-style top-5 sensor selection and PC1 extraction."""
    cond_str = str(condition)
    if cond_str in epochs.event_id:
        sub = epochs[cond_str]
    else:
        sub = epochs[condition]

    sub = sub.copy().apply_baseline(baseline)
    data = sub.get_data()  # (n_ep, n_ch, n_t)
    times = sub.times

    t0, t1 = fit_window
    mask = (times >= t0) & (times <= t1)
    win = data[:, :, mask]  # (n_ep, n_ch, n_win)

    # RMS over time, then average over epochs
    rms = np.sqrt((win**2).mean(axis=-1)).mean(axis=0)  # (n_ch,)
    top_idx = np.argsort(rms)[-5:]

    top = data[:, top_idx, :]  # (n_ep, 5, n_t)
    top_mean = top.mean(axis=1)  # (n_ep, n_t)

    # PCA across epochs on the top-5 average
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(top_mean)[:, 0]
    sign = np.sign(pc1.mean())
    pc1 *= sign

    evoked = sub.average()

    return top_idx, pc1, evoked


def plot_evokeds_with_layout(
    evokeds: Mapping[str, mne.Evoked],
    baseline: Tuple[Optional[float], Optional[float]] = (None, 0.0),
    fit_window: Tuple[float, float] = (0.05, 0.6),
) -> None:
    """Plot evoked responses and a simple topomap in the fit window."""
    evk_list = list(evokeds.values())
    if not evk_list:
        warnings.warn("No evokeds to plot.")
        return

    # Try to get a layout based on the evoked info (optional).
    layout = None
    try:
        layout = mne.channels.find_layout(evk_list[0].info)
    except Exception:  # pragma: no cover - layout is optional
        layout = None

    for cond, evk in evokeds.items():
        evk_bl = evk.copy().apply_baseline(baseline)
        fig = evk_bl.plot(spatial_colors=True, gfp=True, show=False)
        fig.suptitle(f"Evoked response: {cond}")
        plt.show()

        if layout is not None:
            t0, t1 = fit_window
            mask = (evk_bl.times >= t0) & (evk_bl.times <= t1)
            data = evk_bl.data[:, mask].mean(axis=1)
            fig2, ax = plt.subplots()
            mne.viz.plot_topomap(data, layout.pos[:, :2], axes=ax, show=False)
            ax.set_title(f"{cond}: mean {t0*1e3:.0f}-{t1*1e3:.0f} ms")
            plt.show()


# -------------------------------------------------------------------------
# Concatenation and high-level pipeline
# -------------------------------------------------------------------------


def concatenate(raws: Sequence[mne.io.BaseRaw]) -> mne.io.BaseRaw:
    """Concatenate a sequence of Raw objects."""
    return mne.concatenate_raws(list(raws))


def run_pipeline(cfg: Config, msc_mode: str = "fwer") -> mne.io.BaseRaw:
    """Run the full preprocessing pipeline and return cleaned Raw."""
    raws: List[mne.io.BaseRaw] = []
    for block in cfg.blocks:
        fif = cfg.root / cfg.sub / f"{cfg.sub}_{block}.fif"
        raw_block = preprocess_block(fif, cfg, tag=block)
        raws.append(raw_block)

    raws_motion_clean = [
        ica_motion_reject_with_perm(r, cfg, mode=msc_mode, make_plots=True)
        for r in raws
    ]
    raw_clean = concatenate(raws_motion_clean)
    return raw_clean
