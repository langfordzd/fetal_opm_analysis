# fetal_opm_analysis

Minimal preprocessing pipeline for fetal optically-pumped
magnetometer (OPM) data using MNE-Python.

1. LOF-based bad channel removal
2. Heartbeat ICA removal
3. Automatic notch at line-frequency harmonics
4. Low-pass to resample to band-pass
5. Precision-based spatial filter
6. Motion-related ICA using magnitude squared coherence with an accelerometer(s)
7. Simple epoching + peak-to-peak epoch rejection
8. Evoked responses, noisy-channel dropping, and top-5/PC1 summary

Dependencies:
pip install mne numpy scipy scikit-learn matplotlib
