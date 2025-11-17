
# INPUT DATA
dicom_file = "/home/karelo/Desktop/Development/MEGQC_workshop/datasets/Testdata_ConversionApp/raw_data/physiodata/1.3.12.2.1107.5.2.43.66080.202309141024211473240467.0.0.0_0022_000001_169468350014be.dcm"
prefix = "/home/karelo/Desktop/Development/MEGQC_workshop/datasets/Testdata_ConversionApp/converted_data/BIDS/sub-001/ses-pre/func/sub-001_ses-pre_task-rest_run-1"

# CONVERT DICOM/Physio to BIDS
from bidsphysio.dcm2bids import dcm2bidsphysio

# 2) run the conversion
physio_data = dcm2bidsphysio.dcm2bids(dicom_file)
physio_data.save_to_bids_with_trigger(prefix)

# PLOT RESULTS

from pathlib import Path
import json, gzip
import numpy as np
import matplotlib.pyplot as plt

# >>> edit this to your run prefix (without the "_recording-..._physio.*" suffix)
prefix = Path(prefix)

# which recordings to try to plot (change/order as you like)
recordings = ["ecg", "cardiac", "respiratory"]

def load_bids_physio(base_prefix: Path, recording: str):
    """
    Load a BIDS physio file and return:
      t      : time vector (seconds)
      data   : ndarray (n_samples, n_cols)
      cols   : list of column names from JSON
      meta   : JSON dict
    """
    tsv = base_prefix.with_name(base_prefix.name + f"_recording-{recording}_physio.tsv.gz")
    jsn = base_prefix.with_name(base_prefix.name + f"_recording-{recording}_physio.json")
    if not tsv.exists() or not jsn.exists():
        raise FileNotFoundError(f"Missing files for {recording}:\n  {tsv}\n  {jsn}")

    with open(jsn, "r") as f:
        meta = json.load(f)
    fs = float(meta["SamplingFrequency"])
    t0 = float(meta.get("StartTime", 0.0))
    cols = meta.get("Columns", [])

    with gzip.open(tsv, "rt") as f:
        data = np.loadtxt(f)

    # Ensure 2D shape even if there's a single column
    if data.ndim == 1:
        data = data[:, None]

    n = data.shape[0]
    t = t0 + np.arange(n) / fs
    return t, data, cols, meta

def plot_bids_physio(base_prefix: Path, recording: str, seconds=None):
    """
    Plot one recording. If 'seconds' is a tuple (tmin, tmax), zoom to that window.
    """
    t, X, cols, meta = load_bids_physio(base_prefix, recording)
    # pick signal columns (exclude 'trigger'/'Triggers' if present)
    sig_idx = [i for i, c in enumerate(cols) if str(c).lower() not in ("trigger", "triggers")]
    trig_idx = next((i for i, c in enumerate(cols) if str(c).lower() in ("trigger", "triggers")), None)

    # optional time window
    if seconds is not None:
        tmin, tmax = seconds
        keep = (t >= tmin) & (t <= tmax)
        t, X = t[keep], X[keep, :]

    plt.figure(figsize=(11, 3))
    # plot each signal column
    for i in sig_idx:
        plt.plot(t, X[:, i], label=str(cols[i]) if cols else f"signal{i+1}")
    # overlay trigger markers if present
    if trig_idx is not None:
        trig = X[:, trig_idx]
        idx = np.where(trig > 0)[0]
        if idx.size:
            # draw vertical lines at trigger times
            plt.vlines(t[idx], ymin=np.min(X[:, sig_idx]) if sig_idx else np.min(X),
                       ymax=np.max(X[:, sig_idx]) if sig_idx else np.max(X),
                       linewidth=0.5, linestyles="dotted", label="trigger")

    plt.title(f"{recording}  |  Fs={meta['SamplingFrequency']} Hz  |  StartTime={meta.get('StartTime',0):.3f}s")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# ---- Run: plot all three full-length
for rec in recordings:
    try:
        plot_bids_physio(prefix, rec)
    except FileNotFoundError as e:
        print(e)

# ---- Optional: zoom-in example (first 30 seconds)
# for rec in recordings:
#     plot_bids_physio(prefix, rec, seconds=(0, 30))
