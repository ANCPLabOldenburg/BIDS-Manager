
# BIDS Manager

**BIDS Manager** is a **PyQt-based** GUI that converts **DICOM** folders into **BIDS**-compliant datasets and allows easy metadata editing.

---
# Documentation

You can access to documentation through this [link](https://ancplaboldenburg.github.io/bids_manager_documentation/). You will find information on how to install and use BIDS-Manager.


## Requirements

| Software  | Minimum Version | Notes                                                   |
|----------|-----------------|---------------------------------------------------------|
| **Python** | 3.10            | Installed automatically if you use the one-click installers |

---

## Installation

You can install BIDS Manager in two ways:

### 1. One-click installers <sup>(recommended)</sup>

1. **Download** the ZIP package:  
   **[📦 One-click Installers](https://github.com/ANCPLabOldenburg/BIDS-Manager/raw/main/Installers/Installers.zip
)**
2. **Extract** the ZIP file and run the script for your operating system:

| OS               | Script                        | How to Run                         | Duration |
|------------------|-------------------------------|------------------------------------|---------|
| **Windows 10/11**| `install_bids_manager.bat`     | Double-click                        | ≈ 5 min |
| **Linux**        | `install_bids_manager.sh`      | `./install_bids_manager.sh`         | ≈ 5 min |

3. After the installation finishes, you will find two shortcuts on your desktop:

| OS          | Launch                    | Uninstall                      |
|-------------|---------------------------|--------------------------------|
| **Windows** | `run_bidsmanager.bat`      | `uninstall_bidsmanager.bat`    |
| **Linux**   | **BIDS Manager** (launcher)| `uninstall_bidsmanager.sh`     |

---

### 2. Install in a virtual environment (advanced)

```bash
# 1. Create a virtual environment
python3 -m venv <env_name>

# 2. Activate it
source <env_name>/bin/activate          # On Windows: <env_name>\Scripts\activate

# 3. Install BIDS Manager from GitHub
pip install bids-manager
```

The package declares all dependencies including `heudiconv`, so installation
pulls everything required to run the GUI and helper scripts.
All core requirements are version pinned in `pyproject.toml` to ensure
consistent installations.

After installation the following commands become available:

- `bids-manager` – main GUI combining conversion and editing tools
- `dicom-inventory` – generate `subject_summary.tsv` from a DICOM directory
- `build-heuristic` – create a HeuDiConv heuristic from the TSV
- `run-heudiconv` – run HeuDiConv using the generated heuristic
- `post-conv-renamer` – rename fieldmap files after conversion
- `bids-editor` – standalone metadata editor
- `fill-bids-ignore` – interactively update `.bidsignore`

All utilities provide `-h/--help` for details.

### Recent updates

- The TSV produced by `dicom-inventory` can now be loaded directly in the GUI and
  its file name customised before generation.
- The Batch Rename tool previews changes and allows restricting the scope to
  specific subjects.
- A "Set Intended For" dialog lets you manually edit fieldmap IntendedFor lists
  if the automatic matching needs adjustment.
- `run-heudiconv` now keeps a copy of `subject_summary.tsv` under `.bids_manager`
  and generates a clean `participants.tsv` using demographics from that file.
- Re-running `run-heudiconv` on the same dataset now appends new subjects to
  the existing `.bids_manager` records and updates `participants.tsv` instead of
  overwriting them.
- `dicom-inventory` distinguishes repeated sequences by adding `series_uid` and `rep`
  columns and records `acq_time` for each series in `subject_summary.tsv`.
- Fieldmap rows for magnitude and phase images are now merged so each acquisition
  appears once with the combined file count, and their `series_uid` values are
  stored as a pipe-separated list so both sequences are converted.
- `post-conv-renamer` now adds an `IntendedFor` list to each fieldmap JSON so
  fMRI preprocessing tools can automatically match fieldmaps with the relevant
  functional runs.
- The GUI's Tools menu gained actions to refresh `_scans.tsv` files and edit
  `.bidsignore` entries.
- The DPI scale dialog now adjusts values in 25% increments and the DPI button
  appears between the CPU and Authorship buttons.
- On startup the GUI detects the system DPI and applies the matching scale.
- The scanned data table now provides a "Generate unique IDs" button that
  assigns random 3‑letter/3‑digit identifiers to subjects. If an entry already
  exists for the same study in an existing `.bids_manager/subject_summary.tsv`,
  you are prompted to reuse its identifier.
- A "Detect repeats" button can recompute repetition numbers based on
  acquisition time when all BIDS and given names are filled.

