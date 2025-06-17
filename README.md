# BIDS Manager

This repository provides a PyQt-based GUI and helper scripts to convert DICOM folders into BIDS datasets and edit their metadata.

## Installation

```bash
pip install git+https://github.com/karellopez/BIDS-Manager.git
```

The package declares all dependencies including `heudiconv`, so installation
pulls everything required to run the GUI and helper scripts.

After installation the following commands become available:

- `bids-manager` – main GUI combining conversion and editing tools
- `dicom-inventory` – generate `subject_summary.tsv` from a DICOM directory
- `build-heuristic` – create a HeuDiConv heuristic from the TSV
- `run-heudiconv` – run HeuDiConv using the generated heuristic
- `post-conv-renamer` – rename fieldmap files after conversion
- `bids-editor` – standalone metadata editor

All utilities provide `-h/--help` for details.

### Recent updates

- The TSV produced by `dicom-inventory` can now be loaded directly in the GUI and
  its file name customised before generation.
- The Batch Rename tool previews changes and allows restricting the scope to
  specific subjects.
- `run-heudiconv` now keeps a copy of `subject_summary.tsv` under `.bids_manager`
  and generates a clean `participants.tsv` using demographics from that file.
- `dicom-inventory` distinguishes repeated sequences by adding `series_uid` and `rep`
  columns, records `acq_time` for each series, and marks phase images with an
  `is_phase` flag in `subject_summary.tsv`.



### Troubleshooting

If launching `bids-manager` fails with a message like:

```
ModuleNotFoundError: No module named '_bz2'
```

Your Python interpreter was built without bzip2 support. Install the system development package for `libbz2` (e.g. `libbz2-dev` on Debian/Ubuntu) and rebuild Python or use a distribution-provided interpreter with bzip2. Dependencies such as `nibabel` rely on the standard `bz2` module.
