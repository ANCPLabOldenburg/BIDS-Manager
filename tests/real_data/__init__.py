"""Real-data characterisation tests.

Gated on env vars:
* ``BIDS_MANAGER_REAL_MRI_DATA``
* ``BIDS_MANAGER_REAL_MEG_DATA``
* ``BIDS_MANAGER_REAL_EEG_DATA``

Real datasets live at
``/Users/karelo/Development/datasets/BIDS_Manager/raw_data/{MRI,MEG,EEG}``.

Compare against checked-in golden snapshots
(``subject_summary.tsv``, ``dataset_description.json``, etc.).
"""
