"""Per-modality raw-data scanners. Produces ``InventoryRow[]``.

Reference: architecture.md §2.1, §4.1.

Modules:
* ``mri_dicom``    — pydicom + joblib parallel scan; preserves the
  v0.2.5 22-column ``subject_summary.tsv`` contract
  (improvement_plan.md §4).
* ``eeg_meg_raw``  — mne.io.read_raw probe, no preload.
* ``physio``       — Siemens PhysioLog DICOM detector.

Identity inference (subject / study / session / run) lives here too.
See architecture.md §4.1 for the per-modality evidence vectors and
conflict-detection rules.

Stub — not yet implemented.
"""
