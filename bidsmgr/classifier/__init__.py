"""Chained classifiers: ``InventoryRow -> list[Classification]``.

Reference: architecture.md §4.2.

Layers (run in order, each schema-validates its output):

1. ``dcm2niix_bidsguess`` — MRI; uses dcm2niix's ``BidsGuess`` field.
2. ``mne_channel_types`` — EEG/MEG/iEEG; channel kinds from raw.
3. ``sequence_dict``     — fallback regex dict on
   ``SeriesDescription`` (port from BIDS-Manager v0.2.5
   ``dicom_inventory.guess_modality``).

Every layer's output is ``schema.validate_*``-checked before the
planner sees it. Wrong classifier outputs are dropped silently and
we move to the next layer.

Stub — not yet implemented.
"""
