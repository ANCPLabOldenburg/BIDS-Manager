"""Post-conversion file fixups applied per-subject after the backend runs.

Reference: architecture.md §12. The orchestrator (``cli/convert.py``)
calls these in the per-subject Phase 2, sequentially, before the atomic
rename of the staging tree into the live BIDS root.

Modules:

* ``fieldmaps`` — turn dcm2niix's fmap multi-output tokens
  (``_e1`` / ``_e2`` / ``_ph`` / ``_e1_ph`` / ``_e2_ph``) into the BIDS
  fmap suffixes (``magnitude1`` / ``magnitude2`` / ``phasediff`` /
  ``phase1`` / ``phase2``).
* ``intended_for`` — populate ``IntendedFor`` in fmap JSON sidecars
  using the BIDS URI form, time-based binding with all-pairs fallback.
* ``scans_tsv`` — rewrite the ``filename`` column of any ``*_scans.tsv``
  to match renamed files (no-op until bidsmgr emits scans.tsv files).

Planned later: ``derivatives`` (DWI map relocation: FA/ADC/TRACE/ColFA →
``derivatives/<pipeline>/sub-/ses-/dwi/``).
"""

from .companion import attach_companion_files
from .eeg_sidecar import enrich_recording_sidecars
from .identifiers import prune_identifiers
from .mrs_header import clean_mrs_file, clean_mrs_outputs, read_mrs_header
from .pet_sidecar import enrich_pet_sidecars
from .physio_sidecar import enrich_physio_sidecars
from .fieldmaps import apply_fieldmap_renames
from .intended_for import populate_intended_for
from .scans_tsv import update_scans_tsv
from .blood import convert_blood_files
from .deface import deface_staged
from .sidecar_schema import apply_stated_metadata, repair_converter_output

__all__ = [
    "convert_blood_files",
    "apply_fieldmap_renames",
    "populate_intended_for",
    "apply_stated_metadata",
    "deface_staged",
    "repair_converter_output",
    "update_scans_tsv",
    "clean_mrs_file",
    "prune_identifiers",
    "clean_mrs_outputs",
    "enrich_pet_sidecars",
    "read_mrs_header",
    "enrich_physio_sidecars",
    "enrich_recording_sidecars",
    "attach_companion_files",
]
