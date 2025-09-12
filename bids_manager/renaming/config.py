from pathlib import Path

DEFAULT_SCHEMA_DIR = Path("bids_manager/miscellaneous/schema")  # adjust if different
ENABLE_SCHEMA_RENAMER = True
ENABLE_FIELDMap_NORMALIZATION = True
ENABLE_DWI_DERIVATIVES_MOVE = True
DERIVATIVES_PIPELINE_NAME = "dcm2niix"  # or "BIDS-Manager" if you prefer
