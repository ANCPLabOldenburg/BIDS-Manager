import shutil
from pathlib import Path

from bids_manager.schema_renamer import (
    DEFAULT_SCHEMA_DIR,
    DERIVATIVES_PIPELINE_NAME,
    SeriesInfo,
    apply_post_conversion_rename,
    build_preview_names,
    load_bids_schema,
)
from bids_manager.dicom_inventory import guess_modality


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("dummy")


def create_fake_dataset(root: Path):
    _touch(root / "sub-001" / "anat" / "sub-001_orig.nii.gz")
    _touch(root / "sub-001" / "anat" / "sub-001_orig.json")
    _touch(root / "sub-001" / "func" / "sub-001_run1.nii.gz")
    _touch(root / "sub-001" / "func" / "sub-001_run1.json")
    _touch(root / "sub-001" / "dwi" / "sub-001_raw.nii.gz")
    _touch(root / "sub-001" / "dwi" / "sub-001_raw.json")
    _touch(root / "sub-001" / "dwi" / "sub-001_raw.bval")
    _touch(root / "sub-001" / "dwi" / "sub-001_raw.bvec")
    for suffix in ["ADC", "FA", "TRACEW", "ColFA"]:
        _touch(root / "sub-001" / "dwi" / f"sub-001_raw_{suffix}.nii.gz")
        _touch(root / "sub-001" / "dwi" / f"sub-001_raw_{suffix}.json")
    _touch(root / "sub-001" / "fmap" / "sub-001_echo-1.nii.gz")
    _touch(root / "sub-001" / "fmap" / "sub-001_echo-2.nii.gz")
    _touch(root / "sub-001" / "fmap" / "sub-001_fmap.nii.gz")


def test_schema_renamer_end_to_end(tmp_path):
    create_fake_dataset(tmp_path)
    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)
    series = [
        SeriesInfo("001", None, "T1w", "mprage", None, {"current_bids": "sub-001_orig"}),
        SeriesInfo("001", None, "bold", "fmri_rest", None, {"current_bids": "sub-001_run1"}),
        SeriesInfo("001", None, "dwi", "ep2d_diff", None, {"current_bids": "sub-001_raw"}),
    ]
    proposals = build_preview_names(series, schema)
    rename_map = apply_post_conversion_rename(tmp_path, proposals)
    assert (tmp_path / "sub-001" / "anat" / "sub-001_T1w.nii.gz").exists()
    assert (tmp_path / "sub-001" / "func" / "sub-001_task-rest_bold.nii.gz").exists()
    assert (tmp_path / "sub-001" / "dwi" / "sub-001_dwi.nii.gz").exists()
    assert (tmp_path / "sub-001" / "dwi" / "sub-001_dwi.bval").exists()
    assert (tmp_path / "sub-001" / "dwi" / "sub-001_dwi.bvec").exists()
    for suffix in ["ADC", "FA", "TRACEW", "ColFA"]:
        out = tmp_path / "derivatives" / DERIVATIVES_PIPELINE_NAME / "sub-001" / "dwi" / f"sub-001_desc-{suffix}_dwi.nii.gz"
        assert out.exists()
    assert (tmp_path / "sub-001" / "fmap" / "sub-001_magnitude1.nii.gz").exists()
    assert (tmp_path / "sub-001" / "fmap" / "sub-001_magnitude2.nii.gz").exists()
    assert (tmp_path / "sub-001" / "fmap" / "sub-001_phasediff.nii.gz").exists()
    rename_map2 = apply_post_conversion_rename(tmp_path, proposals)
    assert rename_map2 == {}


def test_duplicate_names_numbered(tmp_path):
    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)
    _touch(tmp_path / "sub-001" / "anat" / "sub-001_orig1.nii.gz")
    _touch(tmp_path / "sub-001" / "anat" / "sub-001_orig1.json")
    _touch(tmp_path / "sub-001" / "anat" / "sub-001_orig2.nii.gz")
    _touch(tmp_path / "sub-001" / "anat" / "sub-001_orig2.json")
    series = [
        SeriesInfo("001", None, "T1w", "mprage", None, {"current_bids": "sub-001_orig1"}),
        SeriesInfo("001", None, "T1w", "mprage", 2, {"current_bids": "sub-001_orig2"}),
    ]
    proposals = build_preview_names(series, schema)
    rename_map = apply_post_conversion_rename(tmp_path, proposals)
    assert (tmp_path / "sub-001" / "anat" / "sub-001_T1w.nii.gz").exists()
    assert (tmp_path / "sub-001" / "anat" / "sub-001_T1w_rep-2.nii.gz").exists()


def test_fieldmap_runs_and_task_hits(tmp_path):
    """Fieldmaps with run numbers should keep distinct names and task_hits
    should influence task detection."""

    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)

    # Two fieldmap series with run tokens in their sequence names
    fm1 = SeriesInfo("001", None, "phasediff", "fmap_run-1", None, {})
    fm2 = SeriesInfo("001", None, "phasediff", "fmap_run-2", None, {})

    # Series with custom task hits. The sequence itself has no known task
    # tokens but ``task_hits`` provides a hint.
    task_series = SeriesInfo(
        "001",
        None,
        "bold",
        "customsequence",
        None,
        {"task_hits": "custom"},
    )

    proposals = build_preview_names([fm1, fm2, task_series], schema)

    # Extract basenames for easier assertions
    fmap_bases = [base for (_, dt, base) in proposals[:2]]
    task_base = proposals[2][2]

    assert fmap_bases == [
        "sub-001_run-01_phasediff",
        "sub-001_run-02_phasediff",
    ]
    # Task hit "custom" should be used
    assert task_base == "sub-001_task-custom_bold"


def test_dwi_direction_and_acq_detection():
    """DWI series should capture dir/acq hints from their sequence names."""

    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)
    series = [
        # Modality "dti" should normalise to dwi and pick up LR/RL directions
        SeriesInfo("001", None, "dti", "DTI_LR", None, {}),
        SeriesInfo("001", None, "dti", "DTI_RL", None, {}),
        # Numbers combined with direction should become an acq label
        SeriesInfo("001", None, "dwi", "15_AP", None, {}),
        SeriesInfo("001", None, "dwi", "15b0_AP", None, {}),
    ]

    proposals = build_preview_names(series, schema)
    bases = [base for (_, _, base) in proposals]

    assert bases == [
        "sub-001_dir-lr_dwi",
        "sub-001_dir-rl_dwi",
        "sub-001_acq-15_dir-ap_dwi",
        "sub-001_acq-15b0_dir-ap_dwi",
    ]


def test_sequence_acq_token_preserved_in_preview():
    """Existing ``acq-`` labels in the sequence should be preserved."""

    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)
    series = SeriesInfo("001", None, "T1w", "mprage_acq-HighRes_run-01", None, {})

    proposals = build_preview_names([series], schema)

    assert proposals[0][2] == "sub-001_acq-HighRes_run-01_T1w"


def test_sequence_acq_token_preserved_during_rename(tmp_path):
    """Post-conversion rename keeps the original ``acq-`` discriminator."""

    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)
    nii = tmp_path / "sub-001" / "anat" / "sub-001_mprage_acq-HighRes.nii.gz"
    json = tmp_path / "sub-001" / "anat" / "sub-001_mprage_acq-HighRes.json"
    _touch(nii)
    _touch(json)

    series = [
        SeriesInfo(
            "001",
            None,
            "T1w",
            "mprage_acq-HighRes",
            None,
            {"current_bids": "sub-001_mprage_acq-HighRes"},
        )
    ]

    proposals = build_preview_names(series, schema)
    rename_map = apply_post_conversion_rename(tmp_path, proposals)

    target = tmp_path / "sub-001" / "anat" / "sub-001_acq-HighRes_T1w.nii.gz"
    assert target.exists()
    assert (tmp_path / "sub-001" / "anat" / "sub-001_acq-HighRes_T1w.json").exists()
    # Rename map should reflect the move for downstream updates.
    assert nii in rename_map
    assert rename_map[nii] == target


def test_sequence_multiple_acq_tokens_prefers_richest(tmp_path):
    """When multiple ``acq-`` tokens exist keep the most descriptive one."""

    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)

    # Simulate a diffusion series where the scanner stored two acquisition
    # hints.  The more specific ``acq-15b0`` should win over the generic
    # ``acq-15`` in both preview and post-conversion rename steps.
    sequence = "acq-15_acq-15b0 dir-ap dwi"
    series = SeriesInfo("002", None, "dwi", sequence, None, {})

    proposals = build_preview_names([series], schema)
    assert proposals[0][2] == "sub-002_acq-15b0_dir-ap_dwi"

    # Create fake heudiconv outputs using the verbose sequence name so the
    # renamer must rely on the acquisition token extraction.
    nii = tmp_path / "sub-002" / "dwi" / "sub-002_acq-15_acq-15b0_dir-ap_dwi.nii.gz"
    json = tmp_path / "sub-002" / "dwi" / "sub-002_acq-15_acq-15b0_dir-ap_dwi.json"
    bval = tmp_path / "sub-002" / "dwi" / "sub-002_acq-15_acq-15b0_dir-ap_dwi.bval"
    bvec = tmp_path / "sub-002" / "dwi" / "sub-002_acq-15_acq-15b0_dir-ap_dwi.bvec"
    for path in (nii, json, bval, bvec):
        _touch(path)

    series_with_current = SeriesInfo(
        "002",
        None,
        "dwi",
        sequence,
        None,
        {"current_bids": "sub-002_acq-15_acq-15b0_dir-ap_dwi"},
    )

    rename_map = apply_post_conversion_rename(tmp_path, build_preview_names([series_with_current], schema))

    target = tmp_path / "sub-002" / "dwi" / "sub-002_acq-15b0_dir-ap_dwi.nii.gz"
    assert target.exists()
    assert (tmp_path / "sub-002" / "dwi" / "sub-002_acq-15b0_dir-ap_dwi.json").exists()
    assert (tmp_path / "sub-002" / "dwi" / "sub-002_acq-15b0_dir-ap_dwi.bval").exists()
    assert (tmp_path / "sub-002" / "dwi" / "sub-002_acq-15b0_dir-ap_dwi.bvec").exists()
    assert nii in rename_map
    assert rename_map[nii] == target


def test_sbref_and_physio_detection():
    """SBRef and physio sequences should not be misclassified as bold."""

    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)

    # SeriesDescriptions containing "bold" tokens should still be detected
    # as SBRef or physio when those hints are present.
    sbref_series = SeriesInfo("001", None, "SBRef", "fmri_sbref", None, {})
    phys_series = SeriesInfo("001", None, "physio", "fmri_physio", None, {})

    proposals = build_preview_names([sbref_series, phys_series], schema)

    (_, dt_sbref, base_sbref), (_, dt_phys, base_phys) = proposals

    assert dt_sbref == "func"
    assert base_sbref.endswith("_sbref")

    assert dt_phys == "func"
    assert base_phys.endswith("_physio")


def test_physio_naming_preserves_task_and_run():
    """Physio recordings should share task/run labels with their BOLD runs."""

    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)

    bold = SeriesInfo("001", None, "bold", "task-two_run-01_bold", None, {})
    phys = SeriesInfo("001", None, "physio", "task-two_run-01_physio", None, {})

    proposals = build_preview_names([bold, phys], schema)

    bases = {base for (_, _, base) in proposals}

    assert "sub-001_task-two_run-01_bold" in bases
    assert "sub-001_task-two_run-01_physio" in bases


def test_guess_modality_prefers_sbref_and_physio():
    """When sequences contain bold tokens, SBRef/physio patterns win."""

    assert guess_modality("fmri_sbref") == "SBRef"
    assert guess_modality("BOLD_SBRef") == "SBRef"
    assert guess_modality("fmri_physio") == "physio"
