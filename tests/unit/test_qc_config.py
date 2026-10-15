"""The quality checks' configuration (``bidsmgr.qc.config``): one model for
the viewer, the Editor, the post-convert step and ``bidsmgr-qc``."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bidsmgr.qc import anat, live
from bidsmgr.qc.config import DEFAULT, QcConfig, fast, key
from bidsmgr.viz.settings import QC_SECTIONS, VizSettings, qc_config

from tests.unit.qc_phantoms import anat_phantom, dataset, save


def test_a_hand_edited_file_still_loads() -> None:
    """Out of range is clamped and an unknown choice falls back, as the
    viewer settings do: a settings file never stops the app opening."""
    cfg = QcConfig.model_validate({
        "bold": {"fd_threshold_mm": 99, "detrend": 7, "sample_voxels": 3.7e6},
        "methods": {"brain": "not-a-method"},
        "anat": {"unknown_field": 1}})
    assert cfg.bold.fd_threshold_mm == 5.0
    assert cfg.bold.detrend == 2
    assert cfg.bold.sample_voxels == 100_000 and isinstance(cfg.bold.sample_voxels, int)
    assert cfg.methods.brain == "mindgrab"


def test_fast_switches_every_method_and_keeps_the_thresholds() -> None:
    cfg = QcConfig()
    cfg.bold.fd_threshold_mm = 0.2
    quick = fast(cfg)
    assert (quick.methods.brain, quick.methods.tissues, quick.methods.registration,
            quick.methods.dwi_brain) == ("template", "em", "numpy", "median_otsu")
    assert quick.bold.fd_threshold_mm == 0.2
    assert cfg.methods.brain == "mindgrab", "the original was changed"


def test_the_fingerprint_follows_every_setting() -> None:
    a, b = QcConfig(), QcConfig()
    assert key(a) == key(b)
    b.dwi.spike_z = 9.0
    assert key(a) != key(b)


def test_the_viewer_settings_carry_the_configuration() -> None:
    s = VizSettings()
    assert {"qc_methods", "qc_bold", "qc_anat", "qc_dwi"} <= set(QC_SECTIONS)
    s.qc_anat.defaced_face_pct = 55.0
    cfg = qc_config(s)
    assert cfg.anat.defaced_face_pct == 55.0 and cfg.bold == DEFAULT.bold
    # Stored as JSON and read back.
    again = VizSettings.model_validate_json(s.model_dump_json())
    assert again.qc_anat.defaced_face_pct == 55.0


def test_a_threshold_changes_the_check() -> None:
    """A planted saturation is a finding at the default and not with a
    threshold above it: the thresholds are the configuration's."""
    data, affine = anat_phantom()
    top = float(data.max())
    head = data > 0.2 * top
    idx = np.flatnonzero(head.ravel())[::50]
    data.ravel()[idx] = top
    cfg = fast(QcConfig())
    found = {f.key for f in anat.check(data, affine, suffix="T1w", config=cfg).findings}
    assert "saturation" in found
    cfg.anat.saturation_pct = 10.0
    found = {f.key for f in anat.check(data, affine, suffix="T1w", config=cfg).findings}
    assert "saturation" not in found


def test_each_result_records_its_configuration() -> None:
    data, affine = anat_phantom()
    cfg = fast(QcConfig())
    cfg.anat.work_mm = 3.0
    res = anat.check(data, affine, suffix="T1w", config=cfg)
    assert res.facts["config"]["anat"]["work_mm"] == 3.0
    assert res.facts["config"]["methods"]["brain"] == "template"


def test_fast_methods_chosen_on_purpose_are_not_a_fallback() -> None:
    """The approximate-masks note is for methods that could not run, not
    for the fast ones the user chose."""
    data, affine = anat_phantom()
    res = anat.check(data, affine, suffix="T1w", config=fast(QcConfig()))
    assert "approximate" not in {f.key for f in res.findings}
    assert "tool_note" not in res.facts


def test_the_viewer_cache_is_kept_per_configuration(tmp_path) -> None:
    from bidsmgr.viz.data.volume import open_volume

    root = dataset(tmp_path / "ds")
    data, affine = anat_phantom()
    path = save(data, affine, root / "sub-01" / "anat" / "sub-01_T1w.nii.gz")
    src = open_volume(path)
    src.stream()
    live.forget()
    a = fast(QcConfig())
    b = a.model_copy(deep=True)
    b.anat.saturation_pct = 9.0
    first = live.result_for(src, {}, config=a)
    assert live.cached(src, a) is first and live.cached(src, b) is None
    second = live.result_for(src, {}, config=b)
    assert second is not first and live.cached(src, a) is first
    live.forget(src)
    assert live.cached(src, a) is None and live.cached(src, b) is None


def test_the_cli_prints_and_reads_a_configuration(tmp_path, capsys) -> None:
    from bidsmgr.cli import qc as cli

    assert cli.main(["--print-config", "--fast"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["methods"]["brain"] == "template"
    printed["dwi"]["spike_z"] = 12.0
    path = tmp_path / "qc.json"
    path.write_text(json.dumps(printed), encoding="utf-8")
    assert cli.main(["--config", str(path), "--print-config", "--no-flip-check"]) == 0
    read = json.loads(capsys.readouterr().out)
    assert read["dwi"]["spike_z"] == 12.0 and read["dwi"]["check_flips"] is False
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert cli.main(["--config", str(bad), str(tmp_path)]) == 2


def test_the_cli_still_needs_a_dataset_to_check() -> None:
    from bidsmgr.cli import qc as cli

    with pytest.raises(SystemExit):
        cli.main([])
