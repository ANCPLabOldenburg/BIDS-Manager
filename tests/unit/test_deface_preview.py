"""What defacing would remove, without removing it (``deface.preview``).

The engine is replaced by a fake that blanks a known region, so these run
without niimath and say exactly what the preview must report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from bidsmgr.deface import preview  # noqa: E402
from bidsmgr.deface.run import DefaceResult  # noqa: E402


def _head(path: Path) -> Path:
    """A bright 'head' in a box of faint background noise."""
    data = np.random.default_rng(0).random((20, 20, 20)).astype(np.float32) * 5
    data[3:17, 3:17, 3:17] = 500.0
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def _fake_engine(monkeypatch, blank, *, crop=None):
    """An engine that zeroes ``blank`` (a slice tuple) and the background,
    and with ``crop`` also keeps only that part of the grid."""
    def deface_to_temp(source, **_kw):
        img = nib.load(str(source))
        data = np.asarray(img.dataobj, dtype=np.float32).copy()
        data[data < 100] = 0.0            # engines zero the noise too
        data[blank] = 0.0
        affine = img.affine.copy()
        if crop is not None:
            data = data[crop]
            offset = np.array([s.start or 0 for s in crop] + [1.0])
            affine[:3, 3] = (img.affine @ offset)[:3]
        out = Path(source).with_name("defaced_tmp.nii.gz")
        nib.save(nib.Nifti1Image(data, affine), str(out))
        return DefaceResult(source=Path(source), output=out, engine_id="fake", seconds=0.0)

    monkeypatch.setattr(preview, "deface_to_temp", deface_to_temp)


def test_the_blanked_part_of_the_head_is_the_mask(tmp_path, monkeypatch):
    src = _head(tmp_path / "t1.nii.gz")
    _fake_engine(monkeypatch, (slice(0, 20), slice(0, 8), slice(0, 20)))
    mask, head, affine = preview.removed_mask(src)
    assert mask[10, 5, 10] == 1.0, "the face, blanked"
    assert mask[10, 12, 10] == 0.0, "the brain, kept"
    assert mask[0, 0, 0] == 0.0, "background noise is nobody's face"
    assert head[10, 12, 10] and not head[0, 0, 0]
    assert np.array_equal(affine, np.eye(4))
    assert not (tmp_path / "defaced_tmp.nii.gz").exists(), "the temporary result is removed"


def test_a_cropping_engine_is_compared_in_world_space(tmp_path, monkeypatch):
    """robustfov returns a smaller grid: what it cut away was removed."""
    src = _head(tmp_path / "t1.nii.gz")
    _fake_engine(monkeypatch, (slice(0, 0), slice(0, 0), slice(0, 0)),
                 crop=(slice(0, 20), slice(0, 20), slice(6, 20)))
    mask, _head_mask, _a = preview.removed_mask(src)
    assert mask[10, 10, 4] == 1.0, "below the crop: removed"
    assert mask[10, 10, 12] == 0.0, "inside the crop: kept"


def test_the_summary_counts_the_head_not_the_air(tmp_path, monkeypatch):
    src = _head(tmp_path / "t1.nii.gz")
    _fake_engine(monkeypatch, (slice(0, 20), slice(0, 10), slice(0, 20)))
    mask, head, _a = preview.removed_mask(src)
    text = preview.summary(mask, head)
    share = float(text.split("%")[0]) / 100
    assert share == pytest.approx(7 / 14, abs=0.01), "half the head's rows"


def test_the_viewer_overlay(tmp_path, monkeypatch):
    from bidsmgr.viz.overlays import deface_preview

    src = _head(tmp_path / "t1.nii.gz")
    _fake_engine(monkeypatch, (slice(0, 20), slice(0, 8), slice(0, 20)))
    ov = deface_preview(src)
    assert ov.kind == "mask" and ov.source.in_memory
    assert ov.name == "What defacing would remove"
    assert "Nothing was changed" in ov.note
    assert ov.display.colormap == "red"
