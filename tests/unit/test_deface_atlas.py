"""The atlas is one asset, and it is the one nobody will notice going wrong.

Swapping the template without its mask produces a registration that succeeds
and a mask that lands somewhere else: output that looks defaced and is not.
Nothing downstream can catch that, so it is caught here.

The checksums are the ones recorded in ``atlas/PROVENANCE.md``, which are in
turn the ones BIDSvue records for the same pair.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bidsmgr.deface.engines import ATLAS_DIR, MASK, TEMPLATE

EXPECTED = {
    "avg152T1.nii.gz":
        "e8f5440f0dcec1a4d44384acbdf19c8e6cf94c032c3356ef91c4441fee3aaea8",
    "avg152T1mask.nii.gz":
        "adc275e26e5217189d70acfd43311bc3f66f316321ac61defc88505d2f91a0aa",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_atlas_shipped():
    assert TEMPLATE.is_file(), f"{TEMPLATE} missing from the package"
    assert MASK.is_file(), f"{MASK} missing from the package"


@pytest.mark.parametrize("name, digest", sorted(EXPECTED.items()))
def test_the_atlas_is_the_pair_we_recorded(name, digest):
    assert _sha256(ATLAS_DIR / name) == digest, (
        f"{name} is not the file atlas/PROVENANCE.md describes. If this was "
        "deliberate, update the provenance file AND bump ENGINE_REVISION in "
        "engines.py, so datasets defaced with the old atlas stay "
        "distinguishable."
    )


def test_provenance_records_the_checksums_that_are_actually_shipped():
    """The documentation of the asset cannot drift from the asset."""
    text = (ATLAS_DIR / "PROVENANCE.md").read_text(encoding="utf-8")
    for name in EXPECTED:
        assert name in text, f"{name} is not mentioned in PROVENANCE.md"
    for digest in EXPECTED.values():
        assert digest in text, "PROVENANCE.md records a different checksum"


def test_template_and_mask_share_a_geometry():
    """Different geometry is the failure that still produces an output file."""
    nib = pytest.importorskip("nibabel")
    t = nib.load(str(TEMPLATE))
    m = nib.load(str(MASK))
    assert t.shape == m.shape, "template and mask have different shapes"
    assert (t.affine == m.affine).all(), "template and mask are not aligned"


def test_the_mask_keeps_a_plausible_share_of_the_volume():
    """A sanity bound, not a precise one.

    A mask that keeps everything defaces nothing; a mask that keeps almost
    nothing is a brain mask, which would skull-strip instead. The shipped pair
    keeps about half its bounding box.
    """
    nib = pytest.importorskip("nibabel")
    data = nib.load(str(MASK)).get_fdata()
    keep = float((data >= 0.5).mean())
    assert 0.2 < keep < 0.8, (
        f"the mask keeps {keep:.0%} of its bounding box, which is not a face "
        "mask. A brain mask here would skull-strip rather than deface."
    )


# ---------------------------------------------------------------------------
# The brain mask, used by the atlas skull-strip engine.
#
# It is derived rather than copied (see PROVENANCE.md), so what matters is the
# geometry: a brain mask that no longer shares the template's grid produces a
# registration that succeeds and a mask that lands in the wrong place, which
# deletes brain and looks like it worked.


def test_the_brain_mask_shipped():
    from bidsmgr.deface.engines import BRAIN_MASK

    assert BRAIN_MASK.is_file(), (
        "the skull-strip atlas engine has nothing to register against"
    )


def test_the_brain_mask_shares_the_template_geometry():
    nib = pytest.importorskip("nibabel")
    import numpy as np

    from bidsmgr.deface.engines import BRAIN_MASK, TEMPLATE

    template = nib.load(str(TEMPLATE))
    mask = nib.load(str(BRAIN_MASK))
    assert mask.shape == template.shape
    assert np.allclose(mask.affine, template.affine)


def test_the_brain_mask_keeps_a_plausible_share_and_is_binary():
    nib = pytest.importorskip("nibabel")
    import numpy as np

    from bidsmgr.deface.engines import BRAIN_MASK

    data = np.asarray(nib.load(str(BRAIN_MASK)).dataobj)
    assert set(np.unique(data)) <= {0, 1}, "not a binary mask"
    share = float((data > 0).mean())
    # A brain is a large minority of the template's bounding box. Far below
    # this would clip; far above would keep the skull and defeat the point.
    assert 0.10 < share < 0.40, share


def test_the_two_masks_are_not_the_same_file():
    """Getting these the wrong way round keeps the face and deletes the brain."""
    from bidsmgr.deface.engines import BRAIN_MASK, MASK

    assert BRAIN_MASK != MASK
    assert BRAIN_MASK.read_bytes() != MASK.read_bytes()


def test_provenance_records_the_brain_mask_checksum_actually_shipped():
    import hashlib

    from bidsmgr.deface.engines import ATLAS_DIR, BRAIN_MASK

    digest = hashlib.sha256(BRAIN_MASK.read_bytes()).hexdigest()
    text = (ATLAS_DIR / "PROVENANCE.md").read_text(encoding="utf-8")
    assert digest in text, (
        "PROVENANCE.md does not record the brain mask that is shipped"
    )


def test_the_brain_mask_is_left_right_symmetric():
    """The check that catches a mask built in the wrong axis order.

    The template is a symmetric average, so a correctly oriented brain mask
    on it is very nearly its own mirror image. The first version of this mask
    was built by pairing brainchop's in-memory array with its header, which
    are in OPPOSITE axis orders, and the result was a rotated mask that
    registered cleanly and then cut every subject at an angle. Its symmetry
    was 0.5; a correct one is 0.98.
    """
    nib = pytest.importorskip("nibabel")
    import numpy as np

    from bidsmgr.deface.engines import BRAIN_MASK

    mask = np.asarray(nib.load(str(BRAIN_MASK)).dataobj) > 0
    dice = 2 * (mask & mask[::-1]).sum() / (2 * mask.sum())
    assert dice > 0.90, f"the mask is not symmetric (Dice {dice:.3f})"
