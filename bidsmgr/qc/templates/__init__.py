"""The MNI ICBM 152 2009c template at 2 mm, as 8-bit images.

What the anatomical check registers to (``qc.register``): the T1w and T2w
templates, the head and brain masks, and the CSF, GM and WM tissue maps,
which become the priors of the segmentation and the reference of the
template overlap. Provenance, checksums and the licence notice the files
must travel with: ``PROVENANCE.md`` beside them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np

#: The files, by what they hold.
FILES = {
    "T1w": "mni2009c_2mm_T1w.nii.gz",
    "T2w": "mni2009c_2mm_T2w.nii.gz",
    "brain": "mni2009c_2mm_brain_mask.nii.gz",
    "head": "mni2009c_2mm_head_mask.nii.gz",
    "csf": "mni2009c_2mm_csf.nii.gz",
    "gm": "mni2009c_2mm_gm.nii.gz",
    "wm": "mni2009c_2mm_wm.nii.gz",
}

#: MRIQC's landmarks (MNI mm): the air the artefact measures use is above
#: the plane through the glabella and the inion, so the face, the neck and
#: the shoulders, where folding and motion are normal, are not counted.
GLABELLA_Z_MM = -14.0
#: The face, for the defacing check: in front of and below the brain.
FACE_Y_MM = 30.0
FACE_Z_MM = 10.0


@dataclass(frozen=True)
class Template:
    affine: np.ndarray
    shape: tuple[int, int, int]
    images: dict

    def __getitem__(self, key: str) -> np.ndarray:
        return self.images[key]


@lru_cache(maxsize=1)
def template() -> Template:
    """The template, read once (about 2 MB, a few tens of milliseconds)."""
    import nibabel as nib

    folder = resources.files(__package__)
    images = {}
    affine = None
    for key, name in FILES.items():
        with resources.as_file(folder / name) as path:
            img = nib.load(str(path))
            images[key] = np.asarray(img.get_fdata(dtype=np.float32))
            affine = np.asarray(img.affine, dtype=float)
    shape = images["T1w"].shape
    return Template(affine=affine, shape=tuple(int(s) for s in shape), images=images)


__all__ = ["FACE_Y_MM", "FACE_Z_MM", "FILES", "GLABELLA_Z_MM", "Template", "template"]
