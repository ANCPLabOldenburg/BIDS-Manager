# QC template provenance

The images in this folder are the MNI ICBM 152 Nonlinear Asymmetrical
template, version 2009c (`MNI152NLin2009cAsym`), at 2 mm, as distributed by
TemplateFlow (`https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/`,
downloaded 2026-10-07). The quality check registers anatomical images to it
and uses its tissue maps as priors (`bidsmgr/qc/register.py`, `segment.py`).

## What was changed

To keep the package small, every image was stored as 8-bit (uint8) with a
NIfTI scale slope of 1/255, on the template's own 2 mm grid and affine:

- `T1w`, `T2w`: divided by their 99.9th percentile, clipped to [0, 1].
- `brain_mask`, `csf`, `gm`, `wm`: clipped to [0, 1]. Largest error 1/510.
- `head_mask`: the published `res-02_desc-head_mask` covers a larger field of
  view (273 x 327 x 273); it was smoothed with a 3-voxel box filter and
  resampled (trilinear) onto the 2 mm grid of the others.

The originals, before conversion (SHA-256 prefix of the downloaded file):

| File | Original | SHA-256 (first 16) |
|---|---|---|
| `mni2009c_2mm_T1w.nii.gz` | `tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz` | `7c4e551ae8150ac1` |
| `mni2009c_2mm_T2w.nii.gz` | `tpl-MNI152NLin2009cAsym_res-02_T2w.nii.gz` | `2228a730b59f5a64` |
| `mni2009c_2mm_brain_mask.nii.gz` | `tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz` | `7a71e9ce99d2af7b` |
| `mni2009c_2mm_head_mask.nii.gz` | `tpl-MNI152NLin2009cAsym_res-02_desc-head_mask.nii.gz` | `5d75e94ebd280ff0` |
| `mni2009c_2mm_csf.nii.gz` | `tpl-MNI152NLin2009cAsym_res-02_label-CSF_probseg.nii.gz` | `c98c2fb2beb6ebca` |
| `mni2009c_2mm_gm.nii.gz` | `tpl-MNI152NLin2009cAsym_res-02_label-GM_probseg.nii.gz` | `66aab92950c4256a` |
| `mni2009c_2mm_wm.nii.gz` | `tpl-MNI152NLin2009cAsym_res-02_label-WM_probseg.nii.gz` | `88d15b840383ab37` |

## Licence

The notice below must appear in every copy (it is the template's licence):

> Copyright (C) 1993-2004 Louis Collins, McConnell Brain Imaging Centre,
> Montreal Neurological Institute, McGill University. Permission to use,
> copy, modify, and distribute this software and its documentation for any
> purpose and without fee is hereby granted, provided that the above
> copyright notice appear in all copies. The authors and McGill University
> make no representations about the suitability of this software for any
> purpose. It is provided "as is" without express or implied warranty. The
> authors are not responsible for any data loss, equipment damage, property
> loss, or injury to subjects or patients resulting from the use or misuse
> of this software package.

## References

- Fonov V, Evans AC, Botteron K, Almli CR, McKinstry RC, Collins DL.
  Unbiased average age-appropriate atlases for pediatric studies.
  NeuroImage 2011;54(1):313-327. doi:10.1016/j.neuroimage.2010.07.033
- Fonov VS, Evans AC, McKinstry RC, Almli CR, Collins DL. Unbiased nonlinear
  average age-appropriate brain templates from birth to adulthood.
  NeuroImage 2009;47:S102. doi:10.1016/S1053-8119(09)70884-5
- RRID: SCR_008796

## Replacing them

Replace the whole set at once (the maps share a grid), record the source and
checksums here, and re-run `tests/unit/test_qc_templates.py`.
