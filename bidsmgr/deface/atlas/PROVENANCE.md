# Deface atlas provenance

`avg152T1.nii.gz` and `avg152T1mask.nii.gz` are a matched template/mask pair
used by `niimath -deface`. They are copied byte-for-byte from
[`niivue/deface` commit `a27d54264256531db1cb17019a01319e78506e76`](https://github.com/niivue/deface/commit/a27d54264256531db1cb17019a01319e78506e76),
the revision that moved the reference application to the BSD niimath
modern-allineate `-deface` engine.

| Asset | SHA-256 |
| --- | --- |
| `avg152T1.nii.gz` | `e8f5440f0dcec1a4d44384acbdf19c8e6cf94c032c3356ef91c4441fee3aaea8` |
| `avg152T1mask.nii.gz` | `adc275e26e5217189d70acfd43311bc3f66f316321ac61defc88505d2f91a0aa` |

Verified against the checksums BIDSvue records for the same pair
(`BIDSvue/resources/common/README.md`): identical.

## Treat the pair as one asset

The template and the mask share a geometry. Replacing one without the other
produces a registration that succeeds and a mask that lands in the wrong place,
which is the worst kind of failure here: the output looks defaced and is not.

A replacement must:

1. record its upstream revision and both checksums in this file,
2. keep template and mask geometry identical to each other,
3. be checked on real data, visually, before release, and
4. bump the `CodeValue` in `bidsmgr/deface/engines.py` so a dataset defaced
   with the old atlas stays distinguishable from one defaced with the new one.

`tests/unit/test_deface_atlas.py` enforces 1 and 2 mechanically. It cannot
enforce 3.

## Licence

BSD-2-Clause, via niivue/deface. The images derive from the MNI avg152 T1
template.

## The brain mask, `avg152T1brainmask.nii.gz`

Used by the `strip-atlas` skull-strip engine. niimath's own help spells out
why one mask file is all that differs: "skull-stripping: use -deface with a
brain mask". The mask decides what survives, so the face mask above removes a
face and this one keeps only the brain.

It is **derived**, not copied, because no brain mask for this template ships
with anything we depend on and the obvious source is FSL's, whose licence does
not clearly permit redistribution here. It is built from two things we already
ship and may redistribute: the BSD-2-Clause `avg152T1.nii.gz` above, and the
BSD-2-Clause mindgrab network.

Rebuild it with:

```
python tools/make_brain_mask.py
```

which runs mindgrab on the template, resamples the result onto the template's
own grid with nearest-neighbour interpolation, and closes the pinholes that
downsampling leaves (a hole in a keep-mask punches a hole in the brain).

| Property | Value |
| --- | --- |
| Geometry | identical to `avg152T1.nii.gz` (91 x 109 x 91, 2 mm) |
| Brain voxels | 234,650 |
| Mask voxels that are template background (`< 0.05`) | 0.0 % |
| Left-right symmetry (Dice against its own mirror) | 0.976 |
| SHA-256 | `527b433427b8d306c0338b711507b387205b325622404c6a0ad0a05adac6ad54` |

It is deliberately GENEROUS. On a blurred average template mindgrab keeps more
than it would on an individual scan, and for a keep-mask that is the safe
direction to be wrong in: the engine leaves some dura rather than cutting into
cortex. That, plus the affine fit, is why `strip-atlas` is documented as the
crude option and `mindgrab` as the one to use when the result will be
analysed.

The same four rules as the pair above apply to replacing it, and the geometry
rule matters most: a brain mask that no longer shares the template's grid
produces a registration that succeeds and a mask that lands in the wrong
place, which deletes brain and looks like it worked.
