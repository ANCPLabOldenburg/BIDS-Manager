"""ECAT to NIfTI backend, built on nibabel.

Siemens HRRT and the older CTI/ECAT scanners write ECAT rather than DICOM, and
dcm2niix cannot read it. nibabel can, and it is already a BIDS Manager
dependency, so ECAT support costs no new package and no subprocess.

The backend follows the same contract as every other one: take a
:class:`ConvertTask`, write into the per-subject staging directory, report what
landed on disk. Sidecar enrichment is not its job. It writes only what the ECAT
header states as fact (frame timing, units, radionuclide, institution); the
fields BIDS requires that no scanner records are filled later by
``fixups.pet_sidecar`` from the user's metadata spec.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..types import ConvertResult, ConvertTask

log = logging.getLogger(__name__)


# ECAT main-header ``patient_orientation`` codes 0 to 7 encode a real
# orientation; 8 means the scanner did not record one. nibabel knows 0 to 7 and
# flips the voxel data accordingly, but builds its affine from voxel size and
# offsets ALONE, never consulting the field. So the affine describes one fixed
# arrangement and the data only lands in it when the flip fires.
#
# Code 8 therefore produces an image whose data and affine disagree: nibabel
# applies no flip, while the affine still assumes the flipped arrangement. The
# result is upside down and mirrored.
#
# Siemens HRRT writes 8. Since HRRT is a brain-only scanner that subjects enter
# head first supine, the orientation the scanner failed to record is code 3,
# which is neurological, so apply the flip nibabel would have applied. Verified
# against both the published OpenNeuroPET reference images and an independent
# conversion of the same files.
_ECAT_ORIENTATION_UNKNOWN = 8


def _orient_to_affine(img, data):
    """Put ECAT voxel data into the arrangement nibabel's affine describes.

    A no-op for every file that records a patient orientation, because nibabel
    has already done the flip. Only the unrecorded case is corrected, and it is
    corrected towards head-first supine, the position the scanners writing that
    code physically enforce.
    """
    import numpy as np

    try:
        orientation = int(img.header["patient_orientation"].item())
    except (KeyError, TypeError, ValueError):
        return data
    if orientation != _ECAT_ORIENTATION_UNKNOWN:
        return data

    log.info(
        "ECAT file records no patient orientation (code %d); assuming head "
        "first supine so the voxel data matches the affine",
        orientation,
    )
    return np.flip(data, axis=(0, 1, 2))


class EcatDirect:
    """Convert an ECAT ``.v`` PET recording to NIfTI via nibabel.

    Stateless; safe to instantiate once per ``run_convert`` and reuse.
    """

    name = "ecat_direct"

    def can_handle(self, task: ConvertTask) -> bool:
        if task.datatype != "pet":
            return False
        if not task.source_files or not task.basename:
            return False
        # Signature check, not extension: ".v" is too generic to trust and a
        # valid ECAT file is sometimes shipped without it.
        from ...inventory.pet_ecat import is_ecat_file

        return any(is_ecat_file(fp) for fp in task.source_files if fp.exists())

    def convert(self, task: ConvertTask, staging_dir: Path) -> ConvertResult:
        t0 = time.monotonic()
        try:
            return self._convert_inner(task, staging_dir, t0)
        except Exception as exc:  # noqa: BLE001 - report, never crash the run
            log.exception("ecat_direct: unexpected error for %s", task.basename)
            return ConvertResult(
                task=task,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_s=time.monotonic() - t0,
            )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _convert_inner(
        self, task: ConvertTask, staging_dir: Path, t0: float,
    ) -> ConvertResult:
        import nibabel

        from ...inventory.pet_ecat import is_ecat_file, probe_ecat

        source = next(
            (fp for fp in task.source_files if fp.exists() and is_ecat_file(fp)),
            None,
        )
        if source is None:
            return ConvertResult(
                task=task, success=False,
                error="no readable ECAT file among the task's source files",
                duration_s=time.monotonic() - t0,
            )

        out_dir = staging_dir
        if task.session:
            out_dir = out_dir / f"ses-{task.session}"
        out_dir = out_dir / task.datatype
        out_dir.mkdir(parents=True, exist_ok=True)

        nii_path = out_dir / f"{task.basename}.nii.gz"
        json_path = out_dir / f"{task.basename}.json"

        try:
            img = nibabel.ecat.load(str(source))
            # get_fdata applies the per-frame scale factors ECAT stores in its
            # subheaders. Reading the raw array would silently drop them and
            # leave the image in arbitrary units.
            data = _orient_to_affine(img, img.get_fdata())
            # A single-frame acquisition is a 3-D image. nibabel reports ECAT
            # as (x, y, z, frames) whatever the frame count, and carrying a
            # trailing length-1 axis makes a static scan look dynamic to
            # anything that keys on ndim, the time-series graph included.
            if data.ndim == 4 and data.shape[3] == 1:
                data = data[..., 0]
            nifti = nibabel.Nifti1Image(data, img.affine)
            nifti.header.set_xyzt_units("mm", "sec")
            nibabel.save(nifti, str(nii_path))
        except Exception as exc:  # noqa: BLE001
            return ConvertResult(
                task=task, success=False,
                error=f"nibabel could not convert {source.name}: {exc}",
                duration_s=time.monotonic() - t0,
            )

        json_path.write_text(
            json.dumps(self._sidecar_from_header(source, probe_ecat(source)), indent=2),
            encoding="utf-8",
        )

        return ConvertResult(
            task=task,
            success=True,
            staged_files=(nii_path, json_path),
            duration_s=time.monotonic() - t0,
        )

    @staticmethod
    def _sidecar_from_header(source: Path, probe) -> dict:
        """The sidecar fields the ECAT header states outright.

        Two readers, in order of authority.

        Ours first, from :func:`probe_ecat`: frame timing, radionuclide, tracer
        and institution, the fields the inventory already showed the user, so
        the sidecar and the table can never disagree.

        Then pet2bids fills what it reads and we do not. An ECAT7 header carries
        roughly sixty fields, of which we parsed five, so an ECAT user used to
        get a far thinner sidecar than a DICOM user for no better reason than
        that nobody had written the parser. They had, and it is the reference
        implementation. It adds the scanner model, the reconstruction method
        with its iterations and subsets, per-frame decay and scale factors, the
        dose calibration factor and ``TimeZero``, which BIDS REQUIRES and no
        other part of our ECAT path could supply.

        Still deliberately narrow about invention. Nothing here guesses: every
        value is something a reader found in the file. What no scanner records
        (injected mass, administration mode) stays absent, so it stays visibly
        missing in validation and the form still asks for it.
        """
        out: dict = {"Modality": "PT", "ConversionSoftware": "nibabel"}
        if probe is None:
            return _fill_absent(out, source)

        if probe.frame_durations:
            out["FrameDuration"] = [round(v, 6) for v in probe.frame_durations]
        if probe.frame_starts:
            out["FrameTimesStart"] = [round(v, 6) for v in probe.frame_starts]
        if probe.isotope:
            from ...inventory.pet import normalise_radionuclide

            out["TracerRadionuclide"] = normalise_radionuclide(probe.isotope)
        if probe.tracer:
            out["TracerName"] = probe.tracer
        if probe.facility:
            out["InstitutionName"] = probe.facility
        return _fill_absent(out, source)


def _fill_absent(out: dict, source: Path) -> dict:
    """Add what pet2bids reads, without touching anything already stated.

    Ours wins on every field it sets: those came from the same probe the
    inventory displayed, and a sidecar that disagrees with the table it was
    built from is worse than a thin one. The user's template still overrides
    all of it at the metadata step, which is where stated answers belong.
    """
    from ...inventory.pet_ecat import ecat_sidecar_fields

    for name, value in ecat_sidecar_fields(source).items():
        out.setdefault(name, value)
    return out


__all__ = ["EcatDirect"]
