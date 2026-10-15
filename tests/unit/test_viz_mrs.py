"""The maths a NIfTI-MRS file goes through before it means anything, and the
file as a source: ``bidsmgr.viz.compute.mrs`` and ``bidsmgr.viz.data.spectrum``.

Qt-free. These are the parts that would be wrong in a way nobody could see.
"""

from __future__ import annotations

import numpy as np
import pytest

from bidsmgr.viz.compute import mrs as M  # noqa: E402
from bidsmgr.viz.compute.mrs import (
    DEFAULT_PPM_RANGE_1H,
    METABOLITES_1H,
    apodize,
    coil_combine,
    combine,
    default_ppm_range,
    metabolites_for,
    part,
    peak_ppm,
    spectrum,
    stagger,
)
from bidsmgr.viz.data.formats import is_mrs_path
from bidsmgr.viz.data.spectrum import COIL, DYN, EDIT, read_mrs, reference_for
from tests.fixtures.signals import DWELL, SF_MHZ, fid_at, write_mrs

pytest.importorskip("nibabel")

_fid_at = fid_at


class TestThePpmAxis:
    """The step that is easy to get wrong and impossible to notice."""

    @pytest.mark.parametrize("name,shift", [
        ("NAA", 2.01), ("Cr", 3.03), ("Cho", 3.22), ("mI", 3.56),
    ])
    def test_a_peak_lands_where_it_was_put(self, name, shift):
        ppm, _hz, spec = spectrum(_fid_at(shift), DWELL, SF_MHZ, "1H")
        found = ppm[int(np.argmax(np.abs(spec)))]
        assert found == pytest.approx(shift, abs=0.02), name

    def test_the_axis_is_not_mirrored(self):
        """A spectrum with the sign the other way round is a perfectly
        plausible picture in which every metabolite is on the wrong side of
        the water. Two peaks pin the direction."""
        ppm, _hz, spec = spectrum(
            _fid_at(2.01) + 0.5 * _fid_at(3.22), DWELL, SF_MHZ, "1H",
        )
        mag = np.abs(spec)
        naa = ppm[int(np.argmax(np.where((ppm > 1.8) & (ppm < 2.2), mag, 0)))]
        cho = ppm[int(np.argmax(np.where((ppm > 3.0) & (ppm < 3.4), mag, 0)))]
        assert naa == pytest.approx(2.01, abs=0.02)
        assert cho == pytest.approx(3.22, abs=0.02)
        assert naa < cho          # and NAA really is the lower shift

    def test_water_sits_at_the_reference(self):
        ppm, _hz, spec = spectrum(_fid_at(4.65), DWELL, SF_MHZ, "1H")
        assert ppm[int(np.argmax(np.abs(spec)))] == pytest.approx(4.65, abs=0.02)

    def test_no_spectrometer_frequency_gives_hz_not_nonsense(self):
        """Better an axis in Hz than a ppm scale that means nothing."""
        ppm, hz, _spec = spectrum(_fid_at(2.01), DWELL, 0.0, "1H")
        assert np.allclose(ppm, hz)


class TestProcessing:
    def test_line_broadening_widens_and_shortens_the_peak(self):
        """It trades resolution for signal-to-noise. Both halves checked."""
        fid = _fid_at(2.01)
        _p, _h, plain = spectrum(fid, DWELL, SF_MHZ, "1H")
        _p, _h, broad = spectrum(fid, DWELL, SF_MHZ, "1H", line_broadening_hz=10.0)
        assert np.max(np.abs(broad)) < np.max(np.abs(plain))
        # Wider: more bins above half the maximum.
        def width(s):
            m = np.abs(s)
            return int(np.sum(m > m.max() / 2))
        assert width(broad) > width(plain)

    def test_zero_broadening_changes_nothing(self):
        fid = _fid_at(2.01)
        assert np.array_equal(apodize(fid, DWELL, 0.0), fid)

    def test_phase_rotates_without_changing_magnitude(self):
        fid = _fid_at(2.01)
        _p, _h, a = spectrum(fid, DWELL, SF_MHZ, "1H")
        _p, _h, b = spectrum(fid, DWELL, SF_MHZ, "1H", phase_deg=90.0)
        assert np.allclose(np.abs(a), np.abs(b))
        assert not np.allclose(a.real, b.real)

    def test_the_parts_are_what_they_say(self):
        spec = np.array([1 + 1j, -2 + 0j])
        assert np.allclose(part(spec, "real"), [1, -2])
        assert np.allclose(part(spec, "imaginary"), [1, 0])
        assert np.allclose(part(spec, "magnitude"), [np.sqrt(2), 2])


class TestDynamics:
    def test_the_repeats_average_by_default(self):
        """The repeats exist to be averaged; that is where the SNR is."""
        fid = np.stack([_fid_at(2.01), _fid_at(2.01) * 3], axis=1)
        assert np.allclose(combine(fid), fid.mean(axis=1))

    def test_one_repeat_can_be_singled_out(self):
        """How a corrupted repeat gets found."""
        fid = np.stack([_fid_at(2.01), _fid_at(3.03)], axis=1)
        assert np.allclose(combine(fid, mode="single", index=1), fid[:, 1])

    def test_an_out_of_range_repeat_is_clamped_not_an_error(self):
        fid = np.stack([_fid_at(2.01), _fid_at(3.03)], axis=1)
        assert np.allclose(combine(fid, mode="single", index=99), fid[:, 1])


class TestTheMetaboliteTable:
    def test_the_shifts_are_the_textbook_ones(self):
        table = dict((name, shift) for name, shift, _d in METABOLITES_1H)
        assert table["NAA"] == 2.01
        assert table["Cr"] == 3.03
        assert table["Cho"] == 3.22
        assert table["mI"] == 3.56
        assert table["Lac"] == 1.33

    def test_every_entry_falls_inside_a_readable_window(self):
        for name, shift, _d in METABOLITES_1H:
            assert 0.0 < shift < 5.0, name

    def test_only_1h_gets_labels(self):
        """1H labels on a 31P spectrum would be in the wrong places."""
        assert metabolites_for("1H")
        assert metabolites_for("31P") == ()

    def test_1h_opens_in_the_conventional_window(self):
        ppm = np.linspace(-3.0, 12.0, 100)
        assert default_ppm_range("1H", ppm) == DEFAULT_PPM_RANGE_1H

    def test_another_nucleus_opens_on_its_own_data(self):
        ppm = np.linspace(-20.0, 20.0, 100)
        assert default_ppm_range("31P", ppm) == (-20.0, 20.0)


class TestRouting:
    """Routing a click must never read a volume off disk."""

    def test_an_mrs_file_routes(self, tmp_path):
        p = tmp_path / "sub-01" / "mrs" / "sub-01_svs.nii.gz"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
        assert is_mrs_path(p) is True

    def test_a_bold_does_not(self, tmp_path):
        p = tmp_path / "sub-01" / "func" / "sub-01_task-x_bold.nii.gz"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
        assert is_mrs_path(p) is False

    def test_an_unknown_suffix_in_an_mrs_folder_does_not(self, tmp_path):
        p = tmp_path / "sub-01" / "mrs" / "sub-01_T1w.nii.gz"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
        assert is_mrs_path(p) is False

    def test_the_json_sidecar_does_not(self, tmp_path):
        p = tmp_path / "sub-01" / "mrs" / "sub-01_svs.json"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
        assert is_mrs_path(p) is False


class TestLabelStaggering:
    """Which row each metabolite name goes on, so none covers another.

    The test is in fraction of the VISIBLE span, so the same table needs
    staggering at one zoom and not at another. These lock that in, because
    the failure is silent: the labels simply overlap.
    """

    shifts = [shift for _n, shift, _d in METABOLITES_1H]

    def test_one_label_needs_no_row_but_the_first(self):
        assert stagger([3.03], 4.0) == [0]

    def test_no_labels_is_not_an_error(self):
        assert stagger([], 4.0) == []

    def test_far_apart_labels_share_the_top_row(self):
        # 0.5 and 4.0 across a 4 ppm window are nowhere near each other.
        assert stagger([0.5, 4.0], 4.0) == [0, 0]

    def test_neighbours_are_pushed_apart(self):
        # Creatine 3.03 and choline 3.22 across the whole window: 0.19 ppm
        # is under five per cent of four, so they collide.
        assert stagger([3.03, 3.22], 4.0) == [0, 1]

    def test_zooming_in_gives_them_room_again(self):
        # The same pair across half a ppm has the pane to itself.
        assert stagger([3.03, 3.22], 0.5) == [0, 0]

    def test_the_real_table_fits_in_the_rows_available(self):
        rows = stagger(self.shifts, 4.0, rows=4)
        assert len(rows) == len(self.shifts)
        assert max(rows) < 4

    def test_every_label_keeps_a_row_even_when_impossible(self):
        # Ten labels on top of each other cannot all be separated. Dropping
        # one would be a metabolite the reader cannot find, so each still
        # gets a row.
        rows = stagger([3.0] * 10, 4.0, rows=2)
        assert len(rows) == 10
        assert all(0 <= r < 2 for r in rows)

    def test_a_zero_span_does_not_divide_by_zero(self):
        assert len(stagger(self.shifts, 0.0)) == len(self.shifts)

    def test_the_result_is_stable_for_the_same_input(self):
        first = stagger(self.shifts, 2.0)
        assert stagger(self.shifts, 2.0) == first


class TestFirstOrderPhase:
    def test_it_undoes_a_delayed_acquisition(self):
        """A FID whose first points were lost to a delay comes out with a
        phase that grows with frequency. Giving that delay as first-order
        phase puts every peak back where the undelayed FID had it. The sign
        was once the wrong way round, which doubled the delay instead."""
        delay_ms = 5.0
        shift = int(round(delay_ms / 1000.0 / DWELL))
        full = _fid_at(2.01, n=2048 + shift) + _fid_at(3.22, n=2048 + shift)
        ppm, _hz, truth = spectrum(full[:2048], DWELL, SF_MHZ, "1H")
        delayed = full[shift:]
        _p, _h, raw = spectrum(delayed, DWELL, SF_MHZ, "1H")
        _p, _h, fixed = spectrum(delayed, DWELL, SF_MHZ, "1H", phase1_ms=delay_ms)
        mag = np.abs(truth)
        peaks = [int(np.argmax(np.where((ppm > lo) & (ppm < hi), mag, 0)))
                 for lo, hi in ((1.9, 2.1), (3.1, 3.3))]

        def off(spec, i):
            return abs(np.angle(spec[i] * np.conj(truth[i])))

        for i in peaks:
            assert off(fixed, i) < 0.05
            assert off(raw, i) > 0.3, "and without it the phase is visibly wrong"

    def test_zero_is_a_no_op(self):
        fid = _fid_at(2.01)
        _p, _h, a = spectrum(fid, DWELL, SF_MHZ, "1H")
        _p, _h, b = spectrum(fid, DWELL, SF_MHZ, "1H", phase1_ms=0.0)
        assert np.array_equal(a, b)


class TestCoilCombination:
    def test_coils_out_of_phase_add_up_instead_of_cancelling(self):
        """A plain average of unphased coils cancels the signal it is meant
        to add up. Phasing each coil to its first point fixes that."""
        base = _fid_at(2.01)
        coils = np.stack([base, base * np.exp(1j * np.pi), base * np.exp(0.5j)], axis=1)
        naive = np.abs(coils.mean(axis=1)[0])
        combined = np.abs(coil_combine(coils, axis=1)[0])
        assert combined == pytest.approx(np.abs(base[0]), rel=1e-6)
        assert naive < 0.5 * combined

    def test_a_weak_coil_counts_for_less(self):
        base = _fid_at(2.01)
        noise = np.random.default_rng(3).standard_normal(base.size) * 0.05
        coils = np.stack([base, 0.01 * base + noise], axis=1)
        out = coil_combine(coils, axis=1)
        assert np.abs(out - base).mean() < np.abs(coils.mean(axis=1) - base).mean()


class TestTheSource:
    """The higher dimensions keep their NIfTI-MRS tags, so coils are
    combined, repeats averaged or picked and edit conditions subtracted."""

    def test_a_plain_svs_reads(self, tmp_path):
        path = write_mrs(tmp_path / "sub-01" / "mrs", "sub-01_svs.nii.gz", _fid_at(2.01))
        src = read_mrs(path)
        assert src is not None
        assert src.nucleus == "1H"
        assert src.dwell == pytest.approx(DWELL)
        assert src.spectrometer_mhz == pytest.approx(SF_MHZ)
        assert (src.n_points, src.n_dynamics, src.n_coils, src.n_edits) == (2048, 1, 1, 1)
        ppm, _hz, spec = spectrum(src.select(), src.dwell, src.spectrometer_mhz, src.nucleus)
        assert peak_ppm(ppm, np.abs(spec)) == pytest.approx(2.01, abs=0.02)

    def test_tagged_dimensions_are_understood(self, tmp_path):
        base = _fid_at(2.01)
        # (points, coils=2, dynamics=3, edit=2)
        fid = np.stack([np.stack([np.stack([base * (1 + d), base * (1 + d) * 0.5], axis=-1)
                                  for d in range(3)], axis=-2)] * 2, axis=1)
        path = write_mrs(tmp_path / "sub-01" / "mrs", "sub-01_svs.nii.gz", fid,
                         {"dim_5": "DIM_COIL", "dim_6": "DIM_DYN", "dim_7": "DIM_EDIT"})
        src = read_mrs(path)
        assert src.dims == [COIL, DYN, EDIT]
        assert (src.n_coils, src.n_dynamics, src.n_edits) == (2, 3, 2)
        assert "2 coils combined" in src.describe()
        assert "2 edit conditions" in src.describe()
        one = src.select(dynamic=2, edit=0)
        assert np.abs(one[0]) == pytest.approx(3.0 * np.abs(base[0]), rel=1e-5)
        diff = src.select(dynamic=0, edit=-2)
        assert np.abs(diff[0]) == pytest.approx(0.5 * np.abs(base[0]), rel=1e-5), (
            "the difference is edit-on minus edit-off")
        averaged = src.select()
        assert np.abs(averaged[0]) == pytest.approx(1.5 * np.abs(base[0]), rel=1e-5)

    def test_an_untagged_dimension_counts_as_repeats(self, tmp_path):
        fid = np.stack([_fid_at(2.01), _fid_at(3.03)], axis=1)
        path = write_mrs(tmp_path / "sub-01" / "mrs", "sub-01_svs.nii.gz", fid)
        src = read_mrs(path)
        assert src.n_dynamics == 2
        assert np.allclose(src.select(dynamic=1), fid[:, 1], atol=1e-6)

    def test_a_plain_nifti_is_not_a_spectrum(self, tmp_path):
        import nibabel as nib

        path = tmp_path / "x.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 4), dtype=np.float32), np.eye(4)),
                 str(path))
        assert read_mrs(path) is None

    def test_the_water_reference_is_found_and_read(self, tmp_path):
        folder = tmp_path / "sub-01" / "mrs"
        svs = write_mrs(folder, "sub-01_acq-press_svs.nii.gz", _fid_at(2.01))
        ref = write_mrs(folder, "sub-01_acq-press_mrsref.nii.gz", _fid_at(4.65))
        assert reference_for(svs) == ref
        assert reference_for(ref) is None, "a reference has no reference"
        src = read_mrs(svs)
        assert src.reference is not None
        assert src.reference.reference is None

    def test_a_lone_reference_of_another_name_is_still_found(self, tmp_path):
        folder = tmp_path / "sub-01" / "mrs"
        svs = write_mrs(folder, "sub-01_acq-a_svs.nii.gz", _fid_at(2.01))
        ref = write_mrs(folder, "sub-01_acq-b_mrsref.nii.gz", _fid_at(4.65))
        assert reference_for(svs) == ref

    def test_two_candidate_references_are_not_guessed_between(self, tmp_path):
        folder = tmp_path / "sub-01" / "mrs"
        svs = write_mrs(folder, "sub-01_acq-a_svs.nii.gz", _fid_at(2.01))
        write_mrs(folder, "sub-01_acq-b_mrsref.nii.gz", _fid_at(4.65))
        write_mrs(folder, "sub-01_acq-c_mrsref.nii.gz", _fid_at(4.65))
        assert reference_for(svs) is None


# ---------------------------------------------------------------------------
# Reading a spectrum: height, phase, quality
# ---------------------------------------------------------------------------


class TestReading:
    def _spectrum(self, phase_deg=0.0, water=0.0):
        from tests.fixtures.signals import DWELL, SF_MHZ, fid_at

        fid = fid_at(2.01, 2048, decay_s=0.1) + fid_at(3.03, 2048, decay_s=0.1, amplitude=0.8)
        if water:
            fid = fid + fid_at(4.65, 2048, decay_s=0.1, amplitude=water)
        fid = fid * np.exp(1j * np.deg2rad(phase_deg))
        ppm, _hz, spec = M.spectrum(fid, DWELL, SF_MHZ, "1H")
        return ppm, spec

    def test_the_height_leaves_the_water_out(self):
        ppm, spec = self._spectrum(water=50.0)
        y = spec.real
        with_water = M.y_range(ppm, y, 0.2, 5.0)
        without = M.y_range(ppm, y, 0.2, 5.0, exclude=M.WATER_BAND_1H)
        assert without[1] < with_water[1] / 10

    def test_nothing_visible_is_none(self):
        ppm, spec = self._spectrum()
        assert M.y_range(ppm, spec.real, 20.0, 30.0) is None

    @pytest.mark.parametrize("applied", [-120.0, -40.0, 0.0, 35.0, 150.0])
    def test_auto_phase_undoes_a_phase_error(self, applied):
        ppm, spec = self._spectrum(phase_deg=applied)
        found = M.auto_phase0(ppm, spec)
        assert ((found + applied + 180.0) % 360.0) - 180.0 == pytest.approx(0.0, abs=3.0)

    def test_quality_numbers(self):
        from tests.fixtures.signals import SF_MHZ

        ppm, spec = self._spectrum()
        rng = np.random.default_rng(0)
        noisy = spec + (rng.normal(0, 0.5, spec.size) + 1j * rng.normal(0, 0.5, spec.size))
        qc = M.qc_metrics(ppm, noisy, SF_MHZ, "1H")
        assert qc["naa_ppm"] == pytest.approx(2.01, abs=0.02)
        assert qc["snr"] > 5
        # A 0.1 s decay is a Lorentzian 1 / (pi * 0.1) = 3.2 Hz wide.
        assert qc["fwhm_hz"] == pytest.approx(1 / (np.pi * 0.1), rel=0.35)
        assert "SNR" in M.describe_qc(qc)

    def test_quality_is_for_proton_spectra(self):
        ppm, spec = self._spectrum()
        assert M.qc_metrics(ppm, spec, 120.0, "31P") == {}

    def test_the_standard_window_stays_inside_the_data(self):
        ppm = np.linspace(1.29, 8.02, 1024)
        assert M.default_ppm_range("1H", ppm) == (pytest.approx(1.29), 4.2)
        assert M.default_ppm_range("1H", np.linspace(-3, 12, 100)) == (0.2, 4.2)
