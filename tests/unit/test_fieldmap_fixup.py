"""Unit tests for ``fixups/fieldmaps.py``.

The fixup turns dcm2niix's fmap multi-output tokens (``_e1`` / ``_e2`` /
``_ph`` / ``_e1_ph`` / ``_e2_ph``) into the canonical BIDS fmap
suffixes, applied to every matching atom (``.nii``, ``.nii.gz``,
``.json``, ``.bval``, ``.bvec``) within each ``fmap/`` directory found
under the staging root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bidsmgr.fixups.fieldmaps import apply_fieldmap_renames, rename_for_fmap_token


# ---------------------------------------------------------------------------
# Pure-function unit tests for the name transform
# ---------------------------------------------------------------------------


class TestRenameForFmapToken:
    @pytest.mark.parametrize(
        "name,expected",
        [
            # Basename already has the BIDS suffix; just strip the token.
            ("sub-001_magnitude1_e1.nii.gz", "sub-001_magnitude1.nii.gz"),
            ("sub-001_magnitude1_e2.nii.gz", "sub-001_magnitude2.nii.gz"),
            ("sub-001_magnitude1_ph.nii.gz", "sub-001_phasediff.nii.gz"),
            # Per-echo phase outputs.
            ("sub-001_magnitude1_e1_ph.nii.gz", "sub-001_phase1.nii.gz"),
            ("sub-001_magnitude1_e2_ph.nii.gz", "sub-001_phase2.nii.gz"),
            # Legacy basename ending in "_fmap" placeholder gets normalised.
            ("sub-001_fmap_e1.nii.gz", "sub-001_magnitude1.nii.gz"),
            ("sub-001_fmap_e2.nii.gz", "sub-001_magnitude2.nii.gz"),
            ("sub-001_fmap_ph.nii.gz", "sub-001_phasediff.nii.gz"),
            # JSON sidecar.
            ("sub-001_magnitude1_e1.json", "sub-001_magnitude1.json"),
            # bval / bvec (rare on fmap but we handle them defensively).
            ("sub-001_magnitude1_e1.bval", "sub-001_magnitude1.bval"),
            ("sub-001_magnitude1_e1.bvec", "sub-001_magnitude1.bvec"),
            # Plain .nii (no .gz)
            ("sub-001_magnitude1_e1.nii", "sub-001_magnitude1.nii"),
            # Multi-entity basename (run, ses, acq).
            (
                "sub-001_ses-pre_acq-fm2_run-1_magnitude1_e2.nii.gz",
                "sub-001_ses-pre_acq-fm2_run-1_magnitude2.nii.gz",
            ),
        ],
    )
    def test_renames_known_tokens(self, name: str, expected: str) -> None:
        assert rename_for_fmap_token(name) == expected

    @pytest.mark.parametrize(
        "name",
        [
            "sub-001_magnitude1.nii.gz",       # already canonical
            "sub-001_magnitude2.nii.gz",
            "sub-001_phasediff.nii.gz",
            "sub-001_T1w.nii.gz",              # no token at all
        ],
    )
    def test_returns_none_for_canonical_or_unrelated(self, name: str) -> None:
        # Files with no dcm2niix token return None (caller skips them).
        # Note: ``apply_fieldmap_renames`` only walks ``fmap/`` dirs, so
        # the regex doesn't need to discriminate between a token in an
        # anat vs fmap context — only fmap files reach it.
        assert rename_for_fmap_token(name) is None


# ---------------------------------------------------------------------------
# Directory-walking integration tests
# ---------------------------------------------------------------------------


class TestApplyFieldmapRenames:
    def _make_fmap_dir(self, base: Path, *files: str) -> Path:
        fmap_dir = base / "fmap"
        fmap_dir.mkdir(parents=True)
        for f in files:
            (fmap_dir / f).write_text("x")
        return fmap_dir

    def test_renames_phasediff_triplet_in_lockstep(self, tmp_path: Path) -> None:
        """The classic phasediff fmap: e1 + e2 + ph, with JSON sidecars."""
        fmap_dir = self._make_fmap_dir(
            tmp_path,
            "sub-001_magnitude1_e1.nii.gz", "sub-001_magnitude1_e1.json",
            "sub-001_magnitude1_e2.nii.gz", "sub-001_magnitude1_e2.json",
            "sub-001_magnitude1_ph.nii.gz", "sub-001_magnitude1_ph.json",
        )

        rename_map = apply_fieldmap_renames(tmp_path)

        # Six renames (3 NIfTIs + 3 JSONs).
        assert len(rename_map) == 6
        # Final filenames in the fmap dir:
        assert sorted(p.name for p in fmap_dir.iterdir()) == sorted([
            "sub-001_magnitude1.nii.gz", "sub-001_magnitude1.json",
            "sub-001_magnitude2.nii.gz", "sub-001_magnitude2.json",
            "sub-001_phasediff.nii.gz", "sub-001_phasediff.json",
        ])

    def test_handles_session_subdir(self, tmp_path: Path) -> None:
        """Both ``fmap/`` and ``ses-*/fmap/`` are walked."""
        ses_dir = tmp_path / "ses-pre"
        ses_dir.mkdir()
        self._make_fmap_dir(
            ses_dir,
            "sub-001_ses-pre_magnitude1_e1.nii.gz",
            "sub-001_ses-pre_magnitude1_e2.nii.gz",
        )

        rename_map = apply_fieldmap_renames(tmp_path)

        assert len(rename_map) == 2
        names = sorted(p.name for p in (ses_dir / "fmap").iterdir())
        assert names == [
            "sub-001_ses-pre_magnitude1.nii.gz",
            "sub-001_ses-pre_magnitude2.nii.gz",
        ]

    def test_no_op_when_no_fmap_dir(self, tmp_path: Path) -> None:
        (tmp_path / "anat").mkdir()
        (tmp_path / "anat" / "sub-001_T1w.nii.gz").write_text("x")
        rename_map = apply_fieldmap_renames(tmp_path)
        assert rename_map == {}

    def test_no_op_when_subject_dir_missing(self, tmp_path: Path) -> None:
        rename_map = apply_fieldmap_renames(tmp_path / "does-not-exist")
        assert rename_map == {}

    def test_skips_already_canonical_files(self, tmp_path: Path) -> None:
        """Files already at their BIDS name are left alone."""
        fmap_dir = self._make_fmap_dir(
            tmp_path,
            "sub-001_magnitude1.nii.gz",
            "sub-001_magnitude2.nii.gz",
            "sub-001_phasediff.nii.gz",
        )
        rename_map = apply_fieldmap_renames(tmp_path)
        assert rename_map == {}
        assert sorted(p.name for p in fmap_dir.iterdir()) == [
            "sub-001_magnitude1.nii.gz",
            "sub-001_magnitude2.nii.gz",
            "sub-001_phasediff.nii.gz",
        ]

    def test_refuses_to_overwrite_existing_target(self, tmp_path: Path) -> None:
        """If both tokened and canonical files are present, keep the canonical one."""
        fmap_dir = self._make_fmap_dir(
            tmp_path,
            "sub-001_magnitude1_e1.nii.gz",  # would rename to magnitude1.nii.gz
            "sub-001_magnitude1.nii.gz",     # already exists at the target
        )
        rename_map = apply_fieldmap_renames(tmp_path)
        # No rename happened.
        assert rename_map == {}
        # Both files still present.
        assert sorted(p.name for p in fmap_dir.iterdir()) == [
            "sub-001_magnitude1.nii.gz",
            "sub-001_magnitude1_e1.nii.gz",
        ]

    def test_per_echo_phase_outputs(self, tmp_path: Path) -> None:
        """Two-phase fmap: ``_e1_ph`` → ``phase1``, ``_e2_ph`` → ``phase2``."""
        fmap_dir = self._make_fmap_dir(
            tmp_path,
            "sub-001_magnitude1_e1.nii.gz",
            "sub-001_magnitude1_e2.nii.gz",
            "sub-001_magnitude1_e1_ph.nii.gz",
            "sub-001_magnitude1_e2_ph.nii.gz",
        )
        apply_fieldmap_renames(tmp_path)
        assert sorted(p.name for p in fmap_dir.iterdir()) == sorted([
            "sub-001_magnitude1.nii.gz",
            "sub-001_magnitude2.nii.gz",
            "sub-001_phase1.nii.gz",
            "sub-001_phase2.nii.gz",
        ])
