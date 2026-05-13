"""Unit tests for ``converter/backends/mne_bids.py``.

mne / mne-bids are heavy and not what we're testing — we patch the lazy
imports inside ``MneBidsBackend.convert`` so each test is fast and
isolated. A ``_FakeBIDSPath`` and a fake ``write_raw_bids`` write the
files mne-bids would write, then the backend's output collector exercises
the real path-walking logic.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Sequence

import pytest

from bidsmgr.converter import default_backends, dispatch
from bidsmgr.converter.backends.mne_bids import MneBidsBackend, _coerce_run
from bidsmgr.converter.types import ConvertTask


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_source(tmp_path: Path, name: str = "rec.edf") -> Path:
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    fp = src / name
    fp.write_bytes(b"x")
    return fp


def _make_task(
    tmp_path: Path,
    *,
    datatype: str = "eeg",
    suffix: str = "eeg",
    subject: str = "001",
    session: str | None = None,
    task_name: str = "rest",
    run: str | None = None,
    basename: str = "sub-001_task-rest_eeg",
    sources: Sequence[Path] | None = None,
) -> ConvertTask:
    if sources is None:
        sources = (_make_source(tmp_path),)
    entities = {"task": task_name}
    if run:
        entities["run"] = run
    return ConvertTask(
        row_id="r1",
        series_uid="",  # blank for EEG/MEG rows
        source_files=tuple(sources),
        dataset="study",
        bids_root=tmp_path / "bids" / "study",
        subject=subject,
        session=session,
        datatype=datatype,
        suffix=suffix,
        entities=entities,
        basename=basename,
        expected_outputs=(".edf", ".json"),
    )


class _FakeBIDSPath:
    """Stand-in for ``mne_bids.BIDSPath``.

    We don't need real BIDS validation — just record the kwargs and
    forward them so the fake ``write_raw_bids`` knows where to write.
    """

    def __init__(self, *, subject, session, task, run, datatype, root):
        self.subject = subject
        self.session = session
        self.task = task
        self.run = run
        self.datatype = datatype
        self.root = root


def _patch_mne_bids(
    monkeypatch,
    *,
    write_extensions: Sequence[str] = (".edf", ".json", "_channels.tsv"),
    raw_exception: Exception | None = None,
    write_exception: Exception | None = None,
) -> None:
    """Install fake ``mne`` and ``mne_bids`` modules in ``sys.modules``.

    The fake ``write_raw_bids`` mimics the real one's filesystem effects:
    creates ``<root>/sub-<X>/[ses-<Y>/]<datatype>/sub-<X>..._<datatype>.<ext>``
    plus channels.tsv etc.
    """

    class _FakeRaw:
        """Minimal stand-in for an mne ``Raw`` object.

        Carries enough attributes for the backend's
        ``raw.info["line_freq"]`` injection and ``raw.set_montage(...)``
        calls to no-op cleanly, so each test can focus on the logic
        the backend itself adds (line_freq default, montage application,
        write_raw_bids dispatch).
        """

        def __init__(self) -> None:
            self.info: dict = {"line_freq": None}

        def set_montage(self, *args, **kwargs) -> None:
            self._montage_set = True

    def fake_read_raw(path, preload=False, verbose=None):
        if raw_exception is not None:
            raise raw_exception
        return _FakeRaw()

    def fake_write_raw_bids(raw, bids_path, *, overwrite=False, format="auto", verbose=None):
        if write_exception is not None:
            raise write_exception
        sub_dir = Path(bids_path.root) / f"sub-{bids_path.subject}"
        if bids_path.session:
            sub_dir = sub_dir / f"ses-{bids_path.session}"
        out_dir = sub_dir / bids_path.datatype
        out_dir.mkdir(parents=True, exist_ok=True)
        # Build the BIDS basename (subject + session + task + run).
        parts = [f"sub-{bids_path.subject}"]
        if bids_path.session:
            parts.append(f"ses-{bids_path.session}")
        if bids_path.task:
            parts.append(f"task-{bids_path.task}")
        if bids_path.run is not None:
            parts.append(f"run-{int(bids_path.run)}")
        parts.append(bids_path.datatype)
        stem = "_".join(parts)
        for ext in write_extensions:
            target = out_dir / f"{stem}{ext}"
            target.write_text("{}" if ext.endswith(".json") else "")
        return object()

    fake_mne = types.ModuleType("mne")
    fake_mne_io = types.ModuleType("mne.io")
    fake_mne_io.read_raw = fake_read_raw
    fake_mne.io = fake_mne_io
    fake_mne_bids = types.ModuleType("mne_bids")
    fake_mne_bids.BIDSPath = _FakeBIDSPath
    fake_mne_bids.write_raw_bids = fake_write_raw_bids

    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    monkeypatch.setitem(sys.modules, "mne.io", fake_mne_io)
    monkeypatch.setitem(sys.modules, "mne_bids", fake_mne_bids)


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    @pytest.mark.parametrize("datatype", ["eeg", "meg", "ieeg", "nirs"])
    def test_handles_supported_datatypes(self, tmp_path: Path, datatype: str) -> None:
        b = MneBidsBackend()
        task = _make_task(tmp_path, datatype=datatype, suffix=datatype)
        assert b.can_handle(task) is True

    def test_rejects_mri_datatype(self, tmp_path: Path) -> None:
        b = MneBidsBackend()
        task = _make_task(tmp_path, datatype="anat", suffix="T1w",
                          basename="sub-001_T1w")
        assert b.can_handle(task) is False

    def test_rejects_empty_source_files(self, tmp_path: Path) -> None:
        b = MneBidsBackend()
        task = _make_task(tmp_path).model_copy(update={"source_files": ()})
        assert b.can_handle(task) is False


# ---------------------------------------------------------------------------
# Registry dispatch (with all three backends now)
# ---------------------------------------------------------------------------


class TestRegistryDispatch:
    def test_priority_order(self) -> None:
        backends = default_backends()
        assert [b.name for b in backends] == [
            "physio_dcm", "mne_bids", "dcm2niix_direct",
        ]

    def test_eeg_routes_to_mne_bids(self, tmp_path: Path) -> None:
        backends = default_backends()
        task = _make_task(tmp_path, datatype="eeg", suffix="eeg")
        assert dispatch(backends, task).name == "mne_bids"

    def test_meg_routes_to_mne_bids(self, tmp_path: Path) -> None:
        backends = default_backends()
        task = _make_task(tmp_path, datatype="meg", suffix="meg",
                          basename="sub-001_task-rest_meg")
        assert dispatch(backends, task).name == "mne_bids"

    def test_dcm2niix_declines_eeg(self, tmp_path: Path) -> None:
        from bidsmgr.converter.backends.dcm2niix_direct import Dcm2niixDirect
        b = Dcm2niixDirect()
        task = _make_task(tmp_path, datatype="eeg", suffix="eeg")
        assert b.can_handle(task) is False


# ---------------------------------------------------------------------------
# convert — happy path
# ---------------------------------------------------------------------------


class TestConvertSuccess:
    def test_writes_data_and_sidecars_no_session(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Per-subject staging is ``<tmp_bidsmgr>/sub-001/`` and mne-bids
        writes its own sub-001/eeg/... underneath the parent. We walk
        up so the final tree is ``<tmp_bidsmgr>/sub-001/eeg/...`` (not
        ``sub-001/sub-001/eeg/...``)."""
        b = MneBidsBackend()
        task = _make_task(tmp_path)
        _patch_mne_bids(monkeypatch)

        # Mimic the orchestrator's per-subject staging dir.
        staging = tmp_path / ".tmp_bidsmgr" / "sub-001"
        staging.mkdir(parents=True)
        result = b.convert(task, staging)

        assert result.success is True, result.error
        names = sorted(p.name for p in result.staged_files)
        assert "sub-001_task-rest_eeg.edf" in names
        assert "sub-001_task-rest_eeg.json" in names
        assert "sub-001_task-rest_eeg_channels.tsv" in names
        # Files land directly inside <staging>/eeg/ — no doubled sub-001.
        for p in result.staged_files:
            assert p.parent == staging / "eeg"

    def test_session_layout(self, tmp_path: Path, monkeypatch) -> None:
        b = MneBidsBackend()
        task = _make_task(tmp_path, session="pre",
                          basename="sub-001_ses-pre_task-rest_eeg")
        _patch_mne_bids(monkeypatch)
        # Orchestrator passes the per-session staging when task has a session.
        staging = tmp_path / ".tmp_bidsmgr" / "sub-001" / "ses-pre"
        staging.mkdir(parents=True)
        result = b.convert(task, staging)
        assert result.success is True
        for p in result.staged_files:
            assert p.parent == staging / "eeg"

    def test_run_entity_is_emitted(self, tmp_path: Path, monkeypatch) -> None:
        """A non-None run entity flows through to mne_bids' BIDSPath."""
        b = MneBidsBackend()
        task = _make_task(tmp_path, run="2")
        _patch_mne_bids(monkeypatch)
        staging = tmp_path / ".tmp_bidsmgr" / "sub-001"
        staging.mkdir(parents=True)
        result = b.convert(task, staging)
        assert result.success is True
        names = [p.name for p in result.staged_files]
        assert any("run-2" in n for n in names)

    def test_meg_datatype_lands_in_meg_dir(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        b = MneBidsBackend()
        task = _make_task(tmp_path, datatype="meg", suffix="meg",
                          basename="sub-001_task-rest_meg")
        _patch_mne_bids(monkeypatch, write_extensions=(".fif", ".json"))
        staging = tmp_path / ".tmp_bidsmgr" / "sub-001"
        staging.mkdir(parents=True)
        result = b.convert(task, staging)
        assert result.success is True
        for p in result.staged_files:
            assert p.parent == staging / "meg"

    def test_meg_datatype_skips_standard_montage(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Standard 10-20 montages are an EEG / iEEG / NIRS concept;
        applying one to MEG-Neuromag data routinely collides on channel
        rename. The backend must skip ``_apply_standard_montage``
        entirely for MEG, even when the task or backend default carries
        a montage name.
        """
        calls: list = []
        # Backend-level default carries the EEG montage; the task itself
        # doesn't set one. The MEG-datatype guard in the backend must
        # short-circuit before _apply_standard_montage is reached.
        b = MneBidsBackend(montage="standard_1005")
        task = _make_task(
            tmp_path,
            datatype="meg",
            suffix="meg",
            basename="sub-001_task-rest_meg",
        )
        _patch_mne_bids(monkeypatch, write_extensions=(".fif", ".json"))

        # Spy on _apply_standard_montage — it must NOT be called for MEG.
        from bidsmgr.converter.backends import mne_bids as backend_mod
        original = backend_mod._apply_standard_montage
        def _spy(*a, **kw):
            calls.append((a, kw))
            return original(*a, **kw)
        monkeypatch.setattr(backend_mod, "_apply_standard_montage", _spy)

        staging = tmp_path / ".tmp_bidsmgr" / "sub-001"
        staging.mkdir(parents=True)
        result = b.convert(task, staging)
        assert result.success is True
        assert calls == [], (
            "MEG datatype should not invoke standard-montage logic"
        )


# ---------------------------------------------------------------------------
# convert — failure paths
# ---------------------------------------------------------------------------


class TestConvertFailure:
    def test_missing_source_is_failure(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        b = MneBidsBackend()
        ghost = tmp_path / "src" / "missing.edf"
        task = _make_task(tmp_path).model_copy(update={"source_files": (ghost,)})
        _patch_mne_bids(monkeypatch)
        result = b.convert(task, tmp_path / "staging")
        assert result.success is False
        assert "source not found" in (result.error or "")

    def test_read_raw_exception_is_recorded(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        b = MneBidsBackend()
        task = _make_task(tmp_path)
        _patch_mne_bids(
            monkeypatch,
            raw_exception=ValueError("malformed EDF header"),
        )
        staging = tmp_path / "staging"
        staging.mkdir()
        result = b.convert(task, staging)
        assert result.success is False
        assert "mne.io.read_raw failed" in (result.error or "")
        assert "malformed EDF header" in (result.error or "")

    def test_write_raw_bids_exception_is_recorded(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        b = MneBidsBackend()
        task = _make_task(tmp_path)
        _patch_mne_bids(
            monkeypatch,
            write_exception=RuntimeError("non-BIDS task token"),
        )
        staging = tmp_path / "staging"
        staging.mkdir()
        result = b.convert(task, staging)
        assert result.success is False
        assert "write_raw_bids failed" in (result.error or "")

    def test_no_outputs_is_failure(self, tmp_path: Path, monkeypatch) -> None:
        b = MneBidsBackend()
        task = _make_task(tmp_path)
        _patch_mne_bids(monkeypatch, write_extensions=())  # writes nothing
        staging = tmp_path / "staging"
        staging.mkdir()
        result = b.convert(task, staging)
        assert result.success is False
        # Either "no output dir" or "produced no output files" is acceptable.
        assert (
            "no eeg output dir" in (result.error or "").lower()
            or "produced no output files" in (result.error or "")
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestApplyStandardMontage:
    """Channel-name normalisation for montage matching.

    PhysioNet pads EDF channel names to 16 chars with trailing dots
    (``Fc5.``, ``C5..``); other vendors pad with spaces. The backend
    strips non-alphanumerics in place before applying the montage so
    real coordinates land in ``electrodes.tsv`` instead of ``n/a``.
    """

    def test_strips_trailing_dots_then_applies_montage(
        self, monkeypatch,
    ) -> None:
        from bidsmgr.converter.backends.mne_bids import (
            _apply_standard_montage,
        )

        class _FakeRaw:
            def __init__(self, names):
                self.ch_names = list(names)
                self.set_montage_called_with = None

            def rename_channels(self, mapping, **kwargs):
                self.ch_names = [mapping.get(c, c) for c in self.ch_names]
                self._renames = mapping

            def set_montage(self, montage, **kwargs):
                self.set_montage_called_with = montage

        # Monkey-patch mne.channels.make_standard_montage to a sentinel.
        import sys
        import types
        fake_mne = types.ModuleType("mne")
        fake_mne.channels = types.SimpleNamespace(
            make_standard_montage=lambda name: f"montage:{name}",
        )
        monkeypatch.setitem(sys.modules, "mne", fake_mne)

        raw = _FakeRaw(["Fc5.", "Fc3.", "C5..", "Cz", "Iz"])
        _apply_standard_montage(raw, "standard_1005", source=Path("/x.edf"))

        # Trailing dots stripped.
        assert raw.ch_names == ["Fc5", "Fc3", "C5", "Cz", "Iz"]
        # Original names without dots (Cz, Iz) untouched in the rename map.
        assert "Cz" not in raw._renames
        # Montage set with the resolved object.
        assert raw.set_montage_called_with == "montage:standard_1005"

    def test_collision_skips_offending_rename(self, monkeypatch) -> None:
        """If stripping non-alphanumerics would collide with an existing
        channel name, the rename for that one channel is skipped — the
        rest still go through, and mne is never asked to do a
        duplicate-introducing rename.
        """
        from bidsmgr.converter.backends.mne_bids import (
            _apply_standard_montage,
        )

        class _FakeRaw:
            def __init__(self, names):
                self.ch_names = list(names)

            def rename_channels(self, mapping, **kwargs):
                self._renames = mapping
                self.ch_names = [mapping.get(c, c) for c in self.ch_names]

            def set_montage(self, montage, **kwargs):
                self._montage = montage

        import sys
        import types
        fake_mne = types.ModuleType("mne")
        fake_mne.channels = types.SimpleNamespace(
            make_standard_montage=lambda name: f"montage:{name}",
        )
        monkeypatch.setitem(sys.modules, "mne", fake_mne)

        # ``MEG 0113`` stripped → ``MEG0113`` collides with the existing
        # ``MEG0113`` (no space). The renamer must skip the offending
        # entry instead of crashing.
        raw = _FakeRaw(["MEG 0113", "MEG0113", "MEG 0112"])
        _apply_standard_montage(raw, "standard_1005", source=Path("/x.fif"))

        # ``MEG 0113`` keeps its original name; ``MEG 0112`` gets stripped.
        assert "MEG 0113" in raw.ch_names
        assert "MEG0113" in raw.ch_names
        assert "MEG0112" in raw.ch_names

    def test_unknown_montage_logs_and_no_ops(self, monkeypatch) -> None:
        from bidsmgr.converter.backends.mne_bids import (
            _apply_standard_montage,
        )

        class _FakeRaw:
            ch_names = ["Fp1", "Fp2"]

            def rename_channels(self, *args, **kwargs):
                self._renamed = True

            def set_montage(self, *args, **kwargs):
                self._montage_set = True

        import sys
        import types
        fake_mne = types.ModuleType("mne")

        def _raise(name):
            raise ValueError(f"unknown montage {name!r}")

        fake_mne.channels = types.SimpleNamespace(make_standard_montage=_raise)
        monkeypatch.setitem(sys.modules, "mne", fake_mne)

        raw = _FakeRaw()
        # Must not raise.
        _apply_standard_montage(raw, "no_such_montage", source=Path("/x.edf"))
        assert not getattr(raw, "_montage_set", False)


class TestCoerceRun:
    @pytest.mark.parametrize("value,expected", [
        ("1", 1),
        ("2", 2),
        ("01", 1),
        (None, None),
        ("", None),
        ("nan", None),
        ("n/a", None),
        ("None", None),
        (3, 3),
        (3.0, 3),
    ])
    def test_run_coercion(self, value, expected) -> None:
        assert _coerce_run(value) == expected
