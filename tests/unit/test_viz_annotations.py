"""Bad segments and bad channels: marked in the viewer, written to BIDS.

``bidsmgr.viz.commands.signal`` (the annotation commands), the recording's
own bad segments (``viz.data.signal``) and ``bidsmgr.editor.annotations``
(one undoable write of a recording's review). Qt-free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bidsmgr.editor.annotations import events_table, save_review
from bidsmgr.project.operations import read_log, undo_last
from bidsmgr.viz.commands.signal import bad_label, bad_spans, spans_changed
from bidsmgr.viz.data.events import read_events_tsv
from bidsmgr.viz.data.signal import SignalSource, open_meeg
from bidsmgr.viz.scene import Scene, SignalLayer, SourceRef, Span, TracesState
from bidsmgr.viz.store import SceneStore
from tests.fixtures.signals import write_fif

mne = pytest.importorskip("mne")


def _store(seconds: float = 60.0, sfreq: float = 100.0) -> SceneStore:
    n = int(seconds * sfreq)
    info = mne.create_info(["a", "b"], sfreq, ["eeg", "eeg"], verbose=False)
    raw = mne.io.RawArray(np.zeros((2, n)), info, verbose=False)
    src = SignalSource(path=Path("sub-01_task-x_eeg.edf"), raw=raw, kind="meeg")
    store = SceneStore()
    store.sources = {"sig0": src}
    scene = Scene()
    scene.sources = {"sig0": SourceRef(id="sig0", path=str(src.path), kind="signal")}
    scene.layers = [SignalLayer(id="sig", source="sig0", name=src.path.name)]
    scene.traces = TracesState()
    store.replace_scene(scene)
    return store


class TestCommands:
    def test_labels_are_bad_labels(self):
        assert bad_label("muscle") == "BAD_muscle"
        assert bad_label("bad_eye") == "BAD_eye"
        assert bad_label("BAD_") == "BAD_"

    def test_add_sorts_selects_and_clamps(self):
        store = _store()
        store.run("annotate.add", onset=30.0, duration=2.0)
        store.run("annotate.add", onset=10.0, duration=-3.0, label="muscle")
        spans = bad_spans(store)
        assert [(s.onset, s.duration, s.label) for s in spans] == [
            (7.0, 3.0, "BAD_muscle"), (30.0, 2.0, "BAD_")]
        assert store.scene.traces.selected_span == 0
        store.run("annotate.add", onset=58.0, duration=10.0)
        end = store.sources["sig0"].duration
        last = bad_spans(store)[-1]
        assert last.onset + last.duration == pytest.approx(end), "not past the end"
        with pytest.raises(ValueError):
            store.run("annotate.add", onset=5.0, duration=0.0)

    def test_edit_delete_and_undo(self):
        store = _store()
        store.run("annotate.add", onset=10.0, duration=2.0)
        store.run("annotate.set", onset=12.0, label="eye")
        assert bad_spans(store) == [Span(onset=12.0, duration=2.0, label="BAD_eye")]
        store.run("annotate.remove")
        assert bad_spans(store) == []
        store.undo()
        assert bad_spans(store) == [Span(onset=12.0, duration=2.0, label="BAD_eye")]

    def test_many_at_once_merge_where_they_touch(self):
        store = _store()
        store.run("annotate.add_many", segments=[(2.0, 2.0), (4.0, 2.0), (10.0, 1.0)],
                  label="noise")
        assert [(s.onset, s.duration) for s in bad_spans(store)] == [(2.0, 4.0), (10.0, 1.0)]

    def test_changed_against_the_recording(self):
        store = _store()
        assert not spans_changed(store)
        store.run("annotate.add", onset=1.0, duration=1.0)
        assert spans_changed(store)
        store.run("annotate.remove", index=0)
        assert not spans_changed(store), "back to what the recording says"

    def test_a_reset_of_the_view_keeps_them(self):
        store = _store()
        store.run("annotate.add", onset=1.0, duration=1.0)
        store.run("traces.reset")
        assert len(bad_spans(store)) == 1


class TestTheRecordingsOwn:
    def test_events_tsv_bad_rows_are_the_segments(self, tmp_path):
        path = write_fif(tmp_path, "sub-01_task-x_eeg.fif", sfreq=100.0, seconds=20.0)
        (tmp_path / "sub-01_task-x_events.tsv").write_bytes(
            "﻿onset\tduration\ttrial_type\n1.0\t0\tgo\n5.0\t2.5\tBAD_muscle\n".encode())
        src = open_meeg(path)
        assert [(e.onset, e.duration, e.label) for e in src.bad_spans] == [
            (5.0, 2.5, "BAD_muscle")]
        assert [e.kind for e in read_events_tsv(tmp_path / "sub-01_task-x_events.tsv")] == [
            "", "bad"]

    def test_without_a_table_the_files_annotations(self, tmp_path):
        path = write_fif(tmp_path, "sub-01_task-x_eeg.fif", sfreq=100.0, seconds=20.0)
        raw = mne.io.read_raw_fif(str(path), preload=True, verbose=False)
        raw.set_annotations(mne.Annotations([3.0], [1.0], ["BAD_jump"]))
        raw.save(str(path), overwrite=True, verbose=False)
        src = open_meeg(path)
        assert [(e.onset, e.duration, e.label) for e in src.bad_spans] == [(3.0, 1.0, "BAD_jump")]


class TestWriting:
    def test_bad_rows_replaced_the_rest_kept(self, tmp_path):
        tsv = tmp_path / "sub-01_task-x_events.tsv"
        tsv.write_bytes("﻿onset\tduration\ttrial_type\tsample\n"
                        "1.0\t0\tgo\t100\n5.0\t2.5\tBAD_old\t500\n9.0\t0\tstop\t900\n"
                        .encode("utf-8"))
        text, changed = events_table(tsv, [Span(onset=3.25, duration=1.0, label="BAD_eye")],
                                     sfreq=100.0)
        assert changed == 2
        assert text.startswith("﻿onset\tduration\ttrial_type\tsample\n")
        rows = text.splitlines()[1:]
        assert rows == ["1.0\t0\tgo\t100", "3.25\t1\tBAD_eye\t325", "9.0\t0\tstop\t900"]

    def test_a_run_without_a_table_gets_one(self, tmp_path):
        text, changed = events_table(tmp_path / "none_events.tsv",
                                     [Span(onset=1.0, duration=0.5, label="BAD_")])
        assert changed == 1
        assert text == "onset\tduration\ttrial_type\n1\t0.5\tBAD_\n"

    def test_one_undoable_operation_for_both(self, tmp_path):
        eeg = tmp_path / "sub-01" / "eeg"
        eeg.mkdir(parents=True)
        channels = eeg / "sub-01_task-x_channels.tsv"
        channels.write_text("name\ttype\tstatus\nFz\tEEG\tgood\nCz\tEEG\tgood\n")
        events = eeg / "sub-01_task-x_events.tsv"
        result = save_review(tmp_path, recording="sub-01_task-x_eeg.edf",
                             channels_tsv=channels, bads={"Cz"}, events_tsv=events,
                             spans=[Span(onset=2.0, duration=1.0, label="BAD_")], sfreq=100.0)
        assert result == {"channels": 1, "segments": 1}
        assert "Cz\tEEG\tbad" in channels.read_text()
        assert "BAD_" in events.read_text()
        log = read_log(tmp_path)
        assert len(log) == 1
        assert log[-1]["label"] == "Review of sub-01_task-x_eeg.edf: 1 bad channel, 1 bad segment"
        undo_last(tmp_path)
        assert "Cz\tEEG\tgood" in channels.read_text()
        assert not events.exists(), "the table it created goes with the undo"

    def test_nothing_changed_nothing_written(self, tmp_path):
        channels = tmp_path / "c_channels.tsv"
        channels.write_text("name\tstatus\nFz\tgood\n")
        assert save_review(tmp_path, recording="x", channels_tsv=channels, bads=set()) == {
            "channels": 0, "segments": 0}
        assert read_log(tmp_path) == []


def test_the_same_segment_twice_is_one():
    store = _store()
    store.run("annotate.add", onset=5.0, duration=1.0)
    store.run("annotate.add", onset=5.0, duration=1.0)
    store.run("annotate.add_many", segments=[(5.0, 1.0)])
    assert len(bad_spans(store)) == 1
