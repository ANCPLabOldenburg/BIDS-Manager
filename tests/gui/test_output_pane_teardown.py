"""A scan that comes back after the pane is gone must not raise.

`OutputFsPane` walks the output directory on the global thread pool and hands
the result back through a queued connection. The emit is therefore already
sitting in the event loop when the pane is torn down, and delivering it reaches
a Python wrapper whose C++ tree has been deleted.

This was showing up as a teardown error on an unrelated converter test, which
is the shape this class of bug always takes: it lands on whichever test happens
to run when the garbage collector gets round to the pane. The generation guard
already in the slot does not cover it, because the generation is still current.
The widget is just no longer there.

In normal use the pane outlives the window, so this is mostly a test-suite
symptom. It is still a real crash path: close a project while its output scan
is in flight and the callback lands on a destroyed tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6 import sip

from bidsmgr.gui.output_fs_pane import OutputFsPane

pytestmark = pytest.mark.gui


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    root = tmp_path / "out"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "sub-01" / "anat" / "sub-01_T1w.nii.gz").write_bytes(b"\0" * 16)
    return root


def test_a_scan_landing_after_teardown_is_dropped(qtbot, output_dir) -> None:
    """The exact sequence, forced rather than waited for."""
    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.set_root(output_dir)

    # Capture what the worker would hand back, then destroy the pane and
    # deliver it, which is what the queued connection does.
    generation = pane._scan_generation
    pane.deleteLater()
    pane.setParent(None)
    sip.delete(pane._tree)

    # Must return quietly rather than raising RuntimeError.
    pane._on_scan_done(generation, None)


def test_a_tree_that_dies_MID_render_is_survived(qtbot, output_dir) -> None:
    """The check at the top of the slot is necessary and not sufficient.

    The C++ tree is destroyed with its parent panel, and the parent can be
    collected by Python's cyclic collector, which runs on allocation, which
    means between any two statements in the slot. So the tree can be alive at
    the guard and gone three lines later. Forced here by deleting it from
    inside a call the slot makes.
    """
    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.set_root(output_dir)
    qtbot.waitUntil(lambda: pane._tree.topLevelItemCount() > 0, timeout=4000)

    tree = pane._tree
    original = pane._clear_watcher

    def kill_it():
        original()
        sip.delete(tree)

    pane._clear_watcher = kill_it
    # Passes the top guard, then loses the tree partway through.
    pane._on_scan_done(pane._scan_generation, None)


def test_a_real_error_is_still_raised(qtbot, output_dir) -> None:
    """The catch must not turn a genuine bug into a silent no-op."""
    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.set_root(output_dir)

    def boom(*_a, **_k):
        raise RuntimeError("something actually wrong")

    pane._render_scan = boom
    with pytest.raises(RuntimeError, match="something actually wrong"):
        pane._on_scan_done(pane._scan_generation, None)


def test_the_worker_cannot_emit_into_a_dead_signals_object(
    qtbot, output_dir,
) -> None:
    """The other half, one frame earlier, and it is now a lifetime guarantee
    rather than a guard.

    This test used to parent the signals object to the pane, delete it by
    hand, and assert that ``run()`` did not raise. That pinned the wrong fix
    in place. Guarding the emit cannot work: ``sip.isdeleted`` followed by
    ``emit`` is two steps with a window between them, and no ``except`` clause
    catches the segmentation fault that emitting through a freed QObject
    produces. Destroying a QObject on the GUI thread while a pool thread emits
    its signal is undefined behaviour in Qt, because emission takes the
    sender's connection mutex and that mutex is part of what is being freed.
    It surfaced as a segfault on a loaded Linux runner and never on macOS,
    which is what a timing race looks like.

    So the object is no longer parented to the pane, and the property to
    assert is that the runnable's own reference keeps it alive: the pane can
    go, and the emit is still safe, delivering to a slot PyQt has already
    disconnected.
    """
    import gc

    from bidsmgr.gui.output_fs_pane import _ScanRunnable

    pane = OutputFsPane()
    qtbot.addWidget(pane)
    signals = pane._scan_signals
    runnable = _ScanRunnable(1, output_dir, signals)

    # Destroy the pane the way teardown does, and let the collector run.
    pane.deleteLater()
    pane.setParent(None)
    sip.delete(pane._tree)
    del pane
    gc.collect()

    assert not sip.isdeleted(signals), (
        "the runnable still holds this object, so nothing may have freed it"
    )
    runnable.run()   # must neither raise nor crash


def test_a_real_scan_still_renders(qtbot, output_dir) -> None:
    """The guards must not swallow the ordinary case."""
    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.set_root(output_dir)
    qtbot.waitUntil(lambda: pane._tree.topLevelItemCount() > 0, timeout=4000)
    root_item = pane._tree.topLevelItem(0)
    assert root_item.text(0).startswith("out")
