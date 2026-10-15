"""Diffusion series: shells from b-values, and moving between them."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bidsmgr.viz.bids import shells_of
from bidsmgr.viz.data.volume import array_volume
from bidsmgr.viz.scene import Scene, VolumeLayer
from bidsmgr.viz.store import SceneStore

# b0, three at 1000 (as scanners write them), b0, two at 2000.
BVALS = np.array([5.0, 995.0, 1000.0, 1005.0, 0.0, 2000.0, 1995.0])


def _store(tmp_path: Path, bvals=BVALS) -> SceneStore:
    src = array_volume(np.zeros((2, 2, 2, len(bvals)), np.float32), np.eye(4),
                       path=tmp_path / "dwi", name="dwi")
    src.bvals = np.asarray(bvals, dtype=float)
    store = SceneStore()
    scene = Scene()
    scene.layers = [VolumeLayer(id="base", source="vol0")]
    store.replace_scene(scene)
    store.sources = {"vol0": src}
    return store


def _frame(store) -> int:
    return store.scene.layers[0].frame


def test_shells_are_rounded_b_values():
    assert list(shells_of(BVALS)) == [0, 1000, 1000, 1000, 0, 2000, 2000]


def test_same_shell_steps_through_its_volumes_and_wraps(tmp_path):
    store = _store(tmp_path)
    store.run("frame.set", frame=1)
    store.run("frame.same_shell", n=1)
    assert _frame(store) == 2
    store.run("frame.same_shell", n=1)
    store.run("frame.same_shell", n=1)
    assert _frame(store) == 1, "wrapped back to the first b=1000"
    store.run("frame.same_shell", n=-1)
    assert _frame(store) == 3


def test_shell_goes_to_the_first_volume_of_the_next_b_value(tmp_path):
    store = _store(tmp_path)
    store.run("frame.set", frame=0)
    store.run("frame.shell", n=1)
    assert _frame(store) == 1, "the first b=1000"
    store.run("frame.shell", n=1)
    assert _frame(store) == 5, "the first b=2000"
    store.run("frame.shell", n=1)
    assert _frame(store) == 0, "wrapped to b=0"
    store.run("frame.shell", n=-1)
    assert _frame(store) == 5


def test_without_b_values_nothing_moves(tmp_path):
    store = _store(tmp_path)
    store.sources["vol0"].bvals = None
    assert store.run("frame.shell", n=1) == frozenset()
    assert store.run("frame.same_shell", n=1) == frozenset()


def test_a_shell_of_one_volume_stays(tmp_path):
    store = _store(tmp_path, bvals=[0.0, 1000.0, 1000.0])
    store.run("frame.set", frame=0)
    assert store.run("frame.same_shell", n=1) == frozenset()
