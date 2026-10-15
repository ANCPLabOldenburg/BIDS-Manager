"""Rendering without a window, and the command line that reproduces a view.

``bidsmgr.viz.render2d`` (the slice composition the canvas draws, as pixels
and PNG files, with no Qt) and ``bidsmgr.viz.reproduce`` (the
``bidsmgr-view`` call or Python that gives the same view back).
"""

from __future__ import annotations

import struct
import subprocess
import sys
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from bidsmgr.viz import render2d, reproduce


def _decode_png(data: bytes) -> np.ndarray:
    """The pixels of a PNG that ``encode_png`` wrote (RGBA, filter 0)."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, size = 8, b"", None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if tag == b"IHDR":
            size = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + length
    cols, rows = size
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(rows, cols * 4 + 1)
    assert not raw[:, 0].any()
    return raw[:, 1:].reshape(rows, cols, 4)


@pytest.fixture()
def t1(tmp_path: Path) -> Path:
    """A 40 x 48 x 20 image of 1 x 1 x 3 mm voxels: a bright cube in the middle."""
    data = np.zeros((40, 48, 20), dtype=np.float32)
    data[10:30, 12:36, 5:15] = 100.0
    data += np.linspace(0, 10, 20)[None, None, :]
    path = tmp_path / "sub-01_T1w.nii.gz"
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 3.0, 1.0])), str(path))
    return path


@pytest.fixture()
def stat(tmp_path: Path) -> Path:
    data = np.zeros((40, 48, 20), dtype=np.float32)
    data[18:22, 20:28, 8:12] = 8.0
    path = tmp_path / "sub-01_stat.nii.gz"
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 3.0, 1.0])), str(path))
    return path


class TestRendering:
    def test_pixels_are_square_millimetres(self, t1):
        store = render2d.open_store(t1)
        axial = render2d.panel(store, "axial", px_per_mm=2.0)
        coronal = render2d.panel(store, "coronal", px_per_mm=2.0)
        assert axial.shape[:2] == (96, 80)            # 48 x 40 mm
        assert coronal.shape[:2] == (120, 80), "60 mm of 3 mm slices, not 20 pixels"

    def test_planes_side_by_side_at_one_height(self, t1):
        store = render2d.open_store(t1)
        rgba = render2d.render(store, planes=("sagittal", "coronal", "axial"), height_px=200)
        assert rgba.shape[0] == 200 and rgba.dtype == np.uint8
        assert rgba.shape[1] > 3 * 100

    def test_the_look_is_the_layers(self, t1):
        store = render2d.open_store(t1)
        grey = render2d.panel(store, "axial")
        assert np.allclose(grey[..., 0], grey[..., 1]), "grey by default"
        store.run("layer.set", colormap="hot")
        hot = render2d.panel(store, "axial")
        assert (hot[..., 0] != hot[..., 2]).any(), "hot has colour where grey had none"

    def test_an_overlay_is_drawn_over(self, t1, stat):
        bare = render2d.panel(render2d.open_store(t1), "axial")
        both = render2d.panel(render2d.open_store(t1, overlays=[stat]), "axial")
        changed = np.any(bare != both, axis=-1)
        assert changed.any()
        ys, xs = np.nonzero(changed)
        assert 36 <= xs.mean() <= 44 and 44 <= ys.mean() <= 52, "the blob, where it is"

    def test_the_crosshair_when_asked(self, t1):
        store = render2d.open_store(t1)
        plain = render2d.panel(store, "axial")
        lines = render2d.panel(store, "axial", crosshair=True)
        hit = np.all(lines == render2d.CROSSHAIR, axis=-1)
        assert hit.any() and not np.all(plain == render2d.CROSSHAIR, axis=-1).any()

    def test_an_unknown_plane_is_refused(self, t1):
        with pytest.raises(ValueError):
            render2d.render(render2d.open_store(t1), planes=("oblique",))

    def test_png_round_trip(self, tmp_path):
        rgba = np.random.default_rng(0).integers(0, 255, (7, 5, 4), dtype=np.uint8)
        path = render2d.write_png(tmp_path / "x.png", rgba)
        assert np.array_equal(_decode_png(path.read_bytes()), rgba)


def test_no_qt_is_needed(t1, tmp_path):
    """The library's render path and ``bidsmgr-view --render`` with PyQt6
    unimportable: a server with no display has no Qt either."""
    out = tmp_path / "out.png"
    code = (
        "import sys; sys.modules['PyQt6'] = None\n"
        "from bidsmgr.cli.view import main\n"
        f"raise SystemExit(main([{str(t1)!r}, '--planes', 'axial', 'coronal', "
        f"'--colormap', 'hot', '--render', {str(out)!r}]))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr
    rgba = _decode_png(out.read_bytes())
    assert rgba.shape[0] == 120                     # the coronal plane's 60 mm


class TestReproducing:
    def _view(self, t1, stat):
        store = render2d.open_store(t1, overlays=[stat])
        store.run("layer.set", layer="base", colormap="gray", window=(5.0, 80.0), gamma=1.4)
        store.run("layer.set", layer="overlay1", window=(1.0, 6.0), opacity=0.7)
        store.run("cursor.set_world", x=12.0, y=20.0, z=27.0)
        store.run("view.mode", mode="multi")
        return store

    def test_the_rendered_command_gives_the_same_pixels(self, t1, stat, tmp_path):
        from bidsmgr.cli.view import main

        store = self._view(t1, stat)
        out = tmp_path / "again.png"
        text = reproduce.shell(store, render=str(out), style="posix")
        assert main(reproduce.parse_shell(text)) == 0
        planes = reproduce.plan(store)["planes"]
        assert planes == ("sagittal", "coronal", "axial")
        assert np.array_equal(_decode_png(out.read_bytes()),
                              render2d.render(store, planes=planes))

    def test_the_python_gives_the_same_pixels(self, t1, stat, tmp_path, monkeypatch):
        store = self._view(t1, stat)
        monkeypatch.chdir(tmp_path)
        exec(reproduce.python(store, out="py.png"), {})
        assert np.array_equal(_decode_png((tmp_path / "py.png").read_bytes()),
                              render2d.render(store, planes=reproduce.plan(store)["planes"]))

    def test_the_open_command_says_the_layout(self, t1, stat):
        text = reproduce.shell(self._view(t1, stat), style="posix")
        assert "--layout multi" in text and "--overlay" in text
        assert "window=[5.0,80.0]" in text and "cursor.set_world" in text

    def test_a_computed_layer_is_named_not_dropped(self, t1, stat):
        store = self._view(t1, stat)
        store.scene.sources["ovl1"].path = ""
        text = reproduce.shell(store, style="posix")
        assert text.startswith("# Not reproduced: sub-01_stat.nii.gz")

    def test_windows_gets_one_line_it_can_read(self, t1, stat):
        """cmd and PowerShell read no backslash continuation and no single
        quotes; in PowerShell a bare comma makes an array."""
        text = reproduce.shell(self._view(t1, stat), render="C:\\out dir\\fig.png",
                               style="windows")
        assert "\n" not in text and "'" not in text
        assert '"window=[5.0,80.0]"' in text
        assert '"C:\\out dir\\fig.png"' in text
        assert " x=12.0 " in text, "a plain word stays bare"
