"""``bidsmgr-view``: the viewer standing on its own.

If the viewer runs with no Editor around it, the library's boundary is
clean. The screenshot path is what a shell loop of QC figures uses, so it is
run end to end here, in process.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from bidsmgr.cli import view as cli  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    from bidsmgr.gui.viz.canvases import render

    monkeypatch.setattr(render, "gpu_available", lambda: False)


@pytest.fixture
def image(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps({"Name": "t"}))
    path = root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    data = np.arange(20 * 22 * 18, dtype=np.float32).reshape(20, 22, 18)
    nib.save(nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0])), str(path))
    return path


def test_the_dataset_root_is_found_from_the_file(image) -> None:
    assert cli.find_root(image) == image.parents[2]
    assert cli.find_root(Path("/") / "nowhere.nii.gz") is None


def test_command_parameters_are_read_as_json_where_they_parse() -> None:
    assert cli._parse_params(["n=3", "window=[0,100]", "value=true", "text=A 0 10"]) == {
        "n": 3, "window": [0, 100], "value": True, "text": "A 0 10"}
    with pytest.raises(ValueError):
        cli._parse_params(["no-equals-sign"])


def test_a_missing_file_is_refused(tmp_path, capsys) -> None:
    assert cli.main([str(tmp_path / "missing.nii.gz")]) == 2
    assert "no such file" in capsys.readouterr().err


def test_an_unknown_command_is_refused(qapp, image, capsys) -> None:
    assert cli.main([str(image), "--run", "no.such.command"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_a_screenshot_is_written_and_the_options_are_not_remembered(qapp, image, tmp_path) -> None:
    from bidsmgr.gui.viz.bridge import SettingsHub

    before = SettingsHub.instance().settings.volume.mode
    out = tmp_path / "shot.png"
    code = cli.main([str(image), "--layout", "mosaic", "--colormap", "hot",
                     "--run", "view.mosaic_text", "text=A -4 0 4",
                     "--screenshot", str(out), "--size", "600", "400", "--scale", "2"])
    assert code == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    from PyQt6.QtGui import QImage

    img = QImage(str(out))
    assert img.width() >= 1000            # drawn at twice the window's pixels
    # A one-off: the layout the GUI opens in did not change.
    SettingsHub.reset_instance()
    assert SettingsHub.instance().settings.volume.mode == before


def _atlas(image: Path) -> Path:
    path = image.with_name("sub-01_dseg.nii.gz")
    a = np.zeros((20, 22, 18), np.int16)
    a[4:10, 4:12, :] = 3
    a[11:16, 4:12, :] = 7
    nib.save(nib.Nifti1Image(a, np.diag([2.0, 2.0, 2.0, 1.0])), str(path))
    return path


def test_an_overlay_is_on_the_screenshot(qapp, image, tmp_path) -> None:
    """The figure is taken once the overlay is drawn, not before."""
    from PyQt6.QtGui import QImage

    plain, over = tmp_path / "plain.png", tmp_path / "over.png"
    args = ["--layout", "single", "--plane", "axial", "--size", "500", "400"]
    assert cli.main([str(image), *args, "--screenshot", str(plain)]) == 0
    assert cli.main([str(image), *args, "--overlay", str(_atlas(image)),
                     "--screenshot", str(over)]) == 0
    a, b = QImage(str(plain)), QImage(str(over))
    differ = sum(a.pixelColor(x, y) != b.pixelColor(x, y)
                 for x in range(0, a.width(), 9) for y in range(0, a.height(), 9))
    assert differ > 10


def test_a_saved_scene_opens_from_the_dataset(qapp, image, tmp_path, capsys) -> None:
    from bidsmgr.viz import scenes

    root = image.parents[2]
    scenes.save(root, {"schema": 1, "name": "check", "base": "sub-01/anat/sub-01_T1w.nii.gz",
                       "base_display": {"colormap": "hot"}, "frame": 0,
                       "overlays": [{"path": "sub-01/anat/" + _atlas(image).name,
                                     "origin": "", "name": "atlas", "visible": True,
                                     "display": None}],
                       "view": {"mode": "single", "plane": "coronal"}})
    out = tmp_path / "scene.png"
    assert cli.main([str(root), "--scene", "check", "--screenshot", str(out),
                     "--size", "500", "400"]) == 0
    assert out.is_file()


def test_an_unknown_scene_is_refused(qapp, image, capsys) -> None:
    assert cli.main([str(image.parents[2]), "--scene", "nope"]) == 2
    assert "no scene 'nope'" in capsys.readouterr().err
