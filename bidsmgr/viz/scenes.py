"""Scenes saved with the dataset: what was on screen, to open again.

A saved VIEW (``VizSettings.view_presets``) is a look applied to any image. A
SCENE belongs to a dataset: this image, these overlays with their looks,
this crosshair, this layout. It is what you send a colleague ("open the
'hippocampus check' scene") or come back to a week later, so it lives in the
dataset itself, under ``.bidsmgr/viz/scenes/``, one JSON file each (decision
D4).

Paths are stored relative to the dataset root, POSIX (CROSS_PLATFORM_RULES
1.1), so a scene saved on one computer opens on another. A computed overlay
(a quality map) is stored by how it was made (``origin``) and is computed
again; an MRS voxel by its MRS file.

Qt-free.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Optional

SCHEMA = 1
SCENES = Path(".bidsmgr") / "viz" / "scenes"

#: What a scene's view leaves out: the layers and sources are stored on
#: their own, by path, and the signal and spectrum viewers' state is not the
#: image viewer's.
_VIEW_EXCLUDE = {"sources", "layers", "measurements", "schema_version", "traces", "spectrum"}


def scenes_dir(root: Path) -> Path:
    return Path(root) / SCENES


def _file_name(name: str) -> str:
    try:
        from ..util.paths import safe_path_component

        safe = safe_path_component(name)
    except Exception:  # noqa: BLE001 - a plain fallback is enough here
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()
    return (safe or "scene") + ".json"


def scene_path(root: Path, name: str) -> Path:
    return scenes_dir(root) / _file_name(name)


def to_rel(root: Optional[Path], path: Path) -> str:
    """``path`` relative to the dataset, POSIX; absolute when outside it."""
    if root is not None:
        try:
            return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return Path(path).resolve().as_posix()


def from_rel(root: Optional[Path], rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute() or root is None:
        return p
    return Path(root).joinpath(*rel.split("/"))


def snapshot(scene, sources: dict, root: Optional[Path], name: str) -> dict:
    """The scene on screen as a JSON-able dict (see the module docstring)."""
    base = scene.base_layer()
    if base is None or base.source not in sources:
        raise ValueError("nothing is open to save")
    overlays = []
    for layer in scene.layers:
        if layer.kind != "volume" or layer.id == base.id:
            continue
        src = sources.get(layer.source)
        if src is None:
            continue
        overlays.append({
            "path": to_rel(root, src.path),
            "origin": layer.origin,
            "name": layer.name,
            "visible": layer.visible,
            "in_3d": layer.in_3d,
            "display": layer.display.model_dump(mode="json"),
        })
    return {
        "schema": SCHEMA,
        "name": name,
        "saved": _dt.datetime.now().isoformat(timespec="seconds"),
        "base": to_rel(root, sources[base.source].path),
        "base_display": base.display.model_dump(mode="json"),
        "frame": int(base.frame),
        "overlays": overlays,
        "view": scene.model_dump(mode="json", exclude=_VIEW_EXCLUDE),
    }


def save(root: Path, data: dict) -> Path:
    path = scene_path(root, str(data.get("name") or "scene"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "base" not in data:
        raise ValueError(f"{Path(path).name} is not a saved scene")
    if int(data.get("schema", 0)) > SCHEMA:
        raise ValueError(f"{Path(path).name} was saved by a newer version")
    return data


def list_scenes(root: Optional[Path]) -> list[tuple[str, Path]]:
    """``(name, file)`` of every scene saved in the dataset, by name."""
    if root is None:
        return []
    folder = scenes_dir(root)
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.glob("*.json")):
        try:
            name = str(json.loads(path.read_text(encoding="utf-8")).get("name") or path.stem)
        except (OSError, ValueError):
            continue
        out.append((name, path))
    return sorted(out, key=lambda item: item[0].lower())


__all__ = ["SCENES", "SCHEMA", "from_rel", "list_scenes", "load", "save", "scene_path",
           "scenes_dir", "snapshot", "to_rel"]
