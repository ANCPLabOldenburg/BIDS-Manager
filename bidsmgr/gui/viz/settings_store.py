"""Reading and writing :class:`~bidsmgr.viz.settings.VizSettings`.

One JSON value under one ``AppSettings`` key (``viz/settings``), so the whole
viewer configuration is a single readable blob in the platform's settings
store (plist on macOS, INI on Linux, registry on Windows) and a single thing
to export or reset. Validation clamps, so an edited or older blob still loads.
"""

from __future__ import annotations

import json
import logging

from ...viz.settings import VizSettings

log = logging.getLogger(__name__)


def load_viz_settings() -> VizSettings:
    from ..app_settings import AppSettings, KEYS

    raw = AppSettings._settings().value(KEYS["viz_settings"])
    if not raw:
        return VizSettings()
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        log.warning("viewer settings were not valid JSON; using defaults")
        return VizSettings()
    try:
        return VizSettings.model_validate(data)
    except Exception:  # noqa: BLE001 - never refuse to open a viewer
        log.warning("viewer settings did not validate; using defaults", exc_info=True)
        return VizSettings()


def save_viz_settings(settings: VizSettings) -> None:
    from ..app_settings import AppSettings, KEYS

    AppSettings._settings().setValue(KEYS["viz_settings"], settings.model_dump_json())


def export_json(settings: VizSettings) -> str:
    return settings.model_dump_json(indent=2)


def import_json(text: str) -> VizSettings:
    return VizSettings.model_validate(json.loads(text))


__all__ = ["export_json", "import_json", "load_viz_settings", "save_viz_settings"]
