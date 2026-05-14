"""Cross-platform application-icon helper.

Loads the brand icon into a :class:`PyQt6.QtGui.QIcon` and applies it
to the running ``QApplication``. This is what Linux and Windows use
for the window title bar, the taskbar entry, and the alt-tab
preview. On macOS the icon shown in the Dock and in Spotlight comes
from the ``.app`` bundle's ``Resources/AppIcon.icns`` (placed there
by the macOS installer); calling :meth:`QApplication.setWindowIcon`
on macOS is still safe and improves the rendering inside QDialog
windows that some Qt themes choose to badge.

Sources used (in order, first match wins):

1. ``bidsmgr/gui/assets/macos/AppIcon.icns``
   Native macOS multi-resolution icon container. Qt understands
   ICNS on macOS directly. Used as the highest-quality source if
   present.
2. ``bidsmgr/gui/assets/macos/AppIcon{16,32,64,128,256,512,1024}.png``
   The per-resolution PNG set bundled with the package. Added one
   at a time to a single :class:`QIcon` so Qt picks the right size
   for whatever surface it's painting.
3. ``bidsmgr/gui/assets/windows/AppIcon.ico``
   The multi-resolution Windows icon container. Acceptable to Qt
   on every platform; falls back to it if the macOS PNGs are
   missing for any reason.
4. ``bidsmgr/gui/assets/logo.png``
   Legacy single-resolution logo. Last-resort fallback so a
   stripped install still gets *some* icon rather than the
   platform default.

The function does not change the GUI's internal logo. That stays at
``bidsmgr/gui/assets/logo.png`` and is used by the About dialog and
the brand header inside the running app.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Optional

from PyQt6.QtGui import QIcon

# Per-resolution PNG names in ascending size order. Qt happily
# matches these against the painting surface's DPI (16 px for menu
# bars, 256+ px for the taskbar at HiDPI).
_PNG_SIZES = (16, 32, 64, 128, 256, 512, 1024)


def _assets_root() -> Optional[Path]:
    """Return the on-disk path of ``bidsmgr.gui.assets`` if reachable."""
    try:
        # ``importlib.resources.files`` returns a Traversable. For a
        # wheel install this is a real filesystem path (no .zip),
        # which is exactly what QIcon needs.
        return Path(str(files("bidsmgr.gui").joinpath("assets")))
    except (ModuleNotFoundError, FileNotFoundError):
        return None


def load_app_icon() -> QIcon:
    """Build the application's QIcon from bundled assets.

    Returns an empty :class:`QIcon` if nothing usable is found. The
    caller can still pass that to :meth:`QApplication.setWindowIcon`;
    Qt treats an empty icon as a no-op and falls back to the platform
    default.
    """
    icon = QIcon()
    assets = _assets_root()
    if assets is None:
        return icon

    macos_dir = assets / "macos"
    windows_dir = assets / "windows"

    # 1. Multi-resolution PNG set (preferred). Adding every size to a
    #    single QIcon lets Qt pick the closest match per surface.
    added_any = False
    if macos_dir.is_dir():
        for size in _PNG_SIZES:
            candidate = macos_dir / f"AppIcon{size}.png"
            if candidate.is_file():
                icon.addFile(str(candidate))
                added_any = True
        if added_any:
            return icon

    # 2. The macOS .icns container (single file, Qt unpacks it on macOS).
    icns = macos_dir / "AppIcon.icns"
    if icns.is_file():
        icon.addFile(str(icns))
        return icon

    # 3. The Windows .ico container.
    ico = windows_dir / "AppIcon.ico"
    if ico.is_file():
        icon.addFile(str(ico))
        return icon

    # 4. Last-resort legacy logo.
    legacy = assets / "logo.png"
    if legacy.is_file():
        icon.addFile(str(legacy))

    return icon


def set_app_icon(app) -> None:
    """Apply the bundled icon to ``app`` (a :class:`QApplication`)."""
    icon = load_app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)


__all__ = ["load_app_icon", "set_app_icon"]
