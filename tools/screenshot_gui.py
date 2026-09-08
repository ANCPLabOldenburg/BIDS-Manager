"""Render BIDS Manager's windows to PNG so they can be looked at.

Layout bugs are invisible to a test suite. A pane can pass every assertion about
its widths and still show a field with no box beside it, a column that does not
line up, or a label cut mid-word. All three of those shipped, and all three were
obvious the moment somebody rendered the thing and looked.

This does that without a display: Qt's ``offscreen`` platform draws into memory
and ``QWidget.grab()`` returns the pixels. Run it, open the PNGs, and see.

    QT_QPA_PLATFORM=offscreen python tools/screenshot_gui.py dark
    QT_QPA_PLATFORM=offscreen python tools/screenshot_gui.py light --out /tmp/x

Both themes are worth a look: a colour that reads well on the near-black surface
can vanish on the light one, and only one of them is ever open while you work.

The fixtures are deliberately synthetic. Every widget here renders from an
inventory row and a metadata scaffold, so a made-up row exercises the same code
a scanned one does, and the script stays runnable with no data to hand.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PyQt6.QtWidgets import QApplication, QScrollArea

# One EEG recording, filled in the way a scan would leave it: classified, with
# the scan's read-only hints, and with the fields only a human can answer blank.
ROW = {
    "include": "1",
    "proposed_datatype": "eeg",
    "bids_guess_suffix": "eeg",
    "proposed_basename": "sub-001_task-rest_eeg",
    "source_file": "/raw/sub-001/rest.edf",
    "BIDS_name": "sub-001",
    "task": "rest",
    "modality": "eeg",
    "format": "EDF",
    "line_freq": "",
    "montage": "",
    "eeg_reference": "",
    "eeg_ground": "",
    "PatientSex": "M",
    "PatientAge": "34",
    "Handedness": "R",
    "manufacturer_suggestion": "Brain Products",
    "montage_suggestion": "standard_1005 (60/64)",
    # What the scan read out of this recording, which is what the converter
    # will write for it. Internal column: it never reaches the TSV.
    "_derived_fields": (
        '{"SamplingFrequency": 500.0, "EEGChannelCount": 64, '
        '"RecordingType": "continuous", "MISCChannelCount": 0}'
    ),
}


def _shoot(app, widget, path: Path, width: int, height: int, *, expand=False,
           scroll: float | None = None) -> None:
    """Draw one widget at a given size and save it.

    ``expand`` opens every collapsible section, which is where most of the form
    lives; ``scroll`` moves the view down by a fraction, since the interesting
    part is rarely the top.
    """
    from bidsmgr.gui.widgets.template_form import CollapsibleSection

    widget.resize(width, height)
    widget.show()
    app.processEvents()
    if expand:
        for section in widget.findChildren(CollapsibleSection):
            section.set_expanded(True)
        app.processEvents()
        app.processEvents()
    if scroll is not None:
        area = widget.findChild(QScrollArea)
        if area is not None:
            bar = area.verticalScrollBar()
            bar.setValue(int(bar.maximum() * scroll))
            app.processEvents()
    widget.grab().save(str(path))
    print(f"  {path.name}  ({widget.width()}x{widget.height()})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("theme", nargs="?", default="dark", choices=("dark", "light"))
    parser.add_argument("--out", default=None, help="where to write the PNGs")
    args = parser.parse_args(argv)

    out = Path(args.out or f"/tmp/bidsmgr-shots-{args.theme}")
    out.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv[:1])
    from bidsmgr.gui.theme_manager import ThemeManager

    ThemeManager(app).apply(args.theme)

    from bidsmgr.gui.models import InventoryTableModel
    from bidsmgr.gui.properties_panel import PropertiesPanel
    from bidsmgr.gui.recording_meta_dialog import RecordingMetaDialog

    scratch = Path(tempfile.mkdtemp())
    df = pd.DataFrame([ROW])
    pairs = [("eeg", "eeg")]
    counts = {("eeg", "eeg"): 5}
    example = {("eeg", "eeg"): "sub-001/eeg/sub-001_task-rest_eeg.json"}

    def dialog(name: str):
        return RecordingMetaDialog(
            scratch / f"{name}.recording_meta.json", {"eeg"}, None,
            present_pairs=pairs, pair_counts=counts, example_paths=example,
        )

    def panel():
        p = PropertiesPanel()
        p.bind_model(InventoryTableModel(df.copy()))
        p.set_selected_row(0)
        return p

    print(f"{args.theme} theme -> {out}")
    _shoot(app, dialog("closed"), out / "dialog_closed.png", 620, 760)
    _shoot(app, dialog("open"), out / "dialog_open.png", 640, 900, expand=True)
    _shoot(app, dialog("eeg"), out / "dialog_eeg.png", 640, 900,
           expand=True, scroll=0.25)
    _shoot(app, panel(), out / "panel_wide.png", 380, 900, expand=True)
    _shoot(app, panel(), out / "panel_narrow.png", 240, 900, expand=True)
    _shoot(app, panel(), out / "panel_metadata.png", 380, 900,
           expand=True, scroll=1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
