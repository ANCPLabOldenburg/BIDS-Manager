"""How a figure is saved: its resolution and background."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout,
)

#: (factor, text): drawn again at that factor, never a screen grab enlarged.
SCALES = ((1.0, "Screen (1x)"), (2.0, "Print (2x)"), (3.0, "Poster (3x)"), (4.0, "4x"))


class FigureDialog(QDialog):
    """Resolution and background of a figure about to be saved."""

    def __init__(self, title: str = "Save the figure", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        note = QLabel("The figure is drawn again at the chosen resolution, so text and "
                      "edges stay sharp; the 3-D view is saved at screen resolution.")
        note.setObjectName("dlg-hint")
        note.setWordWrap(True)
        lay.addWidget(note)
        form = QFormLayout()
        self.scale = QComboBox()
        for factor, text in SCALES:
            self.scale.addItem(text, factor)
        self.scale.setCurrentIndex(1)
        form.addRow("Resolution", self.scale)
        self.transparent = QCheckBox("Transparent background")
        self.transparent.setToolTip("Off: the black surround the viewer shows.")
        form.addRow("", self.transparent)
        lay.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setObjectName("tb-btn-primary")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def values(self) -> tuple[float, bool]:
        return float(self.scale.currentData()), self.transparent.isChecked()


__all__ = ["FigureDialog", "SCALES"]
