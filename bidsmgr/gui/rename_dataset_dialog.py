"""What a user means when they change a dataset's name.

Two different intentions wear the same edit, and getting it wrong is annoying
either way.

Sometimes the folder was named in a hurry, ``study2``, and the real name arrives
later: "Verbal Working Memory in Ageing". That is a RENAME, and leaving the
folder called ``study2`` means every path, every backup and every conversation
about the data keeps the wrong name.

Sometimes the folder name IS the name everyone uses, and what is being typed is
the dataset's proper title for publication. That is a TITLE, and renaming the
folder under it would break paths that other people, scripts and backups already
point at.

BIDS Manager cannot tell which from the edit, so it asks, once, at the moment of
the change. It used to guess: whatever went into ``dataset_description.json``
became the name shown everywhere, so typing a publication title silently
renamed the project in the interface while the folder on disk stayed put and the
two drifted apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from .theme_manager import CUR

# What the user chose.
KEEP_FOLDER = "title"     # the folder keeps its name; this is the dataset title
RENAME_ALL = "rename"     # rename the folder too, so the two agree


class RenameDatasetDialog(QDialog):
    """Ask whether a new dataset name renames the project or titles it."""

    def __init__(
        self,
        folder_name: str,
        new_name: str,
        parent: Optional[QWidget] = None,
        *,
        can_rename: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dataset name")
        self.setModal(True)
        pal = CUR()

        outer = QVBoxLayout(self)
        outer.setSpacing(10)

        headline = QLabel(f'You have named this dataset "{new_name}".')
        headline.setWordWrap(True)
        headline.setStyleSheet(
            f"color: {pal['text']}; font-weight: 600; background: transparent;"
        )
        outer.addWidget(headline)

        blurb = QLabel(
            f'The folder holding it is called "{folder_name}", and that is the '
            "name BIDS Manager shows for the project. What should happen to it?"
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet(f"color: {pal['dim']}; background: transparent;")
        outer.addWidget(blurb)

        self._title = QRadioButton(
            f'Keep the folder called "{folder_name}"'
        )
        self._title.setChecked(True)
        title_note = QLabel(
            f'"{new_name}" becomes the dataset title, written to '
            "dataset_description.json and shown beside the project name. "
            "Nothing on disk moves, so anything already pointing at this folder "
            "keeps working."
        )
        title_note.setWordWrap(True)
        title_note.setStyleSheet(
            f"color: {pal['dim']}; background: transparent; margin-left: 20px;"
        )

        self._rename = QRadioButton(f'Rename the folder to "{new_name}"')
        rename_note = QLabel(
            "The project is renamed on disk and everywhere it is listed. Use "
            "this when the folder was named provisionally. Anything pointing at "
            "the old path, a script or a backup, will need updating."
        )
        rename_note.setWordWrap(True)
        rename_note.setStyleSheet(
            f"color: {pal['dim']}; background: transparent; margin-left: 20px;"
        )

        outer.addWidget(self._title)
        outer.addWidget(title_note)
        outer.addSpacing(4)
        outer.addWidget(self._rename)
        outer.addWidget(rename_note)

        if not can_rename:
            # A name the filesystem will not take, or a folder already there.
            self._rename.setEnabled(False)
            rename_note.setText(
                f'"{new_name}" cannot be used as a folder name here, so only '
                "the title option is available."
            )
            rename_note.setStyleSheet(
                f"color: {pal['warning']}; background: transparent; margin-left: 20px;"
            )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons, 0, Qt.AlignmentFlag.AlignRight)

    def choice(self) -> str:
        """``RENAME_ALL`` or ``KEEP_FOLDER``."""
        return RENAME_ALL if self._rename.isChecked() else KEEP_FOLDER


def ask_what_the_name_means(
    folder_name: str, new_name: str, parent: Optional[QWidget] = None,
    *, can_rename: bool = True,
) -> Optional[str]:
    """Put the question. ``None`` if the user backed out."""
    dialog = RenameDatasetDialog(
        folder_name, new_name, parent, can_rename=can_rename,
    )
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.choice()


def can_rename_to(root: Path, new_name: str) -> bool:
    """Could this dataset's folder actually take that name?

    Asked before offering the choice, so the dialog never proposes something
    that will fail: a name with a separator in it, or one already taken by a
    sibling.
    """
    from ..util.paths import safe_path_component

    name = (new_name or "").strip()
    if not name or name != safe_path_component(name):
        return False
    target = Path(root).parent / name
    return target == Path(root) or not target.exists()


__all__ = [
    "KEEP_FOLDER",
    "RENAME_ALL",
    "RenameDatasetDialog",
    "ask_what_the_name_means",
    "can_rename_to",
]
