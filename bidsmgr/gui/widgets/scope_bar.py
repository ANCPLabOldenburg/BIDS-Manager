"""Where a tool acts: what was picked in the tree, or a part of the dataset.

Several Editor tools used to insist on a tree selection. That is right when
you are looking at the file you want to change, and wrong the rest of the
time: "take the face off every anatomical in the dataset" and "put these four
runs into a session" are the same tool, and only one of them starts with a
click on a file.

So every tool that acts on files takes a scope, and a scope is one of:

* **what was picked in the tree**, offered first and chosen by default
  whenever there IS a selection, so opening a tool from the right-click menu
  behaves exactly as it did;
* **the whole dataset**, a subject, or a session, so the same tool can be
  opened from the Tools menu with nothing selected;
* narrowed further by **datatype**, because "every anatomical" and "every
  recording" are different requests and the difference is one combo box.

The scope resolves to a list of paths, which is what the tools already take:
a folder stands for everything under it, and every engine here already
expands a folder (``restructure.expand``, ``remove.plan_delete``, the
defacing walk). So this adds a way to say what to act on without changing
what acting on it means.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .primitives import ElidedLabel

from ... import schema as schema_mod

#: Sentinel for "do not narrow by datatype".
ANY_DATATYPE = ""

#: Sentinel identifying the tree-selection entry in the part combo.
PICKED = "\x00picked"

#: Sentinel identifying the whole-dataset entry.
WHOLE = ""

#: Stop counting files past this. The summary is orientation, not a census,
#: and it is recomputed on the GUI thread every time a combo changes.
_COUNT_CAP = 20000


def _count_files(targets: list[Path]) -> int:
    """How many files the scope covers, giving up at :data:`_COUNT_CAP`."""
    seen = 0
    stack = list(targets)
    while stack and seen < _COUNT_CAP:
        path = stack.pop()
        if path.is_file():
            seen += 1
            continue
        try:
            for entry in os.scandir(path):
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    stack.append(Path(entry.path))
                else:
                    seen += 1
                    if seen >= _COUNT_CAP:
                        break
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
    return seen


class ScopeBar(QWidget):
    """Two combos and a count: which part of the dataset, and which datatype.

    ``changed`` fires whenever the resolved target list could differ, which
    is what a dialog connects its replan to.
    """

    changed = pyqtSignal()

    def __init__(
        self,
        root: Path,
        picked: Optional[Sequence[Path]] = None,
        *,
        allow_dataset: bool = True,
        show_summary: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._picked = [Path(p) for p in (picked or [])]

        # Never squeezed below what its rows need. A widget nested inside a
        # QFormLayout is laid out from its ``sizeHint``, and a form row that
        # wants more room than the dialog has will compress the rest: this
        # bar came out with its summary drawn nine pixels INSIDE its own
        # combo boxes at a 1.6 font scale.
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum,
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(8)

        self._part = QComboBox()
        self._part.setObjectName("ent-input")
        self._part.setToolTip(
            "Which part of the dataset this acts on. The tree selection is "
            "offered first when there is one; otherwise pick the dataset, a "
            "subject or a session."
        )
        if self._picked:
            self._part.addItem(
                f"What you picked in the tree ({len(self._picked)})", PICKED
            )
        if allow_dataset:
            self._part.addItem("The whole dataset", WHOLE)
        for subject in sorted(p for p in self._root.glob("sub-*") if p.is_dir()):
            self._part.addItem(subject.name, subject.name)
            for session in sorted(
                p for p in subject.glob("ses-*") if p.is_dir()
            ):
                self._part.addItem(
                    f"    {subject.name} / {session.name}",
                    f"{subject.name}/{session.name}",
                )
        self._part.currentIndexChanged.connect(self._on_changed)
        row.addWidget(self._part, 2)

        self._datatype = QComboBox()
        self._datatype.setObjectName("ent-input")
        self._datatype.setToolTip(
            "Narrow it to one datatype. Only the datatypes this dataset "
            "actually has are listed."
        )
        self._datatype.addItem("Every datatype", ANY_DATATYPE)
        for name in self._datatypes_present():
            self._datatype.addItem(name, name)
        self._datatype.currentIndexChanged.connect(self._on_changed)
        row.addWidget(self._datatype, 1)

        outer.addLayout(row)

        # ELIDED, not word-wrapped. A word-wrapped QLabel answers with a
        # height that depends on its width, and Qt does not carry that
        # answer out through a nested widget into a QFormLayout: the row is
        # sized for one line, the label asks for two, and the second one
        # lands on top of the combo above it. One elided line cannot do
        # that, and the full text is on the tooltip.
        self._summary = ElidedLabel("")
        self._summary.setObjectName("dlg-hint")
        self._summary.setVisible(show_summary)
        outer.addWidget(self._summary)

        self._refresh_summary()

    # -- what the dataset has ---------------------------------------------

    def _datatypes_present(self) -> list[str]:
        """Datatype folders that exist here, in the schema's order.

        Read from the tree rather than from the schema alone: a list of the
        standard's twenty datatypes on a dataset holding two is a list
        nobody reads. The schema still decides what COUNTS as a datatype, so
        a stray folder named ``notes`` is not offered.
        """
        known = list(schema_mod.list_datatypes())
        present = set()
        for subject in self._root.glob("sub-*"):
            if not subject.is_dir():
                continue
            for child in subject.iterdir():
                if not child.is_dir():
                    continue
                if child.name in known:
                    present.add(child.name)
                elif child.name.startswith("ses-"):
                    for grand in child.iterdir():
                        if grand.is_dir() and grand.name in known:
                            present.add(grand.name)
        return [name for name in known if name in present]

    # -- the answer --------------------------------------------------------

    def targets(self) -> list[Path]:
        """The paths the tool should act on."""
        part = self._part.currentData()
        datatype = self._datatype.currentData() or ANY_DATATYPE

        if part == PICKED:
            base = list(self._picked)
        elif part == WHOLE:
            base = [self._root]
        else:
            base = [self._root / str(part)]

        if not datatype:
            return base
        return self._narrow(base, str(datatype))

    def _narrow(self, base: list[Path], datatype: str) -> list[Path]:
        """``base`` reduced to the parts of it inside ``datatype``.

        A picked FILE survives only if it lives in that datatype; a picked
        FOLDER contributes the datatype folders under it. That way narrowing
        never adds anything the wider scope did not already cover.
        """
        out: list[Path] = []
        for path in base:
            if path.is_file():
                if datatype in path.parts:
                    out.append(path)
                continue
            if path.name == datatype:
                out.append(path)
                continue
            # The two places BIDS puts a datatype folder, rather than
            # ``rglob``: a recursive glob from the dataset root walks
            # ``.bidsmgr/`` and every backup in it to find folders that can
            # only be one or two levels down.
            found = set()
            for pattern in (
                f"{datatype}", f"sub-*/{datatype}", f"sub-*/ses-*/{datatype}",
                f"ses-*/{datatype}",
            ):
                found.update(p for p in path.glob(pattern) if p.is_dir())
            out.extend(sorted(found))
        return out

    def label(self) -> str:
        """The scope in words, for a dialog that wants to state it."""
        part = self._part.currentText().strip()
        datatype = self._datatype.currentData() or ANY_DATATYPE
        return f"{part}, {datatype} only" if datatype else part

    def is_empty(self) -> bool:
        return not self.targets()

    # -- internals ---------------------------------------------------------

    def _on_changed(self, *_a) -> None:
        self._refresh_summary()
        self.changed.emit()

    def _refresh_summary(self) -> None:
        targets = self.targets()
        if not targets:
            self._summary.setText(
                "Nothing in the dataset matches that. Widen the scope."
            )
            self._summary.setToolTip("")
            return
        folders = sum(1 for p in targets if p.is_dir())
        files = _count_files(targets)
        more = "+" if files >= _COUNT_CAP else ""
        if folders:
            text = (
                f"{folders} folder(s), {files}{more} file(s) in all"
                if folders == len(targets) else
                f"{len(targets)} item(s), {files}{more} file(s) in all"
            )
        else:
            # Files only, so the two counts are the same number and saying
            # it twice ("1 file(s), 1 file(s) in all") reads as a bug.
            text = f"{len(targets)} file(s)"
        self._summary.setText(text)
        self._summary.setToolTip(
            "\n".join(str(p) for p in targets[:20])
            + ("\n..." if len(targets) > 20 else "")
        )


__all__ = ["ANY_DATATYPE", "PICKED", "WHOLE", "ScopeBar"]
