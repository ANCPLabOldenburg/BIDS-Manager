"""Write the sidecars recordings are missing, from what is still there.

Over :mod:`bidsmgr.editor.sidecars`. For a sidecar deleted by accident, or a
dataset that arrived without them. Three areas, as in Anonymise metadata
(GUI_LOOK_AND_FEEL 4.37):

- **Read from**: the data each recording was converted from (converted again,
  the most complete, only in a project that converted it), always the
  recording's own header, the answers given in this project and the links
  other sidecars hold; and whether to add to sidecars that are there, and
  mark what is still unknown.
- **Where**: what was picked in the tree, or a part of the dataset.
- **What would be written**: every recording, ticked, and the selected one's
  fields with where each was read from, what a sidecar higher up already
  states, and what nothing could give.

Nothing is written until Write is pressed, a value already in a sidecar is
never changed, and writing is one step of the Editor's history. The plan is
made on a thread, again after every choice; converting source data again is
kept between plans, so each recording is converted once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..editor import sidecars as engine
from .dialog_chrome import (
    WrapLabel,
    build_footer_with,
    build_header,
    card,
    hint,
    scrollable_body,
    stat,
)
from .fs_watch import watchers_released
from .widgets.primitives import ElidedLabel, Segmented
from .widgets.scope_bar import ScopeBar
from .widgets.sidecar_views import FieldList, RecordingTree
from .widgets.spinner import BusySpinner
from .widgets.splitter import Splitter
from .window_flags import give_window_buttons

log = logging.getLogger(__name__)

#: How long the choices settle before the dataset is read again.
_SETTLE_MS = 250

#: Above this many recordings the folders start folded.
_UNFOLDED = 40

#: The placeholder scopes, in the words the metadata step's setting uses.
_SCOPES = (
    ("none", "Nothing", "Leave what is unknown out of the sidecar."),
    ("required", "Required fields",
     "A TODO in every required field nothing could give."),
    ("recommended", "Required and recommended",
     "A TODO in every required and recommended field nothing could give, as "
     "the metadata step marks them by default."),
    ("optional", "Every field the standard declares",
     "A TODO in every field the standard declares for the recording."),
)


def _count(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


class SidecarsDialog(QDialog):
    """Plan the sidecars of the picked recordings, show them, write the ticked."""

    def __init__(self, root: Path, targets: Optional[list] = None,
                 parent: Optional[QWidget] = None, *, complete: Optional[bool] = None,
                 fill_scope: Optional[str] = None) -> None:
        super().__init__(parent)
        # A window people work in: maximisable on Linux and Windows too
        # (CROSS_PLATFORM 5.3), which a large font scale needs.
        give_window_buttons(self)
        self._root = Path(root)
        self._plan: Optional[engine.SidecarsPlan] = None
        self._written: list[Path] = []
        self._generation = 0
        self._planning = False
        #: Converting source data again, by scan row, kept between plans.
        self._converted: dict = {}
        #: What a file outside BIDS naming is read as, by the user's choice.
        self._kinds: dict[str, tuple[str, str]] = {}
        #: The recording shown, kept across plans.
        self._shown: Optional[str] = None
        picked = [Path(t) for t in (targets or [])]
        if complete is None:
            # A right-click on recordings that all have a sidecar means "add
            # to it": opening on "missing only" would show nothing to do.
            complete = bool(picked) and all(_has_sidecar(self._root, p) for p in picked)
        if fill_scope is None:
            from .app_settings import AppSettings

            fill_scope = AppSettings.load().placeholder_scope()
        from .viz.bridge import JobRunner

        self._jobs = JobRunner(self)
        self._jobs.done.connect(self._on_planned)
        self._jobs.failed.connect(self._on_plan_failed)
        self._jobs.progressed.connect(self._on_progress)
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(_SETTLE_MS)
        self._settle.timeout.connect(self._replan)

        self.setWindowTitle("Sidecars from the recordings")
        self.setModal(True)
        self.resize(1240, 820)
        self.setMinimumSize(880, 560)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            "Sidecars from the recordings",
            "Writes the <b>.json</b> sidecar a recording is missing, or adds what an "
            "existing one lacks, from what is still there: the data it was converted "
            "from, the recording's own header, your answers in this project and the "
            "other sidecars that link to it. Any image or recording BIDS Manager reads "
            "gets one, named the BIDS way or not. <b>A value already in a sidecar is "
            "never changed.</b> Every field is shown with where it was read before "
            "anything is written, and writing is one step of the Editor's history.",
        ))

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        bl.setSpacing(10)

        split = Splitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        # The choices scroll rather than squeeze: at a small window or a
        # large font a compressed card overlaps its own text.
        left, ll = scrollable_body()
        ll.setContentsMargins(0, 0, 4, 0)
        ll.addWidget(self._build_read_from(picked, complete, fill_scope))
        ll.addWidget(self._build_where(picked))
        ll.addStretch(1)
        split.addWidget(left)
        split.addWidget(self._build_result())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([350, 870])
        bl.addWidget(split, 1)
        outer.addWidget(body, 1)

        self._status = QLabel("")
        self._status.setObjectName("dlg-hint")
        self._status.setWordWrap(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close | QDialogButtonBox.StandardButton.Ok)
        self._ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setObjectName("tb-btn-primary")
        self._ok.setText("Write")
        buttons.accepted.connect(self._on_write)
        buttons.rejected.connect(self.reject)
        outer.addWidget(build_footer_with(self._status, buttons))

        self._replan()

    # -- building -----------------------------------------------------------

    def _build_read_from(self, picked: list, complete: bool, fill_scope: str) -> QWidget:
        frame, layout = card("Read from")
        converted_here = (self._root / ".bidsmgr" / "provenance").is_dir()
        self._use_source = QCheckBox("The data it was converted from")
        self._use_source.setChecked(converted_here)
        self._use_source.setEnabled(converted_here)
        self._use_source.setToolTip(
            "Converts the recording's source again, with the conversion's own code, "
            "into a temporary folder, and reads the sidecar it writes: every field "
            "dcm2niix, the physio, ECAT and eye-tracking converters read from the "
            "original files. The most complete, and a few seconds per recording. "
            "EEG and MEG recordings are their own source and are read directly.")
        self._use_source.toggled.connect(lambda _on: self._schedule())
        layout.addWidget(self._use_source)
        layout.addWidget(hint(
            "Converted again in a temporary folder; the dataset is not touched."
            if converted_here else
            "Only for a dataset converted in this project: BIDS Manager has no record "
            "of converting this one."))
        always = WrapLabel(
            "Always read: the recording's own header (NIfTI timing, the spectroscopy "
            "header, everything MNE-BIDS reads from an EEG or MEG recording), your "
            "answers in this project, and the links other sidecars hold (a fieldmap's "
            "target, its partner images).")
        always.setObjectName("dlg-hint")
        layout.addWidget(always)
        from .. import schema as schema_mod

        version = WrapLabel(
            f"Fields and values follow BIDS {schema_mod.bids_version()}, the version "
            "chosen in Settings > Validation. A value it does not allow is shown, "
            "not written.")
        version.setObjectName("dlg-hint")
        layout.addWidget(version)

        which = QLabel("Which sidecars")
        which.setObjectName("dlg-section-title")
        layout.addSpacing(4)
        layout.addWidget(which)
        self._which = Segmented((
            ("missing", "Missing ones",
             "Only recordings without a sidecar. One that has a sidecar is left alone."),
            ("complete", "Also add to existing",
             "Also the recordings that have a sidecar: it keeps every value it holds "
             "and gains what it lacks. A TODO is replaced by a value that was found."),
        ), current="complete" if complete else "missing")
        self._which.changed.connect(lambda _k: self._schedule())
        row = QHBoxLayout()
        row.addWidget(self._which)
        row.addStretch(1)
        layout.addLayout(row)

        mark = QLabel("Mark what is still unknown with TODO")
        mark.setObjectName("dlg-section-title")
        layout.addSpacing(4)
        layout.addWidget(mark)
        self._scope = QComboBox()
        self._scope.setObjectName("ent-input")
        for key, text, tip in _SCOPES:
            self._scope.addItem(text, key)
            self._scope.setItemData(self._scope.count() - 1, tip,
                                    Qt.ItemDataRole.ToolTipRole)
        self._scope.setCurrentIndex(max(0, self._scope.findData(fill_scope)))
        # Narrower than its longest choice when the column is: the popup
        # still shows every choice whole.
        self._scope.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._scope.setMinimumContentsLength(10)
        self._scope.setToolTip(
            "As the metadata step does: a TODO says a field is still to be answered, "
            "and the validator lists it. Never written over a value or a TODO already "
            "there.")
        self._scope.currentIndexChanged.connect(lambda _i: self._schedule())
        layout.addWidget(self._scope)
        return frame

    def _build_where(self, picked: list) -> QWidget:
        frame, layout = card("Where")
        self._scope_bar = ScopeBar(self._root, picked, show_summary=False, parent=self)
        self._scope_bar.changed.connect(self._schedule)
        layout.addWidget(self._scope_bar)
        layout.addWidget(hint("A folder stands for every recording in it; a sidecar "
                              "you picked stands for its recording."))
        return frame

    def _build_result(self) -> QWidget:
        frame, layout = card("What would be written")
        stats = QHBoxLayout()
        stats.setSpacing(28)
        self._stat_new = stat("new sidecars", "…")
        self._stat_added = stat("sidecars added to", "…")
        self._stat_fields = stat("fields", "…")
        self._stat_missing = stat("required, still unknown", "…", "warn")
        for tile in (self._stat_new, self._stat_added, self._stat_fields,
                     self._stat_missing):
            stats.addWidget(tile)
        stats.addStretch(1)
        layout.addLayout(stats)

        busy = QHBoxLayout()
        busy.setSpacing(8)
        self._spinner = BusySpinner()
        busy.addWidget(self._spinner)
        self._busy_label = QLabel("")
        self._busy_label.setObjectName("dlg-hint")
        busy.addWidget(self._busy_label, 1)
        layout.addLayout(busy)

        inner = Splitter(Qt.Orientation.Horizontal)
        inner.setChildrenCollapsible(False)

        files = QWidget()
        fl = QVBoxLayout(files)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(6)
        self._tree = RecordingTree()
        self._tree.itemChanged.connect(lambda *_a: self._update_status())
        self._tree.itemSelectionChanged.connect(self._show_selected)
        fl.addWidget(self._tree, 1)
        picks = QHBoxLayout()
        picks.setSpacing(6)
        for text, on in (("All", True), ("None", False)):
            btn = QPushButton(text)
            btn.setObjectName("tb-btn")
            btn.clicked.connect(lambda _c=False, v=on: self._tree.set_all(v))
            picks.addWidget(btn)
        picks.addStretch(1)
        fl.addLayout(picks)
        inner.addWidget(files)

        detail = QWidget()
        dl = QVBoxLayout(detail)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(4)
        self._detail_path = ElidedLabel("")
        self._detail_path.setObjectName("dlg-section-title")
        dl.addWidget(self._detail_path)
        self._detail_read = WrapLabel("")
        self._detail_read.setObjectName("dlg-hint")
        dl.addWidget(self._detail_read)
        self._detail_notes = WrapLabel("")
        self._detail_notes.setObjectName("dlg-hint")
        dl.addWidget(self._detail_notes)
        dl.addWidget(self._build_kind_row())
        self._fields = FieldList()
        dl.addWidget(self._fields, 1)
        inner.addWidget(detail)
        inner.setStretchFactor(0, 2)
        inner.setStretchFactor(1, 3)
        inner.setSizes([330, 530])
        layout.addWidget(inner, 1)
        return frame

    # -- planning ---------------------------------------------------------

    def planning(self) -> bool:
        return self._planning

    def plan(self) -> Optional[engine.SidecarsPlan]:
        return self._plan

    def _schedule(self) -> None:
        self._settle.start()

    def _set_busy(self, on: bool, text: str = "") -> None:
        self._spinner.set_busy(on, message="")
        self._busy_label.setText(text)
        self._busy_label.setVisible(bool(text))

    def _replan(self) -> None:
        self._settle.stop()
        self._generation += 1
        self._planning = True
        self._tree.setEnabled(False)
        self._ok.setEnabled(False)
        self._status.setText("")
        for tile in (self._stat_new, self._stat_added, self._stat_fields,
                     self._stat_missing):
            tile.number.setText("…")
        self._set_busy(True, "Reading the recordings…")
        self._jobs.start(
            "plan", self._generation, engine.plan_sidecars, self._root,
            self._scope_bar.targets(),
            complete=self._which.current() == "complete",
            use_source=self._use_source.isChecked(),
            fill_scope=self._scope.currentData() or "none",
            kinds=dict(self._kinds),
            converted=self._converted,
        )

    def _on_progress(self, _tag: str, generation: int, done: int, total: int) -> None:
        if generation != self._generation or not total:
            return
        verb = ("Converting the source data again and reading"
                if self._use_source.isChecked() else "Reading")
        self._busy_label.setText(f"{verb} recording {min(done + 1, total)} of {total}…")

    def _on_plan_failed(self, _tag: str, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self._planning = False
        self._set_busy(False)
        log.warning("sidecar plan failed: %s", message)
        self._status.setText(
            f"Could not read the dataset: {message.strip().splitlines()[-1]}")

    def _on_planned(self, _tag: str, generation: int, plan) -> None:
        if generation != self._generation:
            return
        self._planning = False
        self._set_busy(False)
        self._plan = plan
        files = list(plan.files)
        self._tree.fill(files, unfold=len(files) <= _UNFOLDED)
        self._tree.setEnabled(True)
        writes = plan.writes()
        self._stat_new.number.setText(f"{sum(1 for p in writes if not p.exists):,}")
        self._stat_added.number.setText(f"{sum(1 for p in writes if p.exists):,}")
        self._stat_fields.number.setText(f"{sum(len(p.added) for p in writes):,}")
        self._stat_missing.number.setText(
            f"{sum(len(p.still_missing) for p in files):,}")
        if files:
            if not (self._shown and self._tree.select_recording(self._shown)):
                self._tree.select_first()
        else:
            self._show_selected()
        self._update_status()

    # -- the selected recording -------------------------------------------

    def _show_selected(self) -> None:
        plan = self._tree.selected_plan()
        self._fields.show_plan(plan)
        self._show_kind(plan)
        if plan is not None:
            self._shown = plan.recording
        if plan is None:
            self._detail_path.setText("")
            self._detail_read.setText(self._empty_text())
            self._detail_notes.setText("")
            self._detail_notes.setVisible(False)
            return
        verb = "Adds to" if plan.exists else "Writes"
        self._detail_path.setText(f"{verb} {plan.sidecar}")
        self._detail_path.setToolTip(plan.sidecar)
        self._detail_read.setText(
            "Read from " + "; ".join(plan.read) + "." if plan.read
            else "Nothing could be read from this recording.")
        self._detail_notes.setText(" ".join(plan.notes))
        self._detail_notes.setVisible(bool(plan.notes))

    # -- what a file outside BIDS naming is read as ---------------------------

    def _build_kind_row(self) -> QWidget:
        """Read as [datatype] [suffix], for a file whose BIDS name does not
        say: the fields the standard declares follow from it."""
        self._kind_row = QWidget()
        row = QHBoxLayout(self._kind_row)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(6)
        label = QLabel("Read as")
        label.setObjectName("dlg-section-title")
        row.addWidget(label)
        self._kind_datatype = QComboBox()
        self._kind_datatype.setObjectName("ent-input")
        self._kind_datatype.setToolTip(
            "The datatype this file is recorded as. It decides which fields the "
            "standard declares, and so which are read, checked and marked.")
        self._kind_suffix = QComboBox()
        self._kind_suffix.setObjectName("ent-input")
        self._kind_suffix.setToolTip("What the recording is (its BIDS suffix).")
        for combo in (self._kind_datatype, self._kind_suffix):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(6)
            row.addWidget(combo)
        self._kind_why = ElidedLabel("")
        self._kind_why.setObjectName("dlg-hint")
        row.addWidget(self._kind_why, 1)
        self._kind_datatype.activated.connect(lambda _i: self._on_kind_datatype())
        self._kind_suffix.activated.connect(lambda _i: self._on_kind_chosen())
        self._kind_row.setVisible(False)
        self._kind_choices: list[tuple[str, list[str]]] = []
        return self._kind_row

    def _show_kind(self, plan) -> None:
        kind = getattr(plan, "kind", None) if plan is not None else None
        if kind is None or kind.bids:
            self._kind_row.setVisible(False)
            return
        self._kind_choices = engine.kind_choices(self._root / plan.recording)
        self._kind_datatype.blockSignals(True)
        self._kind_datatype.clear()
        for datatype, _suffixes in self._kind_choices:
            self._kind_datatype.addItem(datatype, datatype)
        self._kind_datatype.setCurrentIndex(max(0, self._kind_datatype.findData(kind.datatype)))
        self._kind_datatype.blockSignals(False)
        self._fill_suffixes(kind.datatype, kind.suffix)
        why = {engine.KIND_NAME: "from its name", engine.KIND_FOLDER: "from its folder",
               engine.KIND_GUESS: "guessed", engine.KIND_CHOSEN: "your choice"}.get(kind.how, "")
        self._kind_why.setText(f"{why}: {kind.why}" if kind.how != engine.KIND_CHOSEN else why)
        self._kind_why.setToolTip(
            "This file is not named the BIDS way, so what it is was read from "
            f"{kind.why}. Choose another kind if that is wrong: the sidecar is "
            "worked out again.")
        self._kind_row.setVisible(True)

    def _fill_suffixes(self, datatype: str, current: str = "") -> None:
        suffixes = dict(self._kind_choices).get(datatype, [])
        self._kind_suffix.blockSignals(True)
        self._kind_suffix.clear()
        for suffix in suffixes:
            self._kind_suffix.addItem(suffix, suffix)
        index = self._kind_suffix.findData(current) if current else -1
        if index < 0 and suffixes:
            from ..editor.headers import default_suffix

            index = max(0, self._kind_suffix.findData(default_suffix(datatype)))
        self._kind_suffix.setCurrentIndex(index)
        self._kind_suffix.blockSignals(False)

    def _on_kind_datatype(self) -> None:
        self._fill_suffixes(self._kind_datatype.currentData() or "")
        self._on_kind_chosen()

    def _on_kind_chosen(self) -> None:
        plan = self._tree.selected_plan()
        datatype = self._kind_datatype.currentData()
        suffix = self._kind_suffix.currentData()
        if plan is None or not datatype or not suffix:
            return
        if (plan.kind.datatype, plan.kind.suffix) == (datatype, suffix):
            return
        self._kinds[plan.recording] = (datatype, suffix)
        self._schedule()

    def _empty_text(self) -> str:
        if self._planning or self._plan is None:
            return ""
        skipped = self._plan.skipped
        if not skipped:
            return "There are no recordings here."
        have = sum(1 for _r, why in skipped if why == "has a sidecar")
        if have == len(skipped):
            return ("Every recording here has its sidecar. Choose “Also add to "
                    "existing” to add what they lack.")
        return f"{_count(len(skipped), 'recording')} left alone: " + \
            "; ".join(f"{Path(r).name} ({why})" for r, why in skipped[:3])

    # -- choosing and writing ---------------------------------------------

    def _update_status(self) -> None:
        if self._planning or self._plan is None:
            return
        chosen = self._tree.chosen()
        writes = self._plan.writes()
        anon = (f" New fields are anonymised as this dataset was ({self._plan.anonymised})."
                if self._plan.anonymised else "")
        if not writes:
            self._status.setText(self._empty_text() or "Nothing to write.")
            self._ok.setEnabled(False)
            self._ok.setText("Write")
            return
        fields = sum(len(p.added) for p in chosen)
        ticked = (f"{len(chosen)} of {len(writes)} sidecars" if len(chosen) < len(writes)
                  else _count(len(chosen), "sidecar"))
        have = sum(1 for _r, why in self._plan.skipped if why == "has a sidecar")
        left = (f" {_count(have, 'recording')} here already "
                f"{'has its sidecar' if have == 1 else 'have theirs'}." if have else "")
        self._status.setText(
            f"{ticked}, {_count(fields, 'field')}.{left}{anon}" if chosen
            else "No sidecar ticked.")
        self._ok.setEnabled(bool(chosen))
        self._ok.setText(f"Write {_count(len(chosen), 'sidecar')}" if chosen else "Write")

    def _on_write(self) -> None:
        chosen = self._tree.chosen()
        if not chosen or self._planning:
            return
        try:
            with watchers_released():
                self._written = engine.apply_sidecars(self._root, chosen)
        except Exception as exc:  # noqa: BLE001 - reported, nothing half-written
            QMessageBox.warning(self, "Could not write the sidecars",
                                f"Nothing was changed.\n\n{exc}")
            return
        self.accept()

    def written(self) -> list[Path]:
        """The sidecars written, once the dialog has been accepted."""
        return list(self._written)

    def done(self, result: int) -> None:  # noqa: D401 - Qt signature
        self._settle.stop()
        self._jobs.cancel_all()
        super().done(result)


def _has_sidecar(root: Path, path: Path) -> bool:
    """A picked file that is a sidecar, or a recording with one."""
    if path.is_dir() and not engine.is_recording(root, path):
        return False
    if path.name.endswith(".json"):
        return True
    return engine.sidecar_path(path).exists()


__all__ = ["SidecarsDialog"]
