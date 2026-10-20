"""Anonymise the open dataset's metadata: what, where, and every change first.

Over :mod:`bidsmgr.anonymise`. Three questions, one area each:

- **What**: the categories, grouped by purpose (the participants; when it
  happened; the site and its equipment; traces in files), each switched on or
  off with its options and how many changes it makes now. Recommended and
  Strict set them; any change makes the policy Custom.
- **Where**: the dataset files, the subjects and their sessions, ticked.
- **What changes**: every change, by kind or by file, the values only a
  person can judge, and what a format does not let BIDS Manager rewrite. A
  file can be left out of the run.

Nothing is written until Anonymise is pressed; the change is ONE step of the
Editor's history, undone and redone as one. Planning runs on a thread, again
after every choice (debounced), and so does the writing: a FIF goes through
MNE, which takes seconds per recording.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..anonymise import (
    CATEGORIES,
    CATEGORY,
    PRESETS,
    PURPOSES,
    RECOMMENDED,
    apply_anonymise,
    dataset_units,
    plan_anonymise,
)
from ..anonymise.policy import RECORD, RECORD_TITLE
from .dialog_chrome import build_footer_with, build_header, card, hint, stat
from .widgets.anonymise_views import (
    CategoryCard,
    ChangeList,
    Group,
    Row,
    ScopeTree,
    purpose_heading,
)
from .widgets.primitives import Segmented
from .widgets.spinner import BusySpinner
from .widgets.splitter import Splitter

log = logging.getLogger(__name__)

#: How long the choices settle before the dataset is read again.
_SETTLE_MS = 250


def _apply_job(root: Path, plan, label: str, forget: bool) -> tuple[int, int, list]:
    """On the thread: write the plan as one operation; then, if asked,
    forget the history's copies. ``(files written, bytes freed, failed)``."""
    from ..project.operations import begin_operation, forget_history

    with begin_operation(root, label) as op:
        written = apply_anonymise(plan, op=op)
    freed = forget_history(root) if forget else 0
    return len(written), freed, list(plan.failed)


def _files(n: int) -> str:
    return f"{n:,} file" + ("" if n == 1 else "s")


class AnonymiseDialog(QDialog):
    """Choose what and where, read every change, anonymise."""

    VIEWS = (("kind", "By kind", "Every change, under the category it belongs to"),
             ("file", "By file", "Every file and what changes in it; a file can be left "
                                 "out of the run"),
             ("review", "To read by hand",
              "Free text and labels that may name someone: not changed, for a person to "
              "read"),
             ("cannot", "Not changed here",
              "What a format, or the folder it is in, does not let BIDS Manager rewrite"))

    def __init__(self, root: Path, parent: Optional[QWidget] = None, *,
                 scope=None) -> None:
        super().__init__(parent)
        self._root = Path(root)
        #: What to tick once the subjects are read (None: everything).
        self._initial_scope = scope
        self._policy = PRESETS[RECOMMENDED]
        self._plan = None
        self._written = 0
        self._generation = 0
        self._busy_state = False
        self._released = None
        from .viz.bridge import JobRunner

        self._jobs = JobRunner(self)
        self._jobs.done.connect(self._on_done)
        self._jobs.failed.connect(self._on_failed)
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(_SETTLE_MS)
        self._settle.timeout.connect(self._replan)

        self.setWindowTitle("Anonymise metadata")
        self.setModal(True)
        self.resize(1240, 860)
        self.setMinimumSize(900, 600)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            "Anonymise metadata",
            "Removes what identifies a participant from the dataset's metadata, in every "
            "kind of file: sidecars, tables, recording headers, photographs. Choose "
            "<b>what</b>, by purpose, and <b>where</b>; every change is listed before "
            "anything is written. The recorded data are never changed, and the whole run "
            "is one step of the Editor's history: Undo takes it back, Redo makes it again.",
        ))

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        bl.setSpacing(10)

        split = Splitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        left = Splitter(Qt.Orientation.Vertical)
        left.setChildrenCollapsible(False)
        left.addWidget(self._build_what())
        left.addWidget(self._build_where())
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 2)
        split.addWidget(left)
        split.addWidget(self._build_result())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 800])
        left.setSizes([480, 360])
        bl.addWidget(split, 1)

        self._forget = QCheckBox(
            "Then delete the copies the Editor keeps for Undo (they hold the originals)")
        self._forget.setToolTip(
            "To undo, the Editor keeps a copy of every file it changes under .bidsmgr/. "
            "After anonymising, those copies are the identifiable originals: a dataset "
            "shared or backed up with .bidsmgr/ in it still holds them.\n\nTicked, they "
            "are deleted once the anonymisation is written. Nothing done before, this "
            "included, can be undone or redone afterwards.")
        bl.addWidget(self._forget)
        outer.addWidget(body, 1)

        self._status = QLabel("")
        self._status.setObjectName("dlg-hint")
        self._status.setWordWrap(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close | QDialogButtonBox.StandardButton.Ok)
        self._ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setObjectName("tb-btn-primary")
        self._ok.setText("Anonymise")
        buttons.accepted.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        outer.addWidget(build_footer_with(self._status, buttons))

        self._sync_cards()
        self._set_busy(True, "Reading the dataset…")
        self._jobs.start("units", self._generation, dataset_units, self._root)

    # -- building -----------------------------------------------------------

    def _build_what(self) -> QWidget:
        frame, layout = card("What")
        layout.addWidget(hint("What to remove, by purpose. Recommended and Strict set "
                              "these; any change of yours makes it Custom."))
        row = QHBoxLayout()
        row.setSpacing(8)
        self._preset = Segmented([
            ("recommended", "Recommended",
             "Keeps what analyses need: times of day, the site's name, exact ages up to "
             "89."),
            ("strict", "Strict", "Keeps as little as still makes a valid dataset: times "
                                 "to midnight, the site removed, ages in five-year bins."),
            ("custom", "Custom", "Your own choice of categories and options."),
        ], current=RECOMMENDED)
        self._preset.changed.connect(self._on_preset)
        row.addWidget(self._preset)
        row.addStretch(1)
        layout.addLayout(row)

        scroll = QScrollArea()
        scroll.setObjectName("anon-scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setObjectName("anon-scroll-body")
        v = QVBoxLayout(inner)
        v.setContentsMargins(0, 0, 6, 0)
        v.setSpacing(6)
        self._cards: dict[str, CategoryCard] = {}
        for purpose, title in PURPOSES:
            v.addWidget(purpose_heading(title))
            for category in (c for c in CATEGORIES if c.purpose == purpose):
                box = CategoryCard(category, inner)
                box.changed.connect(self._on_category)
                v.addWidget(box)
                self._cards[category.key] = box
        v.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)
        return frame

    def _build_where(self) -> QWidget:
        frame, layout = card("Where")
        self._where = hint("")
        layout.addWidget(self._where)
        self._scope = ScopeTree(frame)
        self._scope.changed.connect(self._on_scope)
        layout.addWidget(self._scope, 1)
        return frame

    def _build_result(self) -> QWidget:
        frame, layout = card("What changes")
        stats = QHBoxLayout()
        stats.setSpacing(28)
        self._stat_changes = stat("changes", "…")
        self._stat_files = stat("files", "…")
        self._stat_review = stat("to read by hand", "…", "warn")
        self._stat_cannot = stat("not changed here", "…")
        for tile in (self._stat_changes, self._stat_files, self._stat_review,
                     self._stat_cannot):
            stats.addWidget(tile)
        stats.addStretch(1)
        layout.addLayout(stats)

        row = QHBoxLayout()
        row.setSpacing(10)
        self._view = Segmented(self.VIEWS, current="kind")
        self._view.changed.connect(lambda _k: self._fill_list())
        row.addWidget(self._view)
        row.addStretch(1)
        self._spinner = BusySpinner()
        row.addWidget(self._spinner)
        self._busy_label = QLabel("")
        self._busy_label.setObjectName("dlg-hint")
        row.addWidget(self._busy_label)
        layout.addLayout(row)

        self._list = ChangeList(frame)
        self._list.left_out_changed.connect(self._after_left_out)
        layout.addWidget(self._list, 1)
        return frame

    # -- state ------------------------------------------------------------------

    def busy(self) -> bool:
        return self._busy_state

    def policy(self):
        return self._policy

    def scope(self):
        return self._scope.scope()

    def _sync_cards(self) -> None:
        for key, box in self._cards.items():
            box.set_state(self._policy.enabled(key), self._policy.option(key))
        self._preset.set_current(self._policy.name)

    def _on_preset(self, name: str) -> None:
        if name not in PRESETS:          # Custom: the choices as they are
            self._preset.set_current(self._policy.name)
            return
        self._policy = PRESETS[name]
        self._sync_cards()
        self._settle.start()
        self._show_pending()

    def _on_category(self, key: str, on: bool, option: str) -> None:
        self._policy = self._policy.with_category(key, on, option or None)
        self._cards[key].set_state(on, self._policy.option(key))
        self._preset.set_current(self._policy.name)
        self._settle.start()
        self._show_pending()

    def _on_scope(self) -> None:
        self._where.setText(self._where_text())
        self._settle.start()
        self._show_pending()

    def _where_text(self) -> str:
        scope = self._scope.scope()
        if self._scope.is_empty():
            return "Nothing ticked: tick the dataset files or a subject."
        return f"Anonymising {scope.describe()}."

    def _show_pending(self) -> None:
        for box in self._cards.values():
            box.set_count(None)

    # -- planning ---------------------------------------------------------------

    def _set_busy(self, on: bool, text: str = "") -> None:
        self._busy_state = on
        self._spinner.set_busy(on, message="")
        self._busy_label.setText(text)
        self._busy_label.setVisible(on)
        for w in (self._forget, self._scope, self._preset):
            w.setEnabled(not on)
        self._ok.setEnabled(not on and bool(self._included()))

    def _replan(self) -> None:
        self._generation += 1
        if self._scope.is_empty():
            self._plan = None
            self._fill()
            return
        self._set_busy(True, "Reading the dataset…")
        self._jobs.start("plan", self._generation, plan_anonymise, self._root, self._policy,
                         scope=self._scope.scope())

    def _on_done(self, tag: str, generation: int, result) -> None:
        if generation != self._generation:
            return
        if tag == "units":
            top, units = result
            self._scope.set_units(top, units)
            if self._initial_scope is not None:
                self._scope.apply_scope(self._initial_scope)
            self._where.setText(self._where_text())
            self._replan()
        elif tag == "plan":
            self._plan = result
            self._set_busy(False)
            self._list.model.left_out.clear()
            self._fill()
        else:
            self._finish_apply(result)

    def _on_failed(self, tag: str, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        if tag == "apply" and self._released is not None:
            self._released.__exit__(None, None, None)
            self._released = None
        self._set_busy(False)
        log.warning("anonymisation %s failed: %s", tag, message)
        last = message.strip().splitlines()[-1] if message.strip() else message
        if tag == "apply":
            QMessageBox.warning(self, "Anonymisation failed",
                                f"Nothing was changed.\n\n{last}")
        self._status.setText(f"Could not {'write' if tag == 'apply' else 'read'}: {last}")

    # -- the result -------------------------------------------------------------

    def _included(self) -> list:
        if self._plan is None:
            return []
        out = self._list.model.left_out
        return [f for f in self._plan.files if self._plan.rel(f.path) not in out]

    def _fill(self) -> None:
        plan = self._plan
        grouped = plan.grouped_reviews() if plan is not None else []
        self._stat_review.number.setText(f"{len(grouped):,}")
        self._stat_cannot.number.setText(f"{len(plan.cannot) if plan else 0:,}")
        self._view.buttons["review"].setText(
            f"To read by hand ({len(grouped)})" if grouped else "To read by hand")
        self._view.buttons["cannot"].setText(
            f"Not changed here ({len(plan.cannot)})" if plan and plan.cannot
            else "Not changed here")
        self._fill_list()
        self._after_left_out()

    def _fill_list(self) -> None:
        plan = self._plan
        view = next((k for k, b in self._view.buttons.items() if b.isChecked()), "kind")
        if plan is None:
            self._list.set_groups([], empty="Nothing in this scope.")
            return
        if view == "kind":
            groups = self._by_kind(plan)
            empty = ("No change: the dataset states nothing the chosen categories "
                     "remove, in this scope.")
        elif view == "file":
            groups = self._by_file(plan)
            empty = "No file changes."
        elif view == "review":
            groups = self._reviews(plan)
            empty = "Nothing to read by hand."
        else:
            groups = self._cannot(plan)
            empty = "Every file in this scope can be cleaned here."
        self._list.set_groups(groups, empty=empty)

    @staticmethod
    def _by_kind(plan) -> list[Group]:
        rows: dict[str, list[Row]] = {}
        files: dict[str, set] = {}
        for planned in plan.files:
            rel = plan.rel(planned.path)
            for change in planned.changes:
                key = change.category or RECORD
                rows.setdefault(key, []).append(Row(
                    "change", change.what, change.before, change.after, rel, rel,
                    change.reason[:1].upper() + change.reason[1:] + "."))
                files.setdefault(key, set()).add(rel)
        order = [c.key for c in CATEGORIES] + [RECORD]
        out = []
        for key in order:
            if key not in rows:
                continue
            title = CATEGORY[key].title if key in CATEGORY else RECORD_TITLE
            out.append(Group(title, key, _files(len(files[key])), rows[key]))
        return out

    @staticmethod
    def _by_file(plan) -> list[Group]:
        out = []
        for planned in plan.files:
            rel = plan.rel(planned.path)
            rows = [Row("change", c.what, c.before, c.after,
                        CATEGORY[c.category].title if c.category in CATEGORY else RECORD_TITLE,
                        rel, c.reason[:1].upper() + c.reason[1:] + ".")
                    for c in planned.changes]
            out.append(Group(rel, rel, "", rows, leavable=True))
        return out

    @staticmethod
    def _reviews(plan) -> list[Group]:
        """One group per field, its reason once on the heading; each value
        with where it is."""
        rows: dict[str, list[Row]] = {}
        reasons: dict[str, list[str]] = {}
        for what, value, why, paths in plan.grouped_reviews():
            where = plan.rel(paths[0]) if len(paths) == 1 else _files(len(paths))
            rows.setdefault(what, []).append(Row(
                "review", repr(value), "", "", where, "",
                f"{why[:1].upper() + why[1:]}. In "
                + ", ".join(plan.rel(p) for p in paths[:5])
                + (f" and {len(paths) - 5:,} more" if len(paths) > 5 else "") + "."))
            if why not in reasons.setdefault(what, []):
                reasons[what].append(why)
        return [Group(what, what, "; ".join(reasons[what]), found)
                for what, found in rows.items()]

    @staticmethod
    def _cannot(plan) -> list[Group]:
        if not plan.cannot:
            return []
        rows = [Row("cannot", path, why, "", "", "", why) for path, why in plan.cannot]
        return [Group("Left as they are", "cannot", "", rows)]

    def _after_left_out(self) -> None:
        """Counts, cards and the button after a file is left out or put back."""
        plan = self._plan
        included = self._included()
        counts: dict[str, int] = {}
        for planned in included:
            for change in planned.changes:
                counts[change.category] = counts.get(change.category, 0) + 1
        for key, box in self._cards.items():
            box.set_count(counts.get(key, 0))
        changes = sum(counts.values())
        self._stat_changes.number.setText(f"{changes:,}")
        self._stat_files.number.setText(f"{len(included):,}")
        left_out = len(self._list.model.left_out)
        if plan is None or not plan.files:
            self._status.setText("Nothing to change in this scope." if plan is not None
                                 or self._scope.is_empty() else "")
            self._ok.setEnabled(False)
            self._ok.setText("Anonymise")
            return
        self._status.setText(
            f"{self._policy.label}: {changes:,} changes in {_files(len(included))}"
            + (f", {left_out:,} left out" if left_out else "") + ".")
        self._ok.setEnabled(bool(included) and not self._busy_state)
        self._ok.setText(f"Anonymise {_files(len(included))}")

    # -- applying -----------------------------------------------------------------

    def _on_apply(self) -> None:
        included = self._included()
        if self._plan is None or not included:
            return
        forget = self._forget.isChecked()
        scope = self._scope.scope()
        what = ", ".join(CATEGORY[k].title.lower() for k in self._policy.categories_on())
        text = (f"Anonymise {_files(len(included))} of {scope.describe()} "
                f"({self._policy.label}: {what})?\n\nIt is one step of the Editor's "
                "history: Undo takes it back, Redo makes it again.")
        if forget:
            text += ("\n\nThe copies kept for Undo are then DELETED: nothing done before, "
                     "this included, can be undone afterwards.")
        answer = QMessageBox.question(
            self, "Anonymise metadata", text,
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Cancel if forget else QMessageBox.StandardButton.Ok)
        if answer != QMessageBox.StandardButton.Ok:
            return
        plan = self._plan
        plan.files = included
        from .fs_watch import watchers_released

        # Windows cannot replace a watched file: the watches go for the write.
        self._released = watchers_released()
        self._released.__enter__()
        self._generation += 1
        self._set_busy(True, "Anonymising…")
        self._ok.setEnabled(False)
        label = f"Anonymise metadata ({self._policy.label}" + (
            "" if scope.whole else f", {scope.describe()}") + ")"
        self._jobs.start("apply", self._generation, _apply_job, self._root, plan, label, forget)

    def _finish_apply(self, result) -> None:
        if self._released is not None:
            self._released.__exit__(None, None, None)
            self._released = None
        self._busy_state = False
        self._written, _freed, failed = result
        if failed:
            listed = "\n".join(f"{p}: {why}" for p, why in failed[:10])
            more = f"\n... and {len(failed) - 10} more" if len(failed) > 10 else ""
            QMessageBox.warning(
                self, "Some files were not anonymised",
                f"{len(failed)} file(s) could not be rewritten and were left as they "
                f"were; the rest is anonymised.\n\n{listed}{more}")
        self.accept()

    def written(self) -> int:
        """How many files were anonymised, once accepted."""
        return self._written


__all__ = ["AnonymiseDialog"]
