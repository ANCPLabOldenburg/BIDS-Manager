"""One validation message row used in the Editor's right pane.

Layout: ``[badge] [rule_label][body][fix button?]``. Body accepts rich
text (``Qt.TextFormat.RichText``) so the validator can highlight code
literals via ``<code>...</code>``. Lift-and-shift from
``inspector_proto/proto.py`` lines 463-483.

The ``ValMessage`` consumes the same shape that
:class:`bidsmgr.editor.types.Issue` produces (severity + rule_id +
message + optional fix label), so the editor view can bind to validator
output with no reshaping.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from .status_badge import StatusBadge


class _Elided(QLabel):
    """A single-line label that shortens rather than setting a width floor.

    A plain QLabel reports its whole text as its MINIMUM width, so one long
    rule id or schema path stopped the whole validation pane from being
    narrowed. This one reports zero and elides to whatever it is given.
    """

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self._full = text
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred,
        )

    def setText(self, text: str) -> None:  # noqa: N802 - Qt override
        self._full = text
        super().setText(text)
        self._elide()

    def minimumSizeHint(self):  # noqa: N802 - Qt override
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        width = self.width()
        if width <= 0 or not self._full:
            return
        super().setText(
            self.fontMetrics().elidedText(
                self._full, Qt.TextElideMode.ElideMiddle, width,
            )
        )

    def full_text(self) -> str:
        return self._full


def _caption(text: str) -> QLabel:
    """The small uppercase word that says what the next line IS.

    The parts of a finding used to run together: a rule id, a message and a
    schema path stacked with nothing saying which was which. Naming them costs
    one quiet line each and makes the block scannable.
    """
    label = QLabel(text)
    label.setObjectName("val-caption")
    return label


_OBJECT_NAME_BY_SEVERITY: dict[str, str] = {
    "ok":   "val-msg-ok",
    "warn": "val-msg-warn",
    "err":  "val-msg-err",
}


class ValMessage(QFrame):
    """One validator finding rendered as a single row.

    ``severity`` ∈ {``"ok"``, ``"warn"``, ``"err"``}. Unknown severities
    fall back to the neutral ``val-msg`` object name (no tint).

    ``fix_label`` is optional — pass a string to render a small button
    next to the body, and connect to :pyattr:`fix_requested` to handle
    clicks. The widget never executes fixes itself (validator returns
    a fix label + opaque token; the controller decides how to apply).
    """

    # Emitted on fix-button click. Carries the issue's ``field`` (the
    # JSON key the finding refers to) so the host panel can focus that
    # row in the sidecar form; empty string when the issue has no
    # specific field.
    fix_requested = pyqtSignal(str)
    # Right-click on a WARNING: record that somebody looked at it and
    # decided to keep it. Never offered on an error: a tool that lets you
    # dismiss wrongness produces broken datasets quietly.
    accept_requested = pyqtSignal(str, str)   # (rule_id, field)

    def __init__(
        self,
        severity: str,
        rule: str,
        body_html: str,
        fix_label: Optional[str] = None,
        field: Optional[str] = None,
        schema_rule: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(_OBJECT_NAME_BY_SEVERITY.get(severity, "val-msg"))

        self._schema_rule = schema_rule or ""
        self._rule_id = rule or ""
        self._field_name = field or ""

        # Vertical, not horizontal. The old layout put the badge in a column
        # beside everything else and the field chip out on the right, which
        # set a width floor: the pane could not be squeezed narrower than
        # "badge + widest line + chip", and the chip collided with the rule
        # name long before that. Stacking means the only floor is the badge,
        # and every part is free to wrap.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(5)

        # A finding with no rule id, no field and no schema path is a plain
        # sentence: the scanner notes the Converter shows are exactly that.
        # Captioning a single sentence with "WHAT IS WRONG" adds a row and
        # says nothing, and the empty rule slot adds another. Those messages
        # get a one-line treatment instead.
        if not (rule or field or self._schema_rule):
            self._build_plain(outer, severity, body_html, fix_label)
            return

        # Line 1: what KIND of thing this is, and what it is about.
        head = QHBoxLayout()
        head.setSpacing(6)
        head.addWidget(StatusBadge(severity), 0, Qt.AlignmentFlag.AlignVCenter)
        rule_l = _Elided(rule)
        rule_l.setObjectName("val-rule")
        rule_l.setToolTip(rule)
        head.addWidget(rule_l, 1)
        if fix_label:
            btn = QPushButton(fix_label)
            btn.setObjectName("val-fix")
            btn.clicked.connect(
                lambda: self.fix_requested.emit(self._field_name)
            )
            head.addWidget(btn, 0, Qt.AlignmentFlag.AlignVCenter)
        outer.addLayout(head)

        # Line 2: WHICH field, on its own row rather than fighting the rule
        # name for the same line. Labelled, because a bare word in a chip
        # does not say what it is a chip of.
        if field:
            field_row = QHBoxLayout()
            field_row.setSpacing(6)
            field_row.addWidget(_caption("Field"))
            chip = QLabel(field)
            chip.setObjectName("val-field")
            chip.setToolTip("The metadata field this finding is about.")
            # A plain label with a Maximum policy: it hugs its text and can be
            # squeezed, but is never given zero width. ``_Elided`` here took
            # an Ignored policy and a maximum width, and between them the chip
            # collapsed to nothing: the caption showed and the field name did
            # not.
            chip.setSizePolicy(
                QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred,
            )
            field_row.addWidget(chip, 0)
            field_row.addStretch(1)
            outer.addLayout(field_row)

        # Line 3: what the validator actually said.
        outer.addWidget(_caption("What is wrong"))
        body_l = QLabel(body_html)
        body_l.setObjectName("val-body")
        body_l.setWordWrap(True)
        body_l.setTextFormat(Qt.TextFormat.RichText)
        body_l.setMinimumWidth(0)
        outer.addWidget(body_l)

        # Line 4: what the rule is FOR. The id lets a user CHECK the claim;
        # this says why the claim matters, which the codes never do.
        from ...editor.rule_help import explain as _explain_rule

        help_text = _explain_rule(rule)
        if help_text:
            meaning, action = help_text
            outer.addWidget(_caption("Why it matters"))
            note = QLabel(meaning + " " + action)
            note.setObjectName("val-explanation")
            note.setWordWrap(True)
            note.setMinimumWidth(0)
            note.setToolTip(meaning + "\n\n" + action)
            outer.addWidget(note)

        # Line 5: where the standard says it, when there is a schema rule
        # behind it. Findings BIDS Manager raises itself have none, and show
        # none rather than an invented provenance.
        if self._schema_rule:
            outer.addWidget(_caption("Defined in"))
            prov = _Elided(self._schema_rule)
            prov.setObjectName("val-provenance")
            prov.setToolTip(
                "Where this comes from in the BIDS schema.\n"
                "Look it up in the specification to see what the standard says."
            )
            outer.addWidget(prov)

        if str(severity).lower().startswith("warn"):
            self.setContextMenuPolicy(
                Qt.ContextMenuPolicy.CustomContextMenu
            )
            self.customContextMenuRequested.connect(self._on_context_menu)


    def _build_plain(
        self,
        outer: QVBoxLayout,
        severity: str,
        body_html: str,
        fix_label: Optional[str],
    ) -> None:
        """Badge and sentence on one line, for findings with no structure.

        No caption, no rule row, no provenance: there is nothing to tell
        apart. Tighter margins too, because these stack several-per-card in
        the Converter's chip dialogs and the padding was most of the height.

        No accept-this-warning menu either. An acceptance is recorded against
        a rule id, and these have none.
        """
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(
            StatusBadge(severity), 0, Qt.AlignmentFlag.AlignTop,
        )
        body = QLabel(body_html)
        body.setObjectName("val-body")
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setMinimumWidth(0)
        row.addWidget(body, 1)
        if fix_label:
            btn = QPushButton(fix_label)
            btn.setObjectName("val-fix")
            btn.clicked.connect(
                lambda: self.fix_requested.emit(self._field_name)
            )
            row.addWidget(btn, 0, Qt.AlignmentFlag.AlignTop)
        outer.addLayout(row)

    def _on_context_menu(self, pos) -> None:
        from PyQt6.QtWidgets import QMenu

        from ..combo_popup import round_menu

        menu = QMenu(self)
        round_menu(menu)
        act = menu.addAction("Accept this warning...")
        act.setToolTip(
            "Record that you looked at this and decided it is fine, with a "
            "note saying why. Stored in the dataset, so the next reviewer "
            "sees it too."
        )
        if menu.exec(self.mapToGlobal(pos)) is act:
            self.accept_requested.emit(self._rule_id, self._field_name)


__all__ = ["ValMessage"]
