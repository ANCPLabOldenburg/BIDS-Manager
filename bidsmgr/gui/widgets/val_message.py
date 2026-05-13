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
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from .status_badge import StatusBadge


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

    def __init__(
        self,
        severity: str,
        rule: str,
        body_html: str,
        fix_label: Optional[str] = None,
        field: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(_OBJECT_NAME_BY_SEVERITY.get(severity, "val-msg"))

        h = QHBoxLayout(self)
        h.setContentsMargins(10, 7, 10, 7)
        h.setSpacing(10)

        h.addWidget(StatusBadge(severity), 0, Qt.AlignmentFlag.AlignTop)

        right = QVBoxLayout()
        right.setSpacing(4)

        # Header row: rule label + optional field chip on the right.
        head = QHBoxLayout()
        head.setSpacing(8)
        rule_l = QLabel(rule)
        rule_l.setObjectName("val-rule")
        head.addWidget(rule_l)
        head.addStretch(1)
        if field:
            field_chip = QLabel(field)
            field_chip.setObjectName("val-field")
            field_chip.setToolTip(
                "JSON field this finding refers to."
            )
            head.addWidget(field_chip, 0, Qt.AlignmentFlag.AlignTop)
        right.addLayout(head)

        body_widget = QHBoxLayout()
        body_widget.setSpacing(8)

        body_l = QLabel(body_html)
        body_l.setObjectName("val-body")
        body_l.setWordWrap(True)
        body_l.setTextFormat(Qt.TextFormat.RichText)
        body_widget.addWidget(body_l, 1)

        self._field_name = field or ""
        if fix_label:
            btn = QPushButton(fix_label)
            btn.setObjectName("val-fix")
            btn.clicked.connect(
                lambda: self.fix_requested.emit(self._field_name)
            )
            body_widget.addWidget(btn, 0, Qt.AlignmentFlag.AlignTop)

        right.addLayout(body_widget)
        h.addLayout(right, 1)


__all__ = ["ValMessage"]
