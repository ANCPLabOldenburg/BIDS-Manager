"""Turn a template field into a control, and back into a value.

The one place in the GUI that knows how a BIDS field becomes a widget. Both
surfaces that ask for metadata render through it: the dataset dialog, which
edits a whole dataset's template, and the inspection table's properties panel,
which edits one recording. Because the mapping lives here, neither of them
contains a BIDS field name, and a field added to a future schema version appears
in both without a line of code.

The control follows from what the schema declares about the field, never from a
list kept here:

===========================  ==========================================
the schema says              the user gets
===========================  ==========================================
a controlled vocabulary      an editable combo box, values preloaded
``boolean``                  unset / true / false, because a checkbox
                             cannot say "not stated"
``number`` / ``integer``     a numeric box with the unit beside it
``array`` of strings         a list editor, one row per entry
``object``                   key and value rows
anything else                a line edit with a shaped placeholder
===========================  ==========================================

One field is special by name rather than by type, and only one: ``Authors`` is a
list of people written "Family, Given". No separator survives that in free text,
so it gets a row per person with the two parts apart.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QComboBox,
    QSizePolicy,
    QHBoxLayout,
    QLabel,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...metadata.template_plan import REGION_AGNOSTIC
from ..theme_manager import CUR

# Fields that are lists of people. The only place a BIDS name appears in the
# GUI, and it earns it: "Lopez, Karel" is one author, so neither a comma nor a
# newline can be used to separate entries.
PEOPLE_FIELDS: frozenset[str] = frozenset({"Authors"})

_UNSET = "(not stated)"


class _RowList(QWidget):
    """A list the user grows a row at a time.

    ``make_row`` builds one row's widgets and returns ``(widget, read, write)``,
    so the same list serves plain strings and two-part names without knowing
    which it is holding.
    """

    # Emitted whenever the list's contents settle: a row added, removed, or
    # finished being typed into. The dialog saves on a button and ignores this;
    # the properties panel commits as you go and needs it.
    changed = pyqtSignal()

    def __init__(
        self,
        make_row: Callable[[], tuple[QWidget, Callable[[], Any], Callable[[Any], None]]],
        add_label: str = "Add",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._make_row = make_row
        self._rows: list[tuple[QWidget, Callable[[], Any], Callable[[Any], None]]] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        self._rows_box = QVBoxLayout()
        self._rows_box.setSpacing(2)
        outer.addLayout(self._rows_box)

        add = QPushButton(add_label)
        add.setObjectName("template-add-row")
        add.clicked.connect(lambda: self.add_row())
        outer.addWidget(add, 0, Qt.AlignmentFlag.AlignLeft)
        self.add_row()

    def add_row(self, value: Any = None) -> None:
        widget, read, write = self._make_row()
        if value is not None:
            write(value)

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(widget, 1)
        remove = QPushButton("−")
        remove.setFixedWidth(26)
        remove.setToolTip("Remove this entry")
        layout.addWidget(remove, 0)
        self._rows_box.addWidget(row)

        entry = (row, read, write)
        self._rows.append(entry)

        def drop() -> None:
            if entry in self._rows:
                self._rows.remove(entry)
            row.setParent(None)
            if not self._rows:
                self.add_row()
            self.changed.emit()

        remove.clicked.connect(drop)
        for edit in widget.findChildren(QLineEdit) or (
            [widget] if isinstance(widget, QLineEdit) else []
        ):
            edit.editingFinished.connect(self.changed)

    def value(self) -> list:
        out = []
        for _row, read, _write in self._rows:
            item = read()
            if item not in (None, "", [], {}):
                out.append(item)
        return out

    def set_value(self, values) -> None:
        for row, _read, _write in list(self._rows):
            row.setParent(None)
        self._rows.clear()
        for item in values or []:
            self.add_row(item)
        if not self._rows:
            self.add_row()


def _string_row() -> tuple[QWidget, Callable[[], Any], Callable[[Any], None]]:
    edit = QLineEdit()
    return edit, lambda: edit.text().strip(), lambda v: edit.setText(str(v))


def _person_row() -> tuple[QWidget, Callable[[], Any], Callable[[Any], None]]:
    """Family and given name apart, composed the way BIDS writes them."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    family = QLineEdit()
    family.setPlaceholderText("Family name")
    given = QLineEdit()
    given.setPlaceholderText("Given name")
    layout.addWidget(family, 3)
    layout.addWidget(given, 2)

    def read() -> str:
        f, g = family.text().strip(), given.text().strip()
        if f and g:
            return f"{f}, {g}"
        return f or g

    def write(value: Any) -> None:
        f, _, g = str(value).partition(",")
        family.setText(f.strip())
        given.setText(g.strip())

    return row, read, write


def _let_it_shrink(combo: QComboBox) -> None:
    """Stop a combo from demanding the width of its widest option.

    By default a QComboBox reports its longest item as its minimum, so one
    45-character enum value sets a floor for the whole pane and the user cannot
    narrow it. The popup is widened separately, so nothing becomes unreadable.
    """
    combo.setMinimumContentsLength(0)
    combo.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    # Expanding, NOT Ignored. Ignored lets Qt give the widget zero width, and it
    # does: HEDVersion and License rendered as a label with nothing beside it.
    # Expanding plus a zero minimum shrinks just as far without disappearing.
    combo.setMinimumWidth(0)
    combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


def fit_popup_to_contents(combo: QComboBox) -> None:
    """Let the dropdown be as wide as its widest option.

    A combo sized to its layout is often much narrower than the values it
    offers, and the popup inherits that width, so "bolus followed by infusion"
    arrives as "bolus followed by...". The box stays compact; only the list
    opens wide enough to read, which is the part that has to be legible to
    choose from.
    """
    if combo.count() == 0:
        return
    metrics = combo.fontMetrics()
    widest = max(
        metrics.horizontalAdvance(combo.itemText(i)) for i in range(combo.count())
    )
    # Room for the frame, the scrollbar and a little breathing space.
    combo.view().setMinimumWidth(widest + 44)


def build_field_widget(field, suggestions: tuple = ()) -> QWidget:
    """The control for one :class:`TemplateField`.

    ``suggestions`` are values BIDS Manager offers that the standard does not:
    the manufacturers we have seen, the tracers a PET lab typically uses, what
    the scan detected in this dataset. They are added to an editable box, never
    used in place of a schema vocabulary, and never restrict what can be typed.
    """
    if field.name in PEOPLE_FIELDS:
        widget = _RowList(_person_row, add_label="Add author")
        widget.setProperty("template_kind", "people")
        return widget

    if field.enum:
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItem("")
        combo.addItems([str(v) for v in field.enum])
        combo.setProperty("template_kind", "enum")
        # The same treatment every other combo here gets, and it was missing
        # only from this branch. Left to itself a vocabulary combo keeps Qt's
        # Preferred policy and sizes to its widest item on first show, which
        # inside a folded section means it is measured while hidden: it came out
        # 49px wide, too narrow to draw "3D" beside the arrow, so a field the
        # conversion had answered read as a stunted box with a fragment of a
        # glyph in it.
        _let_it_shrink(combo)
        fit_popup_to_contents(combo)
        return combo

    if suggestions and field.type in ("", "string", "number", "integer"):
        # Sized to its layout, not to its widest option: fit_popup_to_contents
        # widens the popup instead, so the list stays readable while the box
        # itself can shrink with the pane.
        # Numbers get the list too. The mains frequency is 50 or 60 everywhere
        # on earth, and offering a bare numeric box for it helps nobody: BIDS
        # types the field as a number, which is not a reason to withhold the two
        # answers. The box stays typable, and stays numeric.
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItem("")
        combo.addItems([str(v) for v in suggestions])
        combo.setProperty(
            "template_kind", "number" if field.type in ("number", "integer") else "enum"
        )
        if field.type in ("number", "integer"):
            combo.setValidator(QDoubleValidator())
        combo.setToolTip(
            (field.description or "")
            + ("\n\n" if field.description else "")
            + "Suggestions, not a fixed list: anything may be typed."
        )
        _let_it_shrink(combo)
        fit_popup_to_contents(combo)
        return combo

    if field.type == "boolean":
        combo = QComboBox()
        combo.addItems([_UNSET, "true", "false"])
        combo.setProperty("template_kind", "boolean")
        _let_it_shrink(combo)
        fit_popup_to_contents(combo)
        return combo

    if field.type == "array" and field.item_type in ("", "string"):
        widget = _RowList(_string_row, add_label="Add entry")
        widget.setProperty("template_kind", "list")
        return widget

    edit = QLineEdit()
    edit.setPlaceholderText(field.placeholder)
    if field.type in ("number", "integer"):
        edit.setValidator(QDoubleValidator())
        edit.setProperty("template_kind", "number")
    else:
        edit.setProperty("template_kind", "text")
    return edit


def connect_field_widget(widget: QWidget, on_change) -> None:
    """Call ``on_change`` whenever this control's value settles.

    Which signal that is depends on the kind of control, so it is decided here
    rather than by each caller: the dataset dialog saves on a button and needs
    none of this, while the properties panel commits as you type, and neither
    should have to know that a boolean is a combo and a list is not.
    """
    kind = widget.property("template_kind")
    if kind in ("people", "list"):
        widget.changed.connect(on_change)
        return
    if isinstance(widget, QComboBox):
        widget.activated.connect(lambda _i: on_change())
        if widget.isEditable():
            widget.lineEdit().editingFinished.connect(on_change)
        return
    widget.editingFinished.connect(on_change)


def _as_number(text: str) -> Any:
    """The number the user typed, or the text if it is not one."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def read_field_widget(widget: QWidget, field) -> Any:
    """The widget's value, in the shape the schema declares. ``None`` if unset."""
    kind = widget.property("template_kind")

    if kind in ("people", "list"):
        values = widget.value()
        return values or None

    if kind == "boolean":
        text = widget.currentText()
        return None if text == _UNSET else text == "true"

    text = (
        widget.currentText() if isinstance(widget, QComboBox) else widget.text()
    ).strip()
    if not text:
        return None
    if kind == "number":
        try:
            return int(text) if field.type == "integer" else float(text)
        except ValueError:
            return text
    if not field.type:
        # A field the schema types with anyOf accepts more than one shape, so
        # what the user typed decides which. "60" in PowerLineFrequency means
        # the number 60; writing the string would fail validation for a field
        # that was answered correctly.
        return _as_number(text)
    if field.type == "array":
        if field.item_type in ("number", "integer"):
            # ``write_field_widget`` shows a list as ", "-joined text, so this
            # split is the inverse of the way the value was displayed, not a
            # guess. Coercing the items matters twice over. A numeric array
            # captured as strings fails validation for a field the user
            # answered correctly (``ReconMethodParameterValues`` typed as 0
            # reaching the sidecar as "0"). Worse, an uncoerced read stops
            # comparing equal to what the conversion already wrote, so fields
            # nobody touched, ``FrameTimesStart`` and ``FrameDuration`` among
            # them, get restated in the wrong shape and a template that was
            # meant to add answers takes valid ones away.
            parts = [part.strip() for part in text.split(",")]
            return [_as_number(part) for part in parts if part] or None
        # An array whose items are not numbers: keep what was typed rather than
        # guessing a split that may be wrong.
        return [text]
    return text


def write_field_widget(widget: QWidget, value: Any) -> None:
    """Show ``value`` in the widget, whatever kind it is."""
    kind = widget.property("template_kind")
    if kind in ("people", "list"):
        widget.set_value(value if isinstance(value, list) else [value])
        return
    if kind == "boolean":
        widget.setCurrentText(
            _UNSET if value is None else ("true" if value else "false")
        )
        return
    text = (
        ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
    )
    if isinstance(widget, QComboBox):
        widget.setCurrentText(text)
    else:
        widget.setText(text)


def field_text(field) -> str:
    """The words in a field's label: its name, its level mark, its unit."""
    mark = {"required": " *", "recommended": " \u00b7"}.get(field.level, "")
    unit = " (%s)" % field.unit if field.unit else ""
    return "%s%s%s:" % (field.name, mark, unit)


def _level_colour(field, colour: bool) -> str:
    """The colour that says what the standard asks of this field."""
    pal = CUR()
    if not colour:
        return pal["text"]
    return {
        "required": pal["error"],
        "recommended": pal["warning"],
        "deprecated": pal["muted"],
    }.get(field.level, pal["text"])


def field_label(field, *, colour: bool = True) -> str:
    """A field's label as rich text, for callers that place their own.

    The form itself uses :class:`FieldLabel`, which elides into a fixed column
    so the rows line up.
    """
    text = field_text(field)
    if not colour:
        return text
    return '<span style="color:%s;">%s</span>' % (_level_colour(field, True), text)


def level_legend(*, colour: bool = True) -> QLabel:
    """A one-line key to the marks, so nobody has to infer them."""
    pal = CUR()
    if colour:
        text = (
            f'<span style="color:{pal["error"]};">* required</span> &nbsp; '
            f'<span style="color:{pal["warning"]};">· recommended</span> &nbsp; '
            f'<span style="color:{pal["muted"]};">optional</span>'
        )
    else:
        text = "* required &nbsp; · recommended &nbsp; (unmarked) optional"
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setObjectName("level-legend")
    label.setStyleSheet("#level-legend { background: transparent; }")
    return label


# ---------------------------------------------------------------------------
# Collapsible sections
# ---------------------------------------------------------------------------
#
# NOT PanelFrame, though the plan first said so. That widget collapses a docked
# pane to a 24px vertical strip and can detach it into its own window, which is
# right for the converter's panes and wrong inside a scrolling form: a folded
# section here should simply stop taking room, keeping its heading readable.


class _ShrinkableButton(QPushButton):
    """A section header that cannot stop its pane from narrowing.

    A QPushButton reports the full width of its text as its MINIMUM, and these
    headers carry a title, a filename and a count. One of them therefore set the
    floor for the whole dialog: 367 px in a pane the user wanted at 150. It
    elides instead, and keeps the full text as its tooltip.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt override
        self._full = text
        super().setText(text)
        self._elide()

    def minimumSizeHint(self):  # noqa: N802 - Qt override
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        if not self._full:
            return
        room = max(0, self.width() - 20)
        shown = self.fontMetrics().elidedText(
            self._full, Qt.TextElideMode.ElideRight, room,
        ) if room else self._full
        if shown != super().text():
            QPushButton.setText(self, shown)
        if self.toolTip() in ("", self._full):
            self.setToolTip(self._full)


class CollapsibleSection(QWidget):
    """A titled section that folds away, with a count of what is inside.

    The heading stays legible when folded, and carries how many fields the
    section holds and how many of them BIDS requires, so a user can decide
    whether to open it without opening it.
    """

    def __init__(
        self,
        title: str,
        *,
        subtitle: str = "",
        badge: str = "",
        level: int = 0,
        expanded: bool = True,
        tone: str = "",
        body_factory=None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._expanded = expanded
        # Built on first opening rather than up front. A four-modality dataset
        # is 559 controls, and Qt spends over a millisecond adding each one to a
        # layout, so building them all to show a dozen cost two and a half
        # seconds every time the window opened.
        self._body_factory = body_factory

        outer = QVBoxLayout(self)
        outer.setContentsMargins(level * 10, 0, 0, 0)
        outer.setSpacing(2)

        pal = CUR()
        # A region reads as a heading, a file as a row you open. Giving them the
        # same weight made the tree a flat list of identical bars, which is what
        # made the structure hard to see even though it was correct.
        tint = {
            0: (pal["accent"], pal["surface3"], "700", 1),
            1: (pal["text"], pal["surface2"], "600", 0),
        }.get(level, (pal["text"], "transparent", "500", 0))
        colour, back, weight, rule = tint
        # A section can say what KIND of thing it holds. The settled block is
        # green because nothing in it is outstanding.
        colour = tone or colour

        self._header = _ShrinkableButton()
        self._header.setObjectName("template-section-header")
        self._header.setCheckable(True)
        self._header.setChecked(expanded)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setStyleSheet(
            "QPushButton{text-align:left; padding:5px 8px; border:none;"
            f"border-radius:6px; background:{back}; color:{colour};"
            f"font-weight:{weight};"
            + (f"border-bottom:1px solid {pal['accent_border']};" if rule else "")
            + "}"
            f"QPushButton:hover{{background:{pal['accent_bg']};}}"
        )
        self._title_text = title
        self._subtitle = subtitle
        self._badge = badge
        outer.addWidget(self._header)

        self._body = QWidget()
        self._body.setObjectName("template-section-body")
        self._body_layout = QVBoxLayout(self._body)
        # An indent plus a hairline down the left says "this belongs to the
        # heading above" without drawing a box around every level, which at
        # three levels deep becomes nested boxes and unreadable.
        # Scoped to this widget. A bare "QWidget{...}" rule is a TYPE selector:
        # it applies to every descendant, so the hairline was drawn down the
        # side of each control inside, and the rule reached the tooltips and
        # popups they raise.
        self._body.setStyleSheet(
            f"#template-section-body{{border-left:1px solid {pal['border']};}}"
            if level else ""
        )
        self._body_layout.setContentsMargins(14, 4, 0, 8)
        self._body_layout.setSpacing(6)
        outer.addWidget(self._body)

        self._header.toggled.connect(self._on_toggled)
        self._refresh_header()
        if expanded:
            self._realise()
        self._body.setVisible(expanded)

    def _refresh_header(self) -> None:
        pal = CUR()
        caret = "▾" if self._expanded else "▸"
        parts = [f'<b>{self._title_text}</b>']
        if self._subtitle:
            parts.append(f'<span style="color:{pal["muted"]};">{self._subtitle}</span>')
        if self._badge:
            parts.append(f'<span style="color:{pal["muted"]};">{self._badge}</span>')
        full = f"{caret}  " + "   ".join(_strip_markup(p) for p in parts)
        self._header.setToolTip(self._subtitle or full)
        self._header.setText(full)

    def _realise(self) -> None:
        """Build the body, once."""
        factory, self._body_factory = self._body_factory, None
        if factory is not None:
            self._body_layout.addWidget(factory())

    def is_realised(self) -> bool:
        """Has this section's body been built yet?"""
        return self._body_factory is None

    def _on_toggled(self, checked: bool) -> None:
        self._expanded = checked
        if checked:
            self._realise()
        self._body.setVisible(checked)
        self._refresh_header()
        self.toggled_by_user.emit(checked)

    toggled_by_user = pyqtSignal(bool)

    def body(self) -> QVBoxLayout:
        return self._body_layout

    def add(self, widget: QWidget) -> None:
        self._body_layout.addWidget(widget)

    def set_expanded(self, expanded: bool) -> None:
        self._header.setChecked(expanded)

    def is_expanded(self) -> bool:
        return self._expanded


def _strip_markup(text: str) -> str:
    """A QPushButton shows plain text, so drop the tags but keep the words."""
    import re

    return re.sub(r"<[^>]+>", "", text)


__all__ = [
    "CollapsibleSection",
    "TemplateTree",
    "PEOPLE_FIELDS",
    "build_field_widget",
    "connect_field_widget",
    "fit_popup_to_contents",
    "FieldLabel",
    "field_label",
    "field_label_widget",
    "field_text",
    "level_legend",
    "read_field_widget",
    "write_field_widget",
]


# ---------------------------------------------------------------------------
# The tree, rendered
# ---------------------------------------------------------------------------


# The widest a label column may get. A form fits its column to the longest name
# it holds and stops here; past that the names elide, because one 45-character
# field should not push every control across the pane.
LABEL_COLUMN_PX = 200


class FieldLabel(QLabel):
    """A field name that lines up and never sets a floor.

    Two requirements that pull against each other. Every row in a section shares
    one label column, or the form reads as a jumble. And a field name runs to 33
    characters, which a plain QLabel reports as its MINIMUM width, so one row
    would stop the whole pane from narrowing.

    So the column is fixed and the name ELIDES into it, with the full text in the
    tooltip. An earlier attempt gave the label an Ignored size policy instead,
    which let Qt shrink it to nothing: the names disappeared and the fields drew
    over where they had been.
    """

    def __init__(
        self, text, colour, width=LABEL_COLUMN_PX, mark="", mark_colour="",
        fixed=False,
    ) -> None:
        super().__init__()
        self._full = f"{text}{mark}"
        self._name = text
        self._mark = mark
        self._colour = colour
        self._mark_colour = mark_colour or colour
        self._fixed = fixed
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setToolTip(self._full)
        # Scoped to this label. An unscoped "background: transparent" cascades
        # into the tooltip Qt raises for this widget, and the tooltip renders
        # see-through over whatever is behind it.
        self.setObjectName("field-label")
        self.setStyleSheet(
            "#field-label { color: %s; background: transparent; }" % colour
        )
        self.setMinimumWidth(0)
        self.setMaximumWidth(width)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._elide()

    def sizeHint(self):  # noqa: N802 - Qt override
        """How much room this label would like.

        In a QFormLayout every label is given the column width, which the layout
        takes as the widest of them, so asking for just this name is enough and
        the column ends up as wide as it needs to be and no wider.

        Rows built by hand out of a QHBoxLayout have no such column: each label
        gets what it asks for, so there ``fixed`` makes them all ask for the
        same thing.
        """
        hint = super().sizeHint()
        room = self.fontMetrics().horizontalAdvance(self._full) + 4
        hint.setWidth(
            self.maximumWidth() if self._fixed
            else min(room, self.maximumWidth())
        )
        return hint

    def minimumSizeHint(self):  # noqa: N802 - Qt override
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        """Elide the NAME, keep the mark.

        The mark is the whole point of the label being coloured: it says what
        the standard asks of this field. Eliding the two together would drop it
        first, which is the one part that must not go.
        """
        room = max(0, (self.width() or self.maximumWidth()))
        metrics = self.fontMetrics()
        mark_px = metrics.horizontalAdvance(self._mark) if self._mark else 0
        name = metrics.elidedText(
            self._name, Qt.TextElideMode.ElideRight, max(0, room - mark_px),
        )
        html = f'<span style="color:{self._colour};">{name}</span>'
        if self._mark:
            html += f'<span style="color:{self._mark_colour};">{self._mark}</span>'
        if html != QLabel.text(self):
            QLabel.setText(self, html)


def field_label_widget(field, colour=True, width=LABEL_COLUMN_PX, fixed=False):
    """The label for one field.

    The NAME stays in the ordinary text colour and the MARK carries the level.
    Colouring the whole name by level turns a form of required fields into a
    wall of red, and the mark is what the legend explains.
    """
    pal = CUR()
    mark = {"required": " *", "recommended": " \u00b7"}.get(field.level, "")
    unit = f" ({field.unit})" if field.unit else ""
    tone = pal["muted"] if field.level == "deprecated" else pal["text"]
    return FieldLabel(
        f"{field.name}{unit}:", tone, width,
        mark=mark, mark_colour=_level_colour(field, colour), fixed=fixed,
    )


def _origin_tooltip(origin: str) -> str:
    """Why this field is being asked about."""
    from ...metadata.template_plan import ORIGIN_USUALLY_DERIVED

    if origin == ORIGIN_USUALLY_DERIVED:
        return (
            "dcm2niix or mne-bids usually reads this out of the data, and your "
            "files do not carry it. It may still exist at the scanner, or an "
            "anonymiser may have removed it."
        )
    return "No converter supplies this. If you do not state it, it stays absent."


def _plain_field(name: str, value):
    """A stand-in for a field the schema does not declare for this file.

    The converter can write keys BIDS never mentions. They are still shown, so
    nothing appears from nowhere, but there is no level or vocabulary to show.
    """
    from ...metadata.template_plan import TemplateField

    kind = (
        "boolean" if isinstance(value, bool)
        else "number" if isinstance(value, (int, float))
        else "array" if isinstance(value, list)
        else "object" if isinstance(value, dict)
        else "string"
    )
    return TemplateField(name=name, level="optional", type=kind)


class TemplateTree(QWidget):
    """Renders a template tree: regions, modalities, then one node per file.

    The single widget both metadata surfaces use. It holds no BIDS knowledge of
    its own: what to ask, at what level, in what shape, all arrives in the nodes.

    ``values`` maps a node's storage key to the answers already stored, and
    :meth:`values_by_key` gives them back in the same shape, so a caller reads
    and writes without knowing a field name either.
    """

    def __init__(
        self,
        nodes,
        *,
        values: Optional[dict] = None,
        suggestions: Optional[dict] = None,
        answered: Optional[dict] = None,
        colour_levels: bool = True,
        collapsed_keys: Optional[set] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._colour = colour_levels
        self._suggestions = suggestions or {}
        # What the conversion will fill in by itself, per node key. Shown, not
        # asked: a required field left out of the form looks like a form that
        # forgot it, and an empty box looks like missing metadata, when in fact
        # dcm2niix read the value out of the header before anyone opened this.
        self._answered = answered or {}
        self._widgets: dict[str, dict[str, QWidget]] = {}
        self._sections: dict[str, "CollapsibleSection"] = {}
        self._fields: dict[str, dict] = {}
        # What the conversion will write, per node, so an answer that merely
        # repeats it is not mistaken for a statement.
        self._answered_values: dict[str, dict] = {}
        # The plan behind each node, so a caller can tell what is being ASKED
        # from what is merely reachable.
        self._sections_by_key: dict[str, object] = {}
        collapsed_keys = collapsed_keys or set()
        # Kept so a section nobody opened still saves what it already held: its
        # widgets do not exist to be read back.
        self._stored = dict(values or {})

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        legend = level_legend(colour=colour_levels)
        legend.setContentsMargins(2, 0, 0, 2)
        outer.addWidget(legend)
        for node in nodes:
            outer.addWidget(
                self._render(
                    node, values or {}, collapsed_keys, level=0,
                    # The agnostic region is what a user fills first, so it
                    # opens; the per-file boxes under a modality do not, or a
                    # four-modality dataset greets them with a wall.
                    open_files=(node.key == REGION_AGNOSTIC),
                )
            )
        outer.addStretch(1)

    # -- building ------------------------------------------------------

    def _render(
        self, node, values: dict, collapsed: set, level: int,
        open_files: bool = False,
    ) -> QWidget:
        # What this section is FOR: the fields that will still be missing after
        # conversion. A bare field count said nothing about whether there was
        # any work to do.
        badge = ""
        if node.field_count:
            badge = f"{node.field_count} to answer"
            if node.required_count:
                badge += f", {node.required_count} required by BIDS"
        elif node.is_leaf:
            badge = "nothing to answer"

        section = CollapsibleSection(
            node.label,
            subtitle=node.subtitle,
            badge=badge,
            level=level,
            body_factory=(lambda n=node: self._build_leaf(n)) if node.is_leaf else None,
            # A file's own fields start folded: a dataset with four modalities
            # would otherwise open as a wall of boxes. The groupings above stay
            # open, so what exists is visible at a glance.
            expanded=(node.key not in collapsed)
            and (node.kind != "file" or open_files),
        )
        self._sections[node.key] = section

        if node.is_leaf:
            self._sections_by_key[node.key] = node.section
        for child in node.children:
            section.add(
                self._render(child, values, collapsed, level + 1, open_files)
            )
        return section

    def _build_leaf(self, node) -> QWidget:
        """One file's questions, plus what the conversion answers, on demand."""
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        # What is settled comes first, in green, so the eye starts with "here
        # is what you already have" and then reaches the questions. Putting it
        # last made a long section end on a note nobody read.
        measured = self._answered.get(node.key) or {}
        already = {name: measured.get(name) for name in node.section.supplied}
        already.update(measured)
        if already:
            column.addWidget(self._render_answered(node, already))
        column.addWidget(self._render_fields(node, self._stored.get(node.key, {})))
        return holder

    def build(self, key: str) -> None:
        """Build a section's body now, without waiting for the user to open it."""
        section = self._sections.get(key)
        if section is not None:
            section._realise()

    def widgets_for(self, key: str) -> dict:
        """This node's controls, building the section if nobody has opened it."""
        self.build(key)
        return self._widgets.get(key, {})

    def _render_fields(self, node, stored: dict) -> QWidget:
        holder = QWidget()
        form = QFormLayout(holder)
        form.setContentsMargins(2, 2, 2, 2)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(7)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        widgets: dict[str, QWidget] = {}
        fields: dict[str, object] = {}
        for field in node.section.fields:
            widget = build_field_widget(
                field, tuple(self._suggestions.get(field.name, ())),
            )
            if field.description:
                widget.setToolTip(field.description)
            if field.name in stored:
                write_field_widget(widget, stored[field.name])

            label = field_label_widget(field, colour=self._colour)
            tip = field.description or ""
            origin = getattr(field, "origin", "")
            if origin:
                tip = (tip + "\n\n" if tip else "") + _origin_tooltip(origin)
            if tip:
                label.setToolTip(tip)
                widget.setToolTip(tip)
            form.addRow(label, self._with_origin(widget, origin))
            widgets[field.name] = widget
            fields[field.name] = field

        # Merge, do not replace. The answered block builds first now and puts
        # its controls in the same maps; assigning here wiped every one of them,
        # so a field the conversion fills became unreachable again.
        self._widgets.setdefault(node.key, {}).update(widgets)
        self._fields.setdefault(node.key, {}).update(fields)
        return holder

    def _render_answered(self, node, already: dict) -> QWidget:
        """The fields the conversion answers, shown with the value it will write.

        Folded away, because the point of the form is what will still be
        MISSING. But editable, not read-only: the converter reads these from the
        data, and when the data is wrong, or a legacy file carries nothing, the
        only way to correct it is here. Hiding a field a user wants to set is
        worse than showing one they need not.

        Nothing typed here is stored unless it differs from what the conversion
        would write, so opening the section and closing it again changes nothing.
        """
        pal = CUR()
        box = CollapsibleSection(
            "Already answered by the conversion",
            subtitle="edit only to correct what the data says",
            badge=f"{len(already)} fields",
            level=2,
            expanded=False,
            tone=pal["success"],
        )
        body = QWidget()
        form = QFormLayout(body)
        form.setContentsMargins(2, 2, 2, 2)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        widgets = self._widgets.setdefault(node.key, {})
        fields = self._fields.setdefault(node.key, {})
        # Every field the file may carry, so a required one the conversion
        # answered still shows as required. section.fields holds only the
        # ones being asked, which is why the levels used to be missing here.
        declared = {f.name: f for f in node.section.declared}
        for name, value in sorted(already.items()):
            spec = declared.get(name) or _plain_field(name, value)
            widget = build_field_widget(spec, tuple(self._suggestions.get(name, ())))
            varies = str(value).strip().upper() == "VARIES"
            unknown = value is None
            if not varies and not unknown:
                write_field_widget(widget, value)
            widget.setToolTip(
                (spec.description + "\n\n" if spec.description else "")
                + ("The conversion normally supplies this from the data. Type "
                   "here only if you know it will not, or to override it."
                   if unknown else
                   "The recordings disagree, so the conversion writes each one's "
                   "own value. Type here only to give them all the same one."
                   if varies else
                   "The conversion reads this from the data and will write the "
                   "value shown. Type here only to correct it.")
            )
            label = field_label_widget(spec, colour=self._colour)
            if varies or unknown:
                hint = QLabel(
                    "read from the data" if unknown else "differs per recording"
                )
                hint.setStyleSheet(f"color: {pal['dim']};")
                row = QWidget()
                line = QHBoxLayout(row)
                line.setContentsMargins(0, 0, 0, 0)
                line.setSpacing(6)
                line.addWidget(widget, 1)
                line.addWidget(hint, 0)
                form.addRow(label, row)
            else:
                form.addRow(label, widget)
            widgets[name] = widget
            fields[name] = spec
            if not unknown:
                self._answered_values.setdefault(node.key, {})[name] = value
        box.add(body)
        return box

    def _with_origin(self, widget: QWidget, origin: str) -> QWidget:
        """Put a short note beside a field the data usually answers.

        Only for the surprising case. Almost everything the form asks about is
        simply not in the data, and saying so on every row would be noise; a
        field the converter USUALLY fills and did not is worth a look, because
        the value often exists at the scanner or was stripped by an anonymiser.
        """
        from ...metadata.template_plan import ORIGIN_USUALLY_DERIVED

        if origin != ORIGIN_USUALLY_DERIVED:
            return widget
        pal = CUR()
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(6)
        line.addWidget(widget, 1)
        note = QLabel("not in your files")
        note.setStyleSheet(f"color: {pal['warning']};")
        note.setToolTip(_origin_tooltip(origin))
        line.addWidget(note, 0)
        return row

    # -- reading -------------------------------------------------------

    def values_by_key(self) -> dict[str, dict]:
        """What the user has typed, per node, as the schema shapes it."""
        # Sections nobody opened keep what they already held. Their widgets were
        # never built, so there is nothing to read, and returning nothing would
        # erase the answers on the next save.
        out: dict[str, dict] = {
            key: dict(values)
            for key, values in self._stored.items()
            if key not in self._widgets and values
        }
        for key, widgets in self._widgets.items():
            shown = self._answered_values.get(key, {})
            answers = {}
            for name, widget in widgets.items():
                value = read_field_widget(widget, self._fields[key][name])
                if value in (None, "", [], {}):
                    continue
                # Echoing what the conversion writes is not a statement. Storing
                # it would freeze today's value onto every file of the kind.
                if name in shown and shown[name] == value:
                    continue
                answers[name] = value
            if answers:
                out[key] = answers
        return out

    def asked(self, key: str) -> set:
        """The fields this node actually ASKS about.

        ``_widgets`` holds those plus the ones folded away under "already
        answered by the conversion", because both are editable. This is the
        subset the form is putting to the user.
        """
        section = self._sections_by_key.get(key)
        return {f.name for f in section.fields} if section is not None else set()

    def supplied(self, key: str) -> set:
        """The fields folded away because the conversion supplies them."""
        section = self._sections_by_key.get(key)
        return set(section.supplied) if section is not None else set()

    def section_widget(self, key: str):
        """The section for a node key, so a caller can scroll to or open it."""
        return self._sections.get(key)

    def reveal(self, section) -> None:
        """Open every section above one, so jumping to it actually shows it."""
        widget = section
        while widget is not None:
            if isinstance(widget, CollapsibleSection):
                widget.set_expanded(True)
            widget = widget.parentWidget()

    def collapsed_keys(self) -> set:
        """Which sections the user folded, to restore next time."""
        return {
            key for key, section in self._sections.items()
            if not section.is_expanded()
        }
