"""Inspector prototype — PyQt6 visual fidelity test.

Two views (Converter / Editor) inside a single window, switchable via the
top header. Theme toggle (dark/light) in the same header. The QSS is
generated at runtime from a token-based template (`theme.qss`) so a
palette swap is a single function call.

Run:
    cd inspector_proto
    ../.venv/bin/python proto.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from string import Template

from PyQt6.QtCore import Qt, QSize, QRect, QModelIndex, pyqtSignal
from PyQt6.QtGui import (
    QPalette, QColor, QFont, QPainter, QPen, QStandardItemModel, QStandardItem,
    QFontMetrics, QPainterPath,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QLabel, QLineEdit, QTreeWidget, QTreeWidgetItem,
    QTableView, QHeaderView, QTabWidget, QComboBox, QStyledItemDelegate,
    QStyleOptionViewItem, QStatusBar, QFrame, QSizePolicy, QAbstractItemView,
    QStackedWidget, QButtonGroup, QScrollArea,
)

import data as D


# =====================================================================
#  PALETTES
# =====================================================================
DARK = {
    'bg':         '#0a0e13',
    'surface':    '#11161d',
    'surface2':   '#161b22',
    'surface3':   '#1c2128',
    'border':     '#21262d',
    'subtle':     '#1a1f26',
    'text':       '#e6edf3',
    'dim':        '#8b949e',
    'muted':      '#656d76',
    'accent':     '#58a6ff',
    'success':    '#3fb950',
    'warning':    '#d29922',
    'error':      '#f85149',
    'purple':     '#d2a8ff',
    'teal':       '#39c5cf',

    'accent_bg':       'rgba(88,166,255,0.12)',
    'accent_border':   'rgba(88,166,255,0.40)',
    'success_bg':      'rgba(63,185,80,0.12)',
    'success_border':  'rgba(63,185,80,0.30)',
    'warning_bg':      'rgba(210,153,34,0.12)',
    'warning_border':  'rgba(210,153,34,0.30)',
    'error_bg':        'rgba(248,81,73,0.12)',
    'error_border':    'rgba(248,81,73,0.30)',
    'purple_bg':       'rgba(210,168,255,0.12)',
    'purple_border':   'rgba(210,168,255,0.30)',
    'teal_bg':         'rgba(57,197,207,0.12)',
    'teal_border':     'rgba(57,197,207,0.30)',

    'primary_btn_text': '#0a0e13',
    'pressed_alpha':    'rgba(255,255,255,0.04)',
}

LIGHT = {
    'bg':         '#ffffff',
    'surface':    '#f6f8fa',
    'surface2':   '#ffffff',
    'surface3':   '#eef1f4',
    'border':     '#d0d7de',
    'subtle':     '#e5e7ea',
    'text':       '#1f2328',
    'dim':        '#656d76',
    'muted':      '#8c959f',
    'accent':     '#0969da',
    'success':    '#1a7f37',
    'warning':    '#9a6700',
    'error':      '#cf222e',
    'purple':     '#8250df',
    'teal':       '#1d7a8c',

    'accent_bg':       'rgba(9,105,218,0.08)',
    'accent_border':   'rgba(9,105,218,0.32)',
    'success_bg':      'rgba(26,127,55,0.10)',
    'success_border':  'rgba(26,127,55,0.30)',
    'warning_bg':      'rgba(154,103,0,0.10)',
    'warning_border':  'rgba(154,103,0,0.30)',
    'error_bg':        'rgba(207,34,46,0.10)',
    'error_border':    'rgba(207,34,46,0.30)',
    'purple_bg':       'rgba(130,80,223,0.10)',
    'purple_border':   'rgba(130,80,223,0.30)',
    'teal_bg':         'rgba(29,122,140,0.10)',
    'teal_border':     'rgba(29,122,140,0.30)',

    'primary_btn_text': '#ffffff',
    'pressed_alpha':    'rgba(0,0,0,0.04)',
}

PALETTES = {'dark': DARK, 'light': LIGHT}


def rgba(hex6: str, alpha: float) -> QColor:
    c = QColor(hex6)
    c.setAlphaF(alpha)
    return c


# =====================================================================
#  THEME MANAGER
# =====================================================================
class ThemeManager:
    """Owns the QSS template + active palette. Re-applies on toggle."""

    def __init__(self, app: QApplication):
        self._app = app
        self._template = Template((Path(__file__).parent / 'theme.qss').read_text(encoding='utf-8'))
        self._theme = 'dark'
        self._listeners: list = []

    def add_listener(self, fn):
        """fn(palette: dict) called after theme changes."""
        self._listeners.append(fn)

    @property
    def palette(self) -> dict:
        return PALETTES[self._theme]

    @property
    def name(self) -> str:
        return self._theme

    def apply(self, theme: str):
        if theme not in PALETTES:
            return
        self._theme = theme
        pal = PALETTES[theme]
        self._app.setStyleSheet(self._template.safe_substitute(**pal))
        self._update_qpalette(pal)
        for fn in self._listeners:
            try:
                fn(pal)
            except Exception as exc:
                print(f'[theme listener] {exc}')

    def toggle(self) -> str:
        self.apply('light' if self._theme == 'dark' else 'dark')
        return self._theme

    def _update_qpalette(self, pal: dict):
        p = QPalette()
        p.setColor(QPalette.ColorRole.Window,          QColor(pal['bg']))
        p.setColor(QPalette.ColorRole.WindowText,      QColor(pal['text']))
        p.setColor(QPalette.ColorRole.Base,            QColor(pal['bg']))
        p.setColor(QPalette.ColorRole.AlternateBase,   QColor(pal['surface']))
        p.setColor(QPalette.ColorRole.Text,            QColor(pal['text']))
        p.setColor(QPalette.ColorRole.Button,          QColor(pal['surface3']))
        p.setColor(QPalette.ColorRole.ButtonText,      QColor(pal['text']))
        p.setColor(QPalette.ColorRole.Highlight,       QColor(pal['accent']))
        p.setColor(QPalette.ColorRole.HighlightedText, QColor(pal['primary_btn_text']))
        p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(pal['surface2']))
        p.setColor(QPalette.ColorRole.ToolTipText,     QColor(pal['text']))
        self._app.setPalette(p)


# Live palette accessor used by paint code (delegates etc.)
_THEME_REF: dict = DARK
def CUR() -> dict:
    return _THEME_REF


# =====================================================================
#  PRIMITIVE WIDGETS
# =====================================================================
class Chip(QLabel):
    def __init__(self, text: str, kind: str = '', parent=None):
        super().__init__(text, parent)
        self.setProperty('chipKind', kind or 'default')
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)


class VSep(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('vsep')
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFixedWidth(1)


class PaneHeader(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text.upper(), parent)
        self.setObjectName('pane-h5')
        self.setFixedHeight(28)


class PathBar(QFrame):
    def __init__(self, label: str, value: str, ok: bool = False,
                 trailing_chips: list[tuple[str, str]] | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName('pathbar')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 9, 14, 9)
        lay.setSpacing(10)

        lbl = QLabel(label); lbl.setObjectName('path-label')
        lbl.setMinimumWidth(80)
        lay.addWidget(lbl)

        ico = '✔  ' if ok else '○  '
        field = QLineEdit(f'{ico}{value}'); field.setObjectName('path-field'); field.setReadOnly(True)
        lay.addWidget(field, 1)

        for chip_kind, chip_text in (trailing_chips or []):
            lay.addWidget(Chip(chip_text, chip_kind))

        btn = QPushButton('change…'); btn.setObjectName('tb-btn-ghost')
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        lay.addWidget(btn)


# =====================================================================
#  STATUS BADGES (used in cells, validation rows)
# =====================================================================
KIND_CHAR = {
    'ok': '✓', 'warn': '!', 'err': '✕', 'phys': 'P', 'skip': '−', 'info': 'i',
}
KIND_BG_TOKEN = {  # which palette tokens to use for bg + fg
    'ok':   ('success', 0.18),
    'warn': ('warning', 0.18),
    'err':  ('error',   0.18),
    'phys': ('accent',  0.18),
    'skip': ('muted',   0.20),
    'info': ('accent',  0.18),
}
KIND_FG_TOKEN = {
    'ok': 'success', 'warn': 'warning', 'err': 'error',
    'phys': 'accent', 'skip': 'dim', 'info': 'accent',
}


def _badge_paint(painter: QPainter, rect: QRect, kind: str, size: int = 16):
    pal = CUR()
    bg_tok, alpha = KIND_BG_TOKEN.get(kind, ('muted', 0.20))
    fg_tok = KIND_FG_TOKEN.get(kind, 'dim')
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    cx, cy = rect.center().x(), rect.center().y()
    r = QRect(cx - size // 2, cy - size // 2, size, size)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(rgba(pal[bg_tok], alpha))
    painter.drawEllipse(r)
    painter.setPen(QColor(pal[fg_tok]))
    f = QFont(painter.font()); f.setBold(True); f.setPointSize(8)
    painter.setFont(f)
    painter.drawText(r, Qt.AlignmentFlag.AlignCenter, KIND_CHAR.get(kind, '?'))


class StatusBadge(QLabel):
    def __init__(self, kind: str = 'ok', parent=None):
        super().__init__(parent)
        self._kind = kind
        self.setFixedSize(18, 18)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        _badge_paint(p, self.rect(), self._kind, 16)
        p.end()


# =====================================================================
#  TABLE DELEGATES (Converter inspection table)
# =====================================================================
def _row_bg(painter, option, row_state):
    pal = CUR()
    if row_state == 'selected':
        painter.fillRect(option.rect, rgba(pal['accent'], 0.16))
    elif row_state == 'warn':
        painter.fillRect(option.rect, rgba(pal['warning'], 0.06))
    elif row_state == 'err':
        painter.fillRect(option.rect, rgba(pal['error'], 0.06))
    elif row_state == 'skip':
        painter.fillRect(option.rect, QColor(pal['bg']))


class StatusDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        _row_bg(painter, option, index.data(Qt.ItemDataRole.UserRole + 1) or '')
        kind = index.data(Qt.ItemDataRole.UserRole) or 'ok'
        painter.save()
        _badge_paint(painter, option.rect, kind, 16)
        painter.restore()


class CheckboxDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        _row_bg(painter, option, index.data(Qt.ItemDataRole.UserRole + 1) or '')
        pal = CUR()
        checked = bool(index.data(Qt.ItemDataRole.UserRole))
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = 13
        cx, cy = option.rect.center().x(), option.rect.center().y()
        rect = QRect(cx - size // 2, cy - size // 2, size, size)
        path = QPainterPath()
        path.addRoundedRect(rect.x(), rect.y(), rect.width(), rect.height(), 3, 3)
        if checked:
            painter.fillPath(path, QColor(pal['accent']))
            painter.setPen(QColor(pal['primary_btn_text']))
            f = QFont(painter.font()); f.setBold(True); f.setPointSize(8)
            painter.setFont(f)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, '✓')
        else:
            painter.setPen(QPen(QColor(pal['border']), 1))
            painter.setBrush(QColor(pal['bg']))
            painter.drawPath(path)
        painter.restore()


class CellTextDelegate(QStyledItemDelegate):
    def __init__(self, role: str = 'plain', parent=None):
        super().__init__(parent)
        self._role = role

    def paint(self, painter, option, index):
        pal = CUR()
        row_state = index.data(Qt.ItemDataRole.UserRole + 1) or ''
        _row_bg(painter, option, row_state)

        text = str(index.data(Qt.ItemDataRole.DisplayRole) or '')
        if not text:
            return
        painter.save()
        f = QFont(painter.font())
        if self._role in ('mono', 'basename', 'conf'):
            f.setFamilies(['SF Mono', 'Menlo', 'Monaco', 'Consolas', 'monospace'])
            f.setPointSize(11)
        else:
            f.setPointSize(11)
        painter.setFont(f)

        color = QColor(pal['text'])
        if text == '—':
            color = QColor(pal['muted'])
        elif self._role == 'basename':
            color = QColor(pal['dim'])
            if row_state == 'err':
                color = QColor(pal['error'])
        elif self._role == 'conf':
            try:
                v = float(text)
                if v >= 0.9:   color = QColor(pal['success'])
                elif v >= 0.75: color = QColor(pal['warning'])
                else:           color = QColor(pal['error'])
            except ValueError:
                color = QColor(pal['muted'])
        elif text == 'missing':
            color = QColor(pal['error'])
        if row_state == 'skip':
            color = QColor(pal['muted'])

        painter.setPen(color)
        r = option.rect.adjusted(8, 0, -8, 0)
        fm = QFontMetrics(f)
        elided = fm.elidedText(text, Qt.TextElideMode.ElideRight, r.width())
        painter.drawText(r, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), elided)

        if row_state == 'skip' and self._role == 'basename':
            br = fm.boundingRect(elided)
            y = r.center().y()
            painter.setPen(QPen(QColor(pal['muted']), 1))
            painter.drawLine(r.left(), y, r.left() + min(br.width(), r.width()), y)

        painter.restore()


# =====================================================================
#  BIDS-VALIDATION-TREE DELEGATE (Editor)
# =====================================================================
class BidsTreeDelegate(QStyledItemDelegate):
    """Paints a small validation badge to the right of each tree row."""

    def paint(self, painter, option, index):
        pal = CUR()
        # background: selection highlight + active-file tint
        is_sel = bool(option.state & QStyle.StateFlag.State_Selected) if False else False
        # Qt's QStyle is from QtWidgets — just rely on default selection paint
        super().paint(painter, option, index)
        badge = index.data(Qt.ItemDataRole.UserRole + 2)  # 'ok'|'warn'|'err' or None
        if not badge:
            return
        painter.save()
        size = 8
        margin = 12
        bx = option.rect.right() - margin
        by = option.rect.center().y()
        token = {'ok': 'success', 'warn': 'warning', 'err': 'error'}.get(badge, 'success')
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(pal[token]))
        painter.drawEllipse(bx - size // 2, by - size // 2, size, size)
        painter.restore()


# =====================================================================
#  EDITOR: SIDECAR ROW WIDGET
# =====================================================================
class SidecarRow(QFrame):
    def __init__(self, level: str, key: str, value: str, value_kind: str, parent=None):
        super().__init__(parent)
        self.setObjectName('sc-row')
        self._level = level
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 4, 0, 4)
        h.setSpacing(10)

        # 4px colored bar
        bar = QFrame()
        bar.setFixedSize(4, 18)
        h.addWidget(bar)
        self._bar = bar

        # key
        key_lbl = QLabel(f'"{key}"')
        key_lbl.setObjectName('sc-key-dep' if level == 'dep' else 'sc-key')
        key_lbl.setMinimumWidth(220)
        if level == 'dep':
            f = key_lbl.font(); f.setStrikeOut(True); key_lbl.setFont(f)
        h.addWidget(key_lbl)

        # value (rendered with token coloring)
        if value_kind == 'todo':
            val_lbl = QLabel(value); val_lbl.setObjectName('sc-val-todo')
        elif value_kind == 'num':
            val_lbl = QLabel(value); val_lbl.setObjectName('sc-val-num')
        else:
            val_lbl = QLabel(f'"{value}"'); val_lbl.setObjectName('sc-val-str')
        val_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        h.addWidget(val_lbl, 1)

        self._apply_bar_color(CUR())

    def _apply_bar_color(self, pal):
        token = {'req': 'error', 'rec': 'warning', 'opt': 'muted', 'dep': 'muted'}[self._level]
        opacity = 1.0 if self._level in ('req', 'rec') else (0.4 if self._level == 'opt' else 1.0)
        c = QColor(pal[token])
        c.setAlphaF(opacity)
        self._bar.setStyleSheet(f'background: {c.name(QColor.NameFormat.HexArgb)};')

    def repaint_for_palette(self, pal):
        self._apply_bar_color(pal)


# =====================================================================
#  EDITOR: VALIDATION MESSAGE
# =====================================================================
class ValMessage(QFrame):
    def __init__(self, severity: str, rule: str, body_html: str,
                 fix_label: str | None, parent=None):
        super().__init__(parent)
        obj = {'ok': 'val-msg-ok', 'warn': 'val-msg-warn', 'err': 'val-msg-err'}.get(severity, 'val-msg')
        self.setObjectName(obj)
        h = QHBoxLayout(self); h.setContentsMargins(10, 7, 10, 7); h.setSpacing(10)
        h.addWidget(StatusBadge(severity), 0, Qt.AlignmentFlag.AlignTop)
        right = QVBoxLayout(); right.setSpacing(4)
        rule_l = QLabel(rule); rule_l.setObjectName('val-rule')
        right.addWidget(rule_l)
        body_widget = QHBoxLayout(); body_widget.setSpacing(8)
        body_l = QLabel(body_html); body_l.setObjectName('val-body')
        body_l.setWordWrap(True); body_l.setTextFormat(Qt.TextFormat.RichText)
        body_widget.addWidget(body_l, 1)
        if fix_label:
            btn = QPushButton(fix_label); btn.setObjectName('val-fix')
            body_widget.addWidget(btn, 0, Qt.AlignmentFlag.AlignTop)
        right.addLayout(body_widget)
        h.addLayout(right, 1)


# =====================================================================
#  CONVERTER VIEW (4-column splitter + bottom dock)
# =====================================================================
class ConverterView(QWidget):

    COLS = ['', '', 'id', 'ses', 'mod', 'data', 'suffix', 'task', 'run',
            'conf', 'predicted basename', 'backend']
    COL_WIDTHS = [28, 28, 50, 50, 38, 50, 80, 60, 50, 50, 320, 90]

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(self._toolbar())
        v.addWidget(PathBar('Raw input',
                            '~/datasets/raw_data/MRI/neuroimaging_unit_new', ok=True))
        v.addWidget(PathBar('BIDS output',
                            '~/datasets/BIDS_out/Studyname'))

        v_split = QSplitter(Qt.Orientation.Vertical)
        v_split.setHandleWidth(1); v_split.setChildrenCollapsible(False)
        v.addWidget(v_split, 1)

        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setHandleWidth(1); h_split.setChildrenCollapsible(False)
        h_split.addWidget(self._raw_pane())
        h_split.addWidget(self._filter_pane())
        h_split.addWidget(self._inv_pane())
        h_split.addWidget(self._props_pane())
        h_split.setStretchFactor(0, 0); h_split.setStretchFactor(1, 0)
        h_split.setStretchFactor(2, 1); h_split.setStretchFactor(3, 0)
        h_split.setSizes([240, 220, 720, 320])
        v_split.addWidget(h_split)
        v_split.addWidget(self._bottom_dock())
        v_split.setStretchFactor(0, 1); v_split.setStretchFactor(1, 0)
        v_split.setSizes([580, 200])

    def _toolbar(self):
        bar = QFrame(); bar.setObjectName('toolbar'); bar.setFixedHeight(44)
        lay = QHBoxLayout(bar); lay.setContentsMargins(14, 6, 14, 6); lay.setSpacing(8)
        scan = QPushButton('⌖  Scan…'); scan.setObjectName('tb-btn')
        revert = QPushButton('⟲'); revert.setObjectName('tb-btn-ghost'); revert.setFixedWidth(28)
        clear  = QPushButton('⌫'); clear.setObjectName('tb-btn-ghost');  clear.setFixedWidth(28)
        lay.addWidget(scan); lay.addWidget(revert); lay.addWidget(clear)
        lay.addWidget(VSep())
        s = D.TOOLBAR_STATS
        lay.addWidget(Chip(f'{s["valid"]} valid',   'success'))
        lay.addWidget(Chip(f'{s["warn"]} warnings', 'warn'))
        lay.addWidget(Chip(f'{s["error"]} error',   'err'))
        lay.addWidget(Chip(f'{s["skipped"]} skipped'))
        lay.addWidget(VSep())
        lay.addWidget(Chip('BIDS 1.10.0',           'purple'))
        lay.addWidget(Chip('dcm2niix 1.0.20250506', 'teal'))
        lay.addStretch(1)
        settings = QPushButton('⚙  Settings'); settings.setObjectName('tb-btn')
        run      = QPushButton('▶  Run conversion'); run.setObjectName('tb-btn-primary')
        lay.addWidget(settings); lay.addWidget(run)
        return bar

    def _raw_pane(self):
        pane = QWidget(); pane.setObjectName('pane'); pane.setMinimumWidth(200)
        v = QVBoxLayout(pane); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(PaneHeader('Raw data tree'))
        tree = QTreeWidget(); tree.setObjectName('raw-tree')
        tree.setHeaderHidden(True); tree.setRootIsDecorated(False)
        tree.setIndentation(14); tree.setUniformRowHeights(True)

        stack: list[QTreeWidgetItem] = []
        for depth, kind, label, meta, state in D.RAW_TREE:
            text = label + (f'    {meta}' if meta else '')
            it = QTreeWidgetItem([text])
            if depth == 0:
                tree.addTopLevelItem(it)
            else:
                while len(stack) > depth: stack.pop()
                (stack[-1] if stack else tree).addChild(it) if stack else tree.addTopLevelItem(it)
            if kind == 'series-skip':
                it.setForeground(0, QColor(CUR()['muted']))
                f = it.font(0); f.setStrikeOut(True); it.setFont(0, f)
            elif kind == 'series-active':
                it.setBackground(0, rgba(CUR()['accent'], 0.12))
                it.setForeground(0, QColor(CUR()['accent']))
            else:
                it.setForeground(0, QColor(CUR()['text']))
            if state == 'expanded': it.setExpanded(True)
            stack = stack[:depth] + [it]
        v.addWidget(tree, 1)
        return pane

    def _filter_pane(self):
        pane = QWidget(); pane.setObjectName('pane'); pane.setMinimumWidth(190)
        v = QVBoxLayout(pane); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(PaneHeader('Filter / structure'))
        tree = QTreeWidget(); tree.setObjectName('filter-tree')
        tree.setHeaderHidden(True); tree.setRootIsDecorated(True)
        tree.setIndentation(16); tree.setUniformRowHeights(True)

        stack: list[QTreeWidgetItem] = []
        for depth, label, count, state in D.FILTER_TREE:
            text = f'{label}    {count}'
            it = QTreeWidgetItem([text])
            it.setFlags(it.flags()
                        | Qt.ItemFlag.ItemIsUserCheckable
                        | Qt.ItemFlag.ItemIsAutoTristate)
            it.setCheckState(0, {
                'checked':   Qt.CheckState.Checked,
                'partial':   Qt.CheckState.PartiallyChecked,
                'unchecked': Qt.CheckState.Unchecked,
            }[state])
            if depth == 0:
                tree.addTopLevelItem(it)
            else:
                while len(stack) > depth: stack.pop()
                (stack[-1] if stack else tree).addChild(it) if stack else tree.addTopLevelItem(it)
            stack = stack[:depth] + [it]
        tree.expandAll()
        v.addWidget(tree, 1)

        excl = QFrame(); ev = QVBoxLayout(excl)
        ev.setContentsMargins(12, 8, 12, 10); ev.setSpacing(2)
        h = QLabel('ALWAYS EXCLUDE')
        h.setStyleSheet('color: ' + CUR()['muted'] + '; font-size: 10px; font-weight: 600;')
        ev.addWidget(h)
        for pat in D.ALWAYS_EXCLUDE:
            l = QLabel(pat)
            l.setStyleSheet('color: ' + CUR()['dim'] + '; font-size: 10px; '
                            'font-family: "SF Mono","Menlo","Monaco",monospace;')
            ev.addWidget(l)
        v.addWidget(excl)
        return pane

    def _inv_pane(self):
        pane = QWidget(); pane.setObjectName('pane-dark')
        v = QVBoxLayout(pane); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(PaneHeader('Inspection'))

        model = QStandardItemModel(len(D.INVENTORY), len(self.COLS))
        model.setHorizontalHeaderLabels(self.COLS)
        for r, row in enumerate(D.INVENTORY):
            (status, sub, ses, mod, dt, suf, task, run, conf,
             basename, backend, row_state, included) = row
            it_inc = QStandardItem('')
            it_inc.setData(included, Qt.ItemDataRole.UserRole)
            it_inc.setData(row_state, Qt.ItemDataRole.UserRole + 1)
            it_inc.setEditable(False); model.setItem(r, 0, it_inc)
            it_st = QStandardItem('')
            it_st.setData(status, Qt.ItemDataRole.UserRole)
            it_st.setData(row_state, Qt.ItemDataRole.UserRole + 1)
            it_st.setEditable(False); model.setItem(r, 1, it_st)
            for col, val in enumerate([sub, ses, mod, dt, suf, task, run, conf, basename, backend], start=2):
                it = QStandardItem(str(val))
                it.setData(row_state, Qt.ItemDataRole.UserRole + 1)
                it.setEditable(col in (5, 6, 7, 8))
                model.setItem(r, col, it)

        table = QTableView(); table.setObjectName('inv-table'); table.setModel(model)
        table.setShowGrid(False); table.setAlternatingRowColors(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(26)
        table.horizontalHeader().setHighlightSections(False)
        table.horizontalHeader().setStretchLastSection(False)
        for i, w in enumerate(self.COL_WIDTHS):
            table.setColumnWidth(i, w)
        table.horizontalHeader().setSectionResizeMode(10, QHeaderView.ResizeMode.Stretch)
        table.setItemDelegateForColumn(0, CheckboxDelegate(table))
        table.setItemDelegateForColumn(1, StatusDelegate(table))
        for col, role in [
            (2, 'mono'), (3, 'mono'), (4, 'plain'), (5, 'plain'),
            (6, 'plain'), (7, 'mono'), (8, 'mono'), (9, 'conf'),
            (10, 'basename'), (11, 'mono'),
        ]:
            table.setItemDelegateForColumn(col, CellTextDelegate(role, table))
        for r, row in enumerate(D.INVENTORY):
            if row[11] == 'selected':
                table.selectRow(r); break
        v.addWidget(table, 1)
        return pane

    def _props_pane(self):
        pane = QWidget(); pane.setObjectName('pane'); pane.setMinimumWidth(280)
        v = QVBoxLayout(pane); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(PaneHeader('Properties · 1 row selected'))

        body = QWidget(); body.setObjectName('props-panel')
        bl = QVBoxLayout(body); bl.setContentsMargins(14, 12, 14, 12); bl.setSpacing(7)

        bl.addWidget(self._labeled_row('datatype *', QComboBox, options=['func']))
        bl.addWidget(self._labeled_row('suffix *',   QComboBox, options=['bold']))
        bl.addSpacing(4); bl.addWidget(self._divider()); bl.addSpacing(2)

        ent_title = QLabel('Entities'); ent_title.setObjectName('section-title-bold')
        sub = QLabel('  (schema-driven)'); sub.setStyleSheet(f'color: {CUR()["dim"]}; font-weight: 400; font-size: 10px;')
        h = QHBoxLayout(); h.setSpacing(0); h.addWidget(ent_title); h.addWidget(sub); h.addStretch(1)
        bl.addLayout(h); bl.addSpacing(2)
        for name, value, required, opt in D.SELECTED_PROPS['entities']:
            label = name + (' *' if required else '')
            if opt: label += '   opt'
            bl.addWidget(self._labeled_row(label, QLineEdit, value=value, opt=opt is not None))

        bl.addSpacing(6); bl.addWidget(self._divider())
        sec = QLabel('PREDICTED PATH'); sec.setObjectName('section-title')
        bl.addWidget(sec); bl.addWidget(self._path_preview())

        bl.addSpacing(8)
        for kind, msg in D.SELECTED_PROPS['validation']:
            bl.addWidget(self._valmsg(kind, msg))

        bl.addSpacing(6); bl.addWidget(self._divider())
        why_title = QLabel('WHY THIS NAME?'); why_title.setObjectName('section-title')
        bl.addWidget(why_title)

        why = QFrame(); why.setObjectName('why-panel')
        wl = QVBoxLayout(why); wl.setContentsMargins(11, 9, 11, 9); wl.setSpacing(6)
        for head, body_text in D.SELECTED_PROPS['why']:
            row = QVBoxLayout(); row.setSpacing(2)
            hl = QLabel(head); hl.setStyleSheet(f'color: {CUR()["text"]}; font-size: 11px; font-weight: 500;')
            tl = QLabel(body_text); tl.setStyleSheet(f'color: {CUR()["dim"]}; font-size: 11px;'); tl.setWordWrap(True)
            row.addWidget(hl); row.addWidget(tl)
            wrap = QFrame(); wrap.setLayout(row); wl.addWidget(wrap)
        bl.addWidget(why); bl.addStretch(1)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f'QScrollArea {{ background: {CUR()["surface"]}; }}')
        scroll.setWidget(body); v.addWidget(scroll, 1)
        return pane

    def _labeled_row(self, label_text, widget_cls, value='', options=None, opt=False):
        row = QWidget(); row.setStyleSheet('background: transparent;')
        h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(8)
        is_required = '*' in label_text
        lbl = QLabel(label_text)
        lbl.setObjectName('field-label-req' if is_required else
                          'field-label-opt' if opt else 'field-label')
        lbl.setMinimumWidth(76); lbl.setMaximumWidth(76)
        if is_required:
            lbl.setText(label_text.replace('*', f'<span style="color:{CUR()["error"]}">*</span>'))
            lbl.setTextFormat(Qt.TextFormat.RichText)
        if opt:
            lbl.setText(label_text.replace('opt', f'<span style="color:{CUR()["muted"]};font-size:9px;">opt</span>'))
            lbl.setTextFormat(Qt.TextFormat.RichText)
        h.addWidget(lbl)
        if widget_cls is QComboBox:
            w = QComboBox(); w.setObjectName('ent-input'); w.addItems(options or [])
            w.setMinimumHeight(22)
        else:
            w = QLineEdit(value); w.setObjectName('ent-input'); w.setPlaceholderText('—')
        h.addWidget(w, 1)
        return row

    def _divider(self):
        d = QFrame()
        d.setStyleSheet(f'background: {CUR()["subtle"]}; max-height: 1px; min-height: 1px; border: none;')
        return d

    def _path_preview(self):
        f = QFrame(); f.setObjectName('path-preview')
        l = QVBoxLayout(f); l.setContentsMargins(11, 9, 11, 9); l.setSpacing(0)
        pal = CUR()
        seg, ent, suf, dim, ext = pal['accent'], pal['purple'], pal['teal'], pal['dim'], pal['dim']
        pieces = []
        for kind, val in D.SELECTED_PROPS['predicted_path']:
            if kind == 'plain':   pieces.append(val.replace(' ', '&nbsp;'))
            elif kind == 'newline': pieces.append('<br>')
            elif kind == 'seg':   pieces.append(f'<span style="color:{seg}">{val}</span>')
            elif kind == 'ent':   pieces.append(f'<span style="color:{ent}">{val}</span>')
            elif kind == 'suf':   pieces.append(f'<span style="color:{suf}">{val}</span>')
            elif kind == 'ext':   pieces.append(f'<span style="color:{ext}">{val}</span>')
            elif kind == 'dim':   pieces.append(f'<span style="color:{dim}">{val}</span>')
        lbl = QLabel(''.join(pieces)); lbl.setObjectName('path-preview-text')
        lbl.setTextFormat(Qt.TextFormat.RichText); lbl.setWordWrap(True)
        lbl.setStyleSheet('font-family: "SF Mono","Menlo","Monaco",monospace; '
                          f'font-size: 11px; color: {pal["text"]}; background: transparent;')
        l.addWidget(lbl)
        return f

    def _valmsg(self, kind, text):
        row = QWidget(); row.setStyleSheet('background: transparent;')
        h = QHBoxLayout(row); h.setContentsMargins(0, 1, 0, 1); h.setSpacing(7)
        h.addWidget(StatusBadge(kind))
        lbl = QLabel(text); lbl.setStyleSheet(f'color: {CUR()["dim"]}; font-size: 11px;')
        lbl.setWordWrap(True); h.addWidget(lbl, 1)
        return row

    def _bottom_dock(self):
        tabs = QTabWidget(); tabs.setDocumentMode(True); tabs.setMovable(False)
        tabs.addTab(self._bids_preview(),       '📤  BIDS preview')
        tabs.addTab(QLabel('Log output…'),       '📋  Log')
        tabs.addTab(QLabel('Conflicts (1)'),     '⚠  Conflicts (1)')
        tabs.addTab(QLabel('Statistics'),        '📊  Statistics')
        return tabs

    def _bids_preview(self):
        pal = CUR()
        tree = QTreeWidget(); tree.setObjectName('raw-tree')
        tree.setHeaderHidden(True); tree.setRootIsDecorated(False); tree.setIndentation(14)
        stack: list[QTreeWidgetItem] = []
        for depth, kind, label, badge in D.BIDS_PREVIEW:
            it = QTreeWidgetItem([label + ('   [new]' if badge == 'new' else '')])
            color = {'dir': pal['accent'], 'json': pal['purple'], 'tsv': pal['teal'],
                     'nii': pal['text'], 'other': pal['dim']}.get(kind, pal['text'])
            it.setForeground(0, QColor(color))
            if depth == 0: tree.addTopLevelItem(it)
            else:
                while len(stack) > depth: stack.pop()
                (stack[-1] if stack else tree).addChild(it) if stack else tree.addTopLevelItem(it)
            stack = stack[:depth] + [it]
        tree.expandAll()
        return tree


# =====================================================================
#  EDITOR VIEW (3-column splitter)
# =====================================================================
class EditorView(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sidecar_rows: list[SidecarRow] = []
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(self._toolbar())
        v.addWidget(PathBar(
            'Dataset', '~/datasets/BIDS_out/Studyname', ok=True,
            trailing_chips=[('purple', 'BIDS 1.10.0'), ('default', 'Generated by BIDS-Manager')],
        ))

        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setHandleWidth(1); h_split.setChildrenCollapsible(False)
        h_split.addWidget(self._left_pane())
        h_split.addWidget(self._center_pane())
        h_split.addWidget(self._right_pane())
        h_split.setStretchFactor(0, 0); h_split.setStretchFactor(1, 1); h_split.setStretchFactor(2, 0)
        h_split.setSizes([320, 700, 380])
        v.addWidget(h_split, 1)

    def _toolbar(self):
        bar = QFrame(); bar.setObjectName('toolbar'); bar.setFixedHeight(44)
        lay = QHBoxLayout(bar); lay.setContentsMargins(14, 6, 14, 6); lay.setSpacing(8)
        lay.addWidget(self._tb_btn('📁  Open BIDS root…'))
        lay.addWidget(VSep())
        lay.addWidget(self._tb_btn('✓  Validate file'))
        lay.addWidget(self._tb_btn('📂  Validate folder'))
        lay.addWidget(self._tb_btn('🗂  Validate dataset'))
        lay.addWidget(VSep())
        s = D.EDITOR_STATS
        lay.addWidget(Chip(f'✓ {s["valid"]} valid',  'success'))
        lay.addWidget(Chip(f'⚠ {s["warn"]} warnings', 'warn'))
        lay.addWidget(Chip(f'✕ {s["error"]} errors',  'err'))
        lay.addStretch(1)
        ts = QLabel(f'last validated: {s["last_validated"]}')
        ts.setStyleSheet(f'color: {CUR()["dim"]}; font-size: 11px;')
        lay.addWidget(ts)
        revalidate = QPushButton('↻'); revalidate.setObjectName('tb-btn-ghost'); revalidate.setFixedWidth(28)
        lay.addWidget(revalidate)
        return bar

    def _tb_btn(self, text):
        b = QPushButton(text); b.setObjectName('tb-btn'); return b

    def _left_pane(self):
        pane = QWidget(); pane.setObjectName('pane'); pane.setMinimumWidth(260)
        v = QVBoxLayout(pane); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(PaneHeader('BIDS dataset'))

        tree = QTreeWidget(); tree.setObjectName('raw-tree')
        tree.setHeaderHidden(True); tree.setRootIsDecorated(False)
        tree.setIndentation(14)
        tree.setItemDelegate(BidsTreeDelegate(tree))

        stack: list[QTreeWidgetItem] = []
        for depth, kind, label, badge in D.EDITOR_BIDS_TREE:
            it = QTreeWidgetItem([label])
            pal = CUR()
            color = {'dir': pal['accent'], 'json': pal['purple'], 'tsv': pal['teal'],
                     'nii': pal['text'], 'other': pal['dim']}.get(kind, pal['text'])
            it.setForeground(0, QColor(color))
            if badge:
                it.setData(0, Qt.ItemDataRole.UserRole + 2, badge)
            if depth == 0: tree.addTopLevelItem(it)
            else:
                while len(stack) > depth: stack.pop()
                (stack[-1] if stack else tree).addChild(it) if stack else tree.addTopLevelItem(it)
            stack = stack[:depth] + [it]
        tree.expandAll()
        v.addWidget(tree, 1)
        return pane

    def _center_pane(self):
        pane = QWidget(); pane.setObjectName('pane-dark')
        v = QVBoxLayout(pane); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)

        # File tabs
        ft = QFrame(); ft.setObjectName('file-tabs'); ft.setFixedHeight(36)
        fl = QHBoxLayout(ft); fl.setContentsMargins(8, 0, 8, 0); fl.setSpacing(0)
        group = QButtonGroup(ft); group.setExclusive(True)
        for kind, name, active in D.EDITOR_OPEN_TABS:
            ico = {'nii': '⊞', 'json': '{ }', 'tsv': '⊞'}.get(kind, '·')
            b = QPushButton(f'{ico}  {name}'); b.setObjectName('file-tab')
            b.setCheckable(True); b.setChecked(active)
            group.addButton(b)
            fl.addWidget(b)
        plus = QPushButton('+'); plus.setObjectName('file-tab')
        fl.addWidget(plus); fl.addStretch(1)
        v.addWidget(ft)

        # Schema legend
        legend = QFrame(); legend.setObjectName('schema-legend')
        ll = QHBoxLayout(legend); ll.setContentsMargins(14, 6, 14, 6); ll.setSpacing(14)
        for level, label in D.SCHEMA_LEGEND:
            chip = QFrame()
            ch = QHBoxLayout(chip); ch.setContentsMargins(0, 0, 0, 0); ch.setSpacing(6)
            sw = QFrame(); sw.setFixedSize(8, 12)
            token, alpha = {'req': ('error', 1.0), 'rec': ('warning', 1.0),
                            'opt': ('muted', 0.4), 'dep': ('muted', 1.0)}[level]
            c = QColor(CUR()[token]); c.setAlphaF(alpha)
            sw.setStyleSheet(f'background: {c.name(QColor.NameFormat.HexArgb)}; border-radius: 2px;')
            ch.addWidget(sw)
            t = QLabel(label); t.setObjectName('legend-text'); ch.addWidget(t)
            ll.addWidget(chip)
        ll.addStretch(1)
        ctx = QLabel(f'schema · {D.EDITOR_SIDECAR["datatype"]}/{D.EDITOR_SIDECAR["suffix"]}')
        ctx.setObjectName('legend-text')
        ll.addWidget(ctx)
        v.addWidget(legend)

        # Sidecar form (scrollable)
        body = QWidget()
        bl = QVBoxLayout(body); bl.setContentsMargins(14, 8, 14, 12); bl.setSpacing(0)
        for level, key, value, vk in D.EDITOR_SIDECAR['fields']:
            row = SidecarRow(level, key, value, vk)
            self._sidecar_rows.append(row)
            bl.addWidget(row)
        bl.addStretch(1)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        v.addWidget(scroll, 1)

        # Status footer
        bot = QFrame()
        bl2 = QHBoxLayout(bot); bl2.setContentsMargins(14, 6, 14, 6); bl2.setSpacing(10)
        path_l = QLabel(D.EDITOR_SIDECAR['path'])
        path_l.setStyleSheet(f'color: {CUR()["dim"]}; font-family: "SF Mono","Menlo","Monaco",monospace; font-size: 10px;')
        sum_l = QLabel(D.EDITOR_SIDECAR['summary'])
        sum_l.setStyleSheet(f'color: {CUR()["dim"]}; font-size: 10px;')
        bot.setStyleSheet(f'background: {CUR()["surface2"]}; border-top: 1px solid {CUR()["subtle"]};')
        bl2.addWidget(path_l, 1); bl2.addWidget(sum_l)
        v.addWidget(bot)
        return pane

    def _right_pane(self):
        pane = QWidget(); pane.setObjectName('pane'); pane.setMinimumWidth(320)
        v = QVBoxLayout(pane); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(PaneHeader('Validation'))

        body = QWidget(); body.setObjectName('val-panel')
        bl = QVBoxLayout(body); bl.setContentsMargins(14, 12, 14, 12); bl.setSpacing(10)

        for section in D.EDITOR_VALIDATION:
            head = QHBoxLayout(); head.setSpacing(10)
            t = QLabel(section['title']); t.setObjectName('val-section-title')
            head.addWidget(t)
            head.addStretch(1)
            cnt = QLabel(section['count'])
            cnt.setObjectName({'ok': 'val-count-ok', 'warn': 'val-count-warn', 'err': 'val-count-err'}[section['count_kind']])
            head.addWidget(cnt)
            bl.addLayout(head)
            for sev, rule, body_html, fix in section['messages']:
                bl.addWidget(ValMessage(sev, rule, body_html, fix))
            bl.addSpacing(2)

        # Provenance
        prov_title = QLabel('Provenance · this file'); prov_title.setObjectName('val-section-title')
        bl.addWidget(prov_title)
        prov = QFrame(); prov.setObjectName('why-panel')
        pl = QVBoxLayout(prov); pl.setContentsMargins(11, 9, 11, 9); pl.setSpacing(4)
        for head_text, body_text in D.EDITOR_PROVENANCE:
            hl = QHBoxLayout(); hl.setSpacing(6)
            dot = QLabel('·'); dot.setStyleSheet(f'color: {CUR()["success"]}; font-weight: 700;')
            inner = QVBoxLayout(); inner.setSpacing(0)
            hh = QLabel(head_text); hh.setStyleSheet(f'color: {CUR()["text"]}; font-size: 11px; font-weight: 500;')
            tt = QLabel(body_text); tt.setStyleSheet(f'color: {CUR()["dim"]}; font-size: 11px;'); tt.setWordWrap(True)
            inner.addWidget(hh); inner.addWidget(tt)
            hl.addWidget(dot, 0, Qt.AlignmentFlag.AlignTop)
            wrap = QFrame(); wrap.setLayout(inner); hl.addWidget(wrap, 1)
            pl.addLayout(hl)
        bl.addWidget(prov)
        bl.addStretch(1)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f'QScrollArea {{ background: {CUR()["surface"]}; }}')
        scroll.setWidget(body)
        v.addWidget(scroll, 1)
        return pane

    def repaint_for_palette(self, pal):
        for row in self._sidecar_rows:
            row.repaint_for_palette(pal)


# =====================================================================
#  TOP HEADER (view switcher + theme toggle)
# =====================================================================
class TopHeader(QFrame):
    viewChanged   = pyqtSignal(int)
    themeToggled  = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('top-header'); self.setFixedHeight(40)
        h = QHBoxLayout(self); h.setContentsMargins(14, 6, 14, 6); h.setSpacing(10)

        # Brand
        logo = QLabel('B')
        logo.setFixedSize(24, 24); logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet(
            'background: qlineargradient(x1:0,y1:0,x2:1,y2:1,'
            f'stop:0 {DARK["accent"]}, stop:1 {DARK["purple"]});'
            'color: white; border-radius: 6px; font-weight: 700;'
        )
        name = QLabel('BIDS-Manager'); name.setObjectName('brand-name')
        tag  = QLabel('prototype'); tag.setObjectName('brand-tag')
        h.addWidget(logo); h.addWidget(name); h.addWidget(tag)
        h.addSpacing(20)

        # View pills
        self._converter_btn = QPushButton('Converter'); self._converter_btn.setObjectName('view-pill')
        self._converter_btn.setCheckable(True); self._converter_btn.setChecked(True)
        self._editor_btn = QPushButton('Editor'); self._editor_btn.setObjectName('view-pill')
        self._editor_btn.setCheckable(True)
        grp = QButtonGroup(self); grp.setExclusive(True)
        grp.addButton(self._converter_btn, 0); grp.addButton(self._editor_btn, 1)
        grp.idClicked.connect(self.viewChanged.emit)
        h.addWidget(self._converter_btn); h.addWidget(self._editor_btn)

        h.addStretch(1)

        # Theme toggle
        self._theme_btn = QPushButton('◐'); self._theme_btn.setObjectName('theme-toggle')
        self._theme_btn.setToolTip('Toggle light / dark theme')
        self._theme_btn.setFixedSize(32, 28); self._theme_btn.clicked.connect(self.themeToggled.emit)
        h.addWidget(self._theme_btn)


# =====================================================================
#  MAIN WINDOW
# =====================================================================
class MainWindow(QMainWindow):
    def __init__(self, theme: ThemeManager):
        super().__init__()
        self._theme = theme
        self.setWindowTitle('BIDS-Manager — Studyname.bidsmgr (prototype)')
        self.resize(1480, 900)

        central = QWidget(); central.setObjectName('central')
        self.setCentralWidget(central)
        v = QVBoxLayout(central); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)

        self.header = TopHeader()
        v.addWidget(self.header)

        self.stack = QStackedWidget()
        self.converter = ConverterView()
        self.editor    = EditorView()
        self.stack.addWidget(self.converter)
        self.stack.addWidget(self.editor)
        v.addWidget(self.stack, 1)

        self.setStatusBar(self._statusbar())

        self.header.viewChanged.connect(self.stack.setCurrentIndex)
        self.header.themeToggled.connect(self._on_theme_toggle)
        theme.add_listener(self._on_palette_changed)

    def _statusbar(self):
        sb = QStatusBar(); sb.setSizeGripEnabled(False)
        info = D.DATASET_INFO
        left = QLabel(
            f'{info["schema"]}  ·  {info["rows"]} rows  ·  '
            f'{info["subjects"]} subjects  ·  {info["studies"]} study  ·  '
            f'last scan {info["last_scan"]}'
        )
        sb.addWidget(left, 1)
        pill = QLabel(f'{info["ready"]} ready  ·  {info["percent"]}%')
        pill.setObjectName('status-pill')
        sb.addPermanentWidget(pill)
        return sb

    def _on_theme_toggle(self):
        new = self._theme.toggle()
        self.header._theme_btn.setText('☀' if new == 'dark' else '☾')

    def _on_palette_changed(self, pal):
        global _THEME_REF
        _THEME_REF = pal
        # Notify the editor sidecar rows (left bar colors)
        if hasattr(self, 'editor'):
            self.editor.repaint_for_palette(pal)
        # Force a repaint of the inspection table delegate-painted cells
        for w in self.findChildren(QTableView):
            w.viewport().update()
        for w in self.findChildren(QTreeWidget):
            w.viewport().update()


# =====================================================================
#  Application bootstrap
# =====================================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    f = app.font(); f.setPointSize(13); app.setFont(f)

    theme = ThemeManager(app)
    win = MainWindow(theme)
    theme.apply('dark')   # initial paint
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
