"""Welcome panel - the project-first landing (VS Code-style home tab).

Shown when no project is open. A hero header plus three self-contained section
cards: Create a new dataset, Open an existing one, and Recent projects. Emits
:pyattr:`project_opened` with the opened ``Project`` and its dataset root so
:class:`MainWindow` can bind it to the Converter.

The create/open *logic* lives in small testable methods (``create_project`` /
``open_project``); the controls gather input then call those, so the flow can be
exercised offscreen without driving modal dialogs.

Styling is fully QSS-driven (object names keyed in ``theme.qss``);
``repaint_for_palette`` runs the unpolish/polish dance so a dark<->light swap
recomputes every card's colours (no stale white/black patches).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import shutil

from PyQt6.QtCore import QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from ..cli._scaffold import slugify_name
from ..cli.create import open_or_create_workspace
from .app_settings import AppSettings
from .theme_manager import CUR

log = logging.getLogger(__name__)

# Recent-list item roles: the path lives at ``UserRole`` (handlers read it);
# the dataset display name + a missing flag ride alongside for the delegate.
_RECENT_PATH_ROLE = Qt.ItemDataRole.UserRole
_RECENT_NAME_ROLE = Qt.ItemDataRole.UserRole + 1
_RECENT_MISSING_ROLE = Qt.ItemDataRole.UserRole + 2

# Static resource links surfaced in the "Getting started" card. Texts are
# friendly labels; the raw URLs never show. Theme-aware: rendered as rich-text
# anchors that read ``QPalette.Link`` (set by the theme manager).
_RESOURCE_LINKS: tuple[tuple[str, str], ...] = (
    ("Documentation website", "https://ancplaboldenburg.github.io/bids_manager_documentation/"),
    ("Tutorial walkthrough", "https://ancplaboldenburg.github.io/bids_manager_documentation/tutorial.html"),
    ("Source code on GitHub", "https://github.com/ANCPLabOldenburg/BIDS-Manager"),
)
# Sample datasets (hosted on the UOL cloud) the docs offer for trying the tool.
_SAMPLE_DATASETS: tuple[tuple[str, str], ...] = (
    ("MRI walkthrough dataset", "https://cloud.uol.de/s/g9gMPpwL7Xg49y9/download"),
    ("EEG motor-imagery dataset", "https://cloud.uol.de/s/T66zc5mN4eeZPGK/download"),
    ("MEG Elekta sample dataset", "https://cloud.uol.de/s/btGeke5NNkDcs6G/download"),
    ("Advanced MRI (Siemens) dataset", "https://cloud.uol.de/s/ZxaZCtHJPLjtDbR/download"),
)


def _parse_qcolor(value: str) -> QColor:
    """Build a QColor from a palette token, including CSS ``rgba()`` strings.

    The palette carries translucent tints as ``rgba(r,g,b,a)`` strings, which
    ``QColor(str)`` cannot parse (it returns an invalid/black colour). This
    parses both ``rgba()`` / ``rgb()`` and plain hex so theme-driven tints
    render correctly (the recent-list selection was painting black otherwise).
    """
    s = str(value).strip()
    if s.startswith("rgba(") or s.startswith("rgb("):
        inner = s[s.index("(") + 1: s.rindex(")")]
        parts = [p.strip() for p in inner.split(",")]
        try:
            r, g, b = (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])))
            a = int(round(float(parts[3]) * 255)) if len(parts) > 3 else 255
            return QColor(r, g, b, a)
        except (ValueError, IndexError):
            return QColor(0, 0, 0, 0)
    return QColor(s)


def _dataset_display_name(bids_root: Path) -> str:
    """Dataset Name from ``dataset_description.json``, else the folder name."""
    dd = Path(bids_root) / "dataset_description.json"
    if dd.exists():
        try:
            data = json.loads(dd.read_text(encoding="utf-8"))
            name = str(data.get("Name", "")).strip()
            if name:
                return name
        except (OSError, ValueError):
            pass
    return Path(bids_root).name


class _RecentItemDelegate(QStyledItemDelegate):
    """Paint a recent-project row as a coloured dataset name above its path.

    Palette tokens are read fresh on every paint (via :func:`CUR`), so a theme
    swap recolours the rows once the list viewport repaints.
    """

    def paint(self, painter, option, index) -> None:  # noqa: N802
        pal = CUR()
        name = str(index.data(_RECENT_NAME_ROLE) or "")
        path = str(index.data(_RECENT_PATH_ROLE) or "")
        missing = bool(index.data(_RECENT_MISSING_ROLE))

        painter.save()
        # Theme-aware selection / hover backgrounds. The palette stores these
        # as ``rgba()`` strings, so parse them properly (a bare QColor(str)
        # would render black and the selection looked black in every theme).
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, _parse_qcolor(pal["accent_bg"]))
        elif option.state & QStyle.StateFlag.State_MouseOver:
            painter.fillRect(option.rect, _parse_qcolor(pal["surface3"]))

        rect = option.rect.adjusted(10, 5, -10, -5)
        half = rect.height() // 2

        # Dataset name (accent, bold) — muted when the folder is gone.
        name_font = QFont(option.font)
        name_font.setBold(True)
        painter.setFont(name_font)
        painter.setPen(QColor(pal["dim"] if missing else pal["accent"]))
        label = f"{name}   (missing)" if missing else name
        painter.drawText(
            QRect(rect.x(), rect.y(), rect.width(), half),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            label,
        )

        # Path (dim, just one step smaller than the name and middle-elided).
        # The app font is sized in PIXELS, so ``pointSizeF()`` is -1; derive the
        # smaller size from pixelSize instead or it collapses to a tiny 7pt.
        path_font = QFont(option.font)
        px = option.font.pixelSize()
        if px > 0:
            path_font.setPixelSize(max(11, px - 1))
        else:
            path_font.setPointSizeF(max(10.0, option.font.pointSizeF() - 0.5))
        painter.setFont(path_font)
        painter.setPen(QColor(pal["dim"]))
        path_rect = QRect(rect.x(), rect.y() + half, rect.width(), rect.height() - half)
        elided = painter.fontMetrics().elidedText(
            path, Qt.TextElideMode.ElideMiddle, path_rect.width(),
        )
        painter.drawText(
            path_rect,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            elided,
        )
        painter.restore()

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        s = super().sizeHint(option, index)
        return QSize(s.width(), 50)


def _section_card(title: str, description: str) -> tuple[QFrame, QVBoxLayout]:
    """Build a titled, self-contained section card; return (frame, body layout)."""
    card = QFrame()
    card.setObjectName("welcome-card")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(22, 18, 22, 18)
    lay.setSpacing(10)
    head = QLabel(title)
    head.setObjectName("welcome-section")
    desc = QLabel(description)
    desc.setObjectName("welcome-section-desc")
    desc.setWordWrap(True)
    lay.addWidget(head)
    lay.addWidget(desc)
    return card, lay


class WelcomePanel(QWidget):
    """Create / open / recent for BIDS dataset projects.

    Emits ``project_opened(project, bids_root)`` (a
    :class:`~bidsmgr.project.Project` and a :class:`pathlib.Path`).
    """

    project_opened = pyqtSignal(object, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("welcome-panel")

        # Default the create-location to the last BIDS-output parent (or home).
        last_parent = AppSettings.load().bids_parent
        self._create_location = Path(last_parent) if last_parent else Path.home()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("welcome-scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        body = QWidget()
        body.setObjectName("welcome-body")
        scroll.setWidget(body)

        # Centred, capped-width container that uses the horizontal space: a
        # full-width hero on top, then a two-column row (Create on the left,
        # Open + Recent stacked on the right) so the page reads as a desktop
        # layout rather than a tall single phone-width column.
        body_row = QHBoxLayout(body)
        body_row.setContentsMargins(28, 28, 28, 28)
        body_row.addStretch(1)
        col_host = QWidget()
        col_host.setObjectName("welcome-col")
        col_host.setMaximumWidth(1080)
        host_v = QVBoxLayout(col_host)
        host_v.setContentsMargins(0, 0, 0, 0)
        host_v.setSpacing(18)
        body_row.addWidget(col_host, 6)
        body_row.addStretch(1)

        host_v.addWidget(self._build_hero())

        content = QHBoxLayout()
        content.setSpacing(18)
        left = QVBoxLayout()
        left.setSpacing(16)
        left.addWidget(self._build_create_card())
        left.addWidget(self._build_resources_card())
        left.addWidget(self._build_updates_card())
        left.addStretch(1)
        right = QVBoxLayout()
        right.setSpacing(16)
        right.addWidget(self._build_open_card())
        right.addWidget(self._build_recent_card(), 1)
        content.addLayout(left, 1)
        content.addLayout(right, 1)
        host_v.addLayout(content, 1)

        self.refresh_recent()

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _build_hero(self) -> QWidget:
        hero = QWidget()
        hero.setObjectName("welcome-hero")
        h = QHBoxLayout(hero)
        h.setContentsMargins(2, 0, 2, 4)
        h.setSpacing(14)

        icon = QLabel()
        icon.setObjectName("welcome-logo")
        png = Path(__file__).parent / "assets" / "macos" / "AppIcon128.png"
        if png.exists():
            pix = QPixmap(str(png))
            if not pix.isNull():
                icon.setPixmap(pix.scaledToHeight(
                    56, Qt.TransformationMode.SmoothTransformation,
                ))
                h.addWidget(icon, 0, Qt.AlignmentFlag.AlignVCenter)

        text = QVBoxLayout()
        text.setSpacing(2)
        title = QLabel("Welcome to BIDS-Manager")
        title.setObjectName("welcome-title")
        subtitle = QLabel(
            "Schema-driven BIDS conversion, curation, and editing. "
            "Create a dataset to start, or reopen one to pick up where you left off."
        )
        subtitle.setObjectName("welcome-subtitle")
        subtitle.setWordWrap(True)
        text.addWidget(title)
        text.addWidget(subtitle)
        h.addLayout(text, 1)
        return hero

    def _build_create_card(self) -> QFrame:
        card, lay = _section_card(
            "Create a new dataset",
            "Scaffold a fresh BIDS dataset (dataset_description.json, README, "
            ".bidsignore) and start scanning raw data into it.",
        )

        self._name_edit = QLineEdit()
        self._name_edit.setObjectName("welcome-input")
        self._name_edit.setPlaceholderText("Dataset name, e.g. My Study")
        self._name_edit.returnPressed.connect(self._on_inline_create)
        lay.addWidget(self._name_edit)

        loc_row = QHBoxLayout()
        loc_row.setSpacing(8)
        self._location_edit = QLineEdit(str(self._create_location))
        self._location_edit.setObjectName("welcome-input")
        self._location_edit.setReadOnly(True)
        browse = QPushButton("Browse…")
        browse.setObjectName("tb-btn-ghost")
        browse.setCursor(Qt.CursorShape.PointingHandCursor)
        browse.clicked.connect(self._on_browse_location)
        loc_row.addWidget(self._location_edit, 1)
        loc_row.addWidget(browse)
        lay.addLayout(loc_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._create_btn = QPushButton("Create dataset")
        self._create_btn.setObjectName("tb-btn-primary")
        self._create_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._create_btn.clicked.connect(self._on_inline_create)
        btn_row.addWidget(self._create_btn)
        lay.addLayout(btn_row)
        return card

    def _build_open_card(self) -> QFrame:
        card, lay = _section_card(
            "Open an existing dataset",
            "Continue curating a BIDS-Manager project, or adopt a dataset "
            "created elsewhere (it is opened read-only, never overwritten).",
        )
        btn_row = QHBoxLayout()
        self._open_btn = QPushButton("Open dataset folder…")
        # Blue (accent) text — see ``QPushButton#welcome-open-btn`` in theme.qss.
        self._open_btn.setObjectName("welcome-open-btn")
        self._open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._open_btn.clicked.connect(self._on_open_clicked)
        btn_row.addWidget(self._open_btn)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)
        return card

    def _build_resources_card(self) -> QFrame:
        card, lay = _section_card(
            "Getting started",
            "Documentation, tutorial, source code, and sample datasets to "
            "try the full scan to validate workflow.",
        )
        for text, url in _RESOURCE_LINKS:
            lay.addWidget(self._link_label(text, url))

        sample = QLabel("Sample datasets")
        sample.setObjectName("welcome-subsection")
        lay.addWidget(sample)
        for text, url in _SAMPLE_DATASETS:
            lay.addWidget(self._link_label(text, url))
        return card

    def _build_updates_card(self) -> QFrame:
        card, lay = _section_card(
            "Update notes",
            "See what changed in each release: new features, fixes, and "
            "anything worth knowing before you upgrade.",
        )
        lay.addWidget(self._link_label(
            "Read the update notes",
            "https://ancplaboldenburg.github.io/bids_manager_documentation/updates.html",
        ))
        return card

    @staticmethod
    def _link_label(text: str, url: str) -> QLabel:
        """A clickable rich-text link (opens in the system browser).

        No explicit anchor colour, so it inherits ``QPalette.Link`` (accent)
        and recolours on a theme swap. ``text-decoration:none`` keeps it tidy.
        """
        lbl = QLabel(f'<a style="text-decoration:none" href="{url}">{text}</a>')
        lbl.setObjectName("welcome-link")
        lbl.setOpenExternalLinks(True)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        lbl.setToolTip(url)
        return lbl

    def _build_recent_card(self) -> QFrame:
        card, lay = _section_card(
            "Recent projects",
            "Datasets you recently created or opened. Double-click to reopen.",
        )
        self._recent = QListWidget()
        self._recent.setObjectName("welcome-recent")
        self._recent.setMinimumHeight(160)
        self._recent.setMouseTracking(True)  # hover state for the delegate
        self._recent.setItemDelegate(_RecentItemDelegate(self._recent))
        self._recent.itemActivated.connect(self._on_recent_activated)
        self._recent.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._recent.customContextMenuRequested.connect(self._on_recent_menu)
        lay.addWidget(self._recent, 1)
        hint = QLabel("Right-click a project to remove or delete it.")
        hint.setObjectName("welcome-section-desc")
        lay.addWidget(hint)
        self._recent_empty = QLabel("No recent projects yet.")
        self._recent_empty.setObjectName("welcome-section-desc")
        lay.addWidget(self._recent_empty)
        return card

    # ------------------------------------------------------------------
    # Recent list
    # ------------------------------------------------------------------

    def refresh_recent(self) -> None:
        """Repopulate the recent-projects list from AppSettings.

        Each row carries the dataset name (painted in accent by the delegate)
        and its full path (dim, beneath). Missing folders are kept but muted
        and disabled.
        """
        self._recent.clear()
        recents = AppSettings.load().recent_projects
        for p in recents:
            exists = Path(p).exists()
            item = QListWidgetItem()
            item.setData(_RECENT_PATH_ROLE, p)
            item.setData(_RECENT_NAME_ROLE, _dataset_display_name(Path(p)) if exists else Path(p).name)
            item.setData(_RECENT_MISSING_ROLE, not exists)
            if not exists:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self._recent.addItem(item)
        has = bool(recents)
        self._recent.setVisible(has)
        self._recent_empty.setVisible(not has)

    # ------------------------------------------------------------------
    # Core (testable) actions
    # ------------------------------------------------------------------

    def create_project(self, parent_dir: Path, name: str) -> Optional[Path]:
        """Create ``<parent_dir>/<slug(name)>`` and open it. Returns the root."""
        name = (name or "").strip()
        if not name:
            return None
        bids_root = Path(parent_dir) / slugify_name(name)
        proj = open_or_create_workspace(bids_root, name=name)
        AppSettings.remember_recent_project(bids_root)
        self.refresh_recent()
        self.project_opened.emit(proj, bids_root)
        return bids_root

    def open_project(self, bids_root: Path) -> Optional[Path]:
        """Open (or adopt) the dataset at ``bids_root``. Returns the root."""
        bids_root = Path(bids_root)
        proj = open_or_create_workspace(bids_root)
        AppSettings.remember_recent_project(bids_root)
        self.refresh_recent()
        self.project_opened.emit(proj, bids_root)
        return bids_root

    # ------------------------------------------------------------------
    # Control handlers
    # ------------------------------------------------------------------

    def _on_browse_location(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Choose where to create the dataset",
            str(self._create_location),
        )
        if d:
            self._create_location = Path(d)
            self._location_edit.setText(d)

    def _on_inline_create(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            self._name_edit.setFocus()
            return
        # No spaces in dataset names (cleaner BIDS paths + cross-platform
        # safety). Make the user aware and offer the underscore form instead
        # of silently rewriting it.
        if " " in name:
            suggested = "_".join(name.split())
            QMessageBox.information(
                self,
                "Spaces are not allowed",
                "Dataset names cannot contain spaces, for cleaner BIDS paths "
                "and cross-platform safety. Please use underscores instead.\n\n"
                f"Suggested name: {suggested}",
            )
            self._name_edit.setText(suggested)
            self._name_edit.setFocus()
            return
        parent = self._location_edit.text().strip() or str(Path.home())
        self.create_project(Path(parent), name)
        self._name_edit.clear()

    def _on_open_clicked(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Open BIDS dataset folder", str(self._create_location),
        )
        if not d:
            return
        self.open_project(Path(d))

    def _on_recent_activated(self, item: QListWidgetItem) -> None:
        p = item.data(Qt.ItemDataRole.UserRole)
        if p and Path(p).exists():
            self.open_project(Path(p))

    # ------------------------------------------------------------------
    # Remove / delete projects
    # ------------------------------------------------------------------

    def delete_project(self, bids_root: Path, *, from_disk: bool) -> bool:
        """Forget a project (and optionally delete its folder from disk).

        With ``from_disk=False`` the dataset is only dropped from the recent
        list. With ``from_disk=True`` the folder is permanently removed, but only
        when it actually looks like a BIDS dataset / BM project (has
        ``.bidsmgr/`` or ``dataset_description.json``) so an arbitrary path can
        never be nuked. Returns ``True`` if anything was removed.
        """
        bids_root = Path(bids_root)
        if from_disk and bids_root.exists():
            looks_bids = (
                (bids_root / ".bidsmgr").exists()
                or (bids_root / "dataset_description.json").exists()
            )
            if not looks_bids:
                return False  # refuse to delete a non-dataset folder
            shutil.rmtree(bids_root, ignore_errors=True)
        AppSettings.forget_recent_project(bids_root)
        self.refresh_recent()
        return True

    def _on_recent_menu(self, pos) -> None:
        item = self._recent.itemAt(pos)
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return
        bids_root = Path(path)
        menu = QMenu(self)
        from .combo_popup import round_menu
        round_menu(menu)
        act_open = menu.addAction("Open")
        act_open.setEnabled(bids_root.exists())
        act_forget = menu.addAction("Remove from list")
        act_delete = menu.addAction("Delete project from disk…")
        chosen = menu.exec(self._recent.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is act_open:
            if bids_root.exists():
                self.open_project(bids_root)
        elif chosen is act_forget:
            self.delete_project(bids_root, from_disk=False)
        elif chosen is act_delete:
            confirm = QMessageBox.warning(
                self,
                "Delete project",
                f"Permanently delete this dataset and everything in it?\n\n{bids_root}\n\n"
                "This cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if confirm == QMessageBox.StandardButton.Yes:
                if not self.delete_project(bids_root, from_disk=True):
                    QMessageBox.information(
                        self, "Delete project",
                        "Not deleted: that folder does not look like a BIDS "
                        "dataset, so it was left untouched (removed from the "
                        "list only).",
                    )

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def repaint_for_palette(self, pal: dict) -> None:
        """Force QSS recomputation on a dark<->light swap.

        The same unpolish/polish dance the other panels use, so the cards /
        inputs / recent list pick up the re-applied stylesheet instead of
        keeping stale background colours.
        """
        del pal
        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            style.unpolish(w)
            style.polish(w)
            w.update()
        # Rich-text anchors bake the link colour into their layout at parse
        # time, so a bare update() keeps the old colour. Re-set the text to
        # force a re-parse against the new ``QPalette.Link``.
        for lbl in self.findChildren(QLabel):
            if lbl.objectName() == "welcome-link":
                lbl.setText(lbl.text())
        # The recent-list delegate reads palette tokens at paint time; nudge
        # the viewport so the rows recolour immediately.
        if hasattr(self, "_recent"):
            self._recent.viewport().update()


__all__ = ["WelcomePanel"]
