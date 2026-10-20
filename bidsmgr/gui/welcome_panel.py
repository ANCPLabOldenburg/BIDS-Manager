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

from PyQt6.QtCore import QRect, QSize, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
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
# The dataset's own title, when it differs from the folder name.
_RECENT_TITLE_ROLE = Qt.ItemDataRole.UserRole + 4
_RECENT_MISSING_ROLE = Qt.ItemDataRole.UserRole + 2

# Static resource links surfaced in the "Getting started" card. Texts are
# friendly labels; the raw URLs never show. Theme-aware: rendered as rich-text
# anchors that read ``QPalette.Link`` (set by the theme manager).
_RESOURCE_LINKS: tuple[tuple[str, str], ...] = (
    ("Documentation website", "https://ancplaboldenburg.github.io/bids_manager_documentation/"),
    ("Learn BIDS Manager, step by step", "https://ancplaboldenburg.github.io/bids_manager_documentation/learn/"),
    ("Source code on GitHub", "https://github.com/ANCPLabOldenburg/BIDS-Manager"),
)
_DOCS_URL = _RESOURCE_LINKS[0][1]
# Sample datasets, offered two ways: straight to the download for somebody who
# knows which one they want, and to the documentation section for somebody who
# does not. The section says what each dataset demonstrates and what a full run
# of it produces, which a bare link cannot.
#
# These are hosted on the UOL cloud and must be kept in step with the Sample
# data page of the documentation (``site.yml`` there holds the same links). They
# have drifted before: the list here had four while the documentation offered
# six, and in October 2026 every share was reissued, which left all six links
# here dead. If you add or replace a dataset, change both.
#
# An installed application cannot be corrected, so a share must keep its link:
# replace a zip INSIDE its share on the cloud rather than making a new share.
_SAMPLE_DATA_PAGE = (
    "https://ancplaboldenburg.github.io/bids_manager_documentation/start/sample-data/"
)
_SAMPLE_DATASETS: tuple[tuple[str, str], ...] = (
    ("MRI walkthrough dataset", "https://cloud.uol.de/s/qpE8KeBpqgTn2LB/download"),
    ("Advanced MRI (Siemens) dataset", "https://cloud.uol.de/s/KdFzsDtLA9ijNAb/download"),
    ("PET, DICOM and ECAT with blood", "https://cloud.uol.de/s/Sfmdo6m2GCrKyfi/download"),
    ("EEG motor-imagery dataset", "https://cloud.uol.de/s/8YcmPwFJ5qzfEAq/download"),
    ("MEG Elekta sample dataset", "https://cloud.uol.de/s/XQoEqsa42bNQ4JX/download"),
    ("Multimodal: MRI, MRS, PET, EEG, MEG, eye tracking",
     "https://cloud.uol.de/s/kY9nQwdzgyQF6At/download"),
)
_SAMPLE_DATASETS_URL = _SAMPLE_DATA_PAGE


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
    """What this project is called: its folder name.

    It used to be the ``Name`` from ``dataset_description.json``, falling back
    to the folder. That made the label a user reads as "which project am I in"
    an editable metadata field: typing a publication title into the template
    renamed the project in the interface while the folder on disk stayed put,
    and the two drifted apart with nothing to say so.

    The folder is the project's identity everywhere else, in the recent list, in
    the ``dataset`` column, in the output path, so it is the name shown. The
    title lives beside it, see :func:`_dataset_title`.
    """
    return Path(bids_root).name


def _dataset_title(bids_root: Path) -> str:
    """The dataset's own title, when it says something the folder does not.

    Empty when the two agree, which is the usual case and needs no second
    label, or when there is no dataset description to read.
    """
    dd = Path(bids_root) / "dataset_description.json"
    if not dd.exists():
        return ""
    try:
        data = json.loads(dd.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    title = str(data.get("Name", "") or "").strip()
    return "" if title == Path(bids_root).name else title


class _RecentItemDelegate(QStyledItemDelegate):
    """Paint a recent-project row as a coloured dataset name above its path.

    Palette tokens are read fresh on every paint (via :func:`CUR`), so a theme
    swap recolours the rows once the list viewport repaints.
    """

    def paint(self, painter, option, index) -> None:  # noqa: N802
        pal = CUR()
        name = str(index.data(_RECENT_NAME_ROLE) or "")
        title = str(index.data(_RECENT_TITLE_ROLE) or "")
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
        name_rect = QRect(rect.x(), rect.y(), rect.width(), half)
        painter.drawText(
            name_rect,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            label,
        )

        # The dataset's own title, beside the project name and in a quieter
        # tone: it says what the data IS, where the name says where it lives.
        if title and not missing:
            used = painter.fontMetrics().horizontalAdvance(label) + 10
            title_font = QFont(option.font)
            title_font.setBold(False)
            title_font.setItalic(True)
            painter.setFont(title_font)
            painter.setPen(QColor(pal["muted"]))
            room = rect.width() - used
            if room > 40:
                painter.drawText(
                    QRect(rect.x() + used, rect.y(), room, half),
                    int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                    painter.fontMetrics().elidedText(
                        title, Qt.TextElideMode.ElideRight, room,
                    ),
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
        # Two lines of text and their margins: grows with the font size.
        from PyQt6.QtGui import QFontMetrics

        s = super().sizeHint(option, index)
        line = QFontMetrics(option.font).height()
        return QSize(s.width(), 2 * line + 16)


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
    #: A folder that is not a BIDS dataset, to look at in the Editor without
    #: making it a project (Path).
    view_requested = pyqtSignal(object)

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
            "Continue curating a BIDS-Manager project, or make one of a dataset "
            "created elsewhere (its files are kept as they are). A folder that is "
            "not a BIDS dataset opens in the Editor for viewing only.",
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
        # On the documentation's line, at the right: a request for a star on
        # GitHub, the one sign of use BIDS Manager gets (gui/star_request.py).
        from . import icons

        self._star_button = QPushButton("Help us")
        self._star_button.setObjectName("tb-btn")
        self._star_button.setToolTip("Give BIDS Manager a star on GitHub: why it helps, and how")
        icons.apply_button(self._star_button, "star")
        self._star_button.clicked.connect(self._show_star_request)
        for text, url in _RESOURCE_LINKS:
            link = self._link_label(text, url)
            if url != _DOCS_URL:
                lay.addWidget(link)
                continue
            row = QHBoxLayout()
            row.addWidget(link)
            row.addStretch(1)
            row.addWidget(self._star_button)
            lay.addLayout(row)

        sample = QLabel("Sample datasets")
        sample.setObjectName("welcome-subsection")
        lay.addWidget(sample)
        # One button with a menu rather than six links: the card ran past the
        # fold on a first run. Each item opens its download in the browser.
        from .viz.menus import popup_menu

        self._samples_button = QPushButton("Download a sample dataset")
        self._samples_button.setObjectName("tb-btn")
        self._samples_button.setToolTip("Pick one: the download opens in your browser")
        menu = popup_menu(self._samples_button)
        for text, url in _SAMPLE_DATASETS:
            action = menu.addAction(text)
            action.setData(url)
            action.setToolTip(url)
            action.triggered.connect(lambda _c=False, u=url: QDesktopServices.openUrl(QUrl(u)))
        self._samples_button.setMenu(menu)
        row = QHBoxLayout()
        row.addWidget(self._samples_button)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(self._link_label(
            "Compare them, and see what each one demonstrates",
            _SAMPLE_DATASETS_URL,
        ))
        return card

    def _build_updates_card(self) -> QFrame:
        from .whats_new import NEWS_FOR

        card, lay = _section_card(
            "What's new",
            f"What BIDS Manager {NEWS_FOR} brings, and the notes of every release: "
            "new features, fixes, and anything worth knowing before you upgrade.",
        )
        news = self._link_label(f"See what's new in BIDS Manager {NEWS_FOR}", "#whats-new")
        news.setOpenExternalLinks(False)
        news.setToolTip("The announcement shown when this release first opened")
        news.linkActivated.connect(lambda _href: self._show_whats_new())
        lay.addWidget(news)
        lay.addWidget(self._link_label(
            "Read the update notes",
            "https://ancplaboldenburg.github.io/bids_manager_documentation/updates/",
        ))
        return card

    def _show_whats_new(self) -> None:
        from .whats_new import show_dialog

        show_dialog(self.window())

    def _show_star_request(self) -> None:
        from .star_request import show_dialog

        show_dialog(self.window())

    @staticmethod
    def _link_html(text: str, url: str) -> str:
        """The link in the current theme's accent, written into the anchor:
        a label parses its rich text once, so a link left to the palette
        kept the colour of the theme it was made in."""
        from .theme_manager import CUR

        return (f'<a style="text-decoration:none; color:{CUR()["accent"]}" '
                f'href="{url}">{text}</a>')

    @classmethod
    def _link_label(cls, text: str, url: str) -> QLabel:
        """A clickable rich-text link (opens in the system browser),
        re-written in the accent on every theme change."""
        lbl = QLabel(cls._link_html(text, url))
        lbl.setProperty("link_text", text)
        lbl.setProperty("link_url", url)
        lbl.setObjectName("welcome-link")
        lbl.setOpenExternalLinks(True)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        lbl.setToolTip(url)
        return lbl

    def _build_recent_card(self) -> QFrame:
        from ..viz import keynames

        # Several at once: Shift for a range, Ctrl (Command on a Mac) for
        # one more, then remove or delete them together.
        card, lay = _section_card(
            "Recent projects",
            "Datasets you recently created or opened. Double-click to reopen. "
            "Select several with Shift or " + keynames.mouse(["ctrl"], "click")
            + " to remove or delete them together.",
        )
        # The actions sit above the list, where a long list cannot push them
        # below the fold.
        row = QHBoxLayout()
        row.setSpacing(6)
        self._recent_open_btn = QPushButton("Open")
        self._recent_open_btn.setObjectName("tb-btn")
        self._recent_open_btn.setToolTip("Open the selected project")
        self._recent_open_btn.clicked.connect(self._open_selected)
        self._recent_forget_btn = QPushButton("Remove from list")
        self._recent_forget_btn.setObjectName("tb-btn")
        self._recent_forget_btn.setToolTip(
            "Take the selected projects off this list. Nothing is deleted: open a "
            "project's folder again to bring it back.")
        self._recent_forget_btn.clicked.connect(self._forget_selected)
        self._recent_delete_btn = QPushButton("Delete from disk...")
        self._recent_delete_btn.setObjectName("tb-btn")
        self._recent_delete_btn.setToolTip(
            "Permanently delete the selected projects' folders and everything in "
            "them. Asks first; never the project that is open, and never a folder "
            "that is not a dataset.")
        self._recent_delete_btn.clicked.connect(self._delete_selected)
        for b in (self._recent_open_btn, self._recent_forget_btn, self._recent_delete_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self._recent = QListWidget()
        self._recent.setObjectName("welcome-recent")
        self._recent.setMinimumHeight(160)
        self._recent.setMouseTracking(True)  # hover state for the delegate
        self._recent.setItemDelegate(_RecentItemDelegate(self._recent))
        self._recent.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._recent.itemActivated.connect(self._on_recent_activated)
        self._recent.itemSelectionChanged.connect(self._sync_recent_buttons)
        self._recent.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._recent.customContextMenuRequested.connect(self._on_recent_menu)
        lay.addWidget(self._recent, 1)
        # Delete takes the selection off the list (nothing on disk).
        from PyQt6.QtGui import QKeySequence, QShortcut

        forget_key = QShortcut(QKeySequence(QKeySequence.StandardKey.Delete), self._recent)
        forget_key.setContext(Qt.ShortcutContext.WidgetShortcut)
        forget_key.activated.connect(self._forget_selected)
        self._recent_empty = QLabel("No recent projects yet.")
        self._recent_empty.setObjectName("welcome-section-desc")
        lay.addWidget(self._recent_empty)
        # With the list hidden (no recent projects) the spare height goes
        # here, not to the labels, which centred their text in it.
        lay.addStretch(0)
        return card

    # ------------------------------------------------------------------
    # Recent list
    # ------------------------------------------------------------------

    def refresh_recent(self) -> None:
        """Repopulate the recent-projects list from AppSettings.

        Each row carries the dataset name (painted in accent by the delegate)
        and its full path (dim, beneath). A missing folder is kept, muted, and
        can still be selected and taken off the list.
        """
        self._recent.clear()
        recents = AppSettings.load().recent_projects
        for p in recents:
            exists = Path(p).exists()
            item = QListWidgetItem()
            item.setData(_RECENT_PATH_ROLE, p)
            item.setData(_RECENT_NAME_ROLE, _dataset_display_name(Path(p)) if exists else Path(p).name)
            item.setData(_RECENT_TITLE_ROLE, _dataset_title(Path(p)) if exists else "")
            item.setData(_RECENT_MISSING_ROLE, not exists)
            # A missing folder stays selectable (painted muted), so it can be
            # taken off the list with the rest.
            self._recent.addItem(item)
        has = bool(recents)
        self._recent.setVisible(has)
        self._recent_empty.setVisible(not has)
        for w in (self._recent_open_btn, self._recent_forget_btn, self._recent_delete_btn):
            w.setVisible(has)
        self._sync_recent_buttons()

    def selected_recent(self) -> list[Path]:
        """The selected projects, in list order."""
        rows = sorted(self._recent.row(i) for i in self._recent.selectedItems())
        return [Path(self._recent.item(r).data(_RECENT_PATH_ROLE)) for r in rows]

    def _sync_recent_buttons(self) -> None:
        chosen = self.selected_recent()
        self._recent_open_btn.setEnabled(len(chosen) == 1 and chosen[0].exists())
        self._recent_forget_btn.setEnabled(bool(chosen))
        self._recent_delete_btn.setEnabled(any(p.exists() for p in chosen))
        n = len(chosen)
        self._recent_forget_btn.setText("Remove from list" if n < 2
                                        else f"Remove {n} from list")
        self._recent_delete_btn.setText("Delete from disk..." if n < 2
                                        else f"Delete {n} from disk...")

    #: The project open now, if any (set by the main window): never deleted.
    current_project = None

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
        self.open_folder(Path(d))

    def open_folder(self, folder: Path) -> Optional[str]:
        """Open what the user picked: a BIDS dataset (or an empty folder) as a
        project; any other folder only after asking, because making it a
        project WRITES into it (a dataset description, a README, .bidsignore
        and .bidsmgr/). Returns "project", "view" or None (cancelled)."""
        from ..project.adopt import looks_like_bids

        folder = Path(folder)
        empty = folder.is_dir() and not any(folder.iterdir())
        if empty or looks_like_bids(folder):
            self.open_project(folder)
            return "project"
        choice = self.ask_not_bids(folder)
        if choice == "project":
            self.open_project(folder)
        elif choice == "view":
            self.view_requested.emit(folder)
        return choice or None

    def ask_not_bids(self, folder: Path) -> str:
        """``"view"``, ``"project"`` or ``""`` for a folder that is not a
        BIDS dataset."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Not a BIDS dataset")
        box.setText(f"{folder.name} is not a BIDS dataset.")
        box.setInformativeText(
            "View it in the Editor: its images, tables and documents open as "
            "usual and nothing is written into the folder.\n\n"
            "Or make it a BIDS project: BIDS Manager adds a dataset description, "
            "a README, .bidsignore and its .bidsmgr/ folder to it, so you can "
            "convert into it and edit it with every change undoable.")
        view = box.addButton("View in the Editor", QMessageBox.ButtonRole.AcceptRole)
        project = box.addButton("Make it a project", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(view)
        box.exec()
        clicked = box.clickedButton()
        return "view" if clicked is view else "project" if clicked is project else ""

    def _on_recent_activated(self, item: QListWidgetItem) -> None:
        p = item.data(Qt.ItemDataRole.UserRole)
        if p and Path(p).exists():
            self.open_project(Path(p))

    def _open_selected(self) -> None:
        chosen = self.selected_recent()
        if len(chosen) == 1 and chosen[0].exists():
            self.open_project(chosen[0])

    def _forget_selected(self) -> None:
        chosen = self.selected_recent()
        if chosen:
            self.forget_projects(chosen)

    def _delete_selected(self) -> None:
        chosen = [p for p in self.selected_recent() if p.exists()]
        if not chosen:
            return
        listing = "\n".join(str(p) for p in chosen[:12])
        if len(chosen) > 12:
            listing += f"\n... and {len(chosen) - 12} more"
        what = "this dataset" if len(chosen) == 1 else f"these {len(chosen)} datasets"
        confirm = QMessageBox.warning(
            self, "Delete projects",
            f"Permanently delete {what} and everything in them?\n\n{listing}\n\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        deleted, kept = self.delete_projects(chosen)
        if kept:
            QMessageBox.information(
                self, "Delete projects",
                "Not deleted (taken off the list only):\n\n"
                + "\n".join(f"{p}: {why}" for p, why in kept))

    # ------------------------------------------------------------------
    # Remove / delete projects
    # ------------------------------------------------------------------

    def forget_projects(self, roots: list[Path]) -> None:
        """Take ``roots`` off the recent list (nothing on disk changes)."""
        for root in roots:
            AppSettings.forget_recent_project(Path(root))
        self.refresh_recent()

    def delete_projects(self, roots: list[Path]) -> tuple[list[Path], list[tuple[Path, str]]]:
        """Delete each of ``roots`` from disk (see :meth:`delete_project`):
        ``(deleted, [(kept, why)])``. The project open now is never deleted;
        every one of them leaves the list."""
        open_now = self.current_project() if callable(self.current_project) else None
        deleted: list[Path] = []
        kept: list[tuple[Path, str]] = []
        for root in roots:
            root = Path(root)
            if open_now is not None and root == Path(open_now):
                kept.append((root, "it is the project open now"))
                AppSettings.forget_recent_project(root)
                continue
            if self.delete_project(root, from_disk=True, refresh=False):
                deleted.append(root)
            else:
                kept.append((root, "it does not look like a dataset"))
        self.refresh_recent()
        return deleted, kept

    def delete_project(self, bids_root: Path, *, from_disk: bool, refresh: bool = True) -> bool:
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
                AppSettings.forget_recent_project(bids_root)
                if refresh:
                    self.refresh_recent()
                return False  # refuse to delete a non-dataset folder
            from .fs_watch import watchers_released

            # Windows cannot remove a folder something is watching.
            with watchers_released():
                shutil.rmtree(bids_root, ignore_errors=True)
        AppSettings.forget_recent_project(bids_root)
        if refresh:
            self.refresh_recent()
        return True

    def _on_recent_menu(self, pos) -> None:
        """Right-click: on a selected row it acts on the whole selection, on
        another row on that row alone."""
        item = self._recent.itemAt(pos)
        if item is None:
            return
        if not item.isSelected():
            self._recent.clearSelection()
            item.setSelected(True)
        chosen = self.selected_recent()
        n = len(chosen)
        menu = QMenu(self)
        from .combo_popup import round_menu
        round_menu(menu)
        act_open = menu.addAction("Open")
        act_open.setEnabled(n == 1 and chosen[0].exists())
        act_forget = menu.addAction("Remove from list" if n == 1
                                    else f"Remove {n} from list")
        act_delete = menu.addAction("Delete project from disk..." if n == 1
                                    else f"Delete {n} projects from disk...")
        act_delete.setEnabled(any(p.exists() for p in chosen))
        picked = menu.exec(self._recent.mapToGlobal(pos))
        if picked is act_open:
            self._open_selected()
        elif picked is act_forget:
            self._forget_selected()
        elif picked is act_delete:
            self._delete_selected()

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def repaint_for_palette(self, pal: dict) -> None:
        """Force QSS recomputation on a theme change.

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
        # Rich-text anchors bake the link colour in when parsed (and setText
        # with the same text is ignored), so each link is written again in
        # the new accent.
        for lbl in self.findChildren(QLabel):
            if lbl.objectName() == "welcome-link" and lbl.property("link_url"):
                lbl.setText(self._link_html(lbl.property("link_text"), lbl.property("link_url")))
        if hasattr(self, "_star_button"):
            from . import icons

            icons.apply_button(self._star_button, "star")
        # The recent-list delegate reads palette tokens at paint time; nudge
        # the viewport so the rows recolour immediately.
        if hasattr(self, "_recent"):
            self._recent.viewport().update()


__all__ = ["WelcomePanel"]
