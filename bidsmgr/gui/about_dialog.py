"""About BIDS-Manager dialog.

Reached by clicking the brand logo or wordmark in the top header.
Combines two things:

1. **What is BIDS-Manager** — a short paragraph + the bidsmgr version
   so a new user can orient themselves at a glance.
2. **Authorship** — port of the original BIDS-Manager v0.2.5
   ``AuthorshipDialog`` (lab logo, lead-author bio, head-of-lab bio,
   acknowledgements). Same content the original ships under
   *Help → Authorship*, kept faithful to honour the original work.

Image assets ship in :mod:`bidsmgr.gui.assets`:

* ``ANCP_lab.png`` — Applied Neurocognitive Psychology lab logo.
* ``Karel.jpeg`` — Dr. Karel López Vilaret portrait.
* ``Jochem.jpg`` — Prof. Dr. Jochem Rieger portrait.

All three are optional — the dialog renders gracefully without them.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import bidsmgr


_ASSETS = Path(__file__).parent / "assets"


def _scaled(pixmap_path: Path, target_width: int) -> QPixmap | None:
    """Load + scale an image; return ``None`` if missing or invalid."""
    if not pixmap_path.exists():
        return None
    pix = QPixmap(str(pixmap_path))
    if pix.isNull():
        return None
    return pix.scaledToWidth(
        target_width, Qt.TransformationMode.SmoothTransformation,
    )


class AboutDialog(QDialog):
    """About / Authorship dialog opened from the top-header brand."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("about-dialog")
        self.setWindowTitle("About BIDS-Manager")
        self.resize(560, 760)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scrollable body — the dialog easily exceeds a small laptop's
        # vertical space once both author rows are visible.
        scroll = QScrollArea()
        scroll.setObjectName("about-scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        body = QWidget()
        body.setObjectName("about-body")
        b = QVBoxLayout(body)
        b.setContentsMargins(28, 24, 28, 24)
        b.setSpacing(14)

        # --- What is BIDS-Manager -------------------------------------
        title = QLabel("BIDS-Manager")
        title.setObjectName("about-title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        b.addWidget(title)

        version = QLabel(
            f"version <code>{bidsmgr.__version__}</code>"
        )
        version.setObjectName("about-version")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setTextFormat(Qt.TextFormat.RichText)
        b.addWidget(version)

        intro = QLabel(
            "A schema-driven workspace for converting raw neuroimaging "
            "data to the Brain Imaging Data Structure (BIDS), curating "
            "the resulting tree, and running pre-flight quality checks "
            "before sharing or analysis. Supports MRI (DICOM), EEG, "
            "MEG, iEEG, and physio inputs through a unified inventory "
            "and a single conversion pipeline."
        )
        intro.setObjectName("about-intro")
        intro.setWordWrap(True)
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        b.addWidget(intro)

        b.addWidget(self._divider())

        # --- Lab logo + tagline ---------------------------------------
        lab_pix = _scaled(_ASSETS / "ANCP_lab.png", 320)
        if lab_pix is not None:
            lab_logo = QLabel()
            lab_logo.setPixmap(lab_pix)
            lab_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
            b.addWidget(lab_logo)

        lab_desc = QLabel(
            "Developed in the Applied Neurocognitive Psychology Lab "
            "at the University of Oldenburg, with the objective of "
            "facilitating the conversion to BIDS format, easy "
            "metadata handling, and quality control."
        )
        lab_desc.setObjectName("about-lab-desc")
        lab_desc.setWordWrap(True)
        lab_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        b.addWidget(lab_desc)

        b.addWidget(self._divider())

        # --- Authors --------------------------------------------------
        auth_header = QLabel("Authors")
        auth_header.setObjectName("about-section-header")
        auth_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        b.addWidget(auth_header)

        b.addWidget(self._author_row(
            portrait_path=_ASSETS / "Karel.jpeg",
            html=(
                "<b>Dr. Karel López Vilaret</b><br/>"
                "<i>BIDS-Manager App Lead</i><br/><br/>"
                "I hold a PhD in Neuroscience and currently work as a "
                "scientific software developer. I build BIDS-Manager, "
                "a tool designed to streamline BIDS conversion, "
                "metadata handling, and quality control — enabling "
                "researchers to manage neuroimaging data more "
                "efficiently."
            ),
        ))
        b.addWidget(self._author_row(
            portrait_path=_ASSETS / "Jochem.jpg",
            html=(
                "<b>Prof. Dr. rer. nat. Jochem Rieger</b><br/>"
                "<i>Applied Neurocognitive Psychology</i><br/><br/>"
                "Full Professor of Psychology at the University of "
                "Oldenburg and head of the Applied Neurocognitive "
                "Psychology group. His research focuses on open "
                "science, machine learning, and understanding the "
                "neural basis of perception, cognition, and action "
                "in realistic environments."
            ),
        ))

        b.addWidget(self._divider())

        # --- Acknowledgements -----------------------------------------
        ack_header = QLabel("Acknowledgements")
        ack_header.setObjectName("about-section-header")
        ack_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        b.addWidget(ack_header)
        ack = QLabel(
            "Dr. Jorge Bosch-Bayard<br/>"
            "MSc. Erdal Karaca<br/>"
            "BSc. Pablo Alexis Olguín Baxman<br/>"
            "Dr. Amirhussein Abdolalizadeh Saleh<br/>"
            "Dr. Tina Schmitt<br/>"
            "Dr.-Ing. Andreas Spiegler<br/>"
            "MSc. Shari Hiltner"
        )
        ack.setObjectName("about-ack")
        ack.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ack.setTextFormat(Qt.TextFormat.RichText)
        b.addWidget(ack)

        b.addWidget(self._divider())

        # --- Project links --------------------------------------------
        links = QLabel(
            "<a href='https://github.com/ANCPLabOldenburg/BIDS-Manager'>"
            "Project on GitHub</a> &nbsp;·&nbsp; "
            "<a href='https://ancplaboldenburg.github.io/"
            "bids_manager_documentation/'>Documentation</a>"
        )
        links.setObjectName("about-links")
        links.setAlignment(Qt.AlignmentFlag.AlignCenter)
        links.setOpenExternalLinks(True)
        links.setTextFormat(Qt.TextFormat.RichText)
        b.addWidget(links)

        b.addStretch(1)

        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # --- Footer ---------------------------------------------------
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        footer = QFrame()
        footer.setObjectName("about-footer")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(18, 12, 18, 12)
        fl.addStretch(1)
        fl.addWidget(buttons)
        outer.addWidget(footer)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setObjectName("about-divider")
        line.setFrameShape(QFrame.Shape.HLine)
        return line

    @staticmethod
    def _author_row(
        portrait_path: Path,
        html: str,
        portrait_width: int = 140,
    ) -> QFrame:
        row = QFrame()
        row.setObjectName("about-author-row")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(20)

        pix = _scaled(portrait_path, portrait_width)
        if pix is not None:
            pic = QLabel()
            pic.setPixmap(pix)
            pic.setObjectName("about-author-pic")
            pic.setAlignment(
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            )
            rl.addWidget(pic, 0)

        bio = QLabel(html)
        bio.setObjectName("about-author-bio")
        bio.setTextFormat(Qt.TextFormat.RichText)
        bio.setWordWrap(True)
        rl.addWidget(bio, 1)
        return row


__all__ = ["AboutDialog"]
