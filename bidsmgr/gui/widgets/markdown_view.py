"""A Markdown document read the way a repository page shows it.

Qt parses the Markdown (``QTextDocument.setMarkdown``: CommonMark with
GitHub's tables and task lists); this widget then gives it a reader's
typography in the app's theme: a centred column of readable width, headings
with room above and a rule under the first two levels, code in a rounded
box, quotes with a bar, tables with a grid and a header row, links in the
accent colour.

Qt's rich text cannot draw a rounded box, a bar on one side or a rule in a
colour of our choosing, so those are PAINTED behind the text from the laid
out blocks (``_paint_behind``). Two facts about Qt's layout that this relies
on: a block's bounding rectangle is its text only, without its margins; and
the margins of two neighbouring blocks collapse to the larger one. So a
painted box is the text plus ``pad``, and the blocks around it carry margins
of at least that.

Every colour is a palette token and every size goes through
``gui/viz/fonts``, so a theme switch or a new font size re-renders it
(``ThemeHub``). A link to the web opens in the browser, a link inside the
page scrolls to it, and a link to a file beside the document is announced
(``file_requested``) for the Editor to open.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QLineF, QRectF, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QDesktopServices, QImageReader, QPainter, QPen, QTextBlock,
    QTextBlockFormat, QTextCharFormat, QTextCursor, QTextFormat, QTextFrameFormat,
    QTextLength, QTextListFormat,
)
from PyQt6.QtWidgets import QTextBrowser

#: The text column at scale 1.0: about 90 characters; wider is hard to read.
COLUMN_PX = 860
#: Body text and code at scale 1.0.
BODY_PX = 14
CODE_PX = 13
#: Heading sizes by level at scale 1.0.
HEADING_PX = {1: 28, 2: 22, 3: 18, 4: 16, 5: 14, 6: 13}

_P = QTextFormat.Property
_PROPORTIONAL = QTextBlockFormat.LineHeightTypes.ProportionalHeight.value
#: Unordered list markers by nesting depth, as a repository page draws them.
_BULLETS = (QTextListFormat.Style.ListDisc, QTextListFormat.Style.ListCircle,
            QTextListFormat.Style.ListSquare)


def _colour(value: str) -> QColor:
    from ...viz.theme import parse_colour

    return QColor(*parse_colour(value))


def _each_fragment(block: QTextBlock):
    it = block.begin()
    while not it.atEnd():
        yield it.fragment()
        it += 1


class MarkdownView(QTextBrowser):
    """Rendered Markdown (or HTML) in the theme."""

    #: A link to a file beside the document (an absolute path).
    file_requested = pyqtSignal(Path)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("text-rendered")
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._on_link)
        self._text = ""
        self._kind = "markdown"
        self._base: Optional[Path] = None
        #: What is painted behind the text, by block number.
        self._codes: list[tuple[int, int]] = []
        self._quotes: list[tuple[int, int, int]] = []
        self._rules: list[int] = []
        self._underlined: list[int] = []
        from ..viz.bridge import ThemeHub, connect_while_alive

        connect_while_alive(ThemeHub.instance().changed, self, lambda w, _t: w.rerender())

    # -- content ---------------------------------------------------------------

    def set_markdown(self, text: str, base: Optional[Path] = None) -> None:
        self._text, self._kind, self._base = text, "markdown", base
        self.rerender()

    def set_html(self, text: str, base: Optional[Path] = None) -> None:
        self._text, self._kind, self._base = text, "html", base
        self.rerender()

    def rerender(self) -> None:
        """Lay the document out again in the current theme and font size,
        keeping the reader's place."""
        bar = self.verticalScrollBar()
        at = bar.value() / max(bar.maximum(), 1)
        self.setSearchPaths([str(self._base)] if self._base is not None else [])
        self._codes, self._quotes, self._rules, self._underlined = [], [], [], []
        if self._kind == "html":
            self.setHtml(self._text)
        else:
            self.setMarkdown(self._text)
            self._restyle()
        self._fit_column(force=True)
        self.document().setModified(False)
        bar.setValue(int(round(at * bar.maximum())))
        self.viewport().update()

    # -- styling ---------------------------------------------------------------

    def _restyle(self) -> None:
        from ..viz import fonts
        from ..viz.bridge import ThemeHub

        tok = ThemeHub.instance().theme.token
        doc = self.document()
        doc.setDefaultFont(fonts.font(BODY_PX))
        doc.setDocumentMargin(0)
        doc.setIndentWidth(fonts.px(26))
        gap = fonts.px(14)          # between paragraphs
        pad = fonts.px(12)          # inside a code box
        code_font = fonts.font(CODE_PX, mono=True)
        code_bg = QBrush(_colour(tok("surface3")))
        link, dim = _colour(tok("accent")), _colour(tok("dim"))

        blocks = []
        block = doc.begin()
        while block.isValid():
            blocks.append(block)
            block = block.next()

        def is_code(b: QTextBlock) -> bool:
            f = b.blockFormat()
            return f.hasProperty(_P.BlockCodeFence) or f.hasProperty(_P.BlockCodeLanguage)

        def quote_level(b: QTextBlock) -> int:
            f = b.blockFormat()
            return f.intProperty(_P.BlockQuoteLevel) if f.hasProperty(_P.BlockQuoteLevel) else 0

        code_start = quote_start = None
        level_now = 0
        for i, block in enumerate(blocks):
            n = block.blockNumber()
            prev = blocks[i - 1] if i else None
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            cursor = QTextCursor(block)
            fmt = block.blockFormat()
            code, level = is_code(block), quote_level(block)
            in_table = cursor.currentTable() is not None

            # Runs of code lines (one box) and of quoted blocks (one bar).
            if code and code_start is None:
                code_start = n
            if code and (nxt is None or not is_code(nxt)):
                self._codes.append((code_start, n))
                code_start = None
            if level and quote_start is None:
                quote_start, level_now = n, level
            level_now = max(level_now, level)
            if level and (nxt is None or quote_level(nxt) == 0):
                self._quotes.append((quote_start, n, level_now))
                quote_start, level_now = None, 0

            heading = fmt.headingLevel()
            if heading:
                fmt.setTopMargin(0 if prev is None else fonts.px(30 if heading <= 2 else 22))
                fmt.setBottomMargin(fonts.px(18 if heading <= 2 else 8))
                fmt.setLineHeight(120, _PROPORTIONAL)
                if heading <= 2:
                    self._underlined.append(n)
                size = fonts.px(HEADING_PX.get(heading, BODY_PX))

                def as_heading(cf: QTextCharFormat, size=size, heading=heading) -> None:
                    # The reader's size adjustment is applied AFTER the font
                    # is resolved and would override any size set here.
                    cf.clearProperty(_P.FontSizeAdjustment)
                    cf.setProperty(_P.FontPixelSize, size)
                    cf.setFontWeight(600)
                    if heading == 6:
                        cf.setForeground(dim)

                self._rewrite_chars(block, as_heading)
            elif code:
                first = prev is None or not is_code(prev)
                last = nxt is None or not is_code(nxt)
                fmt.setTopMargin(pad + fonts.px(6) if first else 0)
                fmt.setBottomMargin(pad + gap if last else 0)
                fmt.setLeftMargin(pad + fonts.px(2))
                fmt.setRightMargin(pad)
                fmt.setLineHeight(140, _PROPORTIONAL)

                def as_code(cf: QTextCharFormat) -> None:
                    cf.setFont(code_font)

                self._rewrite_chars(block, as_code)
            elif fmt.hasProperty(_P.BlockTrailingHorizontalRulerWidth):
                # Painted in the border colour; Qt's own is in the text colour.
                fmt.clearProperty(_P.BlockTrailingHorizontalRulerWidth)
                self._rules.append(n)
                fmt.setTopMargin(fonts.px(10))
                fmt.setBottomMargin(fonts.px(10))
            elif not in_table:
                fmt.setLineHeight(150, _PROPORTIONAL)
                the_list = block.textList()
                if the_list is not None:
                    # The next item of this list, or of a list nested in or
                    # around it, follows closely; a new list keeps its distance.
                    nl = nxt.textList() if nxt is not None else None
                    same_next = nl is not None and (
                        nl is the_list or nl.format().indent() != the_list.format().indent())
                    fmt.setTopMargin(fonts.px(3))
                    fmt.setBottomMargin(fonts.px(3) if same_next else gap)
                    lf = the_list.format()
                    if lf.style() in _BULLETS:
                        lf.setStyle(_BULLETS[(max(lf.indent(), 1) - 1) % len(_BULLETS)])
                        the_list.setFormat(lf)
                else:
                    fmt.setTopMargin(0)
                    fmt.setBottomMargin(gap)
                if level:
                    fmt.setLeftMargin(fonts.px(20) * level)
                    fmt.setTopMargin(fonts.px(6) if quote_start == n else 0)

                    def as_quote(cf: QTextCharFormat) -> None:
                        cf.setForeground(dim)

                    self._rewrite_chars(block, as_quote)
            cursor.setBlockFormat(fmt)

            # Inline code and links.
            if not code:
                def inline(cf: QTextCharFormat) -> None:
                    if cf.isAnchor():
                        cf.setForeground(link)
                        cf.setFontUnderline(False)
                    elif cf.fontFixedPitch() or "monospace" in (cf.fontFamilies() or []):
                        cf.setFont(fonts.font(CODE_PX, mono=True))
                        cf.setBackground(code_bg)

                self._rewrite_chars(block, inline)
        self._style_tables(tok)
        self._fit_images()

    def _rewrite_chars(self, block: QTextBlock, change: Callable[[QTextCharFormat], None]) -> None:
        """Apply ``change`` to every fragment of ``block`` by REPLACING its
        format (a merge cannot remove a property)."""
        doc = self.document()
        for frag in list(_each_fragment(block)):
            cf = frag.charFormat()
            if cf.isImageFormat():
                continue
            change(cf)
            c = QTextCursor(doc)
            c.setPosition(frag.position())
            c.setPosition(frag.position() + frag.length(), QTextCursor.MoveMode.KeepAnchor)
            c.setCharFormat(cf)

    def _style_tables(self, tok: Callable[..., str]) -> None:
        from ..viz import fonts

        doc = self.document()
        seen: set[int] = set()
        block = doc.begin()
        while block.isValid():
            table = QTextCursor(block).currentTable()
            if table is not None and table.firstPosition() not in seen:
                seen.add(table.firstPosition())
                fmt = table.format()
                fmt.setBorder(1)
                fmt.setBorderBrush(QBrush(_colour(tok("border"))))
                fmt.setBorderStyle(QTextFrameFormat.BorderStyle.BorderStyle_Solid)
                fmt.setBorderCollapse(True)
                fmt.setCellSpacing(0)
                fmt.setCellPadding(fonts.px(7))
                fmt.setTopMargin(fonts.px(2))
                fmt.setBottomMargin(fonts.px(16))
                fmt.setWidth(QTextLength(QTextLength.Type.VariableLength, 0))
                table.setFormat(fmt)
                head, stripe = _colour(tok("surface3")), _colour(tok("surface"))
                for row in range(table.rows()):
                    for col in range(table.columns()):
                        cell = table.cellAt(row, col)
                        cf = cell.format().toTableCellFormat()
                        if row == 0:
                            cf.setBackground(head)
                        elif row % 2 == 0:
                            cf.setBackground(stripe)
                        cell.setFormat(cf)
                        if row == 0:
                            c = cell.firstCursorPosition()
                            c.setPosition(cell.lastCursorPosition().position(),
                                          QTextCursor.MoveMode.KeepAnchor)
                            bold = QTextCharFormat()
                            bold.setFontWeight(600)
                            c.mergeCharFormat(bold)
            block = block.next()

    def _fit_images(self) -> None:
        """An image wider than the column is shown at the column's width."""
        if self._base is None:
            return
        from ..viz import fonts

        widest = fonts.px(COLUMN_PX)
        doc = self.document()
        block = doc.begin()
        while block.isValid():
            for frag in list(_each_fragment(block)):
                cf = frag.charFormat()
                if not cf.isImageFormat():
                    continue
                img = cf.toImageFormat()
                path = Path(img.name())
                if not path.is_absolute():
                    path = self._base / path
                size = QImageReader(str(path)).size() if path.is_file() else None
                if size is not None and size.width() > widest:
                    img.setWidth(widest)
                    img.setHeight(size.height() * widest / size.width())
                    c = QTextCursor(doc)
                    c.setPosition(frag.position())
                    c.setPosition(frag.position() + frag.length(),
                                  QTextCursor.MoveMode.KeepAnchor)
                    c.setCharFormat(img)
            block = block.next()

    def _fit_column(self, *, force: bool = False) -> None:
        """A centred column no wider than ``COLUMN_PX``, with room around it."""
        from ..viz import fonts

        root = self.document().rootFrame()
        fmt = root.frameFormat()
        room = self.viewport().width()
        side = max(fonts.px(28), (room - fonts.px(COLUMN_PX)) // 2)
        if not force and fmt.leftMargin() == side and fmt.rightMargin() == side:
            return
        fmt.setLeftMargin(side)
        fmt.setRightMargin(side)
        fmt.setTopMargin(fonts.px(22))
        fmt.setBottomMargin(fonts.px(32))
        root.setFrameFormat(fmt)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._fit_column()

    def column(self) -> tuple[float, float]:
        """Left and right of the text column, in document coordinates."""
        doc = self.document()
        root = doc.rootFrame().frameFormat()
        return root.leftMargin(), doc.size().width() - root.rightMargin()

    # -- painting --------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._codes or self._quotes or self._rules or self._underlined:
            self._paint_behind()
        super().paintEvent(event)

    def _rect(self, number: int) -> QRectF:
        doc = self.document()
        return doc.documentLayout().blockBoundingRect(doc.findBlockByNumber(number))

    def _paint_behind(self) -> None:
        from ..viz import fonts
        from ..viz.bridge import ThemeHub

        tok = ThemeHub.instance().theme.token
        border = _colour(tok("border"))
        left, right = self.column()
        p = QPainter(self.viewport())
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.translate(-self.horizontalScrollBar().value(), -self.verticalScrollBar().value())
        pad = fonts.px(12)
        radius = fonts.px(6)
        p.setPen(QPen(border, 1))
        p.setBrush(_colour(tok("surface3")))
        for first, last in self._codes:
            box = QRectF(left, self._rect(first).top() - pad, right - left,
                         self._rect(last).bottom() - self._rect(first).top() + 2 * pad)
            p.drawRoundedRect(box.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        bar = max(3, fonts.px(3))
        p.setPen(QPen(QColor(0, 0, 0, 0)))
        p.setBrush(border)
        for first, last, level in self._quotes:
            top, bottom = self._rect(first).top() - fonts.px(2), self._rect(last).bottom()
            for k in range(level):
                x = left + fonts.px(20) * k
                p.drawRoundedRect(QRectF(x, top, bar, bottom - top + fonts.px(2)),
                                  bar / 2, bar / 2)
        p.setPen(QPen(border, 1))
        for n in self._underlined:
            y = round(self._rect(n).bottom() + fonts.px(7)) + 0.5
            p.drawLine(QLineF(left, y, right, y))
        p.setPen(QPen(border, max(2, fonts.px(2))))
        for n in self._rules:
            y = self._rect(n).center().y()
            p.drawLine(QLineF(left, y, right, y))
        p.end()

    # -- links -----------------------------------------------------------------

    def _on_link(self, url: QUrl) -> None:
        if url.scheme() in ("http", "https", "mailto", "ftp"):
            QDesktopServices.openUrl(url)
            return
        if not url.scheme() and not url.path() and url.fragment():
            self.scrollToAnchor(url.fragment())
            return
        if self._base is None:
            return
        target = Path(url.toLocalFile()) if url.isLocalFile() else self._base / url.path()
        try:
            target = target.resolve()
        except OSError:
            return
        if target.exists():
            self.file_requested.emit(target)


__all__ = ["BODY_PX", "CODE_PX", "COLUMN_PX", "HEADING_PX", "MarkdownView"]
