"""Generate the README workflow figure, in a light and a dark variant.

Run it from anywhere; it writes next to itself:

    python docs/workflow_svg.py

The two files are the same drawing with two palettes, which is the whole reason
this is generated rather than hand-written: GitHub picks between them with a
``<picture>`` element, and a figure edited twice by hand drifts.

Two things drive every number below.

**Type size.** GitHub renders the README at roughly 950px, so a figure wider
than that is scaled down and its text with it. The canvas is therefore kept
close to that width and the type set large, rather than the other way round.
That is also why the six steps wrap onto two rows: across one row the captions
came out at about 8px on the page.

**One grid.** Every band is a full-width card between the same margins, every
band label sits on the same left edge, and the three stage columns share one set
of centres, so the connectors point at the middle of the step they feed.

No dependencies and no fonts to install. Text is placed with an estimated
advance width per character, so every string is measured against the box it has
to sit in rather than trusted to fit.
"""

from __future__ import annotations

from pathlib import Path

# GitHub's README column is 948px wide (1012px body, 32px padding each side),
# so the canvas is exactly that: the figure renders 1:1 and the type is never
# scaled down. Every size below is therefore also the size on the page.
W = 948

LIGHT = {
    "name": "workflow.svg",
    "page": "#f6f8fa",
    "page_edge": "#d8dee4",
    "fg": "#1f2328",
    "muted": "#4a5259",
    "faint": "#7d868f",
    "card": "#ffffff",
    "card_edge": "#d0d7de",
    "rail": "#b8c2cc",
    "accent": "#0b6fd4",
    "accent_ink": "#ffffff",
    "code": "#f6f8fa",
    "code_edge": "#d8dee4",
    "ok": "#1a7f37",
    "ok_bg": "#dafbe1",
    "ok_edge": "#aceebb",
}

DARK = {
    "name": "workflow-dark.svg",
    "page": "#0d1117",
    "page_edge": "#30363d",
    "fg": "#e6edf3",
    "muted": "#a9b1ba",
    "faint": "#7d868f",
    "card": "#161c24",
    "card_edge": "#3d444d",
    "rail": "#4a525c",
    "accent": "#4FC3F7",
    "accent_ink": "#04222f",
    "code": "#0d1117",
    "code_edge": "#30363d",
    "ok": "#3fb950",
    "ok_bg": "#102c17",
    "ok_edge": "#1f6f2e",
}

SANS = ("-apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, Helvetica, "
        "Arial, sans-serif")
MONO = ("ui-monospace, SFMono-Regular, &quot;SF Mono&quot;, Menlo, Consolas, "
        "monospace")

# One type scale, used everywhere. Raising a value here is the only place a
# size changes; the boxes are measured from these.
T_LABEL = 12.5      # band labels
T_CHIP = 14.5       # source name
T_CHIP_SUB = 12.5   # source formats
T_TITLE = 18        # step name
T_BODY = 13.5       # step caption
T_ENGINE = 11       # engine list inside step 3
T_TREE = 13         # output tree
T_PILL = 12.5       # output chips
T_NOTE = 13.5       # output captions and footer
T_BADGE = 14.5      # step number

MARGIN = 28
INNER = W - 2 * MARGIN
PAD = 18            # inside every full-width card

# What the scanner actually recognises. Kept in step with
# ``inventory/eeg_meg._RECOGNISED_EXTS`` and ``inventory/pet_ecat``.
SOURCES = [
    ("MRI", "DICOM", "#4C9AFF"),
    ("PET", "DICOM, ECAT7", "#F2994A"),
    ("EEG / iEEG", "EDF, BDF, BrainVision, EEGLAB", "#57C7A3"),
    ("MEG", "FIF, CTF, KIT", "#B39DDB"),
    ("Physio", "Siemens CMRR", "#8B949E"),
    ("Blood", "PMOD", "#E5534B"),
]

STAGES = [
    ("Scan", "Reads every file rather than trusting its name, and proposes a "
             "BIDS name for each series."),
    ("Review", "One editable table. Fix subjects, sessions, tasks and runs, "
               "and settle name clashes before writing."),
    # Two lines here, not three: step 3 also carries the engine list, and
    # ``check_fits`` fails the build if this grows.
    ("Convert", "Per subject, into staging. Committed only when it "
                "finishes."),
    ("Enrich", "Answer what the files cannot say. The form, and its "
               "requirement levels, come from the schema."),
    # The Editor is where a dataset is actually curated. The viewers are the
    # extra, not the point, so they come second in the sentence.
    ("Curate", "Sidecars as schema-aware forms, tables as spreadsheets, and "
               "viewers for volumes and signals."),
    ("Validate", "Every finding names the schema rule it came from, and the "
                 "fix button opens the field that needs it."),
]

# Shown inside step 3, small, because the point of the figure is that the
# conversion engines are the least interesting part of the tool.
ENGINES = ["dcm2niix  ·  mne-bids  ·  nibabel",
           "bidsphysio  ·  pet2bids"]

# (kind, datatype, filename, sidecar tag). Drawn as separate text elements at
# fixed columns rather than one padded string, because SVG collapses runs of
# spaces and a padded string does not line up.
TREE = [
    ("root", "", "sub-001/", ""),
    ("row", "anat/", "sub-001_T1w.nii.gz", "+ .json"),
    ("row", "func/", "sub-001_task-rest_bold.nii.gz", "+ .json"),
    ("row", "pet/", "sub-001_trc-18FFDG_pet.nii.gz", "+ .json"),
    ("row", "eeg/", "sub-001_task-rest_eeg.edf", "+ .json"),
    ("cont", "", "sub-001_task-rest_channels.tsv", ""),
]

# A chip and what it is, one per line, so the column has the same top and
# bottom as the tree beside it.
RESULT_ROWS = [
    ("dataset_description.json", "written from your answers", False),
    ("participants.tsv", "one row per subject", False),
    ("*_scans.tsv", "the files written, and when", False),
    ("0 errors", "checked against the schema", True),
]

FOOTER = ("Underneath all six steps: every scan is versioned and every edit "
          "recorded, so curation can be resumed or undone.")


# --------------------------------------------------------------------------
# primitives


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def width_of(s: str, size: float, *, bold: bool = False,
             mono: bool = False) -> float:
    """Estimated advance width. Deliberately a little generous."""
    per = 0.60 if mono else (0.575 if bold else 0.522)
    return len(s) * size * per


def wrap(s: str, size: float, box: float) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in s.split():
        trial = f"{current} {word}".strip()
        if current and width_of(trial, size) > box:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def text(x, y, s, *, size=T_BODY, fill="#000", bold=False, anchor="start",
         mono=False, spacing=None):
    style = [f"font-family:{MONO if mono else SANS}", f"font-size:{size}px",
             f"fill:{fill}"]
    if bold:
        style.append("font-weight:600")
    if spacing:
        style.append(f"letter-spacing:{spacing}px")
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'style="{";".join(style)}">{esc(s)}</text>')


def rect(x, y, w, h, *, r=10, fill="none", stroke="none", sw=1):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def band_label(y, s, c):
    """Every band label sits on the same left edge as every card."""
    return text(MARGIN, y, s.upper(), size=T_LABEL, fill=c["faint"], bold=True,
                spacing=1.5)


def down_arrow(x, y0, y1, c):
    """A connector between bands, carrying its own arrowhead so no marker
    definition has to survive GitHub's SVG sanitising."""
    head = 7.0
    return (f'<path d="M{x:.1f} {y0:.1f} V{y1 - head:.1f}" '
            f'stroke="{c["rail"]}" stroke-width="2" fill="none" '
            f'stroke-linecap="round"/>'
            f'<path d="M{x - head:.1f} {y1 - head:.1f} L{x:.1f} {y1:.1f} '
            f'L{x + head:.1f} {y1 - head:.1f} Z" fill="{c["rail"]}"/>')


# --------------------------------------------------------------------------
# the figure

# Vertical rhythm. Each band is (label baseline, card top); the gaps between
# bands are equal so the three sections read as one grid.
LBL_1, SRC_Y = 36, 48
SRC_H = 90
LBL_2, RAIL_A = 194, 236
CARD_H = 144
ROW_GAP = 74                      # rail A cards bottom to rail B
RAIL_B = RAIL_A + 16 + CARD_H + ROW_GAP
LBL_3 = RAIL_B + 16 + CARD_H + 52
RES_Y = LBL_3 + 14
TREE_STEP = 23
RES_H = len(TREE) * TREE_STEP + 18 + 2 * PAD
FOOT_Y = RES_Y + RES_H + 20
FOOT_H = 44
H = FOOT_Y + FOOT_H + 24

PER_ROW = 3
CARD_GAP = 16
CARD_W = (INNER - CARD_GAP * (PER_ROW - 1)) / PER_ROW
CENTRES = [MARGIN + i * (CARD_W + CARD_GAP) + CARD_W / 2
           for i in range(PER_ROW)]


def build(c: dict) -> str:
    o: list[str] = []
    o.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="BIDS Manager workflow: raw MRI, PET, EEG, MEG, iEEG, '
        f'physio and blood recordings are scanned, reviewed, converted, '
        f'enriched, curated and validated into one BIDS dataset, with every '
        f'step recorded in the project log.">'
    )
    o.append(rect(0.5, 0.5, W - 1, H - 1, r=18, fill=c["page"],
                  stroke=c["page_edge"]))

    # ---- band 1: sources ------------------------------------------------
    o.append(band_label(LBL_1, "What goes in", c))
    o.append(rect(MARGIN, SRC_Y, INNER, SRC_H, r=14, fill=c["card"],
                  stroke=c["card_edge"]))

    chip_h, chip_gap, chip_pad = 54, 10, 13
    natural = []
    for name, formats, dot in SOURCES:
        natural.append(max(width_of(name, T_CHIP, bold=True) + 16,
                           width_of(formats, T_CHIP_SUB)) + 2 * chip_pad)
    # Share the slack out evenly so the row's right edge lands on the card's
    # inner edge: seven chips that stop short of it read as a mistake.
    room = INNER - 2 * PAD - chip_gap * (len(natural) - 1)
    slack = (room - sum(natural)) / len(natural)
    x = MARGIN + PAD
    cy = SRC_Y + (SRC_H - chip_h) / 2
    for (name, formats, dot), w in zip(SOURCES, natural):
        w += slack
        o.append(rect(x, cy, w, chip_h, r=10, fill=c["page"],
                      stroke=c["card_edge"]))
        o.append(f'<circle cx="{x + chip_pad + 4:.1f}" cy="{cy + 20:.1f}" '
                 f'r="4" fill="{dot}"/>')
        o.append(text(x + chip_pad + 16, cy + 25, name, size=T_CHIP,
                      fill=c["fg"], bold=True))
        o.append(text(x + chip_pad, cy + 43, formats, size=T_CHIP_SUB,
                      fill=c["muted"]))
        x += w + chip_gap

    o.append(down_arrow(CENTRES[0], SRC_Y + SRC_H + 10, RAIL_A - 26, c))

    # ---- band 2: the six steps -----------------------------------------
    o.append(band_label(LBL_2, "What it does", c))

    def rail(y_rail: float) -> None:
        o.append(f'<path d="M{CENTRES[0]:.1f} {y_rail:.1f} '
                 f'H{CENTRES[-1]:.1f}" stroke="{c["rail"]}" stroke-width="2" '
                 f'stroke-linecap="round"/>')
        for a, b in zip(CENTRES, CENTRES[1:]):
            mid = (a + b) / 2
            o.append(f'<path d="M{mid - 4.5:.1f} {y_rail - 5:.1f} '
                     f'L{mid + 3.5:.1f} {y_rail:.1f} L{mid - 4.5:.1f} '
                     f'{y_rail + 5:.1f} Z" fill="{c["rail"]}"/>')

    rail(RAIL_A)
    rail(RAIL_B)

    for i, (title, body) in enumerate(STAGES):
        row, col = divmod(i, PER_ROW)
        cx = CENTRES[col]
        badge_y = RAIL_A if row == 0 else RAIL_B
        top = badge_y + 16
        o.append(rect(cx - CARD_W / 2, top, CARD_W, CARD_H, r=14,
                      fill=c["card"], stroke=c["card_edge"]))
        o.append(f'<circle cx="{cx:.1f}" cy="{badge_y:.1f}" r="16" '
                 f'fill="{c["accent"]}" stroke="{c["page"]}" '
                 f'stroke-width="3"/>')
        o.append(text(cx, badge_y + 5.5, str(i + 1), size=T_BADGE,
                      fill=c["accent_ink"], bold=True, anchor="middle"))
        o.append(text(cx, top + 40, title, size=T_TITLE, fill=c["fg"],
                      bold=True, anchor="middle"))
        ty = top + 68
        for line in wrap(body, T_BODY, CARD_W - 40):
            o.append(text(cx, ty, line, size=T_BODY, fill=c["muted"],
                          anchor="middle"))
            ty += 20
        if title == "Convert":
            ey = top + CARD_H - 32
            for line in ENGINES:
                o.append(text(cx, ey, line, size=T_ENGINE, fill=c["faint"],
                              mono=True, anchor="middle"))
                ey += 15

    # Steps 3 to 4 wrap from the end of the first row back to the start of the
    # second, routed through the gap between the rows so it crosses no badge.
    r = 16
    a_bottom = RAIL_A + 16 + CARD_H
    turn = a_bottom + (ROW_GAP - 16) / 2
    tip = RAIL_B - 17                 # lands on badge 4's top edge
    o.append(
        f'<path d="M{CENTRES[-1]:.1f} {a_bottom:.1f} V{turn - r:.1f} '
        f'A{r} {r} 0 0 1 {CENTRES[-1] - r:.1f} {turn:.1f} '
        f'H{CENTRES[0] + r:.1f} '
        f'A{r} {r} 0 0 0 {CENTRES[0]:.1f} {turn + r:.1f} '
        f'V{tip - 7:.1f}" stroke="{c["rail"]}" stroke-width="2" '
        f'fill="none" stroke-linecap="round"/>'
        f'<path d="M{CENTRES[0] - 7:.1f} {tip - 7:.1f} '
        f'L{CENTRES[0]:.1f} {tip:.1f} '
        f'L{CENTRES[0] + 7:.1f} {tip - 7:.1f} Z" fill="{c["rail"]}"/>'
    )

    o.append(down_arrow(CENTRES[-1], RAIL_B + 16 + CARD_H + 10, RES_Y - 10, c))

    # ---- band 3: the result --------------------------------------------
    o.append(band_label(LBL_3, "What comes out", c))
    o.append(rect(MARGIN, RES_Y, INNER, RES_H, r=14, fill=c["card"],
                  stroke=c["card_edge"]))

    row_step = TREE_STEP
    tree_x = MARGIN + PAD
    tree_y = RES_Y + PAD
    tree_h = len(TREE) * row_step + 18
    # Three fixed columns: the tick, the datatype, the file. Measured rather
    # than padded with spaces, which SVG collapses.
    dt_x = tree_x + 40
    fn_x = dt_x + 52
    tree_w = fn_x - tree_x + max(
        width_of(fn, T_TREE, mono=True)
        + (width_of(f" {tag}", T_TREE, mono=True) if tag else 0)
        for _kind, _dt, fn, tag in TREE) + 18
    o.append(rect(tree_x, tree_y, tree_w, tree_h, r=10, fill=c["page"],
                  stroke=c["code_edge"]))

    base = tree_y + 26
    rows = [(kind, dt, fn, tag, base + i * row_step)
            for i, (kind, dt, fn, tag) in enumerate(TREE)]
    branch_x = tree_x + 22
    last_tick = max(y for kind, _d, _f, _t, y in rows if kind == "row")
    o.append(f'<path d="M{branch_x:.1f} {rows[0][4] + 8:.1f} '
             f'V{last_tick - 5:.1f}" stroke="{c["faint"]}" '
             f'stroke-width="1.3" fill="none" opacity="0.65"/>')
    for kind, dt, fn, tag, y in rows:
        if kind == "root":
            o.append(text(tree_x + 14, y, fn, size=T_TREE, fill=c["fg"],
                          mono=True, bold=True))
            continue
        if kind == "row":
            o.append(f'<path d="M{branch_x:.1f} {y - 5:.1f} '
                     f'H{branch_x + 12:.1f}" stroke="{c["faint"]}" '
                     f'stroke-width="1.3" opacity="0.65"/>')
            o.append(text(dt_x, y, dt, size=T_TREE, fill=c["fg"], mono=True))
        o.append(text(fn_x, y, fn, size=T_TREE, fill=c["muted"], mono=True))
        if tag:
            o.append(text(fn_x + width_of(f"{fn} ", T_TREE, mono=True), y, tag,
                          size=T_TREE, fill=c["faint"], mono=True))

    # Right column: one chip per line, its first baseline shared with the
    # tree's and its last landing on the tree's last, so the two columns of the
    # card start and finish together. The captions share a left edge, which
    # means the chips need one width rather than each their own.
    col_x = tree_x + tree_w + 26
    chip_w = max(width_of(label, T_PILL, mono=not ok, bold=ok) + 26
                 for label, _note, ok in RESULT_ROWS)
    note_x = col_x + chip_w + 16
    step = (len(TREE) - 1) * row_step / (len(RESULT_ROWS) - 1)

    for i, (label, note, is_ok) in enumerate(RESULT_ROWS):
        y = base + i * step
        o.append(rect(col_x, y - 19, chip_w, 28, r=14,
                      fill=c["ok_bg"] if is_ok else c["page"],
                      stroke=c["ok_edge"] if is_ok else c["code_edge"]))
        if is_ok:
            o.append(text(col_x + chip_w / 2, y + 1, label, size=T_PILL,
                          fill=c["ok"], bold=True, anchor="middle"))
        else:
            o.append(text(col_x + 13, y + 1, label, size=T_PILL,
                          fill=c["muted"], mono=True))
        o.append(text(note_x, y + 1, note, size=T_NOTE, fill=c["muted"]))

    # ---- footer ---------------------------------------------------------
    o.append(rect(MARGIN, FOOT_Y, INNER, FOOT_H, r=12, fill=c["card"],
                  stroke=c["card_edge"]))
    o.append(f'<circle cx="{MARGIN + PAD + 5:.1f}" '
             f'cy="{FOOT_Y + FOOT_H / 2:.1f}" r="4.5" fill="{c["accent"]}"/>')
    o.append(text(MARGIN + PAD + 20, FOOT_Y + FOOT_H / 2 + 5, FOOTER,
                  size=T_NOTE, fill=c["muted"]))

    o.append("</svg>")
    return "\n".join(o) + "\n"


def check_fits() -> None:
    """Fail loudly when a string has outgrown the box drawn for it.

    Every overflow in this figure so far has been silent: text simply drew
    over its neighbour and the only way to notice was to look at the PNG. The
    same arithmetic that positions each element can decide whether it fits, so
    it does, and regenerating raises instead of shipping a broken figure.
    """
    bad: list[str] = []

    # Stage captions, against the space left between the title and either the
    # engine list (step 3) or the bottom of the card.
    for title, body in STAGES:
        lines = wrap(body, T_BODY, CARD_W - 40)
        end = 68 + (len(lines) - 1) * 20
        room = CARD_H - (42 if title == "Convert" else 12)
        if end > room:
            bad.append(f"step {title!r}: {len(lines)} caption lines need "
                       f"{end:.0f}px, card allows {room:.0f}px")
        for line in lines:
            if width_of(line, T_BODY) > CARD_W - 40:
                bad.append(f"step {title!r}: line {line!r} is too wide")
    for line in ENGINES:
        if width_of(line, T_ENGINE, mono=True) > CARD_W - 30:
            bad.append(f"engine line {line!r} is too wide for the Convert card")

    # Source chips, against the width of the band they sit in.
    natural = sum(max(width_of(n, T_CHIP, bold=True) + 16,
                      width_of(f, T_CHIP_SUB)) + 26 for n, f, _ in SOURCES)
    room = INNER - 2 * PAD - 10 * (len(SOURCES) - 1)
    if natural > room:
        bad.append(f"source chips need {natural:.0f}px, band allows "
                   f"{room:.0f}px")

    # Result captions, against what is left after the tree and the chips.
    tree_w = 92 + max(width_of(fn, T_TREE, mono=True)
                      + (width_of(f" {t}", T_TREE, mono=True) if t else 0)
                      for _k, _d, fn, t in TREE) + 18
    chip_w = max(width_of(lb, T_PILL, mono=not ok, bold=ok) + 26
                 for lb, _n, ok in RESULT_ROWS)
    room = INNER - 2 * PAD - tree_w - 26 - chip_w - 16
    for _lb, note, _ok in RESULT_ROWS:
        if width_of(note, T_NOTE) > room:
            bad.append(f"result note {note!r} needs "
                       f"{width_of(note, T_NOTE):.0f}px, column allows "
                       f"{room:.0f}px")

    if width_of(FOOTER, T_NOTE) > INNER - 2 * PAD - 20:
        bad.append("footer line is wider than the footer card")

    if bad:
        raise RuntimeError("workflow figure does not fit:\n  "
                           + "\n  ".join(bad))


def main() -> None:
    check_fits()
    here = Path(__file__).resolve().parent
    for palette in (LIGHT, DARK):
        target = here / palette["name"]
        target.write_text(build(palette), encoding="utf-8")
        print(f"wrote {target.relative_to(here.parent)} "
              f"({target.stat().st_size / 1024:.1f} kB, {W}x{H})")


if __name__ == "__main__":
    main()
