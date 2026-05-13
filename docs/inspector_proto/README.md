# Inspector prototype

Static PyQt6 visual fidelity test for the Inspector GUI proposal
(`gui_mockups.html` proposal 1). **No real conversion logic** — the goal is
purely to confirm Qt6 can render the layout cleanly enough.

## Run

The shared venv at the project root already has PyQt6 6.11.0 installed:

```
cd /Users/karelo/PycharmProjects/superbidsmanager/inspector_proto
../.venv/bin/python proto.py
```

Or, to install as a proper package + entry point:

```
../.venv/bin/pip install -e .
../.venv/bin/inspector-proto
```

A single window opens at 1480×880. Resize the splitters between panes
to see how the layout responds.

## What you're looking at

| Region | Purpose |
|---|---|
| Toolbar | Scan button, status chips (valid/warn/error/skipped), schema pills, Settings, Run. |
| Path bars | Raw input + BIDS output. |
| Col 1 — Raw FS tree | Live mirror of the input directory; skipped sequences struck-through; selected row highlighted. |
| Col 2 — Filter tree | Tri-state checkboxes (`Studyname → sub-XXX → ses-YYY → datatype`) plus an "always exclude" footer. |
| Col 3 — Inspection table | One row per recording. Status-badge column, include-checkbox column, inline editing, row-state highlights for warn/err/skip/selected. Conf column is colored by value. |
| Col 4 — Properties panel | Schema-driven entity form, predicted-path preview with entity coloring, validation messages, "Why this name?" provenance. |
| Bottom dock | Tabbed: BIDS preview / Log / Conflicts / Statistics. |
| Status bar | Schema version + counts + readiness pill. |

## Files

- `proto.py` — all Qt code (single file for ease of inspection).
- `theme.qss` — the visual layer; tweak this file and re-run `proto.py` to
  iterate on styling without touching layout code.
- `data.py` — hard-coded data matching the HTML mockup
  (subjects `OL_0001`/`OL_0002`/`OL_0003` from
  `~/Development/datasets/BIDS_Manager/raw_data/MRI/neuroimaging_unit_new`).

## What's faked

Everything except the visual rendering. No file I/O, no scanning, no
conversion, no real model. Inline editing in the table works (the
underlying `QStandardItemModel` accepts edits), but predicted-basename
recomputation is not wired up. That's deliberate — once the visual
fidelity is approved, the real engine lives in the new package, not
here.

## What to evaluate

1. **Density and spacing** — does it feel as tight as the HTML mockup?
2. **Status-badge legibility** — circle badges painted via `QPainter`.
3. **Tri-state checkboxes** — Qt's `ItemIsAutoTristate` works out of the
   box; the modality tree's three-state propagation is real.
4. **Predicted-path coloring** — entity tokens are colored via inline
   rich-text spans (accent / purple / teal / dim).
5. **Editor-style row highlights** — warn rows tinted amber, err rows
   tinted red, skip rows ghosted with strikethrough, selected row in the
   accent tint.
6. **Splitter behaviour** — dragging the dividers should feel native.
7. **Scrollbar style** — slim, dark, hover-only on track.

If any of these fall flat on real hardware, the gap should be small
enough to close in QSS without a layout redesign.

## Known approximations vs the HTML mockup

- No backdrop-filter blur on the toolbar; solid surface color instead.
- No drop shadows on the toolbar / status bar; subtle borders only.
- "✓" / "○" prefixes in the path bars are inline characters, not icons.
  In the real package these become bundled SVGs.
- The "Run conversion" primary button uses an accent fill with dark
  text instead of an outlined chip. That matches the mockup's intent.
