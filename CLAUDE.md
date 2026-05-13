# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo holds

`/Users/karelo/PycharmProjects/superbidsmanager/BIDS-Manager/` is the **`bids-manager`** package on PyPI — a schema-driven BIDS converter, curator, and editor. v1.0.0 is a complete re-imagination of the 0.x line; the old HeuDiConv / dcm2bids two-engine GUI is gone. The PyPI distribution name is `bids-manager`, the Python import name is `bidsmgr` (same pattern as `pip install scikit-learn` / `import sklearn`).

| Path | Purpose |
|---|---|
| `bidsmgr/` | The importable Python package (~16 sub-packages). |
| `tests/` | ~730 unit + GUI + integration tests; 49 real-data tests gated on env vars. |
| `docs/` | Design documents preserved from the cutover: `architecture.md`, `super_plan.md`, `improvement_plan.md`, `gui_mockups.html`, and the standalone `inspector_proto/` GUI prototype. |
| `miscellaneous/images/` | Non-package image assets (e.g. `Logo_negative_square.png`). |
| `external/` | Vendored Python embed used by the Windows installer. |
| `Installers/` | Packaged Windows installer (`Installers.zip`). |

The historical v0.2.5 codebase (`bids_manager/` Python package, HeuDiConv + dcm2bids drivers, 9.2k-LOC single-file GUI) is gone from `main` but lives in git history through tag/commit `v0.2.5`.

## Architecture (read this before adding features)

The single design bet: **the BIDS schema is the engine**, not a check at the end. Every layer reads from the same `bidsschematools` schema. See `docs/architecture.md` §0–§3 for the rationale and §12 for the module layout.

### Module layout (`bidsmgr/`)

```
bidsmgr/                            ← the importable package
├── __init__.py                     __version__ = "1.0.0"
├── main.py                         GUI entry (also the `bidsmgr` console script)
├── schema/                         keystone — bidsschematools wrapper
├── inventory/                      per-modality scanners (DICOM, EEG/MEG, physio)
├── classifier/                     chained classifiers (BidsGuess, sequence dict)
├── planner/                        EntityPlan + user edits, schema-validated
├── converter/                      pluggable backends
│   └── backends/                   Dcm2niixDirect, MneBidsBackend, PhysioDcmBackend
├── metadata/                       post-conv schema engine
├── fixups/                         fmap, IntendedFor, scans.tsv
├── project/                        event-sourced .bidsmgr/ bundles
├── editor/                         post-conv editor logic (no Qt)
├── gui/                            ← THE ONLY Qt subtree
│   ├── theme.qss, theme_manager.py
│   ├── widgets/  delegates/  models/
├── workers/                        QThread bridges
├── cli/                            CLI verbs (scan / rebuild / convert / metadata / validate)
└── util/                           cross-OS path safety, Qt platform helpers
```

### Architectural prevention guards (don't break these)

These exist to avoid repeating the 0.x architectural problems documented in `docs/improvement_plan.md` §12:

1. `schema/` is the keystone — everything imports from it; it imports nothing.
2. `gui/` is the only Qt-coupled subtree; `workers/` also imports Qt (the QThread bridge); nothing else does.
3. **No `Pipeline` orchestrator.** Orchestration is straight-line code in `cli/<verb>.py` and `gui/<panel>.py`.
4. **No subpackage named `core/`** (the name is poisoned by the 0.x post-mortem).
5. Pure-data types only (Pydantic / dataclass; no I/O methods).
6. Functions over classes where possible.

## Current state (v1.0.0)

Engine:

- **`schema/`** — keystone wrapper over `bidsschematools` 1.2.2 (BIDS 1.11.1).
- **`inventory/`** — `mri_dicom.scan_dicoms_long`, `eeg_meg.scan_eeg_meg` (stamps `bids_guess_*` per row), `subject_identity.cluster_subjects`, `probe_convert.probe_rows`, `rebuild.rebuild_from_entities/columns`.
- **`classifier/`** — `dcm2niix_bidsguess` (M1) + `sequence_dict` fallback + B0-reference reroute + DWI-derivative detection.
- **`converter/`** — per-task backend dispatch. Three backends: `Dcm2niixDirect`, `MneBidsBackend` (MEG datatype skips standard montages; EEG rename is collision-safe), `PhysioDcmBackend`.
- **`fixups/`** — `fieldmaps`, `intended_for`, `scans_tsv`.
- **`metadata/`** — `engine.run_metadata`.
- **`editor/`** — `validator.validate` (two-layer), `validator.validate_file/folder` (layer-1-only partials), `html_report.render_html`.
- **`cli/`** — five verbs: `scan`, `rebuild`, `convert`, `metadata`, `validate`. `convert` takes `--raw-root` for EEG/MEG relative-path resolution.
- **`project/`** — event-sourced `.bidsmgr/` bundles. Events: `ProjectCreated`, `ScanImported`, `UserSet{Cell,Entity}`, `UserToggleInclude`, `TodoAcknowledged`, `StageCompleted`. Wired through the GUI.
- **`workers/`** — `ScanWorker` / `ConvertWorker` / `MetadataWorker` / `ValidateWorker` / `ReportWorker` / `FileReportWorker` / `FolderReportWorker` / `NiftiLoaderWorker`.
- **`util/paths.py`** — cross-OS path safety. `safe_path_component(raw)` rejects invalid chars + Windows device names; caps at 96 chars with SHA-1 disambiguator. `long_path(p)` wraps with `\\?\` on Windows when nearing `MAX_PATH`.

GUI (`bidsmgr.gui.*`, launched via `bidsmgr` console script):

- **MainWindow shell** — top header (clickable brand → `AboutDialog`; `Converter | Editor` pill switcher; theme toggle) + `QStackedWidget` + status bar. Active view persists via `AppSettings`.

**Converter view** (`bidsmgr.gui.converter_panel.ConverterPanel`) — toolbar (Scan / TSV name / chips / spinner / Settings / Run) + 4-col splitter (`RawFsPane` + `OutputFsPane` | `FilterPane` | inspection `QTableView` over `InventoryTableModel` + per-pane footer for `⌬ Highlight aborts` + `✎ Bulk edit…` | `PropertiesPanel`) + bottom dock (Log / Conflicts / BIDS preview / Statistics). Settings dialog has 3 tabs (Display / Scan / Convert + post-convert chain); all persists via `QSettings`.

**Editor view** (`bidsmgr.gui.editor_panel.EditorPanel`) — toolbar (`📁 Open BIDS root` / `✓ Validate file` / `📂 Validate folder` / `🗂 Validate dataset` / `⚡ Strict BIDS` toggle / severity chips / `BusySpinner`) + `PathBar` + 3-col splitter (BIDS tree | center stack | validation pane). Center stack routes by extension:
- `.json` → `SidecarFormPane` (two views: `BIDS view` schema-aware form + `Tree view` 2-col QTreeWidget).
- `.tsv` / `.tsv.gz` → `TsvViewerPane` (editable table).
- `.nii` / `.nii.gz` → `NiftiViewerPane` — single-pane 2-D + Tri-view (sag/cor/ax sharing one crosshair, click-and-drag scrubbing) + 4-D time-series Graph (pyqtgraph grid with Scope/Dot size/Mark neighbors). Crosshair colour + thickness inline (QColorDialog + QSpinBox), persisted via `AppSettings`. Loads on `NiftiLoaderWorker` QThread; `BusySpinner` covers big BOLDs; stale loads discarded on file swap.
- other (MEG, EEG, …) → sidecar pane "no form for this file type" hint.

Theme handling: every palette colour is QSS-driven. Custom widgets with palette-baked styling expose `repaint_for_palette()` running Qt's unpolish/polish dance to force QSS recomputation on dark↔light swap. Pyqtgraph plots read the palette explicitly (no QSS support).

Tests: **~730 unit + 49 real-data** gated on `BIDS_MANAGER_REAL_{MRI,EEG,MEG}_DATA=1`. GUI tests under `QT_QPA_PLATFORM=offscreen` via `pytest-qt`. Known intermittent: pytest-qt teardown can segfault on full-suite run (stale paint after a model goes out of scope); individual test files run clean.

Persistent test output: `/Users/karelo/Development/datasets/BIDS_Manager/bids_manager_outputs/testing/`. The `.tmp_bidsmgr/` scratch tree must never persist there.

## Develop

```bash
cd /Users/karelo/PycharmProjects/superbidsmanager/BIDS-Manager
../.venv/bin/pip install -e ".[dev]"
../.venv/bin/python -c "import bidsmgr; print(bidsmgr.__version__)"
../.venv/bin/pytest
```

The shared venv at `/Users/karelo/PycharmProjects/superbidsmanager/.venv/` has every dep installed: PyQt6 6.11.0, bidsschematools 1.2.2, dcm2niix 1.0.20250506, mne 1.12.1, mne-bids 0.17.0, pyqtgraph 0.14.0, nibabel 5.3+, etc.

## Full CLI surface

```
bidsmgr-scan      <raw_root>     <inv.tsv>      [--dataset NAME] [--line-freq 50|60] [--montage NAME] [-j N] [--probe-convert]
bidsmgr-rebuild   <inv.tsv>                     [--from {entities,columns}] [--dry-run]
bidsmgr-convert   <inv.tsv>      <bids_parent>  [--dataset NAME] [-j N] [--overwrite] [--dry-run]
bidsmgr-metadata  <bids_parent>                 [--inventory-tsv …] [--fill-todos] [--name …]
bidsmgr-validate  <bids_parent>                 [--strict] [--strict-warn] [--html]
```

## Key behaviours to remember (non-obvious)

- Scanner **auto-detects modality** per file extension; multimodal trees produce one unified TSV (51 columns).
- Subject identity for EEG/MEG uses path heuristics (BIDS literal `sub-XXX`, parent-folder fallback). Flat layouts collapse to one subject; user edits `BIDS_name` to reconcile.
- Sessions auto-inferred from `mne.info['meas_date']` when path lacks `ses-XXX` tokens (parity with MRI's StudyDate clustering).
- `bidsmgr-convert` runs `rebuild_from_entities` **in memory** before reading rows, so stale TSVs still convert to the freshest BIDS names.
- Per-subject staging at `<bids_root>/.tmp_bidsmgr/sub-XXX/`; atomic `os.rename` on success; staging always wiped, failure logs go to `.bidsmgr/errors/`.
- The metadata engine and validator are modality-agnostic — they walk the BIDS tree, no modality-specific code paths.

## Next features (roadmap)

1. **`gui/` Project menus (M7)** — File → New / Open / Recent for `*.bidsmgr` bundles; Edit → Undo (`Project.undo_last` primitive already exists); right-click provenance tooltips.
2. **3-D viewer + GIFTI/FreeSurfer surface viewer** — port `Volume3DDialog` (~1500 LOC pyqtgraph/scikit-image) + `Surface3DDialog` from the v0.2.5 codebase (preserved in git history). Would add `scikit-image` to deps.
3. **`fixups/derivatives.py`** (~80 LOC) — DWI scanner-derivatives (FA / ADC / TRACE / ColFA / ExpADC) the converter currently skips with a warning.
4. **Cross-modality subject identity** — extend `inventory/subject_identity.py` so MRI Alice + EEG alice.edf merge to one sub-001 post-scan.
5. **MEG / EEG viewer stubs** in the Editor center pane — currently fall through to "no sidecar form" hint.
6. **Lower priority**: `.cnt` Neuroscan format hint; wire `--project` through CLI verbs; track down the pytest-qt teardown segfault.

## User preferences

- **No `Co-Authored-By: Claude` trailer** on commits.
- User prefers explicit / manual revalidation in the Editor (declined per-file revalidation on Save).
- No commits / pushes without explicit confirmation.
- TSV-driven workflow; spreadsheet editability; per-row overrides via TSV.
- Real-data tests gate on `BIDS_MANAGER_REAL_{MRI,EEG,MEG}_DATA=1`. Datasets at `/Users/karelo/Development/datasets/BIDS_Manager/raw_data/{MRI,EEG,MEG}/`.
- Persistent test output at `/Users/karelo/Development/datasets/BIDS_Manager/bids_manager_outputs/testing/`.
