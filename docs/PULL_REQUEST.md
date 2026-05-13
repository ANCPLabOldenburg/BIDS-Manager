# BIDS Manager v1.0.0: Complete re-imagination

> **PR scope.** This pull request replaces the entire `bids_manager`
> Python package with a new `bidsmgr` package, ships a two-tab PyQt6
> GUI (Converter + Editor), and reorganises the repository around a
> schema-driven engine. The PyPI distribution name (`bids-manager`)
> is preserved. The Python import name changes from `bids_manager` to
> `bidsmgr`. There is no in-place upgrade path for code that imports
> from `bids_manager`; the surface area was redesigned, not renamed.
>
> The change was developed over several months in the sibling
> repository `karellopez/bidsmgr` (granular commit-by-commit history
> preserved there). On this side of the cutover the work lands as
> `14cc042  v1.0.0: complete re-imagination` plus follow-ups.

---

## Table of contents

1. [Motivation](#1-motivation)
2. [Scope of change](#2-scope-of-change)
3. [Architectural shift](#3-architectural-shift)
4. [Repository layout](#4-repository-layout)
5. [Engine subpackages](#5-engine-subpackages)
6. [GUI architecture](#6-gui-architecture)
7. [Editor view in depth](#7-editor-view-in-depth)
8. [NIfTI viewer](#8-nifti-viewer)
9. [Worker thread model](#9-worker-thread-model)
10. [Event-sourced project bundles](#10-event-sourced-project-bundles)
11. [Conversion pipeline](#11-conversion-pipeline)
12. [Validation system](#12-validation-system)
13. [CLI surface](#13-cli-surface)
14. [Schema engine](#14-schema-engine)
15. [Theming](#15-theming)
16. [Cross-platform path safety](#16-cross-platform-path-safety)
17. [Packaging](#17-packaging)
18. [Tests](#18-tests)
19. [Breaking changes for 0.x users](#19-breaking-changes-for-0x-users)
20. [Migration guide](#20-migration-guide)
21. [Known limitations](#21-known-limitations)
22. [Roadmap](#22-roadmap)
23. [Reviewer's guide](#23-reviewers-guide)

---

## 1. Motivation

The 0.x line had three structural problems that compounded over time:

1. **Naming logic was scattered.** The same BIDS filename was assembled
   in four places: `dicom_inventory.py`, `build_heuristic_from_tsv.py`,
   `build_dcm2bids_config.py`, and `schema_renamer.py`, with `post_conv_renamer.py`
   patching whatever the engine got wrong. Drift between these paths
   caused the most user-visible bugs.
2. **Two converter engines.** HeuDiConv and dcm2bids both produced
   BIDS, but with different conventions for fmaps, run numbering, and
   intended-for resolution. Keeping both routes in parity meant the
   post-conversion renamer became a meta-engine that knew about both
   conventions. Adding a new feature required understanding all three.
3. **`gui.py` was a single 9.2k-line module** that mixed window
   composition, business logic, subprocess orchestration, threading,
   QSS theming, and ad-hoc validation. It worked, but adding a feature
   meant scrolling. There was no place to put a feature that touched
   "more than one stage" without making the GUI even bigger.

The 0.3 plan documented in `docs/improvement_plan.md` outlined eight
milestones (M1 to M8) addressing these issues incrementally. After
two attempts at incremental refactor (described in `docs/super_plan.md`
§12), it became clear that the architectural problems were not
local to any module. They sat in the dependency arrows. A schema-first
redesign is the simplest way to undo all three at once.

The single design bet of v1.0.0 is **schema-as-engine**:

> Every layer (classification, naming, GUI form generation, validation,
> sidecar generation, post-conversion auditing) reads from the same
> machine-readable BIDS schema. There is no second source of truth.

The schema source is [`bidsschematools`](https://github.com/bids-standard/bids-specification),
the canonical upstream. `ancpbids` is kept as a complementary graph-reader
for entity / suffix listings, but the rules engine is `bidsschematools`.

## 2. Scope of change

This is a complete codebase replacement, not a refactor. Files that
existed in 0.x and are gone in v1.0.0:

- `bids_manager/__init__.py`
- `bids_manager/gui.py` (the 9.2k-LOC GUI)
- `bids_manager/dicom_inventory.py`
- `bids_manager/build_heuristic_from_tsv.py`
- `bids_manager/build_dcm2bids_config.py`
- `bids_manager/eeg_meg_inventory.py`
- `bids_manager/run_dcm2bids.py`
- `bids_manager/run_heudiconv_from_heuristic.py`
- `bids_manager/run_mne_bids.py`
- `bids_manager/schema_renamer.py`
- `bids_manager/post_conv_renamer.py`
- `bids_manager/bids_metadata_engine.py`
- `bids_manager/scans_utils.py`
- `bids_manager/fill_bids_ignore.py`
- `bids_manager/miscellaneous/...` (except the negative-logo PNG)
- `bids_manager/user_preferences/...`
- All v0.2.5 console scripts (`dicom-inventory`, `build-heuristic`,
  `run-heudiconv`, `build-dcm2bids-config`, `run-dcm2bids`,
  `post-conv-renamer`, `eeg-meg-inventory`, `run-mne-bids`,
  `bids-metadata`, `fill-bids-ignore`)
- The old `tests/` tree (replaced with a new one)
- `MANIFEST.in`

Files preserved from 0.x:

- `external/python-embed/`: vendored Python embed used by the
  Windows installer. Untouched.
- `Installers/Installers.zip`: packaged Windows installer.
  Untouched. Will need rebuilding to wire it to the new console
  scripts.
- `LICENSE`: MIT, unchanged.
- `bids_manager/miscellaneous/images/Logo_negative_square.png` -
  promoted one level to `BIDS-Manager/miscellaneous/images/`
  because the parent package directory was deleted. Currently not
  used by the package; reserved for future use.

Files added in v1.0.0:

- `bidsmgr/`: the new Python package (16 subpackages).
- `tests/`: new test tree (`unit/`, `gui/`, `integration/`,
  `real_data/`, `fixtures/`).
- `docs/`: `architecture.md`, `super_plan.md`, `improvement_plan.md`,
  `gui_mockups.html`, the standalone `inspector_proto/` GUI prototype,
  plus this PR document and the README screenshot list.
- `pyproject.toml`: rewritten for the new package.
- `README.md`: rewritten as a user-facing product page.
- `CLAUDE.md`: auto-loaded project map for AI-assisted contributors.
- `.gitignore`: added.

## 3. Architectural shift

### 3.1 The keystone

`bidsmgr.schema` imports nothing from the rest of the package. Every
other subpackage may import from it. This makes the schema the
*keystone* of the dependency graph; if you remove it, everything else
fails to import. This is a deliberate invariant the type checker can
enforce (no circular import is structurally possible).

### 3.2 Dependency direction

```
gui  ----> workers ----> cli ----> {inventory, classifier, planner,
                                    converter, fixups, metadata,
                                    project, editor}  ----> schema
                                              utility:  ----> util
```

The arrows point inward. `gui` may import from anything below it,
but nothing else may import `gui`. Same for `workers` (everything
below may not import `workers`). The CLI verbs and the GUI panels
sit at the top of the graph; everything else can be exercised
headlessly without instantiating a `QApplication`.

### 3.3 Five prevention guards

These rules are documented in `CLAUDE.md` and enforced by code review
and by the test that imports every module in isolation:

1. **`schema/` is the keystone.** Everything imports from it; it
   imports nothing else from the package.
2. **`gui/` is the only Qt-coupled subtree.** `workers/` imports
   `PyQt6.QtCore.QThread` (the bridge mechanism). No other module
   imports PyQt6.
3. **No `Pipeline` orchestrator.** Orchestration is straight-line
   procedural code in `cli/<verb>.py` and `gui/<panel>.py`. The 0.x
   `Pipeline` god-object that wrapped HeuDiConv + dcm2bids + the
   metadata engine is intentionally absent.
4. **No subpackage named `core/`.** The name is poisoned by the
   v1.0-attempt post-mortem (`docs/improvement_plan.md` §12). It
   accumulated unrelated logic until everything imported it.
5. **Pure-data types only.** Pydantic v2 models and plain dataclasses
   carry data between subpackages. They have no I/O methods, no
   side effects on construction. Loaders / writers are free functions.

A sixth rule, *functions over classes where possible*, is a style
preference rather than a structural invariant.

## 4. Repository layout

```
BIDS-Manager/
├── pyproject.toml                  PEP 621, name = "bids-manager"
├── README.md                       User-facing product page
├── CLAUDE.md                       AI contributor project map
├── LICENSE                         MIT
├── .gitignore
├── docs/                           Design documents and PR notes
│   ├── PULL_REQUEST.md             (this file)
│   ├── architecture.md             Architectural rationale (§0 to §15)
│   ├── super_plan.md               Path-and-decisions document (§13 has the resolved decision matrix)
│   ├── improvement_plan.md         The v0.3 plan + a post-mortem on the v1.0 attempt
│   ├── gui_mockups.html            Five GUI design proposals (Inspector won)
│   ├── inspector_proto/            Standalone PyQt6 prototype the Editor was ported from
│   ├── workflow.svg                Workflow diagram used in README.md
│   └── screenshots/                README screenshot drop zone + shooting list
├── miscellaneous/
│   ├── hero.gif                    README hero
│   ├── see.gif                     Feature 1
│   ├── fix.gif                     Feature 3
│   ├── inspect.gif                 Feature 4
│   ├── validate.gif                Feature 5
│   └── images/Logo_negative_square.png  Preserved from 0.x
├── external/python-embed/          Vendored Python embed (Windows installer)
├── Installers/                     Windows installer ZIP
├── bidsmgr/                        The importable Python package
│   ├── __init__.py                 __version__ = "1.0.0"
│   ├── main.py                     GUI entry point (also the `bidsmgr` console script)
│   ├── schema/                     Keystone: bidsschematools wrapper
│   ├── inventory/                  Per-modality scanners + subject identity
│   ├── classifier/                 Chained classifiers
│   ├── planner/                    Entity plan + user edits
│   ├── converter/                  Pluggable backends
│   │   └── backends/               Dcm2niixDirect, MneBidsBackend, PhysioDcmBackend
│   ├── fixups/                     fieldmaps, IntendedFor, scans.tsv
│   ├── metadata/                   Post-conversion sidecar engine
│   ├── project/                    Event-sourced .bidsmgr/ bundles
│   ├── editor/                     Post-conv editor logic (no Qt)
│   ├── gui/                        Qt-coupled subtree
│   │   ├── theme.qss               Token-driven QSS palette
│   │   ├── theme_manager.py        Palette swap at runtime
│   │   ├── widgets/                Pane widgets used by both views
│   │   ├── delegates/              Table delegates
│   │   ├── models/                 QAbstractTableModel implementations
│   │   ├── assets/                 wordmark, logo, author photos, ANCP lab logo
│   │   ├── main_window.py          Top-level shell
│   │   ├── converter_panel.py      Converter view
│   │   ├── editor_panel.py         Editor view
│   │   └── ...                     dialogs, panes, settings
│   ├── workers/                    QThread bridges (Qt-allowed exception)
│   ├── cli/                        Five CLI verbs
│   └── util/                       Cross-OS path safety, Qt platform helpers
└── tests/
    ├── unit/                       ~25 unit test modules
    ├── gui/                        ~27 GUI test modules (offscreen)
    ├── integration/                Placeholder for cross-stage flows
    ├── real_data/                  ~7 real-data modules, env-gated
    └── fixtures/                   Shared pytest fixtures
```

## 5. Engine subpackages

Engine subpackages are anything below `cli/` in the dependency graph.
They are pure-Python, importable headlessly, free of Qt.

### 5.1 `bidsmgr.schema`

Keystone. Wraps `bidsschematools` (BIDS 1.11.1) behind an internal API
that the rest of the package speaks.

- `loader.py`: caches a `Schema` instance per process, validated on
  first use.
- `engine.py`: public surface: `entities_for(datatype, suffix)`,
  `required_fields(datatype, suffix)`, `recommended_fields(...)`,
  `allowed_suffixes(datatype)`, `field_descriptions(...)`,
  `value_kind(field_name)`, `enum_values(field_name)`.
- `validation.py`: schema-level rule application, used by the
  Editor's layer 1 validator.
- `types.py`: strongly-typed views of schema fragments (`SidecarField`,
  `EntitySpec`, `SuffixDef`).
- `bundled/`: frozen copy of the schema shipped with the wheel so
  installs work offline.

The schema graph is never mutated. Callers receive views or new lists,
not references into the cached schema object.

### 5.2 `bidsmgr.inventory`

Per-modality scanners that produce one unified 51-column TSV.

- `mri_dicom.scan_dicoms_long(...)` walks a DICOM tree with `pydicom`
  + `joblib.Parallel`. One row per `SeriesInstanceUID`. Output
  columns include `subject`, `BIDS_name`, `session`, `include`,
  `modality`, `modality_bids`, demographics, and the eight
  `bids_guess_*` columns stamped by the classifier.
- `eeg_meg.scan_eeg_meg(...)` walks a raw tree, probes files via
  `mne.io.read_raw(..., preload=False)`. Folder-shaped recordings
  (`.ds`, `.mff`) become a single candidate; BrainVision triplets
  collapse to `.vhdr`. Adds the same `bids_guess_*` stamping path
  so the inspection table can colour-code rows consistently across
  modalities.
- `subject_identity.cluster_subjects(...)` is the deduplication
  primitive. Keys on `(PatientID, PatientName)` for DICOM,
  falls back to path heuristics for EEG/MEG (`sub-XXX` literal in
  the path, then parent-folder name). Flat layouts collapse to one
  subject; users disambiguate via `BIDS_name`.
- `probe_convert.probe_rows(...)` performs the cheap dry-run that
  the Scan toolbar's `--probe-convert` option triggers; used to
  refine `bids_guess_*` before the user touches a row.
- `rebuild.rebuild_from_entities(...)` regenerates `BIDS_name` from
  the entity columns. Runs in memory before every CLI convert pass
  so a stale TSV still produces fresh BIDS names.
- `rebuild.rebuild_from_columns(...)` does the inverse: derives
  entity columns from a `BIDS_name`. Used by the GUI when the user
  pastes a basename into the BIDS_name cell.
- `_time.py` carries the longitudinal-session inference: when a path
  has no `ses-XXX` token, `(StudyInstanceUID, StudyDate)` clusters
  produce session labels. Mirrors the v0.2.5 heuristic.

### 5.3 `bidsmgr.classifier`

Chained classifiers run in order; each one stamps `bids_guess_*`
columns it is confident about. Later classifiers do not overwrite
earlier ones.

- `dcm2niix_bidsguess.classify_dicom_row(...)` invokes `dcm2niix` with
  `-ba n` (so `SeriesInstanceUID` survives in the produced JSON) and
  reads the `BidsGuess` field. The classifier strips the DICOM
  SeriesNumber that `dcm2niix` injects as the `run` entity (we
  recompute `run` later from grouping). It also handles the
  fmap multi-output case, the B0 reference reroute (B0 belongs in
  `fmap/_epi`, not `dwi/`), and DWI scanner derivatives (FA, ADC,
  TRACE, ColFA, ExpADC) which BIDS 1.11 treats as raw suffixes.
- `sequence_dict.classify_by_sequence_dict(...)` is the legacy
  fallback used when `dcm2niix_bidsguess` is unable to classify a
  row (rare, mostly non-MR series). It is a curated dictionary of
  sequence descriptions to BIDS suffixes. Kept for compatibility
  with the v0.2.5 sequence dictionary so existing users don't
  lose accuracy on unusual scanners.

The classifier never decides the final BIDS name; it only fills the
classification columns. Naming happens later via the schema engine.

### 5.4 `bidsmgr.planner`

The smallest subpackage. Holds the `EntityPlan` Pydantic model that
the GUI's `PropertiesPanel` binds to. An entity plan is just a typed
view of one row's entities; mutating it through the panel triggers a
schema validation pass and a live BIDS path re-preview.

### 5.5 `bidsmgr.converter`

Backend dispatch. Each row in the inventory carries a `bids_modality`;
the converter picks the right backend for it.

- `registry.py` is the dispatch table: `bids_modality → backend`.
- `types.py` defines `BackendTask` (per-row work item) and
  `BackendResult` (success / failure with provenance).
- `backends/dcm2niix_direct.py` is the default MRI backend. Stages
  the row's DICOMs into `<bids_root>/.tmp_bidsmgr/sub-XXX/`, calls
  `dcm2niix` directly (no HeuDiConv or dcm2bids wrapper), then
  performs an atomic `os.rename` to commit the staged tree into the
  final BIDS path. Symlink staging falls back to `shutil.copyfile`
  on Windows when the user lacks the symlink privilege (`WinError
  1314`).
- `backends/mne_bids.py` is the EEG/MEG backend. Builds a `BIDSPath`
  per row and calls `mne_bids.write_raw_bids(..., format="auto")`.
  The MEG datatype path explicitly skips standard montage
  application (MEG sensor coords come from the recording, not from
  the montage file). The EEG rename pass is collision-safe.
- `backends/physio_dcm.py` handles Siemens PhysioLog DICOMs via
  `bidsphysio`.

Failure surface: every backend's `run(task)` returns a `BackendResult`.
Exceptions are caught at the boundary so one bad row does not abort
the whole convert pass.

### 5.6 `bidsmgr.fixups`

Post-conversion cleanup. Each fixup is idempotent and modality-aware.

- `fieldmaps.py` renames dcm2niix's `_echo-1` / `_echo-2` / plain
  outputs into `_magnitude1` / `_magnitude2` / `_phasediff` and
  rewrites the sidecar accordingly.
- `intended_for.py` resolves `IntendedFor` from BIDS metadata
  (`StudyInstanceUID` and series timestamps) and writes the BIDS
  paths into every fmap sidecar.
- `scans_tsv.py` writes / updates each subject's `*_scans.tsv` to
  keep its `filename` column consistent after any rename or fmap fix.

### 5.7 `bidsmgr.metadata`

The post-conversion dataset-level metadata engine. Modality-agnostic:
it walks the produced BIDS tree, no per-modality code paths.

- `engine.run_metadata(bids_parent, inventory_tsv=None, fill_todos=True)`
  emits `dataset_description.json`, `participants.tsv` +
  `participants.json`, `README`, `CHANGES`, and per-subject
  `*_scans.tsv` files. Merges with what `mne_bids` already wrote
  rather than overwriting it (EEG/MEG datasets have richer
  participant rows than DICOM provides).
- Per-datatype sidecar audit reads required + recommended fields
  from the schema engine. Missing fields become `BIDS TODO`
  comments in the sidecars unless `fill_todos=False`.

### 5.8 `bidsmgr.project`

Event-sourced project bundles. A `*.bidsmgr/` directory contains:

- `events.jsonl` – append-only event log.
- `provenance.json` – derived snapshot of who-did-what-when.
- `inventory.tsv` – the unified inventory TSV the user is editing.
- `state/` – cached derived state (current entity plans, last
  validation report, etc.).

Events (defined in `types.py`):

- `ProjectCreated` – emitted by `bidsmgr-scan` or by the GUI's
  Scan action.
- `ScanImported` – emitted once per scan job.
- `UserSetCell(row, column, old, new)` – every inspector edit.
- `UserSetEntity(row, entity, old, new)` – every Properties form edit.
- `UserToggleInclude(row, value)` – inspector include checkbox.
- `TodoAcknowledged(file, field)` – metadata-engine TODO marked
  resolved.
- `StageCompleted(stage, status, duration_ms)` – every CLI verb
  emits one of these on exit.

`project.Project.append_event(...)` is the only mutator. `replay.py`
reconstructs the current state from the event log; `provenance.py`
derives a human-readable summary for the About dialog and for
`dataset_description.json.GeneratedBy`.

Undo is a primitive: `Project.undo_last()` pops the most recent
user event and re-replays from the start. The GUI surface for it
(Edit > Undo) is in the M7 roadmap.

### 5.9 `bidsmgr.editor`

Post-conversion editor logic. No Qt.

- `validator.validate(bids_root, strict=False)` is the two-layer
  validator. Layer 1 is always on: schema-driven required /
  recommended field checks per (datatype, suffix), basename and
  entity coherence checks, dataset-level layout checks. Layer 2 is
  the official `bidsschematools.validator.validate_bids(...)`,
  enabled by `strict=True` (the GUI's "Strict BIDS" toggle).
- `validator.validate_file(bids_root, path)` and
  `validator.validate_folder(bids_root, folder)` are layer-1-only
  partial helpers used by the Editor's "Validate file" / "Validate
  folder" toolbar buttons. They re-validate only the requested
  scope and merge into the existing report.
- `html_report.render_html(report)` produces a standalone HTML
  report (two sections: "Files with issues" and "All files", the
  latter collapsible with per-file schema audit tables and
  hover-tooltip field descriptions).
- `types.py` defines the report data model: `ValidationReport`,
  `FileVerdict`, `Issue`, `Severity` (OK / WARN / ERR), and the
  schema-audit objects.

### 5.10 `bidsmgr.util`

Cross-platform helpers.

- `paths.safe_path_component(raw)` rejects characters illegal on
  any supported platform (`<>:"/\|?*`, control chars, trailing dot
  or space, Windows reserved device names). Caps strings at 96
  chars with a SHA-1 disambiguator suffix.
- `paths.long_path(p)` wraps a path with the `\\?\` prefix on
  Windows when it approaches `MAX_PATH`. Pass-through everywhere
  else.
- `qt_platform.choose_qpa_platform()` picks the right QPA platform
  for the current environment (`offscreen` for headless CI, native
  otherwise).

## 6. GUI architecture

### 6.1 `gui.main_window`

`MainWindow` is the top-level shell. It composes:

- `_TopHeader` – left side carries a clickable brand logo +
  wordmark (clicks open the `AboutDialog`); centre carries the
  `Converter | Editor` pill switcher; right side carries the
  theme toggle.
- A `QStackedWidget` holding `ConverterPanel` at index 0 and
  `EditorPanel` at index 1.
- A `QStatusBar` that mirrors `log_message` signals from both panels.

The active view, the editor's BIDS root, the editor's sidecar view
mode, the editor's strict-validation flag, the NIfTI viewer
crosshair colour and thickness, and the last-used Converter
settings all persist via `AppSettings` (a typed wrapper over
`QSettings`). The `AppSettings.KEYS` mapping is the single source
of truth for setting keys; the rest of the GUI never names a key
directly.

### 6.2 `gui.converter_panel`

`ConverterPanel` is the Converter view. Top-level layout:

- **Toolbar** with `Scan...`, the dataset-name input, the TSV
  filename input, severity chips (clickable, open `IssuesDialog`
  cards), a `BusySpinner`, a `Settings...` button, and the
  `Run conversion` button.
- **Column 1** – a vertical splitter holding `RawFsPane` over
  `OutputFsPane`. The output pane scans disk via `QThreadPool`
  (a `_ScanRunnable` with a generation counter that drops stale
  results), debounces refreshes via `QFileSystemWatcher`, and
  re-colours items in place on theme swap (palette token stamped
  at `Qt.UserRole + 1`).
- **Column 2** – `FilterPane`, a tri-state tree
  `dataset > subject > session > datatype > sequence`. Each
  recording is its own checkable leaf, labelled with its
  `proposed_basename`.
- **Column 3** – the inspection `QTableView` over
  `InventoryTableModel`. Twelve default columns + fifteen
  toggleable via header right-click. Three delegates handle
  row-tint and purple-highlight overlays. Below the table sits an
  `inspection-footer` strip with `Highlight aborts` and
  `Bulk edit...` actions.
- **Column 4** – `PropertiesPanel`, a schema-driven entity form
  for the selected row. Field labels and value editors come from
  `schema.engine.entities_for(...)`. Predicted BIDS path renders
  live.
- **Bottom dock** – a horizontal splitter with Log + Conflicts
  on the left and BIDS preview + Statistics on the right.

The Settings dialog has three tabs (Display, Scan, Convert + post-
convert chain) and every value persists via `QSettings` (macOS
plist, Linux INI, Windows registry).

### 6.3 `gui.editor_panel`

`EditorPanel` is the Editor view. Top-level layout:

- **Toolbar** with `Open BIDS root...`, `Validate file`,
  `Validate folder`, `Validate dataset`, `Strict BIDS` toggle
  (persisted via `AppSettings.editor_strict_validate`), severity
  chips, and a `BusySpinner`.
- **`PathBar`** showing the current BIDS root.
- **Column 1** – `BidsTreePane` with BIDS-aware coloring
  (subject, session, datatype labels in the accent colour; `.nii`
  in body text, `.json` in purple, `.tsv` in teal; folder-recordings
  like `.ds` and `.mff` collapse to a leaf). `BidsTreeDelegate`
  paints severity badges from `FileVerdict.severity` after
  validation; `set_badges()` rolls each folder up to the severity
  of its worst descendant.
- **Column 2** – a `QStackedWidget` routing by file extension:
  - `.json` -> `SidecarFormPane` (see [§7](#7-editor-view-in-depth)).
  - `.tsv` / `.tsv.gz` -> `TsvViewerPane`.
  - `.nii` / `.nii.gz` -> `NiftiViewerPane` (see [§8](#8-nifti-viewer)).
  - everything else -> sidecar pane shows a "no form for this
    file type" hint pointing the user at the JSON peer.
- **Column 3** – `ValidationPane` with four sections
  (Dataset, Folder, File, Schema audit). Each `Issue` becomes a
  `ValMessage` row with a clickable Fix button.

## 7. Editor view in depth

### 7.1 Schema-aware sidecar form

`SidecarFormPane` is the JSON-sidecar editor. Two switchable views:

- **BIDS view** – one `SidecarRow` per schema field. Each row
  carries a 4 px level bar (red REQUIRED, amber RECOMMENDED, muted
  OPTIONAL, strikethrough DEPRECATED) and an inline editor whose
  type comes from the field's `value_kind`:
  - `string` -> `QLineEdit`.
  - `boolean` -> `QComboBox` with `true` / `false` options.
  - `number` / `integer` -> `QLineEdit` with `QDoubleValidator` or
    `QIntValidator`.
  - `enum` -> `QComboBox` with the schema-supplied values.
  - `array` / `object` -> `QLineEdit` (literal JSON text; no
    implicit promotion to dict / list).
  Rows are sorted required > recommended > optional > deprecated.
- **Tree view** – `JsonTreeView`, a two-column `QTreeWidget`
  (key, value). Recursive: dicts expand as parent rows with their
  contents as children; lists render with `[N]` keys.
  `JsonTreeView.drawRow` paints the level bar via a custom
  delegate so colour-coding survives the tree shape. Toolbar has
  `+ Add field` (always at root), `+ Add subfield` (inside the
  selected container; allowed to promote a leaf, matching the
  original `_add_field` semantics), and `- Delete field` (any
  depth).

Both views share an in-memory `_json_cache`. Edits update the cache
only; the per-pane edit toolbar carries `Save`, `Revert`, and a
dirty chip. Save preserves the on-disk key order, appending new
keys at the end. `file_saved(Path)` fires post-write so listeners
can react (the validation pane re-validates the file).

### 7.2 TSV editor

`TsvViewerPane` loads `.tsv` / `.tsv.gz` via stdlib `csv` + `gzip`
into a `QTableView` over `QStandardItemModel`. Cells are
inline-editable (double-click, F2, or select-then-click). Toolbar:
`+ Add row`, `+ Add column` (prompts for the column name via
`QInputDialog`), `- Delete row`, `- Delete column`, plus
`Revert` / `Save` and a dirty chip. Files larger than the 5000-row
preview cap load read-only; Save refuses to overwrite them
(otherwise the truncated preview would destroy the unread tail).

### 7.3 Validation pane

`ValidationPane` binds against a single in-memory
`ValidationReport`. Four sections:

- **Dataset** – dataset-level issues.
- **Folder** – issues for the currently focused folder (driven by
  the BIDS tree selection).
- **File** – issues for the currently focused file.
- **Schema audit** – only for JSON files; shows counts per level
  plus the names of any missing required / recommended fields.

Each `ValMessage` row carries an accent chip with `Issue.field` and
a Fix button. `fix_requested(path, field)` fires when clicked;
`EditorPanel._on_fix_requested` selects `path` in the tree (cascades
through the panes) and calls `SidecarFormPane.focus_field(field)`
to scroll, focus, and select-all the value cell.

### 7.4 Status chips and issues dialog

The toolbar's `ok` / `warn` / `err` chips show counts and are
clickable. Clicking one opens `EditorIssuesDialog`, a modal list of
`FileCard` rows filtered by the chosen severity. Activating a card
selects the file in the tree.

## 8. NIfTI viewer

`NiftiViewerPane` is the most feature-dense widget in the package.
It routes `.nii` and `.nii.gz` files from the editor's center stack.

### 8.1 Rendering pipeline

The pane keeps a 3-D or 4-D array (`_data`) and a crosshair voxel
(`_cross_voxel`). `_refresh()` dispatches to either the
single-pane renderer (`_render_single_axis()`) or the tri-view
renderer (`_render_axis_into_tri(axis)` for each of the three axes).
Both call `_render_axis_to_label(axis, label)`:

1. Pick the 3-D slice for `axis` at `_cross_voxel[axis]`.
2. Normalise to `[0, 1]`, then apply brightness `b` and contrast `c`
   from the toolbar sliders: `arr = (arr - 0.5) * c + 0.5 + b`.
3. Scale to `uint8`, `np.rot90`, wrap in a `QImage` (grayscale or
   RGB / RGBA depending on `arr.ndim`).
4. Scale to the label's current size with Lanczos.
5. Paint the crosshair: two-pass draw with a semi-opaque dark halo
   underneath (only when thickness >= 2) plus the bright user-chosen
   colour on top.

### 8.2 Tri-view

The Tri-view toggle swaps the canvas to a horizontal `QHBoxLayout`
of three `ImageLabel` widgets, one per axis, sharing one
`_cross_voxel`. Clicking any panel updates the crosshair: the
clicked-axis index stays fixed (you can't change depth by clicking
inside a 2-D slice), the other two coordinates update from the
click position. After every update, all three panels re-render so
the crosshair appears in the same physical location in each.

### 8.3 Click-and-drag

`ImageLabel` overrides `mouseMoveEvent` and fires its callback
whenever the left button is held. Right and middle drags don't
trigger crosshair movement (button mask check). This makes
continuous scrubbing feel natural.

### 8.4 4-D Graph

When the data is 4-D non-RGB, the Graph toggle reveals a
`pyqtgraph.GraphicsLayoutWidget` grid. The toolbar above the grid
carries:

- `Scope` (`QSpinBox` 1..4) – neighbourhood size. `dim = 2 *
  (scope - 1) + 1`, so scope=1 is 1x1, scope=2 is 3x3, scope=3 is
  5x5, scope=4 is 7x7. Neighbours are offset in the plane
  perpendicular to the current orientation.
- `Dot size` (`QSpinBox` 1..20) – marker diameter on each cell;
  applies live without rebuilding the grid.
- `Mark neighbors` (`QCheckBox`) – when off, only the centre cell
  carries the volume-index marker.

Every plot's `ViewBox` is locked: `disableAutoRange()`,
`setMouseEnabled(x=False, y=False)`. The whole `GraphicsLayoutWidget`
swallows wheel events to stop the user from accidentally shrinking
the plot via scroll. Y range is shared across the grid (global min
/ max of all neighbour time-series) so cells are visually
comparable. X axis is pinned to `[0, n_vols - 1]`.

### 8.5 Crosshair styling

Crosshair colour and thickness persist via two new `AppSettings`
keys: `nifti_crosshair_color` (hex string, default `#4FC3F7`,
Material light-blue 300) and `nifti_crosshair_thickness` (int
1..5, default 1). The inline toolbar widgets are a coloured swatch
button (opens `QColorDialog`) and a `QSpinBox`. Both write back to
`AppSettings` and trigger an immediate re-render.

### 8.6 Threaded loading

Big BOLDs can take seconds to read off disk. `set_file(path, root)`
swaps the pane to a loading page with a `BusySpinner` and a
"Loading <name>..." label, then starts a `NiftiLoaderWorker`
(`QThread`). On success, `_on_load_complete` populates the canvas
and emits `loaded(path)`. If the user clicks another file before
the worker finishes, the in-flight worker has its result
suppressed via a cooperative `cancel()` flag (`nibabel.get_fdata`
is C code and can't be interrupted mid-flight, but the stale data
is dropped before it reaches the GUI).

## 9. Worker thread model

`bidsmgr.workers` is the only subpackage outside `gui/` that
imports Qt (specifically `PyQt6.QtCore.QThread` and `pyqtSignal`).
Each worker is a thin bridge wrapping a CLI verb or an editor
function, exposing `progress(str)`, `failed(str)`, and a typed
result signal.

- `ScanWorker(raw_root, tsv_out, options)` wraps `cli.scan.run_scan`.
- `ConvertWorker(inventory_tsv, bids_parent, options)` wraps
  `cli.convert.run_convert`.
- `MetadataWorker(bids_parent, options)` wraps `cli.metadata.run_metadata`.
- `ValidateWorker(bids_parent, options)` wraps `cli.validate.run_validate`.
- `ReportWorker(bids_root, strict)` runs an in-memory validation
  pass and emits `finished_with_report(ValidationReport, bids_root)`.
- `FileReportWorker(bids_root, file_path)` and
  `FolderReportWorker(bids_root, folder_path)` are the partial
  re-validation workers behind the Editor's `Validate file` /
  `Validate folder` buttons.
- `NiftiLoaderWorker(path)` performs the off-thread `nibabel.load(...)`
  + `get_fdata()` call for the NIfTI viewer.

Workers never touch widgets directly. They emit; the GUI's main
thread receives via Qt's queued signal-slot mechanism and updates
its models on the main thread.

## 10. Event-sourced project bundles

A project bundle is a directory tree:

```
mystudy.bidsmgr/
├── events.jsonl                  Append-only event log
├── provenance.json               Derived who-what-when summary
├── inventory.tsv                 Current unified TSV
├── state/
│   ├── entity_plans.json         Per-row entity plans
│   └── validation_report.json    Last validation report
└── errors/                       Per-row conversion failure logs
```

`Project.open(path)` reads `events.jsonl` and replays through
`replay.apply(...)` to reconstruct state. `Project.append_event(event)`
appends to `events.jsonl` and updates the derived files. The
append is atomic (write to `events.jsonl.tmp`, `os.replace`).

The provenance map (`provenance.py`) is written to
`dataset_description.json.GeneratedBy` after every convert pass and
also kept in `provenance.json` for the GUI's right-click "where
did this come from?" feature (M7, future).

The format is deliberately JSON-only for v1. SQLite is an option if
event volume becomes a problem; the architecture document
(`docs/architecture.md` §10) describes the migration path.

## 11. Conversion pipeline

End-to-end:

1. **Scan.** `bidsmgr-scan <raw> <inv.tsv>` walks the tree, picks
   the right inventory module per file, runs the classifier chain,
   writes the 51-column unified TSV. Emits a `ProjectCreated` event
   if `--project <path>` is supplied (default off in CLI; default
   on in the GUI's Scan action).
2. **Optional probe-convert.** With `--probe-convert`, each row
   gets a cheap dcm2niix dry-run to refine the `bids_guess_*`
   columns. Slow on large studies, off by default.
3. **Rebuild.** `bidsmgr-rebuild <inv.tsv> --from entities`
   regenerates `BIDS_name` from the entity columns after a manual
   spreadsheet edit. `--dry-run` shows the diff without writing.
4. **Convert.** `bidsmgr-convert <inv.tsv> <bids_parent>`:
   - Runs `rebuild_from_entities` in memory so stale TSVs still
     produce fresh names.
   - Groups included rows by `(subject, session)` to compute final
     `run` numbers.
   - For each row, dispatches to the right backend via
     `converter.registry`.
   - Stages output to `<bids_root>/.tmp_bidsmgr/sub-XXX/` then
     atomically `os.rename`s into the final BIDS path on success.
     On failure, leaves staging in place for forensics + writes a
     row-specific log to `<bids_root>/.bidsmgr/errors/`.
5. **Fixups.** After every backend run, the relevant fixups fire:
   `fieldmaps` renames echo outputs; `intended_for` resolves the
   IntendedFor map; `scans_tsv` keeps the per-subject scans TSV
   in sync.
6. **Metadata.** `bidsmgr-metadata <bids_parent>` writes
   dataset-level files (`dataset_description.json`,
   `participants.tsv`, `README`, `CHANGES`) and runs the
   per-datatype sidecar audit. Merges with what `mne_bids` wrote.
7. **Validate.** `bidsmgr-validate <bids_parent>` runs the
   two-layer validator. `--strict` enables layer 2 (official BIDS
   validator). `--html` writes a standalone report.

The GUI's `ConverterPanel.run_conversion()` runs steps 4-7
sequentially via separate workers, with the toolbar's busy spinner
covering each. The post-convert chain (metadata + validate) is
configurable per project via the Settings dialog.

## 12. Validation system

Two layers, each optional, each clickable.

**Layer 1** is bidsmgr's own validator (`editor.validator.validate`).
Per-(datatype, suffix) sidecar audit reads required and recommended
fields from the schema engine; entity coherence checks compare the
on-disk basename against the entity columns; dataset-level layout
checks confirm the canonical files exist. Layer 1 always runs.

**Layer 2** is `bidsschematools.validator.validate_bids`. Enabled by
the `--strict` CLI flag and by the Editor's `Strict BIDS` toggle.
It checks every on-disk path against the BIDS schema regexes and
surfaces files that don't match a recognised pattern.

Both layers emit `Issue` objects (`Severity`, `field`, `message`,
`fix_hint`) into a single `ValidationReport`. The Editor's GUI is
agnostic to which layer emitted any given issue. The HTML report
groups by file but does not distinguish layers either; the
distinction matters at issue-generation time, not at display time.

## 13. CLI surface

Five verbs. Each maps to a `cli.<verb>:main` callable that the
`[project.scripts]` entries expose as console scripts.

```
bidsmgr-scan      <raw_root>     <inv.tsv>      [--dataset NAME] [--line-freq 50|60] [--montage NAME] [-j N] [--probe-convert]
bidsmgr-rebuild   <inv.tsv>                     [--from {entities,columns}] [--dry-run]
bidsmgr-convert   <inv.tsv>      <bids_parent>  [--dataset NAME] [-j N] [--overwrite] [--dry-run] [--raw-root PATH]
bidsmgr-metadata  <bids_parent>                 [--inventory-tsv PATH] [--fill-todos] [--name NAME]
bidsmgr-validate  <bids_parent>                 [--strict] [--strict-warn] [--html]
```

Each verb is independent. `cli/<verb>.py` exposes both `run_<verb>(...)`
(the importable function the GUI workers call) and `main()`
(the console-script entry that parses argv and dispatches to
`run_<verb>`).

The `bidsmgr` console script is the GUI entry; it calls
`gui.main_window.MainWindow().show()` inside a `QApplication` and
parses two optional flags:

- `--theme {dark,light}` overrides the persisted theme for this
  launch only.
- `--project PATH.bidsmgr` opens the named bundle.

## 14. Schema engine

`bidsschematools` is the canonical schema source. We pin against
**BIDS 1.11.1** as bundled with `bidsschematools` 1.2.2 (the version
in `pyproject.toml`). The schema is loaded once per process via
`schema.loader.load_schema()` and exposed as a frozen object.

`schema.engine` provides the GUI-friendly API:

```python
entities_for(datatype, suffix) -> list[EntitySpec]
required_fields(datatype, suffix) -> list[SidecarField]
recommended_fields(datatype, suffix) -> list[SidecarField]
allowed_suffixes(datatype) -> list[SuffixDef]
field_descriptions(*field_names) -> dict[str, str]
value_kind(field_name) -> str   # "string" | "number" | "integer" | "boolean" | "enum" | "array" | "object"
enum_values(field_name) -> list[str] | None
```

These are the only schema-touching calls the rest of the package
makes. Everything from the inventory `BIDS_name` builder to the
Editor's sidecar form to the metadata engine's TODO comments goes
through this API.

A schema upgrade is a project-local decision. Each `*.bidsmgr/`
bundle records the schema version it was created with. When opening
an older project against a newer bundled schema, the GUI prompts
the user before upgrading.

## 15. Theming

`gui/theme.qss` is a single QSS file with `$token` placeholders
(`$accent`, `$surface`, `$text`, `$border`, etc.). `theme_manager`
substitutes those tokens at load time from a per-theme palette
(`themes/dark.py`, `themes/light.py`) and applies the result via
`QApplication.setStyleSheet`.

Most widgets are pure QSS. A few custom widgets (the inspection
delegates, the NIfTI crosshair, the pyqtgraph plots) read palette
colours at paint time, so they automatically follow theme swaps.
Anything that caches palette values on construction exposes a
`repaint_for_palette(pal)` method; the swap-cascade in
`MainWindow._on_palette_changed` calls it on every registered
widget. The unpolish / polish dance inside that method forces
QSS recomputation for any rules that depend on dynamic property
state.

## 16. Cross-platform path safety

The cutover added explicit cross-OS path hardening:

- `safe_path_component(raw)` is applied to every path component
  that comes from user data (`subject`, `session`, `task`, etc.).
  Rejects characters and trailing characters illegal on any
  supported platform, caps length at 96 chars with a SHA-1
  disambiguator suffix.
- `long_path(p)` wraps with the `\\?\` Windows prefix only when
  the path nears `MAX_PATH`; pass-through everywhere else.
- The dcm2niix staging path uses both helpers, and the symlink-stage
  in `_stage_dicoms` falls back to `shutil.copyfile` on `WinError
  1314` (developer mode disabled, no symlink privilege).
- A `windows: cross-platform path hardening + version banner`
  commit (`89913e0`) landed in the predecessor repo before the
  v1.0.0 squash.

## 17. Packaging

`pyproject.toml` (PEP 621 only, no `setup.py`):

- `name = "bids-manager"` (unchanged from 0.x).
- `version = "1.0.0"`: - `requires-python = ">=3.10"` (0.x was `>=3.8`).
- `packages` lists every subpackage explicitly (no `find:`).
- `package-data` ships `theme.qss`, the assets directory, and the
  bundled schema files.
- `[project.scripts]` exposes one GUI entry (`bidsmgr`) plus five
  CLI verbs.

Dependency changes:

| Status | Package | Notes |
|---|---|---|
| Added | `bidsschematools >= 1.0.0` | Schema engine, the keystone. |
| Added | `pydantic >= 2.6` | Strongly-typed core abstractions. |
| Added | `pyqtgraph >= 0.13` | NIfTI viewer's 4-D graph. |
| Added | `PyOpenGL >= 3.1` | Reserved for the 3-D viewer (M8). |
| Removed | `heudiconv-ancp` | Replaced by `dcm2niix` invoked directly. |
| Removed | `dcm2bids` | Same reason. |
| Removed | `PyQt6-WebEngine` | The 0.x docs viewer is gone. |
| Removed | `psutil` | Subprocess tree termination is now handled by Python's `subprocess` directly. |
| Removed | `matplotlib` | Replaced by `pyqtgraph` for the graph view. |
| Removed | `scikit-image` | Will return when the 3-D viewer lands (M8). |
| Kept | `PyQt6 == 6.11.0` | |
| Kept | `pydicom == 3.0.1` | |
| Kept | `dcm2niix == 1.0.20250506` | |
| Kept | `nibabel >= 5.3` | |
| Kept | `mne >= 1.6`, `mne-bids >= 0.15`, `edfio` | |
| Kept | `bidsphysio == 21.6.24` | |
| Kept | `setuptools < 81` | `bidsphysio` still uses `pkg_resources.declare_namespace`. |
| Kept | `pandas >= 2.0`, `numpy >= 1.26`, `joblib >= 1.4`, `ancpbids >= 0.3.1` | |

## 18. Tests

Test counts at the time of this PR:

- **~454 unit tests** under `tests/unit/`.
- **~394 GUI tests** under `tests/gui/`, run with
  `QT_QPA_PLATFORM=offscreen` via `pytest-qt`.
- **~49 real-data tests** under `tests/real_data/`, gated on
  `BIDS_MANAGER_REAL_{MRI,EEG,MEG}_DATA=1` against datasets at
  `/Users/karelo/Development/datasets/BIDS_Manager/raw_data/`
  (17 datasets total: 10 MRI, 6 EEG, 1 MEG).
- **0 integration tests** for now; the `tests/integration/`
  directory is a placeholder for cross-stage flows that don't fit
  cleanly under unit or GUI.

Notable test modules:

| Module | What it covers |
|---|---|
| `tests/unit/test_schema_engine.py` | `schema.engine` API contracts. |
| `tests/unit/test_bidsguess_parser.py` | dcm2niix `BidsGuess` parsing. |
| `tests/unit/test_dwi_fmap_classification.py` | DWI derivatives + fmap quirks. |
| `tests/unit/test_abort_detection.py` | Same-name + same-image-type + redo-window heuristic. |
| `tests/unit/test_subject_identity.py` | `(PatientID, PatientName)` keying + EEG/MEG path heuristics. |
| `tests/unit/test_dcm2niix_direct.py` | `Dcm2niixDirect` backend (staging + commit). |
| `tests/unit/test_fieldmap_fixup.py` | echo-1 -> magnitude1, phasediff naming. |
| `tests/unit/test_rebuild.py` | `rebuild_from_entities` + `rebuild_from_columns`. |
| `tests/unit/test_validator.py` | Layer 1 schema-driven checks (466 lines, the heaviest module). |
| `tests/unit/test_project_replay.py` | Event log replay. |
| `tests/unit/test_project_provenance.py` | `GeneratedBy` + `provenance.json` derivation. |
| `tests/unit/test_util_paths.py` | Cross-OS path safety. |
| `tests/gui/test_nifti_viewer.py` | 30 tests covering single, tri-view, graph, threaded load, drag, crosshair persistence. |
| `tests/gui/test_editor_*` | BIDS tree, validation, partial validation, interactivity. |
| `tests/gui/test_sidecar_*` | Schema-aware form, manual save, tree view. |
| `tests/gui/test_theme_refresh.py` | Dark <-> light swap cascade. |
| `tests/real_data/test_all_mri_datasets.py` | End-to-end scan + convert + metadata + validate on each MRI dataset. |

### Known intermittent

Pytest-qt teardown can segfault on a full-suite run when a stale
paint happens after a model goes out of scope. Individual test
files run clean. The `tests/gui/test_inspection_footer_and_about.py`
module was rewritten to use `_TopHeader` directly instead of
`MainWindow` to avoid the issue; further mitigation is on the
roadmap.

## 19. Breaking changes for 0.x users

1. **Import name change.** `from bids_manager.X import Y` is gone.
   The mapping is not 1:1; the new surface is exposed under
   `bidsmgr.X`. See [§20](#20-migration-guide).
2. **Console scripts replaced.** The ten 0.x scripts are gone; the
   five `bidsmgr-*` verbs cover the same workflow. The GUI is now
   launched via `bidsmgr` (was `bids-manager`).
3. **Pipeline engines replaced.** HeuDiConv and dcm2bids are no
   longer involved. The default MRI backend is `dcm2niix` invoked
   directly. If existing scripts pipe through HeuDiConv or dcm2bids
   for non-BIDS reasons (custom heuristics, custom dcm2bids
   configs), they need to be re-implemented as a Python backend
   plugged into `converter.registry`, or run alongside as a
   separate tool.
4. **Python 3.8 / 3.9 no longer supported.** `requires-python =
   ">=3.10"`.
5. **`PyQt6-WebEngine` no longer required.** The 0.x docs viewer
   inside the app is gone; users follow the in-README docs link.
6. **`psutil` no longer required.** Subprocess tree management is
   now stdlib-only.
7. **Settings keys are new.** The 0.x preferences in
   `bids_manager/user_preferences/*.txt` do not migrate. The new
   keys live under `QSettings` (macOS plist, Linux INI, Windows
   registry).
8. **Project file format.** There was no project file in 0.x. The
   new `*.bidsmgr/` bundles are optional in v1.0.0 (the GUI writes
   one automatically but the CLI does not, pending the `--project`
   wiring documented in the roadmap).

## 20. Migration guide

### For users running the GUI

1. Uninstall the old version: `pip uninstall bids-manager`.
2. Install v1.0.0: `pip install --upgrade bids-manager`.
3. Launch via the new GUI entry: `bidsmgr` (was `bids-manager`).
4. Re-open your raw tree via the Scan action. The conversion
   results from a 0.x run are still valid BIDS and can be opened
   directly via the Editor tab without re-converting.

### For users running scripts

The 0.x script names are gone. Map each to the v1.0.0 verb:

| 0.x script | v1.0.0 equivalent |
|---|---|
| `dicom-inventory <dir> <tsv>` | `bidsmgr-scan <dir> <tsv>` |
| `eeg-meg-inventory <dir> <tsv>` | `bidsmgr-scan <dir> <tsv>` |
| `build-heuristic`, `run-heudiconv` | `bidsmgr-convert <tsv> <bids_parent>` (no HeuDiConv involved) |
| `build-dcm2bids-config`, `run-dcm2bids` | `bidsmgr-convert <tsv> <bids_parent>` (no dcm2bids involved) |
| `post-conv-renamer` | Built into `bidsmgr-convert` |
| `run-mne-bids` | `bidsmgr-convert <tsv> <bids_parent>` |
| `bids-metadata` | `bidsmgr-metadata <bids_parent>` |
| `fill-bids-ignore` | Not yet; planned as a `bidsmgr-metadata --fill-bidsignore` flag (issue tracker). |

### For users with custom heuristics

The 0.x HeuDiConv heuristic file (`heuristic_<study>.py`) and the
dcm2bids config (`dcm2bids_config_<study>.json`) are no longer
inputs. The conversion is driven entirely by the inventory TSV's
`BIDS_name` and entity columns. To migrate a custom heuristic:

1. Run `bidsmgr-scan` on the same raw tree to produce
   `inventory.tsv`.
2. Open the TSV in any spreadsheet tool (Excel, Numbers, LibreOffice).
   Edit `subject`, `session`, `task`, `run`, `acq`, `dir`, `echo`,
   etc. directly. The 0.x heuristic logic translates to spreadsheet
   formulas or simple find-replace operations.
3. Run `bidsmgr-rebuild inventory.tsv --from entities` to refresh
   `BIDS_name`.
4. Run `bidsmgr-convert inventory.tsv <bids_parent>`.

For programmatic per-row classification, `bidsmgr.classifier.types.Classifier`
is the plugin interface (an entry-point registry will land in v1.1;
in v1.0.0 plugins are in-tree under `classifier/`).

### For users importing from `bids_manager`

The 0.x code is no longer importable. The v1.0.0 import surface is
incompatible by design; the underlying functions were redesigned,
not renamed. A high-level map:

| 0.x | v1.0.0 |
|---|---|
| `bids_manager.dicom_inventory.run(...)` | `bidsmgr.inventory.mri_dicom.scan_dicoms_long(...)` |
| `bids_manager.eeg_meg_inventory.run(...)` | `bidsmgr.inventory.eeg_meg.scan_eeg_meg(...)` |
| `bids_manager.schema_renamer.build_preview_names(...)` | `bidsmgr.inventory.rebuild.rebuild_from_entities(...)` + `bidsmgr.schema.engine.entities_for(...)` |
| `bids_manager.post_conv_renamer.run(...)` | `bidsmgr.fixups.fieldmaps.fix_fieldmaps(...)` + `bidsmgr.fixups.intended_for.resolve_intended_for(...)` |
| `bids_manager.bids_metadata_engine.run(...)` | `bidsmgr.metadata.engine.run_metadata(...)` |
| `bids_manager.run_mne_bids.run(...)` | `bidsmgr.converter.backends.mne_bids.MneBidsBackend.run(...)` |
| `bids_manager.gui` (everything) | `bidsmgr.gui.main_window.MainWindow` + the panels under `bidsmgr.gui` |

## 21. Known limitations

- **DWI scanner-derivatives** (FA, ADC, TRACE, ColFA, ExpADC) are
  detected but currently skipped during conversion with a warning.
  A small `fixups/derivatives.py` (planned at ~80 LOC) closes the
  gap.
- **Cross-modality subject identity** keys per-modality. An MRI
  `Alice` and an EEG `alice.edf` don't currently merge to one
  `sub-001` post-scan; the user has to set `BIDS_name` manually.
  `inventory/subject_identity.py` extension is on the roadmap.
- **Project menus (M7)** are not wired yet. The event-sourced
  bundle is used by the Converter but the File menu (New / Open /
  Recent) and Edit > Undo are still to ship.
- **3-D viewer + GIFTI/FreeSurfer surface viewer (M8)** are not in
  v1.0.0. The pyqtgraph + OpenGL deps are already shipped so the
  port is unblocked; the Volume3DDialog from the 0.x codebase is
  the porting target.
- **MEG / EEG viewers** in the Editor center pane currently show a
  "no form for this file type" hint. Real viewers (channel list,
  trace browser) are future work.
- **`.cnt` Neuroscan format hint** in the EEG/MEG scanner is
  pending; the file is accepted but its sidecar fields aren't
  pre-filled.
- **`--project` flag** is wired through the GUI but not through
  every CLI verb. The CLI side will land before M7.
- **Pytest-qt teardown segfault** described in [§18](#18-tests).

## 22. Roadmap

| Milestone | Status |
|---|---|
| M1 – `dcm2niix_bidsguess` classifier | Done |
| M2 – Schema-driven Inspector + Properties form | Done |
| M3 – Per-subject staging + atomic commit | Done |
| M4 – Two-layer validation (schema + bidsschematools) | Done |
| M5 – Event-sourced project bundles | Done |
| M6 – Editor view (BIDS tree + sidecar + TSV + NIfTI viewer) | Done |
| M7 – Project menus (File / Open / Recent + Edit / Undo + provenance tooltips) | Planned |
| M8 – 3-D NIfTI viewer + GIFTI/FreeSurfer surface viewer | Planned |
| Cross-modality subject identity | Planned |
| DWI scanner-derivatives | Planned |
| MEG / EEG viewer stubs in the Editor | Planned |
| `.cnt` format hint | Planned |
| Plugin entry-point discovery (`bidsmgr.classifier.plugin` group) | v1.1 |

## 23. Reviewer's guide

A suggested reading order for someone reviewing the diff:

1. **Start with this document** for the overall picture and the
   architectural rationale.
2. **`bidsmgr/__init__.py`** for the package's self-description.
3. **`bidsmgr/schema/engine.py`** to see the schema API the rest
   of the package speaks against.
4. **`bidsmgr/inventory/mri_dicom.py`** and
   **`bidsmgr/inventory/eeg_meg.py`** for the scanner side of the
   pipeline. Compare against the 0.x `dicom_inventory.py` and
   `eeg_meg_inventory.py` to see what changed.
5. **`bidsmgr/classifier/dcm2niix_bidsguess.py`** for the M1
   classifier (the biggest single behaviour change vs 0.x's
   sequence-dict-only classification).
6. **`bidsmgr/converter/backends/dcm2niix_direct.py`** for the
   direct-dcm2niix MRI backend (replaces both HeuDiConv and
   dcm2bids paths).
7. **`bidsmgr/cli/convert.py`** to see the conversion orchestration
   without the GUI layer.
8. **`bidsmgr/gui/main_window.py`** + **`bidsmgr/gui/editor_panel.py`**
   for the GUI shell and the Editor view.
9. **`bidsmgr/gui/widgets/nifti_viewer_pane.py`** for the most
   feature-dense widget; the threaded-loader / tri-view / 4-D
   graph all live here.
10. **`bidsmgr/workers/`** to confirm the Qt boundary is clean.
11. **`bidsmgr/editor/validator.py`** for the two-layer validation
    surface.
12. **`docs/architecture.md`** for any deeper architectural
    rationale you want background on.
13. **`tests/unit/test_validator.py`**, **`tests/gui/test_nifti_viewer.py`**,
    and **`tests/unit/test_subject_identity.py`** for the most
    illustrative test modules.

The full diff is large (~234 files, ~50k insertions, ~16k deletions),
but most of the diff is *additions* in `bidsmgr/` and *deletions* in
`bids_manager/`. The two trees don't overlap line-for-line, so a
file-by-file review of `bidsmgr/` (top-down per the reading order
above) is usually the fastest path.

---

## Appendix A. Commit history reference

The granular development history of the new package is preserved in
the sibling repository
[`karellopez/bidsmgr`](https://github.com/karellopez/bidsmgr). The
final commits before the cutover squash:

```
3584d05 gui: NIfTI viewer in the Editor — slice + tri-view + 4-D graph
2fea217 gui: inspection-pane footer + clickable brand → About dialog
54dbfd7 gui: editor polish — strict toggle, jump-to-file, fix-button focus, Validate file/folder
e148789 gui: Editor view (M6) — BIDS tree, sidecar form, TSV editor, validation pane
89913e0 windows: cross-platform path hardening + version banner
9fe0b18 converter: hash series_uid for the per-series staging dir
793ba79 gui: move output-tree scan off the UI thread + ∞ busy spinner
ba760c5 gui: per-sequence leaves in filter pane + theme-aware brand logo
4877316 gui: full Converter tab (PyQt6) + workers + cross-platform settings
aee8d01 project: event-sourced bundle module (architecture.md §9, §10)
```

The squash commit on this side of the cutover is `14cc042`.

## Appendix B. Acceptance checklist

For the reviewers, things worth confirming explicitly:

- [ ] `pip install -e .` succeeds on Python 3.10, 3.11, 3.12.
- [ ] `bidsmgr` console script launches the GUI.
- [ ] `bidsmgr-scan` produces a 51-column TSV on a real DICOM tree.
- [ ] `bidsmgr-convert` produces a BIDS-valid tree (run
      `bids-validator` from another shell to confirm independently).
- [ ] `bidsmgr-validate --strict` agrees with `bids-validator`.
- [ ] The full test suite passes under
      `QT_QPA_PLATFORM=offscreen pytest`.
- [ ] Real-data tests pass on at least one MRI dataset with
      `BIDS_MANAGER_REAL_MRI_DATA=1`.
- [ ] The Windows installer in `Installers/Installers.zip` still
      assembles (it targets the 0.x console scripts; pending
      rebuild).
- [ ] PyPI metadata renders the README correctly (workflow SVG +
      GIFs).

## Appendix C. File counts

```
bidsmgr/                93 Python files
tests/                  57 test modules
docs/                    8 documents (this file, architecture.md,
                            super_plan.md, improvement_plan.md,
                            gui_mockups.html, workflow.svg,
                            inspector_proto/, screenshots/README.md)
miscellaneous/           5 GIFs + 1 PNG
```

---

*Authored by Karel López Vilaret. ANCP Lab, Carl von Ossietzky
Universität Oldenburg, 2026.*
