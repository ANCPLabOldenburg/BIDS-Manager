# BIDS-Manager — Improvement Plan (v0.3.0)

> **STATUS UPDATE (2026-05-10, commit `cdb97ed`).** The decision was
> made (see `super_plan.md`) to ship these improvements as a sibling
> package `bidsmgr/`, not by branching off v0.2.5 in place. The M1–M8
> features in this document have largely landed in `bidsmgr/`:
> M1 (BidsGuess classifier), M2 (dcm2niix-direct), M3 (metadata engine),
> M4 (validator), M5 (post-conv fixups), M6 (EEG/MEG via mne-bids).
> The v0.2.5 trunk at `BIDS-Manager/` stays untouched as the working
> reference. For current state read `CLAUDE.md` and the auto-memory.
> Remaining items in this doc that aren't yet in `bidsmgr/`: derivatives
> fixup, GUI port, project file format.

> Status: **Draft for approval.**
> Owner: Karel López Vilaret.
> Scope: incremental improvements on top of the existing `BIDS-Manager_previous/`
> (v0.2.5) codebase. **Not a rewrite.** Every change in this plan is a feature
> branch off the v0.2.5 trunk; the monolithic `gui.py` and `dicom_inventory.py`
> stay where they are. Where the v1.0 layered rewrite (`BIDS-Manager/` repo)
> produced a useful module — schema engine, post-conv renamer, dcm2niix
> classifier — we copy that file in as a leaf module, not as a new package
> tree.

---

## Operating constraints (non-negotiable)

- **Never remove, archive, rename, or move aside the existing
  `BIDS-Manager_previous/` codebase.** It is the working software. v0.3
  is built **directly on top of it, in place**. Nothing in this plan
  creates a parallel directory, a fork, an archive tag, a sibling repo,
  or a "next" / "v3" / "rewrite" tree. The directory keeps its current
  name and its current contents; commits land on its existing `main`
  branch. Even if a v0.4 plan later moves the code, that is a v0.4
  decision — until then, the code stays exactly where it lives now and
  every edit is a normal commit.
- **No rewrite.** v0.2.5 is the trunk. The PyQt5 monolithic `gui.py`, the
  `dicom_inventory.py` scanner, the `build_heuristic_from_tsv.py` /
  `run_heudiconv_from_heuristic.py` / `post_conv_renamer.py` /
  `schema_renamer.py` modules, and the `bids-manager` console script all stay.
  Every milestone below adds *to* them or *replaces a single function in
  place*; nothing reorganises directories.
- **No remote pushes.** No `git push`, no PR, no GitHub release, no PyPI
  upload, until the project owner approves explicitly.
- **Same `main` branch, single repo.** No parallel branch, no fork. Each
  milestone is a series of small, reviewable commits on `main`.
- **Real-data tests, not synthetic.** Every scanner / converter / GUI change
  is verified against the user's real datasets at
  `/Users/karelo/Development/datasets/BIDS_Manager/raw_data/{MRI,MEG,EEG}`
  before it ships. The 700-test green M9 suite passed with broken Philips
  support, broken joblib, broken subject IDs, and a broken fmap pipeline
  *because every test used hand-rolled fixtures*. Synthetic tests are
  necessary for CI but **insufficient as evidence**.
- **`subject_summary.tsv` is the contract.** The 22-column TSV v0.2.5
  produces is the canonical scan output. No milestone removes columns,
  reorders them, or changes their semantics. New columns may be appended.

---

## 1. Project vision (what v0.3 is for)

BIDS-Manager remains a **single desktop application that turns raw
neuroimaging data into a BIDS-compliant dataset, with the user in the
loop at every step**. v0.3 sharpens three pillars v0.2.5 already
delivers:

1. **See what you have before you commit.** The scanner walks raw
   DICOM / MEG / EEG, shows every series with its predicted BIDS path,
   demographics, and a real preview. No conversion runs until the user
   confirms the plan.
2. **Edit, don't re-run.** Every classification decision (subject id,
   session, task, suffix, "this goes to derivatives") is editable in the
   review table; downstream columns refresh live.
3. **Inspect what you produced.** The Editor tab clicks through the BIDS
   tree, renders JSON / TSV / NIfTI / DICOM / surface files in one
   auto-routing panel, runs the BIDS validator at file / folder / dataset
   scope, and fixes problems in place.

v0.3 adds: schema-driven pre-conversion validation, dcm2niix's
`BidsGuess` as a third classifier layer, multi-study drag-and-drop
orchestration, post-conversion auto-fill of missing sidecars, parallel
conversion, and validator-driven green/red highlighting in the editor.

---

## 2. Target users

Unchanged from v0.2.5:

- **Researchers and lab members** without programming background who need
  to convert complex raw datasets into BIDS-compliant trees.
- **Students** running their first BIDS conversion who need clear
  warnings and a guided workflow.
- **Power users / pipeline engineers** who want a CLI mirror of every
  GUI action so they can script conversions and integrate with HPC.

---

## 3. Current state of BIDS-Manager (v0.2.5)

What v0.2.5 does well today and v0.3 must preserve verbatim:

- **End-to-end MRI pipeline.** `dicom_inventory.scan_dicoms_long` (joblib
  parallel scan, 22-column `subject_summary.tsv`) →
  `build_heuristic_from_tsv` (per-study HeuDiConv heuristic) →
  `run_heudiconv_from_heuristic` (run HeuDiConv + physio conversion) →
  `post_conv_renamer` (fmap normalisation, `IntendedFor` synthesis,
  `scans.tsv` update). All four stages already work on real Siemens /
  Philips / GE data.
- **Pre-conversion overview.** Scanned-data review table, modality-grouped
  trees, BIDS-name preview, four-tab filter sidebar (General / Specific /
  Edit naming / Always exclude), preview pane with raw↔predicted
  selection link.
- **Detection layer.** Mixed-StudyInstanceUID conflict scanner, repeated
  series detection, "Always exclude" patterns persisted across sessions,
  user-customisable sequence dictionary.
- **Schema-driven naming.** Via `ancpbids.model_v1_10_0`.
- **Editor.** `MetadataViewer.load_file` (`gui.py:7969-8240`) — one
  panel that auto-routes by file extension into the right viewer:
  `_json_view`, `_tsv_view`, `_nifti_view` (with `Volume3DDialog`),
  `_surface_view` (GIFTI + FreeSurfer), `_dicom_view`, plain-text.
- **Multi-row autofill.** `AutoFillTableWidget` (`gui.py:570-1047`):
  Excel-style fill-handle with integer / decimal / datetime / text-pattern
  series.
- **Theming.** 16 palettes (light / dark / contrast).

What's missing or needs sharpening — see Section 5.

---

## 4. Features to preserve (do-not-touch list)

The following v0.2.5 contracts are load-bearing. Every milestone below is
designed not to break them.

- **22-column `subject_summary.tsv`**: `subject, BIDS_name, session,
  source_folder, include, sequence, series_uid, rep, acq_time,
  image_type, modality, modality_bids, n_files, GivenName, FamilyName,
  PatientID, PatientSex, PatientAge, StudyDescription, proposed_datatype,
  proposed_basename, Proposed BIDS name`. Column order, dtypes, and the
  fmap M+P merge into `image_type='MP'` are part of the contract.
- **Deterministic alphabetical subject IDs.** Per-study sort by
  `(StudyDescription, GivenName)` → `sub-001`, `sub-002`, … . Reruns on
  the same data return the same IDs.
- **Per-folder session-label union.** Any `ses-X` token in any series
  description in a folder propagates to every sibling series in that
  folder.
- **One auto-routing editor panel.** Click a file → the right pane
  rebuilds in place to render that file. *No* tabs for "JSON viewer",
  "TSV viewer", "NIfTI viewer".
- **`MetadataViewer` viewers** (`gui.py:5393-7969`): `Volume3DDialog`
  (~1500 lines, pyqtgraph + opengl with shaders, MIP, colormap,
  histogram), `Surface3DDialog`, `FreeSurferSurfaceDialog`. Stubs do not
  satisfy users; ports must be lossless.
- **Filter sidebar's two-way binding** to the review table
  (`onModalityItemChanged`, `gui.py:4296-4319`). In-memory, not
  disk-round-trip.
- **CLI parity.** `bids-manager` console script + `dicom-inventory`,
  `build-heuristic`, `run-heudiconv` subcommands. Every GUI action
  reachable from the CLI.

---

## 5. Problems to solve in v0.3

In priority order:

1. **No schema-driven pre-conversion validation.** v0.2.5 builds BIDS
   names but doesn't check them against the schema's entity ordering or
   suffix/datatype matrix until conversion fails.
2. **Sequence classification is hand-curated and brittle.** v0.2.5's
   `guess_modality` + sequence dictionary regex misses vendor variants
   (real Siemens fmap with `ImageType[2]='M' / 'P'` and `NONE` instead of
   `NORM`; multi-band bold; multi-echo MEGRE). dcm2niix already has a
   `BidsGuess` field per JSON sidecar — we don't consult it.
3. **No post-conversion auto-fill of missing sidecars.** After dcm2bids /
   HeuDiConv finishes, the dataset has gaps the schema marks REQUIRED.
   v0.2.5 doesn't fill them.
4. **Multi-study orchestration is read-only.** v0.2.5 detects multiple
   studies via `StudyDescription` but the Specific tree doesn't let the
   user merge / split / drag-and-drop subjects between studies.
5. **Conversion is single-threaded.** dcm2bids runs per (subject,
   session) sequentially, which costs minutes on multi-subject trees.
   joblib could fan it out.
6. **Validator is run-once-and-forget.** v0.2.5 has the `bids-validator`
   wrapper but no green/red highlighting in `BIDSplorer` and no
   per-file / per-folder validate actions in the editor.
7. **Settings are scattered.** `CpuSettingsDialog`, `DpiSettingsDialog`,
   `AuthorshipDialog`, `BidsIgnoreDialog`, `IntendedForDialog`, theme
   menu. Should be one consolidated dialog.
8. **No `IntendedFor` populator at convert time.** v0.2.5
   `post_conv_renamer` writes fmap names but doesn't synthesise the
   `IntendedFor` array; the user does it by hand.

---

## 6. Improvement strategy

**Branch v0.3 features off v0.2.5's `main`. Each milestone is a series
of small commits, scoped to one of the eight problems above. No
milestone touches more than ~5 files.** When the v1.0 rewrite produced
a useful self-contained module (e.g. `bids_manager/validation/post_conv_renamer.py`,
`bids_manager/scanners/dcm2niix_classifier.py`, `bids_manager/schema/`),
we copy it in as a leaf module and adapt the call site in `gui.py` /
`dicom_inventory.py`. We do not adopt the layered package structure.

**Definition of done for every milestone:**

1. The change works against the real datasets at
   `/Users/karelo/Development/datasets/BIDS_Manager/raw_data/{MRI,MEG,EEG}`,
   not just synthetic fixtures.
2. The 22-column `subject_summary.tsv` shape is unchanged (or extended
   with appended columns only).
3. v0.2.5's existing test suite (`tests/`) still passes.
4. A characterization test verifies the new behaviour against real data,
   gated on a `BIDS_MANAGER_REAL_*_DATA` env var so CI on systems
   without the data still passes.
5. The change is covered in `CHANGELOG.md` with a real-data
   before/after measurement.

---

## 7. Release scope — what v0.3.0 ships

v0.3.0 is the **complete v0.3 release**, not a minimum-viable
slice. All eight milestones below are required for the version tag.
Each one is individually reviewable + individually shippable as a
preview build, but the public release is gated on M1–M8 all landing.

- M1. dcm2niix `BidsGuess` classifier layer.
- M2. Schema-driven pre-conversion validation in the review table.
- M3. Multi-study merge / split / drag-and-drop in the Specific tree.
- M4. Consolidated settings dialog.
- M5. Post-conversion auto-fill of REQUIRED / RECOMMENDED sidecars.
- M6. Auto `IntendedFor` populator wired into the convert flow.
- M7. Validator-driven green/red highlighting in the editor.
- M8. Joblib parallel conversion (configurable in Settings).

A milestone is "done" when its Definition-of-done block (Section 10)
is satisfied **on the user's real datasets**. The release tag goes
on the commit that closes M8.

---

## 8. Out of scope for v0.3.0

The following are deliberately deferred because v0.3.0 delivers a
complete capable converter without them; they are roadmap items for
v0.4+, not gaps in v0.3.

- IDE-style project shell with a "recent projects" launcher.
- Multi-project workspace.
- Desktop binary (`.dmg` / `.exe` via PyInstaller).
- BIDS-validator real-time-as-you-type validation.
- Cloud / S3 raw-data sources.

---

## 9. GUI architecture (v0.3 changes only)

The v0.2.5 two-tab shell (`Converter` tab + `Editor` tab) is preserved.
Detachable panels (`detachTSVWindow`, `detachFilterWindow`,
`detachPreviewWindow`) are preserved. Themes are preserved.

**Changes:**

- **Settings menu** (M4) collapses `CpuSettingsDialog`,
  `DpiSettingsDialog`, `AuthorshipDialog`, `BidsIgnoreDialog`,
  `IntendedForDialog`, and the theme menu into one tabbed
  `SettingsDialog` (sections: Appearance, Performance, Conversion,
  Automation, Schema). The legacy small dialogs are kept as
  thin wrappers for backwards compatibility for two minor versions, then
  removed.
- **Specific filter tree** (M3) gains drag-and-drop between Study nodes,
  a "+ New study" drop target, "Merge into…" and "Split off subject"
  context-menu actions. Outputs one folder per study under the BIDS root.
- **Review-table `Status` column** (M2) renders ✓ / ⚠ / ✗ from the
  schema-driven validation result; tooltip shows the schema rule that
  failed.
- **`BIDSplorer` highlighting** (M7) colours each file row green / orange /
  red based on the last validate run; right-click adds "Validate this
  file" and "Validate folder" actions.
- **Predicted-sidecar pane** (M5 prerequisite) gains the same
  REQUIRED / RECOMMENDED / OPTIONAL / DEPRECATED colour-coding the
  Editor tab uses for actual sidecars.

---

## 10. Milestone roadmap

### M1 — dcm2niix `BidsGuess` classifier layer

**Goal.** Use dcm2niix's hand-curated DICOM → (datatype, suffix,
entities) classifier as a third layer feeding the existing
`guess_modality` + sequence-dictionary heuristics.

**Why.** Verified across real Siemens / PPMI / Philips DICOMs:
dcm2niix's per-sidecar `BidsGuess` field returns
`['anat', '_acq-tfl3_run-5_T1w']`, `['fmap', '_acq-fm2_magnitude2']`,
etc. — exactly the classification we'd otherwise hand-curate. Cost is
~10ms per series (one DICOM, sidecar-only mode).

**Scope.**

- New module `bids_manager/dcm2niix_bidsguess.py` (port of
  `BIDS-Manager/bids_manager/scanners/dcm2niix_classifier.py` from the
  v1.0 rewrite, with the `from bids_manager.scanners.classifier_result`
  import dropped — return plain `(datatype, suffix, entities)` tuples
  instead).
- `dicom_inventory.scan_dicoms_long` calls `bids_guess_batch()` once
  per scan (one representative DICOM per series), uses the result to
  populate `proposed_datatype` / `proposed_basename` / `Proposed BIDS
  name` columns.
- Existing `guess_modality` + sequence dictionary stays as fallback
  when `BidsGuess` is absent or returns an empty datatype.
- New TSV columns appended: `bids_guess_datatype`, `bids_guess_suffix`
  (informational; the user can compare).

**Definition of done.**

- Real-data test on `MRI/neuroimaging_unit_new`: at least one row whose
  v0.2.5 classification was wrong (or generic) gets a more specific
  classification from `BidsGuess`.
- v0.2.5's existing `dicom_inventory` tests still pass.
- Scan time on `MRI/Old_LNF` (51480 DICOMs) is within 20% of the
  v0.2.5 baseline.

**Estimated commits.** 4–6.

---

### M2 — Schema-driven pre-conversion validation

**Goal.** Each row in the review table validates its predicted BIDS
name against the official BIDS schema *before* the user clicks
Convert, surfacing a `Status` column (✓ / ⚠ / ✗) and a tooltip with
the failing rule.

**Why.** v0.2.5 lets impossible names through (e.g. a func/bold row
with no `task` entity, or an anat row with a `dir` entity that BIDS
doesn't allow there). The user only finds out when conversion fails.

**Scope.**

- Copy `bids_manager/schema/` from the v1.0 rewrite into v0.2.5 as a
  flat `bids_manager/bids_schema/` module. Strip the `bidsschematools`
  layering boundary; just expose `load_schema()`,
  `validate_entities(entities, datatype, suffix)`,
  `build_basename(entities, datatype, suffix)`,
  `build_relative_path(...)`.
- Hook into `gui.py` `_onMappingItemChanged` (`gui.py:3258`): after
  every cell edit, re-run `validate_entities` on the row, paint the
  Status cell, and emit a `_row_problems[row_idx] = [...]` list for
  the tooltip.
- Cells that fail validation get a 1px red border (custom delegate).
- Save action refuses to write `subject_summary.tsv` when any row has
  ✗ status (warning dialog with row count + first 5 errors).

**Definition of done.**

- Real-data test: deliberately set a func/bold row's task to empty →
  Status flips to ⚠, tooltip says "func/bold requires `task` entity"
  (verbatim from schema description).
- BIDS-version pin: schema package ships `v1.9.0` initially; user can
  swap via Settings → Schema.

**Estimated commits.** 8–10.

---

### M3 — Multi-study orchestration in the Specific tree

**Goal.** Let the user merge two studies, split a subject off into a
new study, and drag-and-drop subjects between study nodes — and have
Convert respect the resulting per-study layout.

**Why.** v0.2.5 detects studies but the tree is read-only. Real
datasets often have one operator-mislabelled subject under the wrong
study, or two studies that should have been one.

**Scope.**

- `populateSpecificTree` (`gui.py:4183-4291`) gains
  `setDragDropMode(QAbstractItemView.InternalMove)` and a custom
  drop handler that updates the row's `StudyDescription` field in
  memory + repaints downstream columns.
- New context-menu actions on Study nodes: "Merge into…", "Rename…".
- New context-menu action on Subject nodes: "Move to new study…",
  "Move to existing study…".
- `build_heuristic_from_tsv.py` already partitions by study; verify
  it picks up the user's edits without changes.
- HeuDiConv / dcm2bids invocation per study writes to
  `bids_root/<study_subdir>/`. Single-study projects stay flat.

**Definition of done.**

- Real-data test: scan a tree with two studies, drag a subject from
  Study A to Study B, run Convert dry-run → predicted output paths
  reflect the move.
- Existing single-study flow unaffected.

**Estimated commits.** 6–8.

---

### M4 — Consolidated Settings dialog

**Goal.** Replace the five small dialogs (`CpuSettingsDialog`,
`DpiSettingsDialog`, `AuthorshipDialog`, `BidsIgnoreDialog`,
`IntendedForDialog`) and the inline theme menu with one tabbed
`SettingsDialog`, MEGqc-style.

**Why.** Discoverability. Users currently can't find half of these.

**Scope.**

- New `bids_manager/settings_dialog.py` (port of
  `BIDS-Manager/bids_manager/gui/widgets/settings_dialog.py`, adapted
  to v0.2.5's `Preferences` dataclass).
- Sections: **Appearance** (theme picker, DPI scale), **Performance**
  (`scanner_n_jobs`, `converter_n_jobs`), **Conversion**
  (backend per modality), **Automation** (auto-validate, auto-fill,
  auto-IntendedFor, derivatives-for-repeats, save-PII-in-linkage),
  **Schema** (BIDS schema version + "Load schema from folder" button).
- Single entry: `File → Settings…` and a gear icon in the status bar.
- Legacy dialogs kept as deprecated thin wrappers; remove after two
  minor versions.

**Definition of done.**

- All settings reachable in three clicks or fewer.
- Settings round-trip through `platformdirs` `preferences.json` (same
  path v0.2.5 uses).

**Estimated commits.** 5–7.

---

### M5 — Post-conversion sidecar auto-fill

**Goal.** After dcm2bids / HeuDiConv finishes, walk the BIDS output,
compare each sidecar against the schema's REQUIRED / RECOMMENDED
fields, and write any missing fields with values from the source DICOM
or `__TODO__: <description>` placeholders.

**Why.** v0.2.5 produces sidecars only as rich as the converter
emitted. The schema specifies many more fields (e.g.
`MagneticFieldStrength`, `ManufacturersModelName`, `ReceiveCoilName`,
`PulseSequenceType`) that dcm2bids skips. The user fills them by
hand today.

**Scope.**

- New module `bids_manager/sidecar_writer.py` (port of
  `BIDS-Manager/bids_manager/validation/sidecar_writer.py` /
  `dataset_metadata.py`).
- After `run_heudiconv_from_heuristic` completes, call
  `materialise_sidecars(bids_root, raw_dicom_index, schema)` which:
  1. Walks `bids_root/sub-*/[ses-*/]<datatype>/*.json`.
  2. For each sidecar, looks up the schema's required + recommended
     keys for that (datatype, suffix).
  3. Pulls values from the matching raw DICOM (via the per-series
     metadata dict already produced by `dicom_inventory`).
  4. Writes `__TODO__: <description-from-schema>` for keys we can't
     fill from raw headers.
- Validator (M7) flags `__TODO__:` strings in the editor.

**Definition of done.**

- Real-data test on `MRI/neuroimaging_unit_new`: every produced
  sidecar has all REQUIRED keys (filled or TODO), and all RECOMMENDED
  keys present (filled or TODO).
- Re-running Convert is idempotent: existing non-TODO values are
  never overwritten.

**Estimated commits.** 8–10.

---

### M6 — Auto `IntendedFor` populator

**Goal.** After conversion, every fmap JSON sidecar's `IntendedFor`
array is populated with the relative paths of every functional /
diffusion file it should correct.

**Why.** v0.2.5's `post_conv_renamer` renames fmap files but doesn't
fill `IntendedFor`. The user does it by hand or runs an external
script.

**Scope.**

- Port `populate_intended_for(dataset_root)` from
  `BIDS-Manager/bids_manager/validation/post_conv_renamer.py:131-...`
  into v0.2.5's `post_conv_renamer.py` (same file, new function).
- Per `sub-*/[ses-*/]fmap/*_(magnitude1|magnitude2|phasediff).json`,
  list every `func/*_bold.nii(.gz)` and `dwi/*_dwi.nii(.gz)` under
  the same subject + session, write their relative paths into
  `IntendedFor`.
- Setting toggle: `auto_intended_for` (default ON).
- Existing `IntendedFor` arrays are overwritten (the populator
  is the source of truth).

**Definition of done.**

- Real-data test: scan + convert `MRI/neuroimaging_unit_new`, every
  fmap sidecar has a non-empty `IntendedFor` array, paths use
  forward slashes, paths are relative to the subject directory per
  BIDS spec.

**Estimated commits.** 3–4.

---

### M7 — Validator-driven highlighting in the Editor

**Goal.** Run the BIDS validator at file / folder / dataset scope from
the Editor tab; render results as green / orange / red highlighting
in `BIDSplorer` and inline in the open file's viewer.

**Why.** v0.2.5 has the `bids-validator` wrapper but no UI surface
for per-file results.

**Scope.**

- Toolbar buttons in the Editor tab: "Validate this file", "Validate
  folder", "Validate dataset".
- `BIDSplorer.set_validation_results(results)` paints each tree row's
  foreground based on result severity.
- Per-file result feeds a small status strip above the open viewer:
  "✓ valid" / "⚠ 3 warnings" / "✗ 2 errors". Click expands to a
  detail panel.
- For JSON sidecars: REQUIRED / RECOMMENDED / OPTIONAL / DEPRECATED
  field tinting (already done by v1.0's `SidecarPanel` —
  port that widget into v0.2.5).
- For TSV files: missing columns highlighted in the column header.
- Ships a Run-validator-as-you-type debounced mode behind a Settings
  toggle (off by default).

**Definition of done.**

- Real-data test on a deliberately-broken BIDS root (renamed file,
  missing required field): the explorer shows red on the broken file,
  green elsewhere; clicking the broken file shows the detail.

**Estimated commits.** 8–10.

---

### M8 — Joblib parallel conversion

**Goal.** Convert (subject, session) groups in parallel using joblib.

**Why.** dcm2bids / HeuDiConv per-(subject, session) invocations are
independent; serialising them costs minutes on multi-subject trees.

**Scope.**

- `run_heudiconv_from_heuristic.run(...)` gains an `n_jobs` parameter
  (sourced from `Preferences.converter_n_jobs`).
- `Parallel(n_jobs=n_jobs, prefer="processes")` over the
  per-(subject, session) tuples.
- Stdout / stderr aggregated and surfaced in the same per-row
  failure table v0.2.5 already shows.
- One progress bar per active worker in the GUI's Convert pane.

**Definition of done.**

- Real-data benchmark: full conversion of a 6-subject, 2-session
  Siemens dataset finishes in ≤ 50% of the v0.2.5 wall time at
  `n_jobs=4`.
- Output is byte-identical to the serial run.

**Estimated commits.** 4–6.

---

## 11. Testing strategy

Three rings, same shape as the v1.0 dev plan but scoped to v0.2.5's
flat layout:

- **Ring 1 — Unit tests** (`tests/unit/`). Pure-Python helpers:
  `dcm2niix_bidsguess.parse_bids_guess`,
  `bids_schema.validate_entities`,
  `sidecar_writer.fill_required_fields`,
  `post_conv_renamer.populate_intended_for`.
- **Ring 2 — Integration tests** (`tests/integration/`). Synthetic
  DICOM fixtures (`tests/fixtures/synthetic_dicom.py`) → exercise the
  full scan → plan → convert chain in a `tmp_path` BIDS root.
- **Ring 3 — Characterization tests** (`tests/characterization/`).
  Gated on `BIDS_MANAGER_REAL_MRI_DATA` /
  `BIDS_MANAGER_REAL_MEG_DATA` / `BIDS_MANAGER_REAL_EEG_DATA`. Run
  the full pipeline on the user's real datasets, diff the output
  against a checked-in golden `subject_summary.tsv` /
  `dataset_description.json` snapshot.

CI runs Rings 1+2 on Linux + macOS + Windows. Ring 3 runs locally
only.

---

## 12. What we copy from the v1.0 rewrite (`BIDS-Manager/`)

The v1.0 layered rewrite is a graveyard for the *shape* but a useful
source for *individual modules*. Files worth porting in as flat
modules:

- `bids_manager/scanners/dcm2niix_classifier.py` → M1.
- `bids_manager/scanners/heuristics.py` (`assign_fmap_group_id`,
  `collapse_fmap_groups`, `fieldmap_polarity_from_image_type`,
  `extract_entities`, `canonicalise_task`, `TASK_HINT_PATTERNS`) →
  M1 + M5.
- `bids_manager/schema/` whole package → M2.
- `bids_manager/validation/post_conv_renamer.py`
  (`rename_fieldmaps`, `populate_intended_for`,
  `move_repeats_to_derivatives`) → M5 + M6.
- `bids_manager/validation/sidecar_writer.py` /
  `dataset_metadata.py` → M5.
- `bids_manager/validation/unified.py` (schema-rule validator) → M7.
- `bids_manager/gui/widgets/sidecar_panel.py` (REQUIRED /
  RECOMMENDED / OPTIONAL / DEPRECATED tinted JSON tree) → M5
  prerequisite + M7.
- `bids_manager/gui/widgets/autofill_drag_handle.py` (Excel-style
  drag-fill with text-pattern series) → optional polish on top of
  v0.2.5's `AutoFillTableWidget`.

What we do **not** port:

- The `core/`, `io/`, `cli/dispatch.py`, `Pipeline` orchestrator.
  v0.2.5 already has its own equivalents and the layered shape is
  what made the rewrite a disaster.
- The `import-linter` `.importlinter` contracts. Flat layout doesn't
  need them.
- The PyQt6 `MainWindow` / `EditPhase` / `ReviewPhase` widgets. v0.2.5
  is PyQt5, the widgets aren't drop-in compatible, and the v0.2.5
  `gui.py` already has working equivalents.
- The `phases/__init__.py` seven-phase widget hierarchy.
- The synthetic-fixture-only test suite.

---

## 13. Open questions for the project owner

1. **PyQt5 vs PyQt6?** v0.2.5 is PyQt5. Sticking with PyQt5 keeps the
   `gui.py` viewers (`Volume3DDialog`, `Surface3DDialog`,
   `FreeSurferSurfaceDialog`) untouched. Migrating to PyQt6 would let
   us reuse v1.0 widgets but is a multi-week migration of its own.
   **Recommendation: stay on PyQt5 for v0.3, defer PyQt6 migration to
   v0.4.**
2. **Schema source.** Bundle the BIDS schema YAML in the wheel
   (`bids_manager/bids_schema/bundled/`) like the v1.0 rewrite did, or
   require an external `bidsschematools` install? Bundle gives offline
   reproducibility; external keeps the wheel small.
3. **Settings persistence.** Keep v0.2.5's
   `~/.config/bids_manager/preferences.json`, or move to
   `platformdirs.user_config_dir`? `platformdirs` is what v1.0
   used; trivially portable.
4. **MEG / EEG.** v0.2.5 has the `mne-bids` integration but the
   real-data MEG/EEG path was never tested in the v1.0 session. Do
   M1–M8 in MRI first, schedule a separate "MEG/EEG parity" milestone
   afterwards.
5. **Release vehicle.** Tag v0.3.0 on `main` once M1–M8 ship; cut a
   GitHub release; update PyPI. Or skip PyPI and ship a `.dmg` /
   `.exe` via PyInstaller as the headline distribution? PyPI is
   simpler; PyInstaller reaches non-Python users.

---

*End of plan. M1–M8 are roughly 50–70 commits across 4–8 weeks of
work, with each milestone individually shippable. v0.3.0 lands when
M8 closes and the real-data characterization test suite is green.*
