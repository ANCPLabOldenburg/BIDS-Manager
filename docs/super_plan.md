# BIDS-Manager — Super Plan (planning document, no code yet)

> **STATUS UPDATE (2026-05-10, commit `cdb97ed`).** Most of this plan
> has landed in `bidsmgr/`. The full CLI pipeline (scan → rebuild →
> convert → metadata → validate) is implemented and verified on 17 real
> datasets across MRI, EEG, MEG, physio. Refresh from `CLAUDE.md` and
> the auto-memory `bidsmgr current state and next feature` before
> treating any "to-do" line in this document as actionable — many are
> done. §13 (decision log) and §14 (handoff brief) remain historically
> accurate, but a few §13 entries (e.g. "two TSVs") were superseded by
> later decisions (one unified 51-column TSV with auto-detecting scan).
> Remaining work: `project/` event-sourced file, `gui/` Inspector port,
> `fixups/derivatives.py`.

> **Status: draft for owner approval. Nothing is created on disk until you sign off
> on Section 12.** This document is the response to the request "create a new
> local repo project and package […] a super improved and more reliable version
> of bids manager", filtered through what `improvement_plan.md` already says.

---

## 0. Read-this-first: contradiction with `improvement_plan.md`

`improvement_plan.md` (the v0.3 plan you authored) sets five non-negotiable
operating constraints. The two relevant ones:

1. *"Never remove, archive, rename, or move aside the existing
   `BIDS-Manager_previous/` codebase. […] Nothing in this plan creates a
   parallel directory, a fork, an archive tag, a sibling repo, or a 'next' /
   'v3' / 'rewrite' tree."*
2. *"**No rewrite.** v0.2.5 is the trunk."*

The same document records (§12) that **a previous v1.0 layered rewrite already
happened and was a graveyard** — "the layered shape is what made the rewrite a
disaster." The do-not-port list (§12) is itself the post-mortem of that
failure: `core/`, `io/`, `cli/dispatch.py`, `Pipeline` orchestrator, the
PyQt6 `MainWindow`/`EditPhase`/`ReviewPhase` widgets, the seven-phase widget
hierarchy, the `import-linter` contracts, the synthetic-fixture-only test
suite — all listed as "do not port."

The new request — *"create a new local repo project and package in this same
project root at the side of BIDS_Manager folder […] a super improved and more
reliable version of bids manager"* — is, structurally, the parallel-rewrite
tree the v0.3 plan forbids. **It is not a small deviation.** It would also be
the third attempt at the same shape (v0.2.5 trunk, v1.0 layered rewrite that
failed, now v1.x successor package).

This plan therefore presents **two paths** rather than assuming the new
request supersedes the old plan. Pick one in §12.

---

## 1. Setting & terminology

For unambiguity in the rest of this document:

- `BIDS-Manager/` — the working v0.2.5 (PyQt6, after our recent migration)
  package, currently at `/Users/karelo/PycharmProjects/superbidsmanager/BIDS-Manager/`.
  This is the working software. Everything we shipped this week (PyQt6
  migration, dcm2bids engine, EEG/MEG via mne-bids, schema metadata engine)
  landed here, in place.
- `improvement_plan.md` — your v0.3 plan, M1–M8.
- The "v1.0 rewrite" referred to in `improvement_plan.md` §12 is described as
  having existed but is **not on disk in this checkout**. Only `BIDS-Manager/`
  exists.

---

## 2. The two paths

### Path A — *In-place v0.3 (your improvement_plan.md, honoured literally)*

We treat `BIDS-Manager/` as the trunk. Everything from now on lands as small
commits on its `main`. The EEG/MEG-into-MRI-panels work becomes a refactor
of the existing two-tab GUI: the `EEG/MEG` tab gets folded into the existing
Converter tab, with modality-aware columns and per-row engine selection.
Then M1–M8 land sequentially as the v0.3 plan describes.

**Pros**
- Honors the plan you wrote and recently emphasised.
- Avoids re-creating the v1.0 failure mode.
- Every change is small, reviewable, and revertible.
- Existing real-data tests, fixtures, and user habits keep working without
  any "is the new one ready yet?" question.

**Cons**
- The 9.2k-LOC `gui.py` stays a monolith. Reliability improvements are
  bounded by what's tractable in a single file.
- "Super improved" remains aspirational — improvements are incremental.

### Path B — *New successor package next to `BIDS-Manager/` (your latest request, taken literally)*

A new sibling directory at the project root holds a successor package designed
from a clean slate. `BIDS-Manager/` stays untouched as the working software
(the v0.3 plan's first non-negotiable: "never remove, archive, rename, or
move aside" — this is the only way to literally satisfy that and still have a
new package). When the successor is at parity, you can decide whether to
deprecate the original.

**Pros**
- Lets us redesign hot spots (the GUI monolith, the per-modality TSV split,
  the inventory→plan→convert orchestration) cleanly.
- Lets us use a more aggressive test posture from day one.

**Cons**
- This is **literally the v1.0 rewrite pattern that already failed once**. The
  v0.3 plan's §12 do-not-port list is the receipts. Repeating the shape
  without a concrete reason it's different this time has a high prior of the
  same outcome.
- Two diverging codebases until parity. Real users keep using `BIDS-Manager/`
  for months. Bugs found in `BIDS-Manager/` need fixes in two places, or get
  abandoned.
- "Super improved and more reliable" is currently undefined — without a
  measurable definition we cannot tell when the successor is ready to take
  over.

### Path C — *In-place v0.3 with a new subpackage for the parts that actually need rewriting*

Middle ground. Stay in-place per Path A for the GUI and the entry points.
Where a *single subsystem* genuinely needs a clean rewrite (currently I'd
argue: the inventory→planner abstraction, so MRI/EEG/MEG can share one
review table), introduce a new subpackage *inside* `BIDS-Manager/bids_manager/`
(e.g. `BIDS-Manager/bids_manager/scanning/`). Keep the existing flat modules
working until the new subpackage is exercised on real data and proves out.

**Pros**
- Captures the "we want to rewrite *that part*" instinct without committing
  to a parallel tree.
- Honours the plan's "no parallel directory" constraint while still allowing
  meaningful structural improvements.
- Smallest blast radius if the rewrite of the subsystem fails.

**Cons**
- Not a "new package" in the literal sense the latest request used.
- Discipline-dependent: the team has to actually keep the subpackage scoped
  to one subsystem; scope creep turns it into Path B in disguise.

**My recommendation: Path C.** It captures the energy of the new request
(redesign the parts that hurt) without re-running the failed pattern, and it
directly satisfies the unified-MRI+EEG/MEG-panels requirement because the
shared inventory abstraction is exactly the part that benefits from a clean
rewrite. Decide in §12.

---

## 3. Vision (regardless of path)

`BIDS-Manager` is a single desktop application that turns raw neuroimaging
data — DICOM (MRI/PET), raw EEG (.edf/.bdf/.vhdr/.set/…), raw MEG
(.fif/.ds/.con/.sqd/…), iEEG, and physio — into a BIDS-compliant dataset, with
the user reviewing every classification decision before commit and inspecting
every file afterwards.

The **invariants** worth carving in stone:

1. **One review table.** Every recording, regardless of modality, is one row
   the user can edit. The visible columns adapt to the modality.
2. **Engine is a per-row attribute, not a global toggle.** A study can mix
   heudiconv, dcm2bids, mne-bids, and physio rows; each row's engine is
   chosen automatically with user override.
3. **Schema-driven naming and validation.** `schema_renamer.build_preview_names`
   is the single source of truth for filenames. The schema metadata engine
   audits sidecars after conversion.
4. **Reruns are deterministic and idempotent.** Same input + same TSV ⇒ same
   output. Re-running conversion never silently overwrites a hand-edited
   sidecar.
5. **Real data is the test.** Synthetic fixtures gate CI, but a release is
   only release-eligible if it has been run end-to-end against the user's
   `MRI/`, `MEG/`, `EEG/` real datasets.

---

## 4. Unified Converter panel (the new UX requirement)

This is the part of the new request that is unambiguous and shouldn't be
contentious. Currently the GUI has:

- `Converter` tab — DICOM workflow, with an Engine combo (HeuDiConv | dcm2bids)
- `EEG/MEG` tab — raw EEG/MEG workflow, separate inventory, separate Run

The user wants **one panel** that handles both. The proposed shape:

| Element | Behaviour |
|---|---|
| **Input picker** | Single picker. Drop a folder; the scanner sniffs whether it contains DICOMs, raw EEG/MEG files, or both, and dispatches to the right per-modality scanner. |
| **Inventory table** | One `QTableWidget` (or `QTreeView` over a `QAbstractItemModel`) with **modality-grouped rows**. Visible columns are the union of MRI columns + EEG/MEG columns; per-row, columns that don't apply render as blank/grey. A modality column ("MRI", "EEG", "MEG", "iEEG", "physio") is sticky-leftmost. |
| **Engine column** | Per-row drop-down: `auto`, `heudiconv`, `dcm2bids`, `mne-bids`, `bidsphysio`. `auto` = the obvious choice for the row's modality (DICOM→dcm2bids/heudiconv-by-setting; EEG/MEG→mne-bids; physio→bidsphysio). |
| **Filter/group sidebar** | Same as today's modality tree, extended with EEG/MEG modalities. |
| **Preview pane** | Shows predicted BIDS path for the selected row, regardless of modality. |
| **Run** | One button. Walks the TSV in modality+engine groups. Each group dispatches to the right pipeline (build heuristic / build dcm2bids config / mne-bids write / bidsphysio). |
| **Schema-metadata engine** | Runs once after all groups finish. Same as today. |

**Single-TSV-vs-two-TSV decision.** Today the DICOM and EEG/MEG inventories
have very different columns. Two reasonable shapes:

- **Single TSV**, columns are the union (DICOM + EEG/MEG), modality-irrelevant
  cells are blank. *Pro:* one source of truth. *Con:* wide, ~30 columns.
- **Two physical TSVs**, virtual unified table in the GUI. *Pro:* keeps the
  v0.2.5 22-column DICOM contract intact (the v0.3 plan's load-bearing
  assertion). *Con:* GUI joins them at runtime.

I'd pick the second to honour `improvement_plan.md`'s "22-column TSV is the
contract" promise. Decide in §12.

---

## 5. Architecture (Path B/C only — Path A keeps current shape)

### 5.1. Module layout (concrete)

```
<package>/
  pyproject.toml
  README.md
  src/<package_name>/
    __init__.py
    cli.py                          # all CLI entry points dispatch here
    schema/                         # schema-driven naming and validation
      __init__.py
      naming.py                     # build_preview_names, normalize_study_name
      sidecar_audit.py              # required-fields table + auditor
      validator.py                  # ancpbids soft validation wrapper
    inventory/                      # scanners
      __init__.py                   # registry of per-modality scanners
      types.py                      # InventoryRow dataclass
      mri_dicom.py                  # walks DICOM, emits InventoryRow[]
      eeg_meg_raw.py                # walks raw EEG/MEG via mne, emits rows
      physio.py                     # detects physio DICOMs
    planning/                       # TSV → engine-specific plan
      heudiconv_heuristic.py
      dcm2bids_config.py
      mne_bids_plan.py              # trivial: BIDSPath per row
    conversion/                     # actually runs the converters
      heudiconv_runner.py
      dcm2bids_runner.py
      mne_bids_runner.py
      physio_runner.py
    metadata/                       # post-conversion metadata engine
      engine.py                     # the existing schema-driven generator
      participants.py
      readme.py
      scans.py
    fixups/                         # post-conversion fixups
      fieldmaps.py                  # rename echo-1→magnitude1 etc
      intended_for.py               # populate IntendedFor
      derivatives.py                # move DWI maps to derivatives/
    gui/                            # the only Qt-coupled subtree
      __init__.py
      main_window.py                # entry, holds tabs
      converter_panel.py            # the unified panel (§4)
      editor_panel.py               # BIDS browser + auto-routing viewer
      widgets/
        inventory_table.py
        modality_tree.py
        preview_tree.py
        sidecar_viewer.py
        nifti_viewer.py
        surface_viewer.py
        ...
    workers/                        # QThread workers, no Qt-importing logic
      scan_worker.py
      convert_worker.py
      validate_worker.py
  tests/
    unit/
    integration/
    real_data/                      # gated on env vars
```

The discipline that the v1.0 rewrite missed (per §12 of `improvement_plan.md`):

- **`gui/` imports everything else; nothing imports `gui/`.**
- **`workers/` imports core logic, never widgets.**
- **No Pipeline orchestrator.** Modules are functions called from
  `cli.py` or `workers/`. The orchestration is explicit per-call code, not a
  framework. This is the single biggest lesson from §12 of `improvement_plan.md`.
- **No phases hierarchy.** The GUI has tabs and panels, not a state machine
  of "phases."

### 5.2. What we *port*, what we *rewrite*, what we *delete*

**Ports (from current `BIDS-Manager/bids_manager/`, minimal change):**
- `schema_renamer.py` → `schema/naming.py`. Already clean.
- `bids_metadata_engine.py` → `metadata/engine.py`. Already clean.
- `build_dcm2bids_config.py` → `planning/dcm2bids_config.py`.
- `run_dcm2bids.py` → `conversion/dcm2bids_runner.py`.
- `eeg_meg_inventory.py` → `inventory/eeg_meg_raw.py`.
- `run_mne_bids.py` → `conversion/mne_bids_runner.py`.
- `dicom_inventory.py` → `inventory/mri_dicom.py`. Big file (~580 LOC), needs
  some splitting but logic stays.
- `build_heuristic_from_tsv.py` → `planning/heudiconv_heuristic.py`.
- `run_heudiconv_from_heuristic.py` → `conversion/heudiconv_runner.py`. Some
  helpers move into `inventory/physio.py` and `conversion/physio_runner.py`.
- `post_conv_renamer.py` → split across `fixups/fieldmaps.py`,
  `fixups/intended_for.py`, `fixups/derivatives.py`.
- `scans_utils.py` → `metadata/scans.py`.
- `fill_bids_ignore.py` → `metadata/bids_ignore.py`.

**Rewrites (the parts that are the actual reason for a redesign):**
- `gui.py` (9.2k LOC). The single biggest reliability liability. Splits into
  `gui/main_window.py`, `gui/converter_panel.py`, `gui/editor_panel.py`, and
  `gui/widgets/*.py`. **The unified Converter panel from §4 is the centrepiece.**
  This is roughly 60% of the rewrite work.
- The two-TSV → unified-table glue. New code in
  `gui/widgets/inventory_table.py` + `inventory/__init__.py` registry.
- Worker layer. The existing GUI calls QProcess for some paths and threads
  for others; standardise on a `workers/` subpackage with one consistent
  signal protocol.

**Deletes (do not port):**
- The `_bz2` shim. PyQt6 + Python 3.11+ have working `_bz2`; the shim is
  a v0.2.x workaround whose root cause is no longer present.
- Anything from the v1.0 rewrite §12 do-not-port list (we don't have it
  on disk anyway, but worth re-stating: no `Pipeline`, no phases, no
  `import-linter`).

---

## 6. Reliability commitments (the "more reliable version" requirement)

If "super improved and more reliable" is going to mean anything we can verify,
it has to be a list of measurable improvements over current `BIDS-Manager/`.
Proposed targets:

1. **No silent failures.** Every code path that catches an exception must
   either re-raise, surface to the GUI log, or return a typed
   `Result`/`Report`. No bare `except Exception: pass`.
2. **Re-runs are byte-identical for unchanged inputs.** A real-data test
   asserts this: convert twice, diff the BIDS root.
3. **Crash recovery.** A conversion that dies mid-stream (kill -9, OOM, disk
   full) leaves the BIDS root in a state that the next run can resume from.
   Currently it doesn't.
4. **Schema validation is a precondition for "Convert succeeded."** If the
   metadata engine reports any required-field violation that isn't `n/a`-able,
   the run is marked as **partial success** in the GUI, not green-tick.
5. **Real-data CI gate.** Tag a release only if all three real-data
   characterisation suites (`MRI/`, `MEG/`, `EEG/`) pass on this machine,
   end-to-end, no manual edits.
6. **GUI smoke test.** Headless Qt smoke test (`pytest-qt`) that exercises
   each tab/panel without real data. Catches regressions like the
   `Qt.AlignCenter`-needs-scoping disaster from the PyQt6 migration.

---

## 7. Feature roadmap

The successor package starts from current `BIDS-Manager/` parity (everything
we have today, including the recent PyQt6/dcm2bids/mne-bids/metadata-engine
work) and then implements `improvement_plan.md`'s M1–M8 *plus* the unified
panel work.

### Phase 0 — Bootstrap (Path B/C only)

1. Create the package skeleton (Path B: sibling repo; Path C: subpackage).
2. Set up pyproject, tests config, CI.
3. Empty modules with docstrings only.
4. **Gate:** `pytest` runs on an empty suite. `python -c "import <pkg>"` works.

### Phase 1 — Parity port

5. Port the inventory + planner + converter modules from §5.2.
6. Port the metadata engine.
7. Port the EEG/MEG modules.
8. **Gate:** every CLI entry point in current `BIDS-Manager/` works in the
   successor. Real-data MRI + EEG conversions produce identical output.

### Phase 2 — Unified Converter panel (the new request)

9. Implement the modality-agnostic inventory model (§4, single TSV or two,
   per §12 decision).
10. Implement the unified `ConverterPanel`.
11. Wire scan workers, convert workers, log mirror.
12. **Gate:** drop a folder containing both DICOMs and `.edf` files, scan,
    review, run, and end up with a single BIDS root containing both
    `sub-001/anat/...` and `sub-001/eeg/...`.

### Phase 3 — `improvement_plan.md` M1–M8 in the successor

13. M1: dcm2niix `BidsGuess` classifier layer.
14. M2: schema-driven pre-conversion validation.
15. M3: multi-study merge / split / drag-and-drop.
16. M4: consolidated Settings dialog.
17. M5: post-conversion sidecar auto-fill.
18. M6: auto `IntendedFor` populator.
19. M7: validator-driven highlighting in the editor.
20. M8: joblib parallel conversion.

### Phase 4 — Reliability commitments (§6)

21. Re-run idempotence test.
22. Crash-recovery test.
23. Real-data CI gate setup.
24. GUI smoke test setup.

### Phase 5 — Cutover decision

25. **Gate:** all of Phase 0–4 pass. Owner decides: deprecate `BIDS-Manager/`,
    keep both, or merge changes back into `BIDS-Manager/` and delete the
    successor.

The total is roughly **eight to twelve weeks of focused work** if Path B is
chosen. Path A (the in-place v0.3 plan) is **four to eight weeks** per its
own estimate. Path C is somewhere in between depending on which subsystems
get rewritten.

---

## 8. Migration & coexistence (Path B specifically)

While the successor is being built, real users keep using current
`BIDS-Manager/`. That means:

- Bug fixes that land in `BIDS-Manager/` need to be back-ported to the
  successor. This is a real cost.
- The `subject_summary.tsv` v0.2.5 contract (22 columns,
  `improvement_plan.md` §4) is preserved verbatim in the successor's MRI
  scanner output.
- The successor and `BIDS-Manager/` use distinct config dirs
  (`~/.config/bids_manager_next/` vs `~/.config/bids_manager/`) so user
  preferences don't collide.

For Path C this entire section collapses — there's no parallel tree.

---

## 9. Risk register

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| 1 | Repeats v1.0 failure mode (Path B) | High | Keep `gui/` strictly downstream; no `Pipeline` orchestrator; commit a §12 do-not-port list to README.md before any code lands |
| 2 | Drift between `BIDS-Manager/` and successor | High (Path B), N/A (Path C) | Bug-fix policy: any change in `BIDS-Manager/` either ships in successor same week or files an issue in the successor's tracker |
| 3 | "Super improved" is undefined | High | §6's six measurable commitments. Add to release-tag preconditions |
| 4 | EEG/MEG-into-MRI-panels stalls | Medium | Phase 2 lands before any M1–M8 milestone; if Phase 2 won't land in 2 weeks, abort |
| 5 | Real-data tests gated locally only | Medium | `improvement_plan.md` already accepts this. Document the local-only test machine |
| 6 | PyQt6/PySide6/PyQt5 churn | Low | Pin PyQt6==6.11.0 (we just migrated). Successor inherits |
| 7 | mne import time blocks GUI startup | Low | Already lazy-imported. Successor preserves this |

---

## 10. Open questions for the project owner

These need answers **before any directory is created**.

### Path-level decisions

- **Q1.** Path A (in-place v0.3, your existing plan), Path B (new sibling
  package), or Path C (in-place + new subpackage)?

  **My recommendation: Path C.** It satisfies the new EEG/MEG-into-MRI-panels
  requirement with a focused subsystem rewrite, without re-running the v1.0
  failure mode, and without breaking your own "no parallel directory" rule.

- **Q2.** Is `improvement_plan.md` still the source of truth for *features*
  (M1–M8), even if the *architecture* path changes?

  Default assumption: yes.

### If Path B is chosen

- **Q3.** Folder name. Suggestions: `bids-bench/`, `bids-manager-next/`,
  `bids-studio/`, `bids-forge/`. None of these are great. Owner picks.
- **Q4.** Package name on PyPI (when we get there). Same trap.
- **Q5.** Cutover policy. When the successor reaches parity and Phase 4
  closes, does `BIDS-Manager/` get archived (forbidden by your plan), kept
  as a frozen reference, or merged back?

### If Path C is chosen

- **Q6.** Subpackage name inside `bids_manager/`. Suggestions:
  `bids_manager/scanning/` (if that's the only rewrite),
  `bids_manager/core/` (if more grows there). I'd avoid `core/` because
  `improvement_plan.md` §12 specifically lists `core/` as a do-not-port name
  from the v1.0 rewrite — using it again would be confusing.

### Cross-cutting

- **Q7.** Single TSV (modality columns merged) or two TSVs (DICOM v0.2.5
  contract preserved, EEG/MEG separate, GUI joins). Per §4 my pick is the
  second.
- **Q8.** Is `BIDS-Manager/` allowed to be renamed at any point? The plan
  says no. The new request implies yes (it called it `BIDS_Manager` with an
  underscore, suggesting it might be moved). If the answer is no, Path B's
  sibling has a different name; Path C is unaffected.
- **Q9.** Real-data dataset paths still
  `/Users/karelo/Development/datasets/BIDS_Manager/raw_data/{MRI,MEG,EEG}`?

---

## 11. What I am explicitly not committing to in this document

To avoid scope creep and to keep this honest:

- I am not promising the successor will be "more reliable" without §6's
  measurable commitments being signed off and instrumented.
- I am not promising the unified Converter panel can fully replicate the
  current GUI's 9.2k-LOC behaviour by week N. Phase 2 is the riskiest piece
  and gets a hard 2-week budget; if it slides, we re-plan.
- I am not promising feature parity with M1–M8 from `improvement_plan.md` in
  one pass; each milestone is individually shippable.
- I am not promising that Path B is a good idea. It is one of three options
  presented honestly. My recommendation is Path C.

---

## 12. Sign-off block

Fill in and tell me to proceed. Until then, **no directory is created, no
code is moved, no commits land.**

```
Path:                    [ ] A (in-place v0.3)
                         [ ] B (sibling package, name: ____________)
                         [ ] C (in-place + new subpackage, name: ____________)

Single or two TSVs:      [ ] one merged TSV
                         [ ] two TSVs, GUI joins

improvement_plan.md M1–M8 features still in scope:  [ ] yes  [ ] no

Real-data dataset paths still {MRI,MEG,EEG} under
/Users/karelo/Development/datasets/BIDS_Manager/raw_data/?  [ ] yes  [ ] no

If Path B:
  Cutover for BIDS-Manager/ when successor reaches parity:
    [ ] keep both indefinitely
    [ ] frozen reference
    [ ] merge back into BIDS-Manager/, delete successor
```

---

## 13. Owner decisions on record (sign-off increment, latest first)

These are the answers received in conversation. They override anything
above that contradicts them.

**Path:** **Path B** — new sibling package next to `BIDS-Manager/`. The
existing `BIDS-Manager/` stays as the working software; nothing is
removed, archived, or renamed. The `improvement_plan.md` non-negotiable
"never remove or move aside" still holds for `BIDS-Manager/` — we are
just building a successor alongside, not in place of.

**Schema source:** `bidsschematools` (upstream canonical), with
`ancpbids` retained for in-memory dataset navigation in the editor.

**Default MRI converter:** `dcm2niix` invoked directly. The schema
engine computes the basename; we pass it to `dcm2niix -f`. No
`dcm2bids`/`heudiconv` wrapper as the default path. Both can ship as
optional plugins later for users who have existing configurations they
want to keep.

**GUI binding:** PyQt6.

**Build system:** `pyproject.toml` only (PEP 621), `setuptools` backend.

**Layer requirements (owner emphasis):**

1. **BIDS schema is the soul.** Architecture.md §0–§3 — the rules engine
   is the keystone module every other layer reads from. Confirmed.
2. **Per-modality scanners.** `pydicom` for MRI; `mne.io.read_raw` for
   EEG/MEG/iEEG, covering all formats `mne` supports (EDF, BDF, GDF,
   BrainVision, EEGLAB, FIF, CTF, KIT, 4D, EGI, etc.). Architecture.md §2,
   §12 (`inventory/`).
3. **Guessing algorithm with explicit, accurate criteria** for both
   classification (datatype/suffix/entities) **and** identity inference
   (subject / study / session / run). Architecture.md §4 — now expanded
   with §4.1 dedicated to identity inference and the per-modality
   evidence vectors / conflict detection rules.
4. **Fully integrated curation GUI** that surfaces *what happened during
   acquisition* — the reasoning behind each classification, the evidence
   for each subject/session grouping, the conflicts to resolve. The five
   GUI proposals in `gui_mockups.html` are the response to "what could
   that look like in practice"; we pick one for the implementation.
5. **Real-time validation before conversion.** Architecture.md §6 —
   three scopes (entity / basename / cross-row), all running off the
   same schema rules engine, every keystroke validated.
6. **Real-time validation after conversion.** Architecture.md §6 —
   dataset-level + per-file validation in the Editor, same rules engine,
   green/red highlighting in the BIDS browser.

**GUI shape:** **Proposal 1 — The Inspector.** Three-pane Converter
(modality/filter tree with checkboxes, inspection table, properties
panel) plus path bars on top and a tabbed bottom dock (BIDS preview /
log / conflicts / stats). The Editor is a second top-level tab with the
same three-pane shape (validation-aware BIDS tree, type-routed file
viewer, file/folder/dataset validation panel). See
`gui_mockups.html` proposal 1 for the comprehensive layout reference.

**Still open** (need answers before scaffolding):
- Package folder name (e.g. `bids-bench/`, `bids-manager-next/`,
  `bids-studio/`, `bids-forge/`). My recommendation: `bids-manager-next/`
  for clarity — name change later is cheap.
- Cutover policy for `BIDS-Manager/` once successor reaches parity:
  keep both indefinitely / frozen reference / merge back. My
  recommendation: **frozen reference** — preserves the v0.3 plan's
  "never remove" constraint while letting the successor become the
  active codebase.

**Defaulted (overridable later, not blocking)**:
- Single vs two TSVs → **two TSVs, GUI joins them at runtime**. Honors
  the v0.2.5 22-column contract from `improvement_plan.md` §4.
- Project file format → **JSON event log** first; SQLite if scale demands
  it.
- Type system → **Pydantic v2** for the five core abstractions in
  architecture.md §2.
- Schema upgrade policy → **pin per project**, prompt on open if a
  newer schema is bundled.
- Plugin discovery → **in-tree registry** for v1; entry-points later.
- Provenance storage → **both**: `GeneratedBy` in
  `dataset_description.json` + `.bidsmgr/provenance.json` for the
  per-cell audit trail.

---

## 14. Agent handoff brief (canonical)

> **Read this section in full before writing any code in `bidsmgr/`.**
> It is the source of truth for the project's current state and is
> kept in sync with the actual filesystem. If anything below conflicts
> with the code on disk, the code wins — fix this section.

### 14.1 What's on disk right now

```
/Users/karelo/PycharmProjects/superbidsmanager/
├── BIDS-Manager/          ← v0.2.5 working software (PyQt6, real users)
├── bidsmgr/               ← NEW successor package, freshly scaffolded
├── inspector_proto/       ← visual reference (working PyQt6 prototype)
├── .venv/                 ← shared venv with all deps installed
├── CLAUDE.md              ← auto-loaded; project map for new agents
├── super_plan.md          ← this file
├── architecture.md        ← architectural rationale (the "why")
├── gui_mockups.html       ← five GUI design proposals
└── improvement_plan.md    ← original v0.3 feature plan (M1–M8 still scope)
```

### 14.2 What was decided (final)

| | |
|---|---|
| **Path** | B — sibling package, no rewrite of `BIDS-Manager/` |
| **Successor folder name** | `bidsmgr/` (provisional; becomes canonical "BIDS-Manager" at cutover) |
| **Successor Python package** | `bidsmgr` |
| **Cutover policy** | `BIDS-Manager/` becomes frozen reference once `bidsmgr` reaches parity |
| **Schema source** | `bidsschematools` (canonical) + `ancpbids` (graph reads only) |
| **Default MRI converter** | `dcm2niix` invoked directly (no wrapper) |
| **GUI binding** | PyQt6 6.11.0 |
| **GUI shape** | Inspector — `gui_mockups.html` proposal 1, validated visually via `inspector_proto/` |
| **Build** | PEP 621 `pyproject.toml` only; `setuptools` backend |
| **Layout** | Flat `bidsmgr/bidsmgr/<modules>` (no `src/` indirection) |
| **TSV strategy** | Two TSVs (DICOM 22-col + EEG/MEG separate); GUI joins at runtime |
| **Project file format** | JSON event log first; SQLite later if scale demands |
| **Types** | Pydantic v2 |
| **Schema versioning** | Pinned per project |
| **Plugin discovery** | In-tree registry for v1 |
| **Provenance** | Both `GeneratedBy` + `.bidsmgr/provenance.json` |

### 14.3 What was scaffolded (current state of `bidsmgr/`)

`bidsmgr/` is **a working pip-installable package skeleton with
zero feature code**. Specifically:

- `pyproject.toml` — every dependency pinned per the decisions above:
  PyQt6 6.11.0, bidsschematools >=1.0, ancpbids, dcm2niix
  1.0.20250506, pydicom 3.0.1, mne >=1.6, mne-bids >=0.15, edfio,
  bidsphysio 21.6.24, setuptools<81, pydantic >=2.6, pydicom, nibabel,
  pandas, numpy, joblib, pyqtgraph, PyOpenGL.
- CLI entry points declared but stubbed: `bidsmgr` (GUI),
  `bidsmgr-scan`, `bidsmgr-classify`, `bidsmgr-plan`,
  `bidsmgr-convert`, `bidsmgr-validate`.
- All 17 subpackages exist with meaningful docstrings stating each
  module's expected role + cross-references to architecture.md.
- `bidsmgr/bidsmgr/gui/theme.qss` — the working token-based stylesheet,
  copied verbatim from `inspector_proto/theme.qss`.
- `bidsmgr/bidsmgr/gui/theme_manager.py` — the working `ThemeManager`
  class, lifted from the prototype; supports dark + light palettes,
  listener subscription, runtime swap.
- `tests/` directories all exist; `pytest` runs cleanly against zero
  tests.
- Verified: `pip install -e ".[dev]"` succeeds in `.venv/`,
  `python -c "import bidsmgr.<every_subpackage>"` succeeds.

**Nothing else is implemented.** Every other module is an empty
`__init__.py` with a docstring describing what should live there.

### 14.4 Architectural prevention guards

These are how we avoid repeating the v1.0 layered-rewrite failure
documented in `improvement_plan.md` §12. Treat them as load-bearing:

1. `schema/` is the keystone — everything imports from it; it imports
   nothing from this codebase.
2. `gui/` is the only Qt-coupled subtree. `workers/` bridges GUI signals
   to core, never the other way.
3. **No `Pipeline` orchestrator.** Orchestration is explicit code in
   `cli/<verb>.py` and `gui/converter_panel.py`.
4. **No subpackage named `core/`** (poisoned by v1.0 post-mortem).
5. Pure-data types (Pydantic / dataclass; no I/O methods on them).
6. Functions over classes where possible.

### 14.5 Where to start (first feature)

`improvement_plan.md` M1 — **dcm2niix `BidsGuess` classifier layer**.
Self-contained, high-leverage, validates the keystone. Sequence:

1. `bidsmgr/schema/` — port the public API shape from
   `BIDS-Manager/bids_manager/schema_renamer.py` but rebuild it on top
   of `bidsschematools` instead of the inline rule tables. Expected
   API listed in the module's `__init__.py` docstring.
2. `bidsmgr/inventory/types.py` — `InventoryRow` (Pydantic).
3. `bidsmgr/classifier/types.py` — `Classification` (Pydantic).
4. `bidsmgr/classifier/dcm2niix_bidsguess.py` — M1.
5. `bidsmgr/inventory/mri_dicom.py` — port
   `dicom_inventory.scan_dicoms_long` preserving the v0.2.5
   22-column TSV contract.
6. `bidsmgr/cli/scan.py` — wire into the `bidsmgr-scan` CLI verb.
7. `tests/real_data/test_mri_scan.py` — characterisation test on
   `MRI/neuroimaging_unit_new/` confirming output identical to the
   v0.2.5 baseline (with appended `bids_guess_*` columns). Gate on
   `BIDS_MANAGER_REAL_MRI_DATA`.

After that loop is green: port `metadata/` (`bids_metadata_engine.py`
is mostly schema-aware already and ports nearly verbatim), then
`converter/backends/dcm2niix_direct.py`, then GUI.

**Don't start GUI work until at least one end-to-end CLI conversion
works on real data.**

### 14.6 What porting from `BIDS-Manager/` looks like

The v0.2.5 trunk is a useful source for individual modules
(architecture.md §15 has a full list). Items most likely to port
verbatim or with light edits:

- `BIDS-Manager/bids_manager/schema_renamer.py` →
  `bidsmgr/schema/`. Rebuild on `bidsschematools`; preserve the public
  API (`build_preview_names`, etc.).
- `BIDS-Manager/bids_manager/dicom_inventory.py` →
  `bidsmgr/inventory/mri_dicom.py`. Preserve 22-column TSV. Add
  `bids_guess_*` columns from M1.
- `BIDS-Manager/bids_manager/bids_metadata_engine.py` →
  `bidsmgr/metadata/`. Schema-aware already; port nearly verbatim.
- `BIDS-Manager/bids_manager/post_conv_renamer.py` →
  split across `bidsmgr/fixups/fieldmaps.py` +
  `bidsmgr/fixups/intended_for.py` + `bidsmgr/fixups/derivatives.py`.
- `BIDS-Manager/bids_manager/eeg_meg_inventory.py` →
  `bidsmgr/inventory/eeg_meg_raw.py` (already schema-driven).
- `BIDS-Manager/bids_manager/run_mne_bids.py` →
  `bidsmgr/converter/backends/mne_bids.py`.

**Do NOT port** the v0.2.5 monolithic `gui.py` — its 9.2k-LOC shape
doesn't fit the new layered architecture. Instead, port the
*behaviour* (auto-routing viewer, themes, drag-fill) into the new
`bidsmgr/gui/widgets/` modules using the prototype's QSS + delegate
patterns as the visual baseline.

### 14.7 Real data

| Modality | Path | Notes |
|---|---|---|
| MRI | `/Users/karelo/Development/datasets/BIDS_Manager/raw_data/MRI/neuroimaging_unit_new/` | OL_0001/0002/0003. Siemens MAGNETOM Prisma 3T. Real series names hard-coded into `inspector_proto/data.py` for visual comparison. |
| MRI (large) | `MRI/Old_LNF/` | ~51k DICOMs. Use for scan-time benchmarks (M1 acceptance: within 20% of v0.2.5 baseline). |
| EEG | `EEG/eegmmidb/` (EDF), `EEG/sternberg/` (BrainVision), others | |
| MEG | `MEG/Klingelbach driving/` | CTF folders |

Real-data tests gate on `BIDS_MANAGER_REAL_{MRI,MEG,EEG}_DATA` env
vars. CI runs `tests/unit/` + `tests/integration/` only; real-data
characterisation runs locally.

### 14.8 What's still open (need owner sign-off)

Nothing critical is blocking. The remaining open questions are
implementation details that can be decided as you encounter them:

- Whether to bundle the BIDS schema snapshot into the wheel
  (`bidsmgr/schema/bundled/v1.10.0/`) or always read from the
  `bidsschematools` install. Easy to flip later.
- Concrete `Project` event-log schema versioning rules (file format
  version field, migration path).
- Whether the editor's NIfTI viewer ports the v0.2.5 `Volume3DDialog`
  pyqtgraph+OpenGL+shaders renderer verbatim or starts simpler. Port
  verbatim if it ports cleanly; rebuild if pyqtgraph compatibility
  bites.

### 14.9 Sanity checks before declaring a feature done

1. `pytest tests/unit/ tests/integration/` is green.
2. The corresponding real-data test (`tests/real_data/`) passes when
   the env var is set.
3. The change works against the real datasets at the paths in 14.7,
   not just synthetic fixtures.
4. The architectural guards in 14.4 are not broken (run
   `grep -r "from bidsmgr.gui" bidsmgr/{schema,inventory,classifier,planner,converter,metadata,fixups,project,editor,workers,cli}/`
   should return zero results).
5. The decision log here is updated if any defaulted-but-overridable
   choice from 14.2 was changed.
