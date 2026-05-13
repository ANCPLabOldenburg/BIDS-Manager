# Architecture — schema-as-engine BIDS converter

> **STATUS UPDATE (2026-05-10, commit `cdb97ed`).** The architecture
> described here is implemented in `bidsmgr/` as of this commit. Module
> layout (§12) and prevention guards are current. `inventory/`,
> `classifier/`, `converter/` (with three backends), `fixups/`,
> `metadata/`, `editor/` and all five `cli/` verbs exist and are
> tested. `project/` and `gui/` remain pending — they are the next
> milestones. Decision log (§15) is current.

> Companion to `super_plan.md`. This document answers the design question:
> "if you want to build the best BIDS converter, curator, and editor on the
> market, how would you do it?" It does not assume Path A/B/C from the
> super-plan; it describes the architecture the package would have, leaving
> the question of *where* it lives (sibling repo vs subpackage) to that
> document.

---

## 0. The single design bet

**The BIDS schema is the engine, not a check.**

Every layer of the tool — classification, naming, GUI forms, real-time
validation, sidecar generation, post-conversion auditing — reads from the
same machine-readable BIDS schema. We do not hand-curate which entities go
with which suffix, which fields are required for `func/bold`, what order
entities take in a filename, or what a valid `task` value looks like. We
ask the schema. When the schema updates, the tool updates.

Every existing BIDS converter I'm aware of fails this test in the same way:
schema-as-validation-step. heudiconv is heuristic-coded Python. dcm2bids is
regex on `SeriesDescription` plus `sidecar_changes`. BIDScoin has a YAML
ruleset hand-curated by humans. ezBIDS has automation rules. bids-validator
is purely a checker. None of them treat the schema as the central rules
engine that drives the rest.

The cost of this bet: we depend hard on the schema's machine-readable
quality. The benefit: the tool is correct by construction whenever the
schema is correct, and updates with the spec.

---

## 1. What the BIDS schema actually gives us, machine-readable

`bidsschematools` (the official Python distribution of the schema) and
`ancpbids` (already a dependency) expose:

| Machine-readable artefact | What we get from it |
|---|---|
| `objects.datatypes` | All datatypes (anat, func, dwi, fmap, perf, eeg, meg, ieeg, pet, nirs, mrs, micr, motion, beh) |
| `objects.suffixes` | All suffixes (T1w, bold, dwi, phasediff, eeg, meg, ...) with their displayName + description |
| `objects.entities` | All entities (sub, ses, task, acq, ce, rec, dir, run, mod, echo, flip, inv, mt, part, recording, ...) with name, format (label/index), regex |
| `objects.formats` | Value formats (label = `[A-Za-z0-9]+`, index = `[0-9]+`, etc.) |
| `objects.metadata` | All sidecar JSON fields, with display names, descriptions, types, units, enums |
| `rules.files.raw.<datatype>.<suffix_group>` | Per-datatype rules: which suffixes are valid, which entities are required/optional, which extensions (`nii.gz`/`json`/`bval`/`bvec`/...), which datatypes share a directory |
| `rules.entities` | Canonical entity ordering in filenames |
| `rules.sidecars.<datatype>.<sidecar_group>` | Per-(datatype, suffix-group) sidecar field requirements: REQUIRED, RECOMMENDED, OPTIONAL, DEPRECATED, with conditional rules ("if X is set, Y is required") |
| `rules.errors` / `rules.warnings` | The validator's own rules — same source we should use |
| `rules.tabular_data` | Required columns for `participants.tsv`, `*_scans.tsv`, `*_channels.tsv`, `*_events.tsv`, `*_electrodes.tsv`, etc. |
| `rules.dataset_metadata` | `dataset_description.json` field requirements |

That is enough machine-readable structure to drive every UX surface in the
tool from a single source.

The two questions you asked map onto this:

- *"…build a real-time programmatical ruler that helps to classify raw
  metadata patterns, give accurate BIDS names (that user can edit using
  entity-based templates in GUI), validate names and file structure
  before conversion, but also files, structure, and metadata after
  conversion"* → exactly what falls out of treating the schema as the
  engine. Sections 4–7 below.
- *"…fully interactive GUI that gives full control of conversion to the
  user"* → consequences of the entity-template form pattern + real-time
  schema validation. Section 8.

---

## 2. Five core abstractions

The whole codebase is structured around five typed objects. Strict
Pydantic / `dataclass` models. Pure data, no I/O methods.

```python
# 2.1  Raw recording, no BIDS interpretation yet.
class InventoryRow:
    row_id: UUID                     # stable across edits
    modality: Modality               # 'mri' | 'eeg' | 'meg' | 'ieeg' | 'pet' | ...
    source: Path                     # canonical on-disk pointer (file or dir)
    raw_metadata: dict               # everything the scanner extracted: DICOM headers,
                                     # mne raw.info, EDF header, etc.
    n_files: int
    discovered_at: datetime

# 2.2  A *candidate* (datatype, suffix) classification with confidence + reason.
class Classification:
    classifier: str                  # 'dcm2niix_bidsguess' | 'sequence_dict' | 'mne_channel_types' | ...
    datatype: Datatype               # validated against schema
    suffix: Suffix                   # validated against schema
    candidate_entities: dict[str, str]   # {task: 'rest', acq: 'mb6', ...}
    confidence: float                # 0.0–1.0
    rationale: str                   # human-readable, e.g. "BidsGuess: ['anat','_acq-tfl3_T1w']"

# 2.3  The *committed* plan for one row. Schema-validated.
class EntityPlan:
    row_id: UUID
    classification: Classification   # the chosen one
    entities: dict[str, str]         # final values, user-edited if needed
    extension: str                   # '.nii.gz', '.edf', etc.
    derived_basename: str            # built from schema rules, never hand-formatted
    derived_path: Path               # relative to BIDS root; built by schema
    backend: ConverterBackend        # 'auto' | 'heudiconv' | 'dcm2bids' | 'mne-bids' | ...
    include: bool
    overrides: dict[str, Any]        # any user override of classifier output
    last_validated: ValidationVerdict

# 2.4  A validation result, scoped to a single check.
class ValidationVerdict:
    severity: Severity               # 'ok' | 'info' | 'warning' | 'error'
    scope: Scope                     # 'entity' | 'basename' | 'sidecar' | 'file' | 'dataset'
    rule_id: str                     # 'BIDS-FUNC-001' or schema rule name
    message: str
    suggestion: str | None
    autofix: Callable | None         # if present, GUI shows "Fix" button

# 2.5  A conversion outcome, one per row.
class ConversionResult:
    row_id: UUID
    backend: ConverterBackend
    output_paths: list[Path]
    started_at: datetime
    finished_at: datetime
    stdout: str
    stderr: str
    return_code: int
    sidecar_fills: list[SidecarFill]
    follow_up: list[ValidationVerdict]
```

These five types travel through the system. Every module is a function on
them.

---

## 3. The schema rules engine (the keystone)

A single module — call it `<pkg>/schema/` — wraps `bidsschematools` and
exposes a strongly-typed API the rest of the codebase consumes. It never
hardcodes a label, suffix, or rule. It loads the schema once, caches it,
and provides:

```python
def list_datatypes() -> list[Datatype]
def list_suffixes(datatype: Datatype) -> list[Suffix]
def list_extensions(datatype: Datatype, suffix: Suffix) -> list[str]

def required_entities(datatype: Datatype, suffix: Suffix) -> list[Entity]
def optional_entities(datatype: Datatype, suffix: Suffix) -> list[Entity]
def deprecated_entities(datatype: Datatype, suffix: Suffix) -> list[Entity]
def entity_order() -> list[Entity]
def entity_format(entity: Entity) -> EntityFormat   # regex + type

def required_sidecar_fields(datatype: Datatype, suffix: Suffix) -> list[Field]
def recommended_sidecar_fields(...) -> list[Field]
def optional_sidecar_fields(...) -> list[Field]
def deprecated_sidecar_fields(...) -> list[Field]
def conditional_sidecar_rules(...) -> list[ConditionalRule]
def field_metadata(field: Field) -> FieldMetadata    # type, units, enum, description

def required_columns(table: TabularName) -> list[Column]
# ...

def build_basename(entities: dict, datatype: Datatype, suffix: Suffix, extension: str) -> str
def build_relative_path(entities: dict, datatype: Datatype, suffix: Suffix, extension: str) -> Path

def validate_basename(basename: str, datatype: Datatype) -> list[ValidationVerdict]
def validate_entity_set(entities: dict, datatype: Datatype, suffix: Suffix) -> list[ValidationVerdict]
def validate_sidecar(sidecar: dict, datatype: Datatype, suffix: Suffix) -> list[ValidationVerdict]
def validate_file_structure(path: Path, dataset_root: Path) -> list[ValidationVerdict]
def validate_dataset(dataset_root: Path) -> list[ValidationVerdict]
```

Everything else in the codebase is downstream of this module. The classifier
uses it to know what `(datatype, suffix)` combinations exist. The GUI uses
it to render entity-form widgets. The metadata engine uses it to know which
sidecar fields are REQUIRED. The validator uses it to validate. There is no
duplicated knowledge of BIDS rules anywhere else.

The schema version is **pinned per project** (a project file remembers
which BIDS version it was created against). Schema upgrades are an explicit
user action, with diff-of-rules surfaced in the UI.

---

## 4. Layered guessing: identity inference + classification

`InventoryRow → list[Classification]` is one half of the guessing problem.
The other half — equally hard, more often wrong in current tools — is
**identity inference**: deciding which rows belong to the same *subject*,
which subjects belong to the same *study*, and which acquisitions form
the same *session*. The two halves run independently and feed the
planner together.

### 4.1 Identity inference (subject / study / session / run)

This is where current tools quietly fail on real datasets. v0.2.5's
`dicom_inventory.scan_dicoms_long` infers identity from a fixed set of
DICOM tags + a folder-name heuristic; a single mislabelled patient or a
shared `StudyDescription` quietly assigns two scans to one subject.

The successor uses **layered evidence with explicit confidence and
provenance**, scoped per modality.

#### MRI / DICOM

Per-series evidence vector:

| Field | What it tells us | Reliability |
|---|---|---|
| `PatientID` (0010,0020) | Subject identifier | High when populated; sometimes anonymized to constants |
| `PatientName` (0010,0010) | Subject identifier | Medium — may be hashed, anonymized, repeated |
| `PatientBirthDate` (0010,0030) | Subject discriminator | High when populated; often blank in research data |
| `StudyInstanceUID` (0020,000D) | Study group | **Highest** — the canonical "all these scans belong to the same imaging session" key |
| `StudyDate` (0008,0020) | Session discriminator | Medium — not unique across subjects |
| `StudyDescription` (0008,1030) | Study group label | Low — operator-typed free text |
| `AccessionNumber` (0008,0050) | Site-assigned study key | High when used |
| `AcquisitionDate` / `AcquisitionTime` | Per-series time | High |
| `OperatorsName` | Operator hint | Low |
| Source folder hierarchy | Operator-supplied grouping hint | Variable |

Inference rules:

- **Subjects.** Group rows by `(PatientID, PatientName, PatientBirthDate)`
  triple, with fallback to `(PatientID, PatientName)` when birth date is
  blank, and finally to source folder when both are anonymised to the
  same constant. **Conflicts** (same triple in two folders, different
  triple in one folder) are flagged for the user with the v0.2.5 mixed-
  StudyInstanceUID detector machinery already in place.
- **Studies.** Group subjects + their series by `StudyInstanceUID` first.
  When `StudyInstanceUID` is shared across multiple subjects (multi-
  subject study), keep the UID as the study group key. When two folders
  have different `StudyInstanceUID` but the same `StudyDescription`, the
  user sees both as candidate studies and can merge.
- **Sessions.** A session is **one acquisition visit**, not one
  scanner-session. Heuristic: rows with the same `(StudyInstanceUID,
  StudyDate)` form a session; gaps > 4 hours between acquisitions in the
  same study split into multiple sessions; a session label hint in the
  folder path (e.g. `ses-pre`, `ses-post`) overrides.
- **Runs.** Repeated `(SeriesDescription, datatype, suffix, entities)`
  tuples within a session get auto-numbered `run-1`, `run-2`, … unless
  the user assigns a different entity manually. The v0.2.5 repetition
  detector already does this; we keep its rules.
- **Subject IDs.** Auto-generated per-study, deterministic per
  `(StudyDescription, GivenName)` sort — the v0.2.5 contract from
  `improvement_plan.md` §4. `sub-001`, `sub-002`, … . Stable across
  re-scans.

#### EEG / MEG / iEEG

DICOM tags don't apply. Inference signals are:

| Source | What it tells us | Reliability |
|---|---|---|
| File path tokens (`sub-XXX`, `ses-YYY`, `task-ZZZ`) | All four entities | High when present (BIDS-style raw layouts) |
| `mne.Info.subject_info` | Subject identity | High when populated by acquisition software |
| `mne.Info.meas_date` | Session timestamp | High |
| File naming convention | Inferred entities | Variable per site |
| Folder hierarchy | Operator grouping hint | Variable |

Rules:

- **Subjects.** Per-modality folder-and-name heuristic with `subject_info`
  override. If two recordings share the same `subject_info.id` but are
  in different folders, they're the same subject (rare but happens).
- **Sessions.** Recordings with the same `subject_info.id` clustered by
  `meas_date` (gap > 4h splits). User-overridable.
- **Tasks.** From filename `task-` token, falling back to the path's
  parent directory name, falling back to a sanitised filename stem. The
  user always sees the chosen value and can edit it.

#### What the GUI shows for identity

The Modality tree has a row per inferred (Study, Subject, Session)
group. Hovering shows the evidence vector with each rule's contribution
("subject grouped by PatientID + PatientName because PatientBirthDate
was blank"). Wrong groupings are fixed by drag-and-drop in the tree
(re-parents the rows; events are recorded).

#### Conflict detection (the hard cases)

A conflict scanner — extension of v0.2.5's `_ConflictScannerWorker` —
runs after the inventory and surfaces:

- Two subjects with the same `(PatientID, PatientName)` triple but
  different `PatientBirthDate` (almost certainly a mislabel).
- One subject (by triple) with two different `StudyInstanceUID`s on the
  same date (possibly two scanner-sessions; usually one logical
  session).
- One `StudyInstanceUID` split across two folders (operator copy/paste
  error in raw-data layout).
- Mixed `(PatientID, PatientName)` triples inside a single folder
  (operator stuffed two subjects into the same folder).

Each conflict is a row in a "Conflicts" panel with one-click resolution
("merge these two as the same subject", "split this folder by
StudyInstanceUID", etc.).

### 4.2 Classification (datatype, suffix, candidate entities)

`InventoryRow → list[Classification]`, ranked.

Three layers, each with full provenance:

1. **`dcm2niix_bidsguess`** (MRI). dcm2niix already emits `BidsGuess` per
   sidecar. Highest-fidelity classifier we have for DICOM. Run once per
   series in sidecar-only mode (~10ms per series).
2. **`mne_channel_types`** (EEG/MEG/iEEG). Channel kinds in the raw object
   give datatype directly. mne-bids' `_handle_datatype` already does this
   work; we wrap it.
3. **`sequence_dict`** (fallback). User-editable JSON dictionary mapping
   `SeriesDescription` regexes to `(datatype, suffix, entity_hints)`.
   Today's `dicom_inventory.guess_modality` regex collection becomes one
   layer here, not the only layer.

Each layer is a function:

```python
def classify(row: InventoryRow) -> list[Classification]
```

Each returns zero, one, or more candidates with confidence. The planner
picks the highest-confidence one (or the user picks manually).

When the schema is the canonical source, **every classifier output is
validated against the schema before it reaches the user**. A classifier
returning `(anat, bold)` is rejected by the schema (the schema's
`anat.suffixes` doesn't include `bold`); we log it and move to the next
classifier.

This is what lets us be aggressive on classifier additions without risk.
We can ship an LLM-assisted classifier, a vendor-specific Siemens
classifier, a Philips classifier — they can all be wrong, the schema
catches it.

---

## 5. The planner: where user edits live

`Classification + user edits → EntityPlan`.

The planner is the entry point for *user agency*. It is the only place
that turns "the classifier says T1w" into "this row produces this exact
path." The user can:

- Accept the classifier's choice.
- Override the datatype + suffix (dropdown of schema-valid options).
- Edit any entity value.
- Add an optional entity that's allowed but not required.
- Mark the row as `include=False`.
- Override the converter backend.
- Bulk-apply a change to N rows at once.

Every edit is **validated by the schema in real time** (Section 6). The
planner refuses to produce a plan that the schema rejects; the user sees
red, with the failing rule.

The planner is pure: `(state + edit) → state`. This is what enables the
event-sourced project file (Section 9) and undo/redo at no extra cost.

---

## 6. Validation: three scopes, one rules engine

The schema's `rules.errors` and `rules.warnings` are the same source we
use, run at three different scopes:

| Scope | When it runs | What it checks |
|---|---|---|
| **Entity** | Every keystroke in the GUI form. Sub-millisecond. | Entity name in the schema's allow-list for this `(datatype, suffix)`. Value matches `entity_format(entity).regex`. Required entities present. Order correct. |
| **Plan / basename** | After every entity edit, post-debounce. <10ms. | Schema's `validate_basename` + cross-row checks (no two plans produce the same path). |
| **Dataset / files** | On demand from the editor; auto after conversion. | Full schema validation against on-disk files. Sidecar required fields. Pairs (`.nii.gz` ↔ `.json`, `.bval`/`.bvec`/`.json` for dwi). `IntendedFor` integrity. |

All three scopes return the same `ValidationVerdict` shape, so the GUI has
one renderer for all of them.

The validator runs **continuously, not as a checkpoint**. The user
literally cannot leave a row in an invalid state without seeing red.

---

## 7. Conversion as a plugin layer

Converter backends register against an interface:

```python
class ConverterBackend(Protocol):
    name: str
    supported_modalities: set[Modality]

    @classmethod
    def can_handle(cls, plan: EntityPlan) -> bool: ...
    def convert(self, plans: list[EntityPlan], dataset_root: Path,
                progress: ProgressCallback) -> list[ConversionResult]: ...
```

Backends ship in-tree as plugins:

| Backend | Modalities | Notes |
|---|---|---|
| `dcm2niix_direct` | MRI | Skip dcm2bids; talk to dcm2niix directly with our own filename via `-f`. The most reliable path. |
| `dcm2bids` | MRI | Optional, kept for users who want dcm2bids behaviour. |
| `heudiconv` | MRI | Optional, kept for back-compat. |
| `mne_bids` | EEG/MEG/iEEG/NIRS | What we already have. |
| `bidsphysio` | physio | Stays. |
| `passthrough` | any | For files already in BIDS shape; just copy + add to scans.tsv. |

Per-row backend is part of `EntityPlan`. The conversion orchestrator groups
plans by backend, dispatches each group, and merges `ConversionResult[]`.

**Critical: the converter never decides BIDS names.** Filenames are
computed by the schema rules engine (`build_basename`/`build_relative_path`)
from the `EntityPlan.entities`. The backend's job is "produce this file at
this path." For dcm2niix-direct that's literally `-o <dir> -f <basename>`.
For dcm2bids we generate a config that targets exactly that filename. For
mne-bids we pass a `BIDSPath` with our entities.

This solves the dcm2bids problem you ran into earlier in the week
(criteria-by-SeriesInstanceUID workarounds): we don't ask the converter to
guess; we tell it.

---

## 8. The GUI — fully interactive, three-pane inspector

Single Converter panel for all modalities. Layout:

```
┌──────────────────────────────────────────────────────────────────────┐
│  toolbar:  [scan] [add files...]  [save project]  [run...]  [editor]│
├────────────┬────────────────────────────────┬────────────────────────┤
│ Modality   │  Inventory table               │  Entity-template form  │
│ tree       │  (modality-aware columns,      │  (the selected row)    │
│            │   one row per recording,       │                        │
│ ─ MRI      │   live status badge)           │  Datatype:   [func ▾]  │
│   ─ anat   │                                │  Suffix:     [bold ▾]  │
│   ─ func   │  ✓ 001 mri func bold task=rest │  Entities (schema-     │
│   ─ dwi    │  ⚠ 001 mri func bold task=…    │   driven, in order):   │
│   ─ fmap   │  ✗ 002 mri anat (no suffix)    │  ◯ sub:    001         │
│ ─ EEG      │  ✓ 002 eeg eeg eeg task=audio  │  ◯ ses:    pre         │
│ ─ MEG      │  ✓ 003 meg meg meg task=rest   │  ◉ task:   rest *req   │
│            │  …                             │  ○ acq:    [____]      │
│ filters:   │                                │  ○ run:    [_]         │
│ ─ status   │                                │  …                     │
│ ─ datatype │                                │                        │
│            │                                │  Backend:  [dcm2niix ▾]│
│            │                                │  Include:  [✓]          │
│            │                                │                        │
│            │                                │  Predicted path:       │
│            │                                │  sub-001/ses-pre/func/ │
│            │                                │  sub-001_ses-pre_task- │
│            │                                │  rest_run-1_bold.nii.gz│
│            │                                │                        │
│            │                                │  Validation:           │
│            │                                │  ✓ entity set valid    │
│            │                                │  ✓ basename valid      │
│            │                                │                        │
├────────────┴────────────────────────────────┴────────────────────────┤
│ Preview tree (the BIDS tree that will exist after Run)               │
└──────────────────────────────────────────────────────────────────────┘
```

Things this layout enables that current tools don't:

1. **Entity-template form, not free text.** When the user selects a row
   classified as `func/bold`, the form is *generated from the schema*: one
   field per allowed entity for `(func, bold)`, in the schema's canonical
   order, each typed (label vs index, regex constraint), each marked
   required/optional/deprecated. **It is impossible to type an invalid
   value into a valid entity, or an invalid entity name at all.**
2. **Live preview.** Predicted basename + path update on every keystroke.
3. **Live validation badges**, all three scopes (entity / basename / cross-row).
4. **Bulk edits.** Multi-select rows in the inventory; the form switches
   to "edit common entities for N rows." Apply once, all rows update.
5. **Modality-agnostic.** The same form pattern handles MRI, EEG, MEG,
   iEEG, PET. The form for `eeg/eeg` shows different fields than the form
   for `func/bold` — both come from the schema.
6. **Preview tree.** Shows the BIDS layout that will be produced. Diff
   against the current state of `bids_root/` if it exists.
7. **Run is dry-run by default.** "Run" produces a diff (paths added /
   modified). User confirms; we commit.

The Editor panel (separate tab) inherits the same schema engine for
post-conversion file inspection. Selecting a sidecar shows a form keyed by
schema field metadata — REQUIRED red if missing, RECOMMENDED orange,
OPTIONAL grey, DEPRECATED struck through. Changes are validated in place.

---

## 9. Project file: event-sourced, durable

A project is a file (`.bidsmgr` or similar) on disk. Two competing shapes:

- **JSON event log + cached state.** Each user action is one event:

  ```json
  {"v": 1, "type": "scan_complete", "ts": "...", "input": "...", "rows": [...]}
  {"v": 1, "type": "auto_classify", "ts": "...", "row_id": "...", "classifier": "...", ...}
  {"v": 1, "type": "user_set_entity", "ts": "...", "row_id": "...", "entity": "task", "value": "rest"}
  {"v": 1, "type": "convert", "ts": "...", "row_ids": [...], "results": [...]}
  ```

  Replaying the event log produces the current state. Undo = pop event,
  re-derive state. Audit = read log.

- **SQLite + last-state cache.** Same idea, in a database. Scales to
  100k-row inventories without loading the whole file.

For "huge heterogeneous raw data" both work; SQLite scales further.
Probably start JSON, migrate to SQLite when one user hits a real ceiling.

The benefits of event sourcing here are not theoretical:

- Re-running a scan that previously crashed mid-way is trivial — replay
  events, skip what's done.
- Sharing a project file with a colleague gives them the full audit trail
  ("why is this T1w classified as `acq-mprage`?" → see the
  `auto_classify` event with classifier='dcm2niix_bidsguess').
- The "undo" stack is essentially free.
- Bulk operations are themselves single events with N affected rows.

---

## 10. Provenance everywhere

Every value that ends up in the BIDS dataset has a recorded source:

- "This `RepetitionTime` came from DICOM tag (0018,0080)."
- "This `task` entity came from regex `task-([a-z]+)` matched against
  `SeriesDescription`."
- "This subject id came from the auto-numbering rule, not the user."
- "This `IntendedFor` array was populated by `auto_intended_for` after
  conversion."

Provenance is a side-table in the project file. The GUI has a "Where did
this come from?" right-click action on every cell, sidecar field, and
filename token. This is the difference between a converter that gives you
data and a converter that gives you data **plus the full audit trail** —
which is what BIDS is supposed to deliver.

---

## 11. Scale: huge heterogeneous raw data

Concrete targets:

| Concern | Target | Strategy |
|---|---|---|
| Inventory scan time | ≤ 60s for 100k DICOMs | joblib parallel (already used). Sidecar-only DICOM read. |
| Live validation latency | < 50ms per keystroke for any row | Validation runs on the changed row only, then cross-row checks debounced. |
| Inventory render | Smooth scrolling at 50k rows | Virtualised `QTableView` on `QAbstractTableModel`, not `QTableWidget` (current `gui.py` uses `QTableWidget`, which is the bottleneck above ~5k rows). |
| Conversion throughput | n_jobs * single-job rate | joblib parallel over per-(subject, session) groups (improvement_plan M8). |
| Project file load | < 2s for 50k events | Cached state snapshot at every Nth event; replay from snapshot, not from epoch. |
| Memory | < 1 GB for 50k-row project | Inventory rows are dataclasses, not pandas frames in memory. |

The current `gui.py` uses `QTableWidget` for the review table. It works
but degrades visibly past a few thousand rows because every cell is its
own `QTableWidgetItem`. The successor uses `QTableView` + a custom model.
This is one of the load-bearing rewrite reasons.

---

## 12. Module layout (concrete, schema-as-engine version)

**Flat layout — no `src/` indirection.** The repository contains the package
directly, matching the existing `BIDS-Manager/bids_manager/` shape so
v0.2.5 contributors don't need to relearn navigation.

```
<repo>/                               # e.g. bids-manager-next/
    pyproject.toml
    README.md
    LICENSE
    <pkg>/                            # the importable Python package
        __init__.py
        main.py                       # GUI entry
        schema/                       # §1, §3 — the keystone
            __init__.py
            engine.py                 # rules engine API
            loader.py                 # bidsschematools wrapper, schema cache
            validation.py             # validate_* functions
            types.py                  # Datatype, Suffix, Entity, Field enums
            bundled/                  # vendored schema for offline use
                v1.10.0/
                v1.9.0/
        inventory/                    # §2.1
            __init__.py               # registry of per-modality scanners
            types.py                  # InventoryRow
            mri_dicom.py              # parallel DICOM scan
            eeg_meg_raw.py            # mne probe
            physio.py                 # physio DICOM detection
        classifier/                   # §4
            __init__.py
            chain.py                  # runs layers in order
            dcm2niix_bidsguess.py
            mne_channel_types.py
            sequence_dict.py          # legacy regex dictionary, last layer
            types.py                  # Classification
        planner/                      # §5
            __init__.py
            plan.py                   # EntityPlan + state machine
            edits.py                  # edit operations the GUI calls
            bulk.py                   # bulk apply + cross-row validation
        converter/                    # §7
            __init__.py
            registry.py               # backend registration
            backends/
                dcm2niix_direct.py    # primary
                dcm2bids.py           # optional plugin
                heudiconv.py          # optional plugin
                mne_bids.py
                bidsphysio.py
                passthrough.py
            orchestrator.py           # groups plans by backend, runs in parallel
        metadata/                     # post-conversion sidecar engine
            __init__.py
            sidecar_audit.py
            sidecar_fill.py
            participants.py
            scans.py
            readme.py
            intended_for.py
        fixups/                       # post-conversion file fixups
            fieldmaps.py
            derivatives.py
        project/                      # §9
            __init__.py
            events.py
            store.py
            replay.py
            provenance.py             # §10
        editor/                       # post-conv editor (BIDS browser logic)
            __init__.py
            validator.py
            sidecar_view.py
            nifti_view.py
            surface_view.py
        gui/                          # the only Qt-coupled subtree
            __init__.py
            theme.qss                 # token-based stylesheet (from prototype)
            theme_manager.py          # palette swapping at runtime
            main_window.py
            top_header.py             # view switcher + theme toggle
            converter_panel.py        # §8 layout — Inspector view
            editor_panel.py           # validation-aware BIDS browser + viewer
            widgets/
                inventory_table.py    # QTableView + QAbstractTableModel
                entity_form.py        # schema-generated form
                modality_tree.py      # tri-state checkbox filter
                preview_tree.py
                sidecar_form.py       # schema-generated sidecar editor
                validation_badge.py
                path_bar.py
                chip.py
            delegates/
                entity_delegate.py    # per-entity widget per schema format
                badge_delegate.py     # status circle painter
                cell_delegate.py      # row-aware text painter
            models/
                inventory_model.py
        workers/                      # QThread bridges (no Qt logic in core)
            scan_worker.py
            classify_worker.py
            convert_worker.py
            validate_worker.py
        cli/
            __init__.py
            scan.py
            classify.py
            plan.py
            convert.py
            validate.py
    tests/
        unit/                         # core logic, no Qt
        integration/                  # synthetic fixtures, end-to-end
        real_data/                    # gated on env vars, against /MRI /EEG /MEG
        gui/                          # pytest-qt smoke tests
        fixtures/
```

The architectural rules (the v1.0-rewrite-failure prevention guards):

1. **`schema/` is the keystone.** Everything imports from it; it imports
   nothing from this codebase.
2. **`gui/` is the only Qt-coupled subtree.** Nothing else imports
   `PyQt6`. Workers (`workers/`) bridge GUI signals to core, never the
   other way.
3. **No `Pipeline` orchestrator.** Orchestration is explicit code in
   `cli/<verb>.py` and `gui/converter_panel.py`. (The v1.0 rewrite's
   "phases" hierarchy is on the do-not-port list.)
4. **No subpackage `core/`.** That name is poisoned by the v1.0
   post-mortem. The closest equivalent is `schema/` + `planner/`.
5. **Pure-data types.** `InventoryRow`, `EntityPlan`, etc. are dataclasses
   with no methods that touch I/O.
6. **Functions, not classes**, where possible. Classes only for stateful
   things (worker threads, GUI widgets, plugin registries).

---

## 13. What this lets us do that current tools don't

| Capability | heudiconv | dcm2bids | BIDScoin | ezBIDS | bids-validator | This design |
|---|---|---|---|---|---|---|
| Schema-driven naming | ✗ | ✗ | partial (YAML) | partial | ✗ (read-only) | **✓** |
| Schema-driven sidecar fill | ✗ | ✗ | ✗ | partial | ✗ | **✓** |
| Real-time entity validation | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Entity-template GUI form | ✗ | ✗ | ✗ | partial | ✗ | **✓** |
| Multi-modality (MRI + EEG + MEG) | ✗ | ✗ | ✗ | partial | ✓ | **✓** |
| Per-row backend selection | ✗ | ✗ | ✗ | ✗ | n/a | **✓** |
| Event-sourced project + undo | ✗ | ✗ | ✗ | partial | ✗ | **✓** |
| Provenance for every value | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Live validation in editor | ✗ | ✗ | ✗ | ✗ | per-run | **✓** |
| Schema-versioned per project | ✗ | ✗ | ✗ | ✗ | ✓ | **✓** |
| Pluggable converter backends | n/a | n/a | ✗ | ✗ | n/a | **✓** |
| Bulk edits across rows | partial | ✗ | partial | partial | ✗ | **✓** |
| Scales to 50k-row inventories | n/a | n/a | ✗ | unknown | ✓ | **✓** (via QTableView) |

The "✓" column isn't a list of features; it's a list of consequences of
the single bet in §0.

---

## 14. What this design *does not* commit to

To be honest about the boundaries:

- **No machine learning.** Adding an ML classifier as a fourth layer is
  trivial in §4 (it's just another `classify(row) -> Classification[]`),
  but ML isn't load-bearing for the design.
- **No web frontend.** Desktop Qt only. Web frontend would mean a
  re-wrap of the schema engine + a separate UI; not on the table.
- **No remote storage.** Local filesystems only. Cloud raw data (S3,
  XNAT) is roadmap, not core.
- **No real-time multi-user editing.** Project file is single-writer.
  Concurrent edit detection on save, no live collaboration.
- **No converter rewrites.** dcm2niix, mne, mne-bids, bidsphysio are
  treated as black boxes. We don't try to outdo them; we orchestrate
  them.
- **No "convert without review."** The tool always renders the plan and
  requires explicit user confirmation before commit. (Headless CLI
  conversion is supported but `cli/convert.py` runs the same validation
  and refuses to run with errors.)

---

## 15. Architectural decisions (resolved + open)

Owner-confirmed decisions are marked **[resolved]**; everything else is
still on the table.

1. **Schema source — [resolved]: `bidsschematools`.** Upstream, canonical,
   pip-installable. `ancpbids` stays as a dep for in-memory dataset
   navigation in the editor (it's a graph reader). They're complementary:
   `bidsschematools` gives us the rules engine, `ancpbids` gives us the
   read-side BIDS object model.
2. **Project file format.** JSON event log first (simplest, debuggable),
   migrate to SQLite at the first project that hurts. Or SQLite from day
   one. My pick: JSON first.
3. **Pydantic vs `@dataclass`.** Pydantic gives schema validation for the
   types in §2 for free, at a 50ms-import-cost. My pick: Pydantic v2.
4. **Schema upgrades.** Auto-update the bundled schema on package
   upgrade, or pin per project? My pick: pin per project, prompt on
   open if a newer schema is available.
5. **Backend for MRI by default — [resolved]: `dcm2niix` direct.** No
   `dcm2bids`/`heudiconv` wrapper layer. The schema engine builds the
   filename (`schema.build_basename`); we invoke `dcm2niix` with `-f` set
   to that exact basename and `-o` set to the right datatype directory.
   Full control over filename, sidecar, BVAL/BVEC. The dcm2bids and
   heudiconv backends from `BIDS-Manager/` get ported as **optional
   plugins** for users who already have configurations they want to keep,
   but they are not the default path.
6. **Qt6 binding — [resolved]: PyQt6.** We already migrated; the editor's
   volume/surface viewers depend on it; the only argument for PySide6 is
   licensing, which doesn't matter here.
7. **Plugin discovery.** Entry-points in `pyproject.toml` (third-party
   plugins possible) vs in-tree registry only? My pick: in-tree registry
   for v1; entry-points later.
8. **Dataset-level provenance.** Embed in `dataset_description.json`
   `GeneratedBy` (BIDS-blessed), or a separate `.bidsmgr/provenance.json`?
   My pick: both — `GeneratedBy` for BIDS consumers, our own file for
   the per-cell audit trail.
9. **Build / packaging — [resolved]: PEP 621 `pyproject.toml` only.** No
   `setup.py`, no `setup.cfg`. Build backend is `setuptools>=61` (the
   default that already works for `BIDS-Manager/`).
10. **GUI layout — [resolved]: Proposal 1 (Inspector).** See
    `gui_mockups.html`. Three-pane Converter (modality/filter tree with
    tri-state checkboxes, inspection table, schema-driven properties
    panel) + tabbed bottom dock (BIDS preview / log / conflicts /
    stats); Editor as a second top-level tab with the same three-pane
    shape (validation-aware BIDS browser, type-routed file viewer,
    file/folder/dataset validation panel). Curator-style split-screen
    detail (proposal 5) may be added later as an opt-in drill-in for
    "explain this row" but is not v1 scope.

---

## 16. Two-line summary

The single design bet is "the BIDS schema is the engine," and that bet
collapses every UX surface — classification, naming, GUI form generation,
validation, sidecar generation, editor highlighting — onto one rules
source. Everything else (event-sourced project file, plugin backends,
provenance everywhere, virtualised table model, three-pane inspector
GUI) is downstream of that bet.

If you sign off on §0 (schema-as-engine), the rest of this document is
deterministic. If you don't, the design is up for grabs and we should
talk before §12.
