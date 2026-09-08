# How metadata flows through BIDS Manager

What the metadata template is, where an answer goes when you type it, and which
step writes it into a file. Read this before changing anything under
`metadata/`, `recording_meta/`, `fixups/` or the two metadata forms.

---

## 1. The one-page version

```mermaid
flowchart LR
    RAW[Raw data] --> SCAN[bidsmgr-scan]
    SCAN --> INV[(inventory.tsv)]
    SCAN --> SCAF[(inventory.tsv.recording_meta.json<br/>the scaffold)]
    SCAF -.->|what the conversion<br/>already answers| FORM
    INV --> FORM[Metadata template<br/>dataset dialog + properties panel]
    FORM -->|your answers| SCAF
    INV --> CONV[bidsmgr-convert]
    CONV --> BIDS[(BIDS tree)]
    SCAF --> META[bidsmgr-metadata]
    BIDS --> META
    META --> BIDS
    BIDS --> VAL[bidsmgr-validate]
```

Three files matter. The **inventory** is the table of what was found. The
**scaffold** beside it holds everything you state that is not a column in that
table. The **BIDS tree** is the output. The template reads the first two and
`bidsmgr-metadata` writes the third.

**Conversion never applies your answers.** `bidsmgr-convert` produces a faithful
conversion and repairs only what the converter itself got wrong.
`bidsmgr-metadata` applies what you stated. The GUI runs metadata immediately
after convert, so from the Run button it is one step.

---

## 2. A section speaks for every file of its kind

This is the thing to understand before anything else. A section headed

> **every `*_bold.json`**  ·  in `func/`  ·  61 files  ·  e.g. `sub-003_task-bernd_bold.json`

is not a form about `sub-003`. It is the statement you are making about all 61
functional runs in the dataset. Answer it once and it is written to every one of
them.

The real filename is shown only as an example of what the section covers. To say
something about ONE recording, select its row in the inspection table and use the
properties panel; that answer overrides this one for that recording alone.

## 3. What the template asks, and why it is not everything

```mermaid
flowchart TD
    S[The BIDS schema at your chosen version] -->|every field this file may carry| A
    A[Fields declared for datatype/suffix] --> B{Does the conversion<br/>already answer it?}
    B -->|yes, on YOUR data| SHOWN[Folded away, with the value:<br/>Already answered by the conversion<br/>still editable, to correct the data]
    B -->|no| C{Does the rule depend on<br/>a scenario this file<br/>may not be in?}
    C -->|yes, speculative| DROP[Not asked]
    C -->|no| ASK[Asked, at the schema's<br/>requirement level]
```

Nothing in that diagram is a list kept in code. The fields, their types, their
vocabularies, their requirement levels and their descriptions all come from the
schema, at the version chosen in **Settings, BIDS version**.

A field the conversion answers is not hidden. It moves into a folded, green
**Already answered by the conversion** block at the TOP of the section, showing
the value that will be written, and it stays editable: the converter reads these
out of the data, and when the data is wrong or a legacy file carries nothing,
that block is the only place to correct it. Typing nothing there stores nothing,
so opening it and closing it again changes no file.

### What counts as "the conversion answers it"

Measured on your data, during the scan:

* **DICOM**: a probe conversion runs dcm2niix once per series and reads the
  sidecar it produced.
* **EEG / MEG**: the scanner opens every recording anyway, and asks it the same
  questions mne-bids will ask: channel counts by type, sampling frequency,
  duration, recording type, and the line frequency and manufacturer where the
  header states them.
* **The row itself**: an entity already on the row settles a field. A task label
  answers `TaskName`, because the converter writes whatever the row says.

Each recording gets its own measurement as well as its kind's, because "what
every `*_bold.json` answers" and "what THIS run answers" are different
questions. The per-file form asks the second one, so a functional run opens
showing its actual repetition time, echo time and slice timing rather than
asking for them.

A field only some files of a kind answer reads as `VARIES` rather than being
asked. Requiring every file to answer was the first rule and it was too strict:
seventy-five functional runs from several studies agree on fourteen fields and
no more, so the form asked for thirty others that dcm2niix reads out of the
header for almost all of them. Nothing is lost, because the block is editable
and a value stated there covers the recordings whose header lacked one.

Values arrive as text from all three, and from every box a user types into. Each
one is put into the shape the schema declares before it is written, so the mains
frequency reaches the sidecar as the number 50 and not as the string "50", which
BIDS rejects.

Two rules keep this honest:

1. **A value of `n/a` is not an answer.** BIDS requires `EEGReference` in every
   EEG sidecar, so mne-bids writes `"n/a"` when it does not know. Counting that
   as answered is what once hid reference, ground and filters from the form.
2. **A field only some files answer is still asked.** Half a dataset answered is
   not answered, and the recordings that lack it need a way to get one.

If a scan could not measure (no probe, no readable recording) a small built-in
list is used as a fallback. It is a fallback, not the authority: it was measured
once, on one tree, with one scanner.

---

## 4. Where an answer goes when you type it

```mermaid
flowchart TD
    subgraph Forms
      DLG[Dataset metadata dialog]
      PANEL[Properties panel<br/>one selected row]
    end
    DLG -->|agnostic section| DD[spec.dataset_description]
    DLG -->|a file's section| ST["spec.sequence_templates<br/>datatype/suffix"]
    PANEL -->|a field the table<br/>also carries| CELL[(inventory column)]
    PANEL -->|any other field| RT["spec.row_templates<br/>row id"]
    DD & ST & RT --> SCAF[(the scaffold)]
```

Both forms are rendered by the same code from the same definition
(`gui/widgets/template_form.py`), so a field means the same thing in both. The
dialog states what is true of every `*_eeg.json`; the panel states what is true
of one recording.

A few fields are inventory COLUMNS as well as BIDS fields, because you need them
in the table to sort and bulk-edit: `line_freq` is `PowerLineFrequency`,
`eeg_reference` is `EEGReference`, `eeg_ground` is `EEGGround`. Answering them
in a form writes the column, so the table and the form can never disagree.

**One thing is not from the schema, and it is the only one: `montage`.** It
names an MNE electrode layout to APPLY during conversion so that
`electrodes.tsv` and `coordsystem.json` get written. It is an instruction to the
converter, not a fact about the recording, which is why BIDS has no field for it.

---

## 5. Which answer wins

Every layer is resolved by one function, `recording_meta/chain.py`, weakest
first:

```mermaid
flowchart LR
    D[dataset<br/>the site, shared by all] --> M[modality<br/>this instrument]
    M --> T["template<br/>datatype/suffix"]
    T --> TT["template@task"]
    TT --> R[row<br/>this recording]
    R --> C[cell<br/>this row in the table]
```

Later beats earlier, and all of it beats what the converter wrote.

Nothing in that chain is a guess. Every layer of it is somebody having typed an
answer into a form, and the form only offers a field when it is worth asking
about. So a value you state replaces what the conversion put there, which is the
whole reason the "already answered by the conversion" block is editable: opening
it and correcting the manufacturer is you saying the header is wrong.

Everything nobody stated is left exactly as the conversion wrote it, which is
almost all of it.

`VARIES` is a fourth state, beside a value, blank, and not-applicable. It says
"this differs per recording, the answer lives further down". It is never itself
written to a sidecar.

---

## 6. One project at a time

Everything a project states lives inside that project. The scaffold sits beside
the inventory, and in project mode the inventory is inside
`<dataset>/.bidsmgr/project/scans/<version>/`, so two projects can never read
each other's answers.

One thing did leak, and it is worth knowing why. The steps that run after a
conversion are handed the PARENT of the dataset, because that is where convert
puts each one, and the metadata engine walks every BIDS root it finds under its
target. Projects are normally kept side by side, so that parent is the home of
all of them: a dataset name typed into one project's template was written into
every neighbour.

The post-convert steps now name the datasets they are allowed to touch, which in
project mode is exactly the project. A deliberate `bidsmgr-metadata <folder>`
over a folder of datasets still touches all of them, because that is what the
command means when the caller does not say otherwise.

What is deliberately shared between projects, because it is a preference rather
than data: the theme, the BIDS version, the scan and convert defaults, the scan
rules, and the validation settings. They live in the application's settings and
apply to whatever you open next.

---

## 7. Who writes what

```mermaid
flowchart TD
    subgraph convert [bidsmgr-convert]
      C1[dcm2niix / mne-bids / bidsphysio]
      C2[fieldmap renames, IntendedFor, scans.tsv]
      C3[channels.tsv retyping, event labels, task protocol]
      C4[repair_converter_output:<br/>key casing, scalar-where-array]
    end
    subgraph metadata [bidsmgr-metadata]
      M1[apply_stated_metadata:<br/>templates, row answers, table cells,<br/>the dataset-wide block]
      M2[dataset_description.json, README, CHANGES]
      M3[participants.tsv, phenotype tables]
      M4[fill TODO placeholders, audit]
    end
    convert --> metadata
```

The line between them: **convert fixes what the converter wrote; metadata
applies what you stated.** That is why `bidsmgr-convert` alone gives you a
conversion with no opinions in it, and why running `bidsmgr-metadata` is what
puts your template into the files.

---

## 8. The BIDS version

One control, in **Settings, BIDS version**, and one flag, `--schema`, on every
CLI verb. It governs:

| | at 1.8.0 | at 1.11.1 |
|---|---|---|
| fields declared for `anat/T1w` | 64 | 76 |
| questions the template asks for `anat/T1w` | 31 | 36 |
| entities `func/bold` may carry | 10 | 11 |
| datatypes that exist | 12 | 16 |
| `BIDSVersion` stamped into `dataset_description.json` | 1.8.0 | 1.11.1 |
| the version validation judges by | 1.8.0 | 1.11.1 |

Set once, it holds for the whole session: forms, filenames, enrichment and
validation. Several versions ship with BIDS Manager; the default is the newest.

---

## 9. Reading the form

| mark | meaning |
|---|---|
| `*` red | the standard REQUIRES this field for this file |
| `·` amber | the standard RECOMMENDS it |
| unmarked | optional |
| greyed value | inherited, from the layer named in the tooltip |
| **Already answered by the conversion** | folded green block at the TOP of a section: what is settled, with the value that will be written; editable, to correct what the data says |
| `N to answer, M required by BIDS` | the section heading: what is still missing, which is the point of the form |
| `differs per recording` | the probed files disagreed; the answer is per recording |

Colour can be turned off in Settings; the marks remain, so the information does
not depend on being able to see colour.

---

## 10. Looking at it

Layout bugs are invisible to a test suite. A pane can pass every assertion about
its widths and still show a field with no box beside it, a column that does not
line up, or a name cut mid-word. All three of those shipped.

```bash
QT_QPA_PLATFORM=offscreen python tools/screenshot_gui.py dark
QT_QPA_PLATFORM=offscreen python tools/screenshot_gui.py light
```

Qt's offscreen platform draws into memory and `QWidget.grab()` returns the
pixels, so this needs no display. It writes the metadata dialog and the
properties panel at several widths, opened and scrolled. Look at them before
believing a layout change worked, and look at both themes: a colour that reads
on the near-black surface can vanish on the light one.

---

## 11. Where the code is

| you want | look in |
|---|---|
| which fields a file may carry | `bidsmgr/schema/` (a thin adapter over `bidsval.schema`) |
| what to ask, arranged as a tree | `bidsmgr/metadata/template_plan.py` |
| what the conversion answers by itself | `bidsmgr/metadata/converter_preview.py`, fallback in `derivable.py` |
| a field becomes a widget | `bidsmgr/gui/widgets/template_form.py` |
| which layer wins | `bidsmgr/recording_meta/chain.py` |
| answers reach the files | `bidsmgr/fixups/sidecar_schema.py`, called by `bidsmgr/metadata/engine.py` |
| which BIDS version is in force | `bidsmgr/schema/loader.py` |
