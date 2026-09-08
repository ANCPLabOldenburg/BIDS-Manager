<h1 align="center">
  <br>
  <img src="miscellaneous/images/app-icon.png" alt="" width="132">
  <br>
  BIDS Manager
  <br>
</h1>

<h3 align="center">Making raw-to-BIDS less painful.</h3>

<p align="center">
  <a href="https://pypi.org/project/bids-manager/"><img alt="PyPI" src="https://img.shields.io/pypi/v/bids-manager?color=4FC3F7&label=PyPI"></a>
  <a href="https://pypi.org/project/bids-manager/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/bids-manager?color=4FC3F7"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/pypi/l/bids-manager?color=4FC3F7"></a>
  <a href="https://ancplaboldenburg.github.io/bids_manager_documentation/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-4FC3F7"></a>
</p>

<p align="center">
  <img src="miscellaneous/hero.gif" alt="Reviewing an inventory, editing a sidecar, and inspecting a volume, in one window" width="100%">
</p>

## What is BIDS Manager?

BIDS Manager is a desktop application that turns raw MRI, PET, EEG, MEG and
iEEG recordings, together with PET blood curves and the physiological traces
Siemens CMRR sequences write beside an MRI series, into a validated BIDS dataset.

It scans your raw data, shows you every conversion decision in a table you can
edit, runs the conversion, and then opens the result for metadata editing,
signal and volume inspection, and validation. One application, no scripts, no
hand-editing JSON.

The thing it is really built around is not the conversion. Converting a DICOM
series is a solved problem, and BIDS Manager uses the same engines everyone else
does: `dcm2niix` for MRI and PET DICOM, `mne-bids` for electrophysiology,
`nibabel` for ECAT, `bidsphysio` for physio, and `pet2bids` for blood curves and
the PET metadata no converter writes. What it adds is everything around them:
seeing what you have before anything is written, saying what the files cannot
say for themselves, and being told what is wrong in terms of the standard rather
than a stack trace.

## Get started

The quickest route is the one-click bootstrap installer for macOS, Linux and
Windows. It bundles a portable Python with every dependency and registers a
native desktop launcher, so no existing Python install is required. The
[install guide][install] walks through it.

With Python already set up, `pip install bids-manager` works too.

Launch the interface with `bidsmgr`. Prefer the command line? Seven verbs cover
the whole pipeline:

```
bidsmgr-create     scaffold a dataset and its project
bidsmgr-scan       walk a raw tree and build the inventory
bidsmgr-rebuild    rebuild BIDS names from edited entities
bidsmgr-convert    convert, routing each row to the right engine
bidsmgr-metadata   dataset_description, participants, phenotype
bidsmgr-validate   validate, with a report you can hand to a colleague
bidsmgr-project    list a project's saved scan versions
```

The [documentation][docs] has the full GUI walkthrough, a reference for every
flag, and a tutorial per modality with a sample dataset you can download and
work through.

[docs]:    https://ancplaboldenburg.github.io/bids_manager_documentation/
[install]: https://ancplaboldenburg.github.io/bids_manager_documentation/installation.html

## The workflow

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/workflow-dark.svg">
    <img src="docs/workflow.svg" alt="Raw MRI, PET, EEG, MEG, iEEG, physio and blood recordings enter at Scan, then Review, Convert, Enrich, Curate and Validate, producing one BIDS dataset with every step recorded in the project log" width="100%">
  </picture>
</p>

The features below follow those steps.

## Features

### 1. See what you have, before anything is written

Point the application at a folder of DICOM, EEG, MEG or PET recordings, or all
of them at once. It walks the tree, works out what each series is, and shows the
proposed BIDS name for every one in a table you can sort, filter and bulk-edit.

Formats are recognised by reading the file, not by its extension, so a Philips
export whose filename is a bare identifier still converts and a renamed ECAT is
still an ECAT. Anything it sets aside says why: a scanner report with no image
data in it, a localiser, a series it cannot classify.

<p align="center">
  <img src="miscellaneous/see.gif" alt="The inventory table: every series, its proposed BIDS name, and the properties panel" width="100%">
</p>

### 2. Convert with confidence

Change any cell before you commit. Subjects, sessions, tasks, runs: the BIDS
filename updates as you type, so what you see is what will be written.

Two recordings that would land on the same filename are caught before anything
is written. Genuine repeats are given a `run` number where the standard allows
one; where it does not, both are shown in red and the conversion refuses to
start rather than write one file over another.

Conversion runs per subject into a staging folder and is committed only when
that subject finishes, so a failure never leaves a half-converted tree.

<p align="center">
  <img src="miscellaneous/convert.gif" alt="Bulk-editing entities in the inventory and watching the predicted filenames update" width="100%">
</p>

### 3. Say what the files cannot say

Some things are simply not in the data. An EEG file has nowhere to record its
reference or its ground. A PET scanner records how it reconstructed an image but
not how much tracer went into the person, in what form, or when.

The metadata form is generated from the BIDS schema, so it asks exactly what the
standard declares for each kind of file, at its real requirement level, with the
standard's own description on hover. Answer once for the study, override for the
one recording that differs. What the conversion already worked out is folded
away, so you are only asked what nobody could answer for you.

<p align="center">
  <img src="miscellaneous/fix.gif" alt="A JSON sidecar as a schema-aware form, colour-coded by requirement level" width="100%">
</p>

### 4. Curate the dataset you converted

Every sidecar opens as a schema-aware form and every table as a spreadsheet, so
correcting a converted dataset does not mean editing JSON by hand. Edits are
undoable, and validation can be re-run against them without leaving the window.

Viewers come with it. Images open one plane at a time, as three planes sharing a
crosshair, or in a GPU renderer with clipping, lighting and colour-FA. A 4-D run
gains a time-series graph, and on PET its axis is real seconds taken from the
frame times, because PET frames are not evenly spaced and a frame index flattens
the part worth looking at.

EEG, MEG and iEEG open as an interactive signal viewer with channel filtering,
per-segment filtering, an in-application power spectrum and events overlaid from
the `events.tsv` beside them.

<p align="center">
  <img src="miscellaneous/inspect.gif" alt="The NIfTI viewer: sagittal, coronal and axial sharing one crosshair" width="100%">
</p>

### 5. Be told what is wrong, and where the rule came from

Validation is part of the same application, reads the same BIDS schema the
metadata form was built from, and runs on a dataset of any modality at once.

Every finding names the schema rule it comes from, so you can check the claim
rather than take it on trust, and carries the standard's suggested fix. The fix
button takes you to the field or the cell that needs the answer, not merely to
the file.

It also reports things most tools miss, because they are invisible one file at a
time: a perfectly named file sitting in a folder that is not a datatype, an
entity the standard does not allow for that kind of file, a sidecar left behind
next to no data file at all.

<p align="center">
  <img src="miscellaneous/validate.gif" alt="Validating a dataset: findings by scope, each naming its schema rule" width="100%">
</p>

### 6. Provenance built in

Every edit is recorded in the project, and every scan is kept as a version. Undo
a decision taken in a session weeks ago, or reopen last month's scan and convert
it again against the same answers. Curation is resumable rather than something
you redo from the raw files each time.

## Modalities

| | Read from | Converted by |
|---|---|---|
| **MRI** | DICOM | `dcm2niix` |
| **PET** | DICOM | `dcm2niix` |
| **PET** | ECAT7, detected by its header rather than a `.v` name | `nibabel` |
| **PET blood** | PMOD `.bld` | `pet2bids` |
| **EEG / iEEG** | EDF, BDF, BrainVision, EEGLAB, MEF, NWB | `mne-bids` |
| **EEG / iEEG** | Neuroscan, GDF, EGI and other non-BIDS formats | `mne-bids`, re-encoded to EDF |
| **MEG** | FIF, CTF `.ds`, KIT `.con`/`.sqd` | `mne-bids` |
| **Physio** | Siemens CMRR log, written beside an MRI series | `bidsphysio` (vendored) |

## Authors

**Karel López Vilaret** and **Jochem Rieger**, ANCP Lab,
Carl von Ossietzky Universität Oldenburg.

## License

[MIT](LICENSE).

Physio conversion code under `bidsmgr/vendor/bidsphysio/` is derived
from [`bidsphysio`](https://github.com/cbinyu/bidsphysio) by Pablo
Velasco and Chrysa Papadaniil (NYU Center for Brain Imaging), used
under the MIT License. See `bidsmgr/vendor/bidsphysio/LICENSE` and
`bidsmgr/vendor/README.md` for the full attribution and what
changed during vendoring.

## Citation

```
López Vilaret, K. M. and Rieger, J.
BIDS Manager (v1.2.6). 2026. https://github.com/ANCPLabOldenburg/BIDS-Manager
```

<p align="center">
  <a href="https://ancplaboldenburg.github.io/bids_manager_documentation/">Documentation</a>
  &middot;
  <a href="https://ancplaboldenburg.github.io/bids_manager_documentation/updates.html">What changed</a>
  &middot;
  <a href="https://github.com/ANCPLabOldenburg/BIDS-Manager/issues">Report a bug</a>
  &middot;
  <a href="https://github.com/ANCPLabOldenburg/BIDS-Manager/issues/new">Suggest a feature</a>
</p>
