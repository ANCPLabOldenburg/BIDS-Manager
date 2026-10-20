# Vendored third-party code

This directory holds verbatim copies of upstream Python packages that
bidsmgr needs to ship. Each vendored sub-package keeps the original
per-file copyright header and is governed by the license stored at
`bidsmgr/vendor/<subpkg>/LICENSE`.

## Why vendor at all

Most of bidsmgr's dependencies are pulled in by `pip install` and
upgraded freely. Vendoring is reserved for packages where the
external install path is fragile and the in-tree alternative is
clean. The current motivation list:

* **Unmaintained upstream.** A pinned dependency is stuck on an old
  Python API that is being removed by the standard library or by
  pip itself. We need the code today and have no way to push an
  upstream fix.
* **Transitive constraint pollution.** A direct dependency forces
  us to cap an otherwise unrelated package (for example,
  `setuptools<81`) and the cap reaches every install.
* **Small surface, well-defined behaviour.** The vendored code is
  small enough to read end to end and stable enough that we are
  comfortable owning future maintenance.
* **We are the upstream, and the round trip is the cost.** A package
  written for this tool whose fixes cannot be tested from this tool
  without publishing a release first.

## Currently vendored

### `bidsmgr.vendor.bidsval`

**Upstream:** `bidsval` (`karellopez/bidsval`, MIT), vendored at **0.1.1 plus
local additions**, so the copy reports `0.1.1+bidsmgr.3`.

"Ahead of upstream" applies to exactly one of the three differences below.
The other two are ADAPTATIONS that vendoring requires and that would be wrong
upstream: a package that names itself absolutely cannot be copied, and a copy
has no installed distribution to read a version from. Only the `schema/fields.py`
addition is functionality bidsval does not have, and that is the one that
should eventually go upstream rather than live here.

**Why:** it is BIDS Manager's own package and the single implementation of two
things the tool rests on: validation, and the interpretation of the BIDS
schema. `bidsmgr/schema/` is a thin adapter over `bidsval.schema` (CLAUDE.md
guard 8) and `bidsmgr/editor/validator.py` a thin adapter over
`bidsval.validate`, precisely so neither fact is derived twice. Depending on it
through PyPI made that arrangement expensive: changing a schema rule meant
editing bidsval, bumping, building, uploading, and only then testing. In tree
it is one edit and one test run.

**What changed during the copy: the files below.** Each is marked
`VENDORED CHANGE` in the source. Keep this list in step, because the refresh
policy below is "wholesale", and a divergence that is not written down here is
one that gets silently deleted the next time.

| File | Kind | Change |
|---|---|---|
| `schema/resolve.py` | adaptation | `_bundled_dir()` uses `resources.files(__package__)`, was `resources.files("bidsval.schema")` |
| `__init__.py` | adaptation | `__version__` stated literally, was `importlib.metadata.version("bidsval")` |
| `schema/fields.py` | **ahead** | `FieldSpec.accepts` / `accepts_free_text` / `accepts_na` / `format`, the `_variants` helpers, `_string_format`, and an `anyOf` lookup in `_item_type`. A string with a `format` whose pattern refuses an arbitrary word (`time`, `bids_uri`, `hed_version`) is not free text, so the metadata step writes no `"TODO"` there. Not in upstream at all |
| `rules/filenames.py` | **ahead** | `match_file_rule` and `_fits`: the raw file rule a path fits, strictly (extension, folder, every entity, required entities, fixed values). BIDS Manager's scans and sessions tables and the coherence checks ask it which files are recordings (`metadata/tables.py`), so there is one answer, the standard's. Also `.bval`/`.bvec` are inheritable (a dataset-root `dwi.bval` was INVALID_LOCATION) |
| `rules/values.py` | **ahead** | JSON `format` checked against `objects.formats` patterns (`"TODO"` in HEDVersion or a time, `"9h28"`, were passed) |
| `rules/engine.py` | **ahead** | `json` evaluated (every `coordsystem.json` rule); `_NULL_IS_FAILURE` (eye tracking without StimulusPresentation is reported, as the reference does); on a derivative dataset, raw-only sidecar rules not enforced (as the reference 3.0.2); rules flattened once per schema (speed) |
| `rules/tables.py` | **ahead** | gzipped TSVs column-checked (named from the sidecar's `Columns`); `TSV_COLUMN_TYPE_REDEFINED` on the `.json` that redefines the column |
| `context/loaders.py` | **ahead** | `load_columns(names=)` for headerless tables; a byte-order mark is `INVALID_JSON_ENCODING` |
| `context/builder.py` | **ahead** | `bids::` URIs into the dataset are resolved (the reference answers true for any `bids:` string, so a dangling IntendedFor passed both); headerless tables read with the sidecar's `Columns` |
| `context/entities.py` | speed | `parse_filename` memoised per schema (4.5 million parses on a 1,500-file folder) |
| `files/tree.py` | speed | `BIDSFile.name` from the relpath; per-folder lists sorted once |
| `rules/integrity.py` | **ahead** | the `INVALID_JSON_ENCODING` message |
| `validate.py` | **ahead** | no bespoke `NO_SUBJECTS` (the schema's `SUBJECT_FOLDERS` says it, scoped by DatasetType); a file no rule names (`NOT_INCLUDED`) is not given sidecar rules; `validate_file` honours `.bidsignore` (an ignored file validated alone was reported as not BIDS) |
| `files/bidsignore.py` | **ahead** | `compile_pattern` public and `BidsIgnore.deciding` names the line that hides a path, so the Editor's ignore list and tree use THIS matcher (it had one of its own that disagreed); `[...]` character classes read as gitignore reads them; a file inside an ignored folder stays ignored whatever a later `!` line says (gitignore's rule, and the reference validator's `ignore` package), folder answers cached |

Measured against the reference validator (3.0.2, same schema) on the 61
single-defect copies of a BIDS Manager dataset from the 1.5.1 audit: 55
defects caught, as the reference (52 before). Not caught: HED (needs
`hedtools`). Caught where the reference is not: a dangling `bids::`
IntendedFor. 1,000 files in one folder: 64.6 s before, 17.7 s after, the same
30,005 findings.

1. **The absolute package name.** A package that is going to be copied cannot
   name itself absolutely, because the absolute name reaches whatever is
   INSTALLED rather than the copy. Here the package is
   `bidsmgr.vendor.bidsval.schema`, so `resources.files("bidsval.schema")`
   read the installed package's schema files on a developer machine with
   bidsval pip-installed, and raised `ModuleNotFoundError` on a machine
   without it, before the GUI could start.
2. **The version lookup.** There is no `bidsval` distribution here, so
   `version("bidsval")` described the installed copy or fell back to
   `"0.0.0"`. Stated literally now, as `0.1.1+bidsmgr.3`. The local segment
   is PEP 440's local version identifier and is load-bearing: plain `0.1.1`
   would name a PyPI artefact that does not contain change 3 below. Bump the
   local segment when the local delta changes, and drop it once upstream
   carries the addition.
3. **`anyOf`-aware field typing.** `FieldSpec` gained three fields saying what
   a field ACCEPTS rather than what its single `type` is, because the schema
   types a great many fields as `anyOf`: `PowerLineFrequency` is a number or
   the string `"n/a"`, `EchoTime` a number or an array of them, and `type` is
   empty for all of those. `_item_type` also had to resolve through `anyOf`,
   since the array branch carries its own `items`. Without it,
   `bidsmgr.metadata.engine.todo_value_for` read "no item type", assumed
   strings, and wrote `["TODO"]` into `EchoTime`, turning a missing field into
   a validation error.

   This one is a local ADDITION, and it is the one to be careful with.
   "Which types does this field accept" is a fact about the standard, and
   CLAUDE.md guard 8 says those are interpreted once, in bidsval. It belongs
   upstream; until it gets there, re-apply it after every refresh.

Note that none of the first two is an import statement, which is why the
static check in `tests/unit/test_vendored_bidsval.py` did not see them: one
names the module in a string argument and the other in a metadata lookup.
That file now also blocks the installed package outright and does real work
through the copy, which is the only check that covers every way a module can
be named.

**The bundled schemas travel too.** `schema/bundled/*.json` (2.9 MB, six BIDS
versions) is the data the resolver resolves against. Copying the code without
it ships a validator with nothing to validate against; it is listed in
`pyproject.toml` under `[tool.setuptools.package-data]`, because package data
is opt-in and a correct tree otherwise still builds a broken wheel.

Dependencies are unchanged: `bidsschematools`, `pydantic`, `nibabel`,
`pandas`, `mne` and `pyyaml`, all already shipped.

**This copy is the version BIDS Manager maintains.** It is not kept in step
with anything: `bidsval` exists as a standalone repository and on PyPI, but
BIDS Manager neither depends on it nor tracks it, and changes are made HERE.
The table above therefore records history rather than a to-do list, and the
copy is free to move ahead of the standalone package.

There used to be a `tools/check_vendored_bidsval.py` that diffed this tree
against a checkout of that repository. It was removed once the copy became the
maintained version: with nothing to be in step with, every local change looked
like drift to be written down. What the copy must satisfy is now stated
entirely by `tests/unit/test_vendored_bidsval.py`, which asks the question that
still matters. Not "does this match somewhere else" but "is this complete and
self-sufficient": nothing reaches an installed `bidsval`, the imports are
relative, the schemas and licence travelled, the wheel carries them, and
validation runs with the standalone package absent.

`tests/unit/test_vendored_bidsval.py` also blocks the installed package and
does real work through the copy, so a lookup that reaches the wrong one fails
even on a machine that has both.


### `bidsmgr.vendor.bidsphysio`

**Upstream:** `bidsphysio` (Pablo Velasco, Chrysa Papadaniil; NYU
Center for Brain Imaging). MIT licensed.
Source: <https://github.com/cbinyu/bidsphysio>. Last upstream
release `21.6.24`, June 2021. Effectively abandoned.

**Why vendored:** the upstream package uses
`pkg_resources.declare_namespace(__name__)` in each sub-package's
`__init__.py`. setuptools 81 removed `pkg_resources.declare_namespace`,
so a bare `import bidsphysio` crashes on any environment with
setuptools 81+ (the default on Python 3.14 and modern installations
of older Pythons too). Forcing `setuptools<81` at the bidsmgr level
solved the symptom but constrained every user's environment for a
problem in one transitive dep. Vendoring lets us drop that cap
entirely.

**What changed during vendoring:**

1. Each sub-package's `__init__.py` lost its
   `__import__('pkg_resources').declare_namespace(__name__)` line.
   The vendored layout is a regular Python package, not a namespace
   package.
2. Cross-package imports were rewritten from absolute
   (`from bidsphysio.base.bidsphysio import ...`) to relative
   (`from ..base.bidsphysio import ...`) so the tree relocates
   cleanly under `bidsmgr.vendor`.
3. One performance fix in `base/bidsphysio.py`:
   `PhysioSignal.plug_missing_data` was rewritten from the upstream
   one-insertion-at-a-time loop (each step a full `np.concatenate` plus
   a re-scan from index 0, i.e. O(n^2)) to a vectorised single-pass
   build. The upstream version hangs for minutes on large CMRR logs
   (a real 22M-sample ECG with many gaps); the rewrite handles 1M
   samples / ~1M gaps in well under a second. Output is identical to
   the original for every input where the original actually fills gaps;
   it additionally fixes an upstream `np.argmax` edge bug that left a
   gap in the very first sampling interval unfilled. Covered by
   `tests/unit/test_bidsphysio_plug_missing_data.py`.
4. One correctness fix in `base/bidsphysio.py`, with a matching change
   to the `bids_label` call in `base/utils.py`: the recording label is
   now sanitised into a valid BIDS entity label before it goes into a
   filename. Upstream puts the SIGNAL'S OWN NAME straight in, and a
   source calls a channel whatever it likes: a Siemens PMU dump yields
   `external_trigger`, producing

       sub-001_task-x_recording-external_trigger_physio.tsv.gz

   which is not a BIDS name. `recording` is an entity whose value must
   match the standard's `label` format, and the underscore is not in it.
   Worse, the underscore is the entity SEPARATOR, so the name does not
   merely fail validation: every tool that parses BIDS names reads an
   entity `recording-external` followed by a stray token and sees a
   different file from the one that was written.

   New `bids_label` and `unique_bids_labels` produce
   `recording-externalTrigger`: characters the standard disallows are
   removed and the word boundary they marked is kept by capitalising
   what followed, so the label still reads as the words it came from.
   Names that clean up to the same label are numbered apart, because two
   recordings with one name means the second overwrites the first.

   Alphanumeric-only, deliberately. The `label` pattern differs by BIDS
   version: `[0-9a-zA-Z]+` through 1.10.0 and `[0-9a-zA-Z+]+` from
   1.10.1, where the plus sign means "several applicable labels" rather
   than being a separator. Alphanumeric is the one spelling valid under
   every version. `tests/unit/test_physio_labels.py` checks the produced
   labels against every schema the installed `bidsval` ships, so this
   claim is verified rather than asserted.

5. One correctness fix in `base/bidsphysio.py`, `save_bids_data`: a
   missing sample is written `n/a`, the standard's word for it. numpy's
   `savetxt` wrote `nan`, which a validator reads as a value of the wrong
   type (`TSV_VALUE_INCORRECT_TYPE` on the multimodal sample's cardiac
   channel). Written in chunks of 100,000 rows, so a 22-million-sample
   recording is never all text at once. The viewers read `n/a` as a gap,
   as they read `nan` (`tests/unit/test_physio_na.py`).

6. Otherwise no behavioural changes. Every other function and class
   body is verbatim. Original per-file MIT headers are preserved.

**What's in the tree:**

| Sub-package    | Purpose                                                       | Optional 3rd-party dep |
|---------------|--------------------------------------------------------------|------------------------|
| `base`         | `PhysioSignal`, `PhysioData` core classes + helpers         | none                   |
| `dcm2bids`     | Siemens CMRR Multiband physio DICOM to BIDS                 | none (pydicom)         |
| `acq2bids`     | BioPac AcqKnowledge `.acq` to BIDS                          | `bioread`              |
| `pmu2bids`     | Siemens PMU `.log` to BIDS                                  | none                   |
| `physio2bids`  | Generic dispatcher across acq / dcm / pmu                   | depends                |
| `edf2bids`     | EDF event-channel to BIDS events                            | `pyedfread`            |
| `events`       | Event-base classes shared by `edf2bids`                     | none                   |
| `session`      | Session-level (multi-recording) orchestration               | none                   |

bidsmgr's `bidsmgr.converter.backends.physio_dcm` currently uses
only `bidsmgr.vendor.bidsphysio.dcm2bids.dcm2bidsphysio.dcm2bids`.
The other sub-packages are vendored proactively so future bidsmgr
backends (BioPac, PMU, EDF) can wire up without re-doing the
vendoring exercise.

`bioread` and `pyedfread` stay out of bidsmgr's `[project.dependencies]`
because most users do not have BioPac or EDF physio data. If you
need those formats, `pip install bioread` or `pip install pyedfread`
separately.

**Maintenance policy:** keep the in-tree copy frozen unless an
upstream patch is genuinely worth chasing. If the upstream resumes
work, we can re-sync. If we extend the code, we keep it in-tree and
upstream is welcome to take the diff back. Either way the file
headers stay attributed to the original authors.


### `bidsmgr.vendor.dcm2niix_win`

**Upstream:** `dcm2niix` (Chris Rorden and contributors). BSD 2-Clause,
shipped unchanged as `LICENSE` beside the binary, as clause 2 requires of
any redistribution in binary form. Source:
<https://github.com/rordenlab/dcm2niix>, branch `development`, commit
`fda9c11`.

**What it is:** not source, unlike the other two. One compiled
`dcm2niix.exe` for Windows x86-64, built with MinGW-w64 GCC and a linker
stack reserve of exactly 16,777,216 bytes. The MinGW runtime is linked
statically: the executable imports only `KERNEL32`, `ADVAPI32` and the
Universal CRT, all present on any Windows that can run Python 3.10, so it
does not depend on a toolchain being installed.

**Why vendored:** the released Windows dcm2niix cannot convert MR
spectroscopy. It reserves 16,388,608 bytes of stack, the frames an MSVC
build emits for a Siemens `svs_se` series need more, and Windows kills the
process with `0xC00000FD` (`STATUS_STACK_OVERFLOW`) and an empty stderr.
Measured, how it was built and why a GCC build at the SAME reserve works:
`dcm2niix_win/PROVENANCE.md`.

**How it is used:** as a fallback only, never in preference to the wheel.
`classifier.dcm2niix_bidsguess.run_dcm2niix` runs the pinned wheel binary
first and retries with this one only on that exact exit code, so a genuine
conversion failure is reported as itself. It is a narrower build (JPEG 2000,
JPEG-LS, TurboJPEG, Jasper and Zstandard are off), which is why it is never
the first choice. `vendored_dcm2niix()` returns `None` off Windows and on
Windows for any architecture but x86-64.

**When to delete it:** when a released `dcm2niix` wheel reserves
16,777,216. The fix is submitted upstream from `karellopez/dcm2niix`, branch
`fix/windows-msvc-stack-spectroscopy`. Delete the directory, its
`package-data` entry in `pyproject.toml` and the fallback branch of
`run_dcm2niix` together.


### `bidsmgr.vendor.niivue_colormaps`

**Upstream:** NiiVue (`niivue/niivue`, BSD 2-Clause, shipped unchanged as
`LICENSE` beside the files). Copied from `packages/niivue/src/cmaps/*.json`
at commit `ee9aefc` (2026-08-18).

**What it is:** data, not code. 72 JSON files, each a colour map as control
points: `R`, `G`, `B`, `A` (0-255) at intensity indices `I` (0-255). The CT
maps also carry a suggested window in Hounsfield units (`min`, `max`).

**Why vendored:** the viewer library (`bidsmgr.viz`) offers the same colour
maps NiiVue does, so a figure or a scene described in NiiVue's terms ("hot",
"ct_bones", "batlow") means the same thing here. 72 small files are cheaper
to ship than a dependency, and NiiVue is a JavaScript package with no Python
distribution to depend on.

**What changed during the copy:** nothing. Two upstream files (`bcgwhw.json`,
`bcgwhw_dark.json`) list 64 positions in `I` but only 60 colours; the files
are kept as they are and `bidsmgr.viz.compute.colormaps.lut` uses the
positions that have a colour. One (`afni_blues_inv.json`) carries a
placeholder window of `(0, 0)`, which `suggested_window` does not offer.

**How it is used:** `bidsmgr.viz.compute.colormaps` reads the files through
`importlib.resources` relative to this package, interpolates each to a
256-entry table once and caches it. The `pyproject.toml` package-data entry
is what puts them in the wheel; without it every map is missing.

**Maintenance policy:** refresh wholesale from a newer NiiVue commit when a
map we want is added; record the commit here.
