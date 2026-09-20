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
one local addition**, so the copy reports `0.1.1+bidsmgr.1`.

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

**What changed during the copy: THREE files.** Each is marked
`VENDORED CHANGE` in the source. Keep this list in step, because the refresh
policy below is "wholesale", and a divergence that is not written down here is
one that gets silently deleted the next time.

| File | Kind | Change |
|---|---|---|
| `schema/resolve.py` | adaptation | `_bundled_dir()` uses `resources.files(__package__)`, was `resources.files("bidsval.schema")` |
| `__init__.py` | adaptation | `__version__` stated literally, was `importlib.metadata.version("bidsval")` |
| `schema/fields.py` | **ahead** | `FieldSpec.accepts` / `accepts_free_text` / `accepts_na`, the `_variants` helpers, and an `anyOf` lookup in `_item_type`. Not in upstream at all |

1. **The absolute package name.** A package that is going to be copied cannot
   name itself absolutely, because the absolute name reaches whatever is
   INSTALLED rather than the copy. Here the package is
   `bidsmgr.vendor.bidsval.schema`, so `resources.files("bidsval.schema")`
   read the installed package's schema files on a developer machine with
   bidsval pip-installed, and raised `ModuleNotFoundError` on a machine
   without it, before the GUI could start.
2. **The version lookup.** There is no `bidsval` distribution here, so
   `version("bidsval")` described the installed copy or fell back to
   `"0.0.0"`. Stated literally now, as `0.1.1+bidsmgr.1`. The local segment
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
   `bidsmgr.metadata.engine._todo_value_for` read "no item type", assumed
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
4. Otherwise no behavioural changes. Every other function and class
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
