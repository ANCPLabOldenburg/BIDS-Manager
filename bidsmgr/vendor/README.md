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

## Currently vendored

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
3. No behavioural changes. Every function and class body is
   verbatim. Original per-file MIT headers are preserved.

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
