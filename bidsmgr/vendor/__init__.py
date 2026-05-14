"""Third-party code vendored into bidsmgr.

Each sub-package here is a verbatim (or near-verbatim) copy of an
upstream project, kept inside bidsmgr to avoid the installation
fragility of a transitive dependency. Vendored code keeps its
original license header per file; the umbrella license terms for
each vendored project live in ``<subpkg>/LICENSE``.

Currently vendored:

* ``bidsmgr.vendor.bidsphysio``  - MIT, originally
  ``bidsphysio`` by Pablo Velasco and Chrysa Papadaniil
  (NYU Center for Brain Imaging). Vendored to drop the
  ``pkg_resources.declare_namespace`` runtime requirement that
  forced ``setuptools<81`` on every install. See
  ``bidsmgr/vendor/README.md`` for the full vendoring story.
"""
