"""Build the per-file context the rule engine evaluates against.

A *context* is a mapping of the names a BIDS schema expression may reference
(``entities``, ``datatype``, ``suffix``, ``sidecar``, ``nifti_header`` ...) to
their values for one file. Its shape follows the schema's own ``meta.context``
definition, so selectors and checks evaluate against exactly what the schema
expects.

:class:`~bidsval.context.builder.ContextBuilder` assembles it: parse the
filename, find the datatype, merge the inheritance-principle sidecars, and
lazily load file content (JSON, TSV columns, NIfTI headers).
"""

from __future__ import annotations

from .builder import ContextBuilder

__all__ = ["ContextBuilder"]
