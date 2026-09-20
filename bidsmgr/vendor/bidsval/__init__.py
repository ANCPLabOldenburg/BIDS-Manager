"""bidsval - a schema-driven, pydantic-typed, in-process BIDS validator.

The public surface grows as the validator does. Today it exposes the two pieces
that the rest of the engine is built on:

* the schema resolver (:func:`bidsval.schema.resolve`), the single place that
  turns a schema selector into one in-memory schema object, and
* the expression evaluator (:func:`bidsval.expr.evaluate_string`), which runs a
  BIDS schema expression against a context.

Result types (:class:`~bidsval.issues.Issue`, :class:`~bidsval.report.ValidationReport`)
are re-exported here so consumers can ``from bidsval import Issue, ValidationReport``.
"""

from __future__ import annotations

from .expr import evaluate_string
from .issues import DatasetIssues, Issue, Severity
from .report import FileVerdict, ValidationReport
from .schema import available_versions, bids_version, resolve, schema_version
from .validate import validate, validate_file, validate_subject

# VENDORED CHANGE. Upstream reads this from installed package metadata with
# ``version("bidsval")``. There is no such distribution here: this tree is a
# copy inside BIDS Manager. That lookup reported the version of a SEPARATELY
# INSTALLED bidsval when the developer happened to have one, and "0.0.0"
# otherwise, so the number never described the code actually running.
#
# The local segment is not decoration. This copy is bidsval 0.1.1 PLUS an
# addition that upstream does not have (the anyOf-aware field typing in
# schema/fields.py), so calling it plain "0.1.1" would name a released
# artefact that does not contain this code. `+bidsmgr.N` is PEP 440's local
# version identifier and says exactly that: 0.1.1 with local changes on top.
#
# Bump the local segment when the local delta changes; drop it entirely once
# upstream carries the addition and this is a clean copy again. See the
# divergence table in ``bidsmgr/vendor/README.md``.
__version__ = "0.1.1+bidsmgr.1"

__all__ = [
    "Severity",
    "Issue",
    "DatasetIssues",
    "FileVerdict",
    "ValidationReport",
    "resolve",
    "available_versions",
    "schema_version",
    "bids_version",
    "evaluate_string",
    "validate",
    "validate_subject",
    "validate_file",
    "__version__",
]
