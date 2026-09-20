"""The BIDS schema expression language: evaluation.

Parsing a schema expression into an abstract syntax tree is already solved by
``bidsschematools.expressions.parse``. What the ecosystem lacks - and what this
subpackage provides - is an *evaluator* that walks that tree against a runtime
context and returns a value.

* :func:`~bidsval.expr.evaluator.evaluate` walks an already-parsed tree.
* :func:`~bidsval.expr.evaluator.evaluate_string` parses then evaluates (parsing
  is memoized).

The helper functions and value coercions live in
:mod:`bidsval.expr.functions`.
"""

from __future__ import annotations

from .evaluator import EvaluationError, UnknownFunction, evaluate, evaluate_string

__all__ = ["evaluate", "evaluate_string", "EvaluationError", "UnknownFunction"]
